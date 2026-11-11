"""Evaluate the published ClaimLens NLI model on a fixed VitaminC test sample.

Usage (CPU is fine):
    python eval/run_eval.py --n 1500 --seed 42 --out eval/results

What it does
  1. Downloads the VitaminC test split (tals/vitaminc, parquet export) and draws
     a simple random sample of n rows with a fixed seed.
  2. Runs Zulfhagez/claimlens-deberta-v3-nli with the same input order the
     pipeline uses (premise = evidence, hypothesis = claim, max_length 512).
  3. Decodes the logits under BOTH candidate label orders (the Hub config
     id2label and the old in-code LABEL_MAP) so the correct order is decided by
     data, not by assumption.
  4. Runs two zero-shot NLI baselines with no fact-check fine-tuning:
     cross-encoder/nli-deberta-v3-base (same size and architecture, SNLI +
     MultiNLI) and facebook/bart-large-mnli (the repo's own `huggingface`
     backend), mapping entailment -> SUPPORTED, contradiction -> REFUTED,
     neutral -> NEI, plus a majority-class baseline.
  5. Writes metrics with bootstrap 95% confidence intervals, confusion
     matrices and per-example predictions.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["SUPPORTED", "REFUTED", "NEI"]
VITAMINC_TO_LABEL = {"SUPPORTS": "SUPPORTED", "REFUTES": "REFUTED", "NOT ENOUGH INFO": "NEI"}

CLAIMLENS_MODEL = "Zulfhagez/claimlens-deberta-v3-nli"
# Zero-shot baselines: off-the-shelf NLI models with no fact-check fine-tuning.
# entailment -> SUPPORTED, contradiction -> REFUTED, neutral -> NEI.
BASELINES = {
    "zero_shot_deberta_v3_base": "cross-encoder/nli-deberta-v3-base",  # same size/arch, SNLI+MultiNLI
    "zero_shot_bart_large_mnli": "facebook/bart-large-mnli",  # the repo's own `huggingface` backend
}
BASELINE_MAP = {"entailment": "SUPPORTED", "contradiction": "REFUTED", "neutral": "NEI"}

# Old in-code order (claimlens/models/nli_placeholder.py before this branch).
CODE_ORDER_OLD = {0: "SUPPORTED", 1: "REFUTED", 2: "NEI"}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sample(n: int, seed: int) -> pd.DataFrame:
    path = hf_hub_download(
        "tals/vitaminc",
        "default/test/0000.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    df = pd.read_parquet(path, columns=["unique_id", "label", "claim", "evidence", "page", "revision_type", "FEVER_id"])
    df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    df["gold"] = df["label"].map(VITAMINC_TO_LABEL)
    assert df["gold"].notna().all()
    return df


def predict_logits(model_id: str, premises, hypotheses, batch_size: int, cache_dir: Path, key: str):
    """Run (or load cached) logits so extra baselines do not force a full re-run."""
    cache = cache_dir / f"logits_{model_id.replace('/', '__')}_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["logits"], json.loads(str(z["id2label"])), float(z["seconds"]), True
    t0 = time.time()
    logits, id2label = _predict_logits(model_id, premises, hypotheses, batch_size)
    dt = time.time() - t0
    np.savez(cache, logits=logits, id2label=json.dumps(id2label), seconds=dt)
    return logits, id2label, dt, False


@torch.no_grad()
def _predict_logits(model_id: str, premises, hypotheses, batch_size: int = 16):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    out = []
    # Sort by length so batches pad less; restore order afterwards.
    order = np.argsort([len(p) + len(h) for p, h in zip(premises, hypotheses)])
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        enc = tok(
            [premises[j] for j in idx],
            [hypotheses[j] for j in idx],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        out.append((idx, model(**enc).logits.float().numpy()))
    logits = np.zeros((len(premises), model.config.num_labels), dtype=np.float32)
    for idx, lg in out:
        logits[idx] = lg
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return logits, id2label


def metrics(gold, pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class": {
            lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(LABELS)
        },
        "confusion_matrix": {
            "labels_rows_gold_cols_pred": LABELS,
            "matrix": confusion_matrix(gold, pred, labels=LABELS).tolist(),
        },
    }


def bootstrap(gold, preds: dict, n_boot: int, seed: int) -> dict:
    """Percentile bootstrap 95% CIs over examples, plus paired differences."""
    rng = np.random.default_rng(seed)
    gold = np.asarray(gold)
    preds = {k: np.asarray(v) for k, v in preds.items()}
    samples = {k: {"accuracy": [], "macro_f1": []} for k in preds}
    n = len(gold)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        g = gold[idx]
        for k, p in preds.items():
            samples[k]["accuracy"].append(accuracy_score(g, p[idx]))
            samples[k]["macro_f1"].append(f1_score(g, p[idx], labels=LABELS, average="macro", zero_division=0))
    ci = {}
    for k in preds:
        ci[k] = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in samples[k].items()}
    for k in preds:
        if k.startswith("zero_shot"):
            d = {}
            for m in ("accuracy", "macro_f1"):
                diff = np.array(samples["claimlens"][m]) - np.array(samples[k][m])
                d[m] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
            ci[f"claimlens_minus_{k}"] = d
    return ci


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=str, default="eval/results")
    ap.add_argument("--cache", type=str, default="eval/.cache", help="logit cache (not committed)")
    args = ap.parse_args()

    set_seeds(args.seed)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    df = load_sample(args.n, args.seed)
    prem, hyp = df["evidence"].tolist(), df["claim"].tolist()
    gold = df["gold"].tolist()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    key = f"vitaminc_test_n{args.n}_seed{args.seed}"
    runtimes = {}
    cl_logits, cl_id2label, runtimes["claimlens_inference"], _ = predict_logits(
        CLAIMLENS_MODEL, prem, hyp, args.batch_size, cache, key)
    cl_id2label = {int(k): v for k, v in cl_id2label.items()}
    cl_argmax = cl_logits.argmax(1)
    pred_config = [cl_id2label[i] for i in cl_argmax]
    pred_code_old = [CODE_ORDER_OLD[i] for i in cl_argmax]

    baseline_preds = {}
    for name, model_id in BASELINES.items():
        bl_logits, bl_id2label, runtimes[f"{name}_inference"], _ = predict_logits(
            model_id, prem, hyp, args.batch_size, cache, key)
        bl_id2label = {int(k): v.lower() for k, v in bl_id2label.items()}
        baseline_preds[name] = [BASELINE_MAP[bl_id2label[i]] for i in bl_logits.argmax(1)]

    majority = pd.Series(gold).mode()[0]
    pred_majority = [majority] * len(gold)

    cl_probs = torch.softmax(torch.tensor(cl_logits), -1).numpy()
    results = {
        "benchmark": "tals/vitaminc test split (parquet export), simple random sample",
        "n": len(df),
        "seed": args.seed,
        "gold_distribution": pd.Series(gold).value_counts().to_dict(),
        "input_format": "tokenizer(evidence, claim, truncation=True, max_length=512)",
        "models": {
            "claimlens": CLAIMLENS_MODEL,
            **{k: v + " (no fact-check fine-tuning)" for k, v in BASELINES.items()},
            "majority_class": f"always '{majority}'",
        },
        "claimlens_hub_id2label": {int(k): v for k, v in cl_id2label.items()},
        "label_order_check": {
            "decoded_with_hub_config_id2label": metrics(gold, pred_config),
            "decoded_with_old_code_LABEL_MAP": metrics(gold, pred_code_old),
        },
        "metrics": {
            "claimlens": metrics(gold, pred_config),
            **{k: metrics(gold, v) for k, v in baseline_preds.items()},
            "majority_class": metrics(gold, pred_majority),
        },
        "bootstrap_95ci": bootstrap(
            gold,
            {"claimlens": pred_config, **baseline_preds, "majority_class": pred_majority},
            args.n_boot,
            args.seed,
        ),
        "n_boot": args.n_boot,
        "runtime_seconds": {
            **{k: round(v, 1) for k, v in runtimes.items()},
            "note": "inference times are from the run that produced each cached logit file",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "cpu_threads": args.threads,
            "platform": platform.platform(),
        },
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    with open(out / "predictions.jsonl", "w") as fh:
        for i, row in df.iterrows():
            fh.write(
                json.dumps(
                    {
                        "unique_id": row["unique_id"],
                        "page": row["page"],
                        "revision_type": row["revision_type"],
                        "claim": row["claim"],
                        "evidence": row["evidence"],
                        "gold": gold[i],
                        "claimlens_pred": pred_config[i],
                        "claimlens_probs": {cl_id2label[j]: round(float(cl_probs[i, j]), 4) for j in range(3)},
                        **{f"{k}_pred": v[i] for k, v in baseline_preds.items()},
                    }
                )
                + "\n"
            )
    print(json.dumps({k: results[k] for k in ("n", "gold_distribution", "runtime_seconds")}, indent=2, default=str))
    for k, v in results["metrics"].items():
        print(f"{k:20s} acc={v['accuracy']:.4f} macroF1={v['macro_f1']:.4f}")
    for k, v in results["label_order_check"].items():
        print(f"{k:40s} acc={v['accuracy']:.4f} macroF1={v['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
