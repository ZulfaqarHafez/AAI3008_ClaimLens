"""Slice metrics and draw failure candidates from eval/results/predictions.jsonl.

Usage: python eval/analyse.py --results eval/results
Writes slices.json and failure_candidates.jsonl. The failure categories in
RESULTS.md were assigned by hand from these candidates (first ones per
confusion type after a seed-42 shuffle).
"""
import argparse, collections, json, random, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="eval/results")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
R = Path(a.results)
P = [json.loads(l) for l in open(R / "predictions.jsonl")]
models = ["claimlens_pred", "zero_shot_deberta_v3_base_pred", "zero_shot_bart_large_mnli_pred"]


def acc(xs):
    return {m: round(sum(p["gold"] == p[m] for p in xs) / len(xs), 4) for m in models} | {"n": len(xs)}


slices = {
    "revision_type=real": acc([p for p in P if p["revision_type"] == "real"]),
    "revision_type=synthetic": acc([p for p in P if p["revision_type"] == "synthetic"]),
    "claim_has_digit": acc([p for p in P if re.search(r"\d", p["claim"])]),
    "claim_has_no_digit": acc([p for p in P if not re.search(r"\d", p["claim"])]),
    "claimlens_pred_distribution": dict(collections.Counter(p["claimlens_pred"] for p in P)),
    "gold_distribution": dict(collections.Counter(p["gold"] for p in P)),
    "claimlens_confusions": dict(collections.Counter(
        f"{p['gold']}->{p['claimlens_pred']}" for p in P if p["gold"] != p["claimlens_pred"])),
}
(R / "slices.json").write_text(json.dumps(slices, indent=2))

F = [p for p in P if p["gold"] != p["claimlens_pred"]]
random.seed(a.seed)
random.shuffle(F)
with open(R / "failure_candidates.jsonl", "w") as fh:
    for p in F[:60]:
        fh.write(json.dumps(p) + "\n")
print(json.dumps(slices, indent=2))
