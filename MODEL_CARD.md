# Model card: Zulfhagez/claimlens-deberta-v3-nli

## Model

- Hub id: [`Zulfhagez/claimlens-deberta-v3-nli`](https://huggingface.co/Zulfhagez/claimlens-deberta-v3-nli)
- Architecture: `DebertaV2ForSequenceClassification` (DeBERTa-v3 base: 12 layers, hidden size 768, 184M parameters), saved with transformers 4.57.6.
- Task: three-way claim verification. Input is a pair (evidence, claim); output is one of three labels.
- Label order (from `config.json`, verified empirically, see below):

| id | label | ClaimLens verdict |
|---|---|---|
| 0 | SUPPORTED | `SUPPORTED` |
| 1 | NEI | `NOT_ENOUGH_INFO` |
| 2 | REFUTED | `REFUTED` |

- Input format used by the pipeline and the evaluation: `tokenizer(evidence, claim, truncation=True, max_length=512)`.

## Intended use

The verifier stage of the ClaimLens pipeline: score each retrieved web
snippet against one atomic claim, then aggregate by weighted vote with
relevance and credibility, with LLM cross-checks on uncertain cases. It is
meant as one signal inside that pipeline, reviewed by a person, not as a
standalone judge of truth.

Out of scope: medical, legal or financial decisions; claims that need
multi-hop reasoning across several documents; languages other than English;
long documents (inputs past 512 tokens are truncated).

## Training data

Not documented. The Hub repository has no card and the training notebook is
not in this repository. `training_args.bin` shows only the output directory
(`verifier_out_v12`), checkpoint selection on `macro_f1`, and seed 42. The
label names (SUPPORTED / NEI / REFUTED) suggest a FEVER-style source. Until
the training set is recorded, assume possible overlap with public FEVER-family
benchmarks.

## Evaluation

Full method, slices and failure analysis: [`eval/README.md`](eval/README.md).

Benchmark: VitaminC test split ([tals/vitaminc](https://huggingface.co/datasets/tals/vitaminc)),
simple random sample of **n = 3,000** pairs (seed 42) out of 55,197; gold mix
SUPPORTED 1,556, REFUTED 1,040, NEI 404. CPU, 11 November 2026.
Intervals are 95% percentile bootstrap (1,000 resamples).

| Model | Accuracy (95% CI) | Macro F1 (95% CI) |
|---|---|---|
| This model | 65.8% (64.0 to 67.5) | 0.625 (0.607 to 0.642) |
| `cross-encoder/nli-deberta-v3-base`, zero-shot | 59.4% (57.8 to 61.1) | 0.572 (0.555 to 0.589) |
| `facebook/bart-large-mnli`, zero-shot | 59.1% (57.5 to 60.9) | 0.561 (0.544 to 0.578) |
| Majority class | 51.9% (50.1 to 53.6) | 0.228 (0.222 to 0.233) |

Per class for this model: SUPPORTED F1 0.788 (P 0.895, R 0.704), REFUTED F1
0.650 (P 0.887, R 0.513), NEI F1 0.435 (P 0.292, R 0.849).

## Limitations and known failure modes

- **Over-predicts NEI.** It predicted NEI 1,173 times against 404 gold NEI;
  830 of 1,027 errors are SUPPORTED or REFUTED pairs called NEI. High
  precision when it does commit (0.89 to 0.90 for SUPPORTED and REFUTED).
- **Entity or attribute swaps** (a different airline, conference or genre in
  the evidence) tend to come out as NEI rather than REFUTED.
- **Near-miss numbers** ("over $55.05 million" against "$55.1 million") are
  misjudged in both directions.
- **Paraphrase and coreference** ("screens" versus "theaters"; "this festival"
  when the entity is named only in the claim) are often missed with
  single-sentence evidence.
- The gain over zero-shot NLI comes from real Wikipedia revisions and claims
  with numbers. On synthetic contrastive edits and on claims without digits
  it is level with or behind the zero-shot baselines.
- Measured on one evidence sentence per claim, not on the full pipeline.
- Training data unknown (see above), so in-distribution versus out-of-
  distribution status is unknown.

## Label-order bug (fixed November 2026)

Before branch `eval-2026-11`, the pipeline decoded index 1 as REFUTED and 2 as
NEI, the reverse of `config.json`. On the same 3,000 examples that decoding
scores 51.1% accuracy and 0.405 macro F1, against 65.8% and 0.625 with the
config order, so the config is right and the code was wrong. The verifier now
reads the order from the checkpoint and a test pins it.
