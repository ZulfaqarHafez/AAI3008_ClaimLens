# ClaimLens NLI evaluation (November 2026)

This folder holds a reproducible evaluation of the published verifier model
[`Zulfhagez/claimlens-deberta-v3-nli`](https://huggingface.co/Zulfhagez/claimlens-deberta-v3-nli)
(DeBERTa-v3 base, 184M parameters, last updated on the Hub 6 March 2026).
It evaluates the NLI model on its own, one claim and one evidence sentence per
example. It does not evaluate the full agentic pipeline (search, gating, LLM
cross-check).

## Headline

VitaminC test split, simple random sample, **n = 3,000**, seed 42.
95% confidence intervals are percentile bootstrap over examples (1,000 resamples).

| Model | Accuracy | 95% CI | Macro F1 | 95% CI |
|---|---|---|---|---|
| **ClaimLens** (`Zulfhagez/claimlens-deberta-v3-nli`) | **65.8%** | 64.0 to 67.5 | **0.625** | 0.607 to 0.642 |
| Zero-shot `cross-encoder/nli-deberta-v3-base` (same size, SNLI + MultiNLI) | 59.4% | 57.8 to 61.1 | 0.572 | 0.555 to 0.589 |
| Zero-shot `facebook/bart-large-mnli` (the repo's `huggingface` backend) | 59.1% | 57.5 to 60.9 | 0.561 | 0.544 to 0.578 |
| Majority class (always SUPPORTED) | 51.9% | 50.1 to 53.6 | 0.228 | 0.222 to 0.233 |

Paired bootstrap difference, ClaimLens minus baseline (95% CI):

| Baseline | Accuracy (points) | Macro F1 |
|---|---|---|
| nli-deberta-v3-base | +4.5 to +8.0 | +0.035 to +0.070 |
| bart-large-mnli | +4.7 to +8.4 | +0.044 to +0.082 |

So the README claim that the fine-tuned model beats `bart-large-mnli` holds on
this benchmark, by about 7 accuracy points, and the interval excludes zero.

## Why VitaminC

ClaimLens checks a claim against a retrieved evidence snippet and returns
SUPPORTED, REFUTED or NOT_ENOUGH_INFO. VitaminC
([tals/vitaminc](https://huggingface.co/datasets/tals/vitaminc), Schuster et al.,
NAACL 2021) has exactly that shape: a claim, one Wikipedia evidence sentence,
and the labels SUPPORTS / REFUTES / NOT ENOUGH INFO. It is harder than plain
FEVER because many pairs are contrastive (the same claim with a revised
sentence that flips the label), which is the failure mode that matters for
web evidence. MNLI or ANLI would test generic entailment rather than
claim checking. The parquet export of the test split (55,197 rows) was sampled
without stratification, so the class mix is the natural one:
SUPPORTED 1,556, REFUTED 1,040, NEI 404.

## Per-class results (ClaimLens)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTED | 0.895 | 0.704 | 0.788 | 1,556 |
| REFUTED | 0.887 | 0.513 | 0.650 | 1,040 |
| NEI | 0.292 | 0.849 | 0.435 | 404 |

Confusion matrix (rows gold, columns predicted):

| gold \ pred | SUPPORTED | REFUTED | NEI |
|---|---|---|---|
| SUPPORTED | 1,096 | 40 | 420 |
| REFUTED | 96 | 534 | 410 |
| NEI | 33 | 28 | 343 |

The dominant error is **over-predicting NEI**: the model says NEI 1,173 times
against 404 gold NEI. 830 of its 1,027 errors are SUPPORTED or REFUTED pairs
called NEI. When it does commit to SUPPORTED or REFUTED it is usually right
(precision 0.89 to 0.90). In the pipeline this is the conservative direction,
and it is exactly what Trigger B (NEI with high relevance goes to the LLM
cross-check) is there to catch.

Slices (accuracy; from `results/slices.json`):

| Slice | n | ClaimLens | nli-deberta-v3-base | bart-large-mnli |
|---|---|---|---|---|
| Real Wikipedia revisions | 1,887 | 64.4% | 52.4% | 50.4% |
| Synthetic contrastive edits | 1,113 | 68.1% | 71.4% | 73.9% |
| Claim contains a digit | 1,363 | 66.1% | 52.3% | 50.3% |
| Claim has no digit | 1,637 | 65.5% | 65.4% | 66.5% |

The whole advantage comes from real revisions and claims with numbers. On
synthetic edits and on claims without digits the fine-tuned model is level
with or behind the zero-shot models. These slices were chosen after looking
at the failures, so treat them as descriptive, not as tested hypotheses.

## Label order: finding and fix

The model's `config.json` on the Hub says:

```
id2label = {0: "SUPPORTED", 1: "NEI", 2: "REFUTED"}
```

The pipeline code (`claimlens/models/nli_placeholder.py`, `ClaimLensVerifier.LABEL_MAP`)
and the README decoded index 1 as REFUTED and index 2 as NEI. The two maps
agree on SUPPORTED and swap REFUTED and NEI.

The data decides which is right. Decoding the same logits both ways on the
same 3,000 examples:

| Decoding | Accuracy | Macro F1 |
|---|---|---|
| Hub config `id2label` (0 SUPPORTED, 1 NEI, 2 REFUTED) | 65.8% | 0.625 |
| Old in-code `LABEL_MAP` (0 SUPPORTED, 1 REFUTED, 2 NEI) | 51.1% | 0.405 |

Under the old map, 534 gold REFUTED examples come out as NEI and 343 gold NEI
examples come out as REFUTED, which is the signature of a swap. The config is
correct and the code was wrong. Effect in production before this fix: every
time the NLI model meant NEI the pipeline counted a REFUTED vote, and every
time it meant REFUTED the pipeline counted NEI. Given the NEI over-prediction
above, that turned many "not enough evidence" cases into refutations. The
docstring note that DeBERTa "commonly misclassifies as REFUTED" near-miss
numbers was most likely this swap.

Fix (on this branch): `LABEL_MAP` now matches the config, and `_load_model`
rebuilds the map from the checkpoint's own `id2label` so a future retrain with
a different order cannot silently swap verdicts. A regression test
(`tests/test_pipeline.py::test_claimlens_label_map_matches_published_config`)
pins the order. The README label table is corrected.

## Ten failures, by category

Drawn from `results/failure_candidates.jsonl` (all ClaimLens errors shuffled
with seed 42, then the first examples of each confusion type, in proportion to
how common that confusion is). Categories are my judgement after reading each
pair.

**A. Entity or attribute swapped, model says NEI instead of REFUTED (3)**

| id | Claim | Evidence (excerpt) | Gold | Pred (p) |
|---|---|---|---|---|
| 5ed50882c9e77c000848ed50_4 | The Airbus A380 entered commercial service with Singapore Airlines. | ...entered commercial service on 25 October 2007 with American Airlines. | REFUTED | NEI (0.64) |
| 5ed50881c9e77c000848e42c_4 | Northwestern University is the only private university of the Big Three Conference. | ...founding member of the Big Ten Conference and remains the only private university in the conference. | REFUTED | NEI (0.66) |
| 5ea2d97dc9e77c0009cdaeaa_3 | Immigrant Union plays folk music. | DeBoer now fronts the country band Immigrant Union. | REFUTED | NEI (0.73) |

The model sees that the evidence names a different airline, conference or
genre but treats it as missing information rather than a contradiction. The
genre case is arguably NEI (country and folk overlap), which shows how thin
the REFUTED/NEI line is.

**B. Numeric comparison (2)**

| id | Claim | Evidence (excerpt) | Gold | Pred (p) |
|---|---|---|---|---|
| 5ec7d50dc9e77c0008441d74_4 | Happy Death Day collected over $55.05 million in the US. | ...has grossed $55.1 million in the United States and Canada... | SUPPORTED | REFUTED (0.77) |
| 5ee3932ac9e77c0008cc9bbf_3 | More than 1,010 cases of COVID-19 were confirmed in Bosnia and Herzegovina. | As of 12 April 2020 ... there were 1000 confirmed coronavirus cases... | REFUTED | SUPPORTED (0.44) |

"Over X" and "more than X" against a nearby number are decided the wrong way
in both directions.

**C. Paraphrase or coreference not resolved (2)**

| id | Claim | Evidence (excerpt) | Gold | Pred (p) |
|---|---|---|---|---|
| 5ee3949bc9e77c0008cd0063_4 | 2,454 theaters across North America screened P.S. I Love You. | The film opened on 2,454 screens in North America... | SUPPORTED | NEI (0.97) |
| 5ec55990c9e77c000842d522_3 | A forum for women's freedom of expression is symbolized by Teej. | ...this festival these day symbolizes a forum for women's freedom of expressions. | SUPPORTED | NEI (0.97) |

"Screens" versus "theaters", and "this festival" with Teej named only in the
claim. Single-sentence evidence without the page title hurts here; the
pipeline's context agent may help in practice.

**D. Probable benchmark label noise (3)**

| id | Claim | Evidence (excerpt) | Gold | Pred (p) |
|---|---|---|---|---|
| 5ed50881c9e77c000848e622_4 | Simon Pegg is a writer. | Pegg, a full-time screenplay writer, is responsible for the screenplay of... | REFUTED | SUPPORTED (0.999) |
| 5ee39419c9e77c0008ccf73f_2 | Ronaldinho was born in Porto Alegre, Brazil. | Ronaldinho was born in the city of Porto Alegre ... Brazil. | NEI | SUPPORTED (0.99) |
| 5ee39419c9e77c0008ccf8eb_4 | More than 500 blacks participated in the Selma to Montgomery marches of 1964. | ...John Lewis led 50 black citizens to the courthouse... | SUPPORTED | NEI (0.94) |

In these the model's answer looks better than the gold label. Some of the
measured error is therefore benchmark noise, so the true accuracy is probably
a little higher than reported, but this sample is too small to estimate by how
much.

## Limitations of this evaluation

- **Training data is undocumented.** The Hub repo has no model card and the
  training notebook is not in this repository; `training_args.bin` only shows
  output dir `verifier_out_v12`, `metric_for_best_model=macro_f1`, seed 42. If
  the model was trained on VitaminC or FEVER, this is an in-distribution test.
  VitaminC test is disjoint from VitaminC train, but contamination with the
  test split cannot be ruled out without the training script.
- One evidence sentence per claim. The pipeline votes over up to 5 web
  snippets and then applies gates and an LLM cross-check, so pipeline accuracy
  will differ.
- Zero-shot baselines are run as pair classifiers (premise = evidence,
  hypothesis = claim). The repo's `huggingface` backend instead calls
  `bart-large-mnli` through the zero-shot-classification pipeline with the
  label names as candidate labels, which is a weaker use of that model.
- n = 3,000 of 55,197 test rows to fit a CPU budget.

## Reproduce

```
python -m venv .venv && . .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.57.6 sentencepiece protobuf pandas pyarrow scikit-learn huggingface_hub==0.36.2
python eval/run_eval.py --n 3000 --seed 42 --n-boot 1000 --out eval/results
python eval/analyse.py --results eval/results
```

Run on 11 November 2026, CPU only (4 threads, Linux x86_64), Python 3.13.16,
torch 2.15.0+cpu, transformers 4.57.6, numpy 2.5.4, pandas 3.1.1,
scikit-learn 1.9.1. Inference time for 3,000 pairs: ClaimLens 226 s,
nli-deberta-v3-base 169 s, bart-large-mnli 478 s; wall clock for the whole
run 15 min 1 s.

Files: `results/metrics.json` (all metrics, CIs, confusion matrices, both
label decodings), `results/predictions.jsonl` (per-example predictions and
probabilities), `results/slices.json`, `results/failure_candidates.jsonl`.
