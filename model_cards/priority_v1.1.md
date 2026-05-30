# priority v1.1

## Intended use
Predicts an email's priority — one of `urgent | high | normal | low` — from its
subject and body, for the PriorMail inbox-triage product. Consumed by
`prior-mail-backend` at inference (ML_PIPELINE.md §6).

**Change from v1.0:** trained on the **Indonesian** translation of the dataset
instead of the English original. This is the meaningful upgrade — the model now
processes Indonesian text natively, matching the product domain. v1.0 (English)
remains published and unchanged; this is a separate version.

**Still a proxy model.** The labels are unchanged: the same 6 topical categories
mapped to priority (`verify_code→urgent`, `updates→high`, …). So this learns
*topic→priority on Indonesian text*, not yet *true urgency understanding*. The
language gap is fixed; the topic-vs-urgency proxy is not. Promotion to production
is still a deliberate team decision (§11).

## Training data
- `chairulridjal/high-accuracy-email-classifier-indonesian` — 13,477 emails
  (Indonesian translation of `jason23322/high-accuracy-email-classifier`; same 6
  categories, same row count). Mapping fixed in `src/data/loaders.py`. Re-split
  from scratch, stratified, fixed seed 42: train 9,433 / val 2,022 / test 2,022.
- Internal human-labeled set (§7): **NOT yet included** — still the key gap for a
  model that judges real urgency rather than topic.

## Evaluation
- **Test set:** held-out stratified 15% (2,022 Indonesian emails), fixed split,
  never seen in training (`data/processed/priority` → `test`).
- **Macro F1: 0.988** on the test set (gate ≥ 0.80 ✅)
- **Per-class recall** on the test set (gate ≥ 0.65 ✅, all pass):

  | class | recall (test) |
  |---|---|
  | urgent | 0.997 |
  | high | 0.982 |
  | normal | 0.987 |
  | low | 0.988 |

- **Inference latency p95: 49 ms** per email on CPU (gate < 500 ms ✅, measured
  over 200 test emails on a local CPU as a Render proxy).
- Full metrics + confusion matrix: `eval/results/priority/eval_report.json`.

## Known limitations
- **Topic→urgency proxy, not true urgency.** The 0.988 reflects how separable the
  source *categories* are (OTP codes vs promos), now in Indonesian. Real urgency
  judgement on genuine work email is unmeasured.
- The Indonesian data is a *translation* of English emails, not natively-authored
  Indonesian work mail — some phrasing may be translationese.
- Comparable F1 to v1.0 (0.988) is expected: same labels/rows, different language.
  The gain is domain fit (Indonesian comprehension), not a higher score.
- Long emails truncated to 512 tokens from the end (subject preserved).

## Threshold (phishing only)
N/A — priority classifier (argmax over 4 classes, no threshold).

## Trained by
Insan, 2026-05-30 (07:41 WIB), on Google Colab (T4 GPU, wandb offline).
git SHA: **3d1de42** (clean tree — `_git_dirty: false`, auto-recorded in
`training_config.yaml`). Base model: `indobenchmark/indobert-base-p1` · config:
`configs/priority_v1.yaml`.

> ✅ **Reproducible:** trained from committed SHA `3d1de42` on a clean tree.
> `make data --indonesian && make train config=configs/priority_v1.yaml`
> reproduces it within the §8 tolerance, modulo GPU nondeterminism.
