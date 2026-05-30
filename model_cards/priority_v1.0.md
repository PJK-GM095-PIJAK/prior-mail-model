# priority v1.0

## Intended use
Predicts an email's priority — one of `urgent | high | normal | low` — from its
subject and body, for the PriorMail inbox-triage product. Consumed by
`prior-mail-backend` at inference time (see ML_PIPELINE.md §6).

**This is a BASELINE (v0-quality) model.** It was trained only on a public
*English, topical-folder* dataset whose categories were mapped to priority as a
proxy. It has **not** been domain-adapted to Indonesian work email yet. Treat its
output as a starting signal, not a finished urgency judgment. Promotion to
production is a deliberate team decision (ML_PIPELINE.md §11), not implied by the
passing gates below.

## Training data
- `jason23322/high-accuracy-email-classifier` — 13,477 emails, 6 topical
  categories mapped to 4 priority classes (mapping + rationale fixed in
  `src/data/loaders.py`). Re-split from scratch, stratified, fixed seed 42:
  train 9,433 / validation 2,022 / test 2,022.
- Internal Indonesian labeled set (ML_PIPELINE.md §7): **NOT yet included** —
  this is the key gap and the next priority for a v1.1.

## Evaluation
- **Test set:** held-out stratified 15% (2,022 emails), fixed split, never seen
  in training (`data/processed/priority` → `test`).
- **Macro F1: 0.989** (gate ≥ 0.80 ✅)
- **Per-class recall** (gate ≥ 0.65 ✅, all pass):

  | class | recall (val) |
  |---|---|
  | urgent | 0.997 |
  | high | 0.979 |
  | normal | 0.994 |
  | low | 0.991 |

- **Inference latency p95: 167 ms** per email on CPU (gate < 500 ms ✅, measured
  over 200 test emails on a Colab CPU as a Render proxy).
- Full metrics + confusion matrix: `eval/results/priority/eval_report.json`.

## Known limitations
- Trained on **English topical-folder data**, not Indonesian work email — the
  0.99 macro F1 reflects how *separable the source categories are*, not true
  urgency understanding. Real-world performance on the target domain is unknown.
- `urgent` / `high` priority is inferred from `verify_code` / `updates`
  categories — a coarse proxy that likely over-prioritizes routine updates.
- No `[URL]`/`[EMAIL]`-heavy Indonesian text in training; preprocessing was
  validated on this dataset only.
- Long emails are truncated to 512 tokens from the end (subject preserved).

## Threshold (phishing only)
N/A — this is the priority classifier (argmax over 4 classes, no threshold).

## Trained by
Insan, 2026-05-30 (~01:45 WIB), on Google Colab (T4 GPU, wandb offline).
git SHA: **unknown** — the Colab session was recycled before the SHA was
captured, and training (~01:45) predates all current commits (earliest 05:36),
so the exact code state cannot be reconstructed. wandb run:
offline-run-20260530_014553-zrld2xgv.
Base model: `indobenchmark/indobert-base-p1` · config: `configs/priority_baseline.yaml`

> ⚠️ **Provenance gap (reproducibility, ML_PIPELINE.md §8):** this artifact was
> trained from an uncaptured code state. It is fine as an exploratory baseline,
> but **before any real release this model should be RE-TRAINED from a known,
> committed SHA** so the run is reproducible. Future runs will record the SHA
> automatically (train_priority.py writes it into training_config.yaml).
