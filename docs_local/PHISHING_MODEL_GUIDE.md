# Phishing Detector — Build Guide

> Handoff for whoever builds the phishing model (Faiz, per CLAUDE.md §15). The
> **priority classifier is done end-to-end** (v1.0 English, v1.1 Indonesian, both
> published to HF). The phishing detector follows the same pipeline, with the
> §3-specific differences called out below. This doc lives in `docs_local/` — it
> is OUR working note, not the shared `docs/` submodule (that's upstream-owned).

---

## 0. Read first
- `CLAUDE.md` — repo rules (LOCKED stack, branch/commit conventions, do-nots).
- `docs/ML_PIPELINE.md` §3 — the phishing contract (architecture, gates, threshold,
  adversarial testing). **This is authoritative; this guide just operationalizes it.**
- `docs/DATA_MODELS.md` — the backend fields you feed: `is_phishing` (bool),
  `phishing_score` (float 0–1).

## 1. What's already built (REUSE, don't rewrite)

The priority work left a lot you can lean on:

| Already exists | Where | Use it for |
|---|---|---|
| `build_phishing_input()` | `src/data/preprocess.py` | The §3 input format `FROM: … [SEP] SUBJECT: … [SEP] BODY: …` — already implemented + tested. Sender is intentionally NOT masked (carries signal). |
| `PHISHING_LABELS`, `PHISHING_LABEL2ID` | `src/utils/constants.py` | `("legit", "phishing")`, phishing = index 1. |
| `clean_text()` | `src/data/preprocess.py` | HTML strip, URL/email masking — same cleaning as priority. |
| `stratified_split()` | `src/data/splits.py` | Reuse as-is for train/val/test. |
| Export + HF upload | `src/exporter/` | `export.py` already knows phishing needs `threshold.json` (`PHISHING_ONLY`). `upload_hf.py` handles `priormail-phishing`. Pass `--phishing` to the CLI. |
| Training skeleton | `src/training/train_priority.py` | Copy its structure (config-driven, seeded, SHA-stamped, wandb, weighted CE, early stopping). |
| Eval skeleton | `src/eval/eval_priority.py` + `benchmarks.py` | Copy the gate-checking + latency pattern. |
| Latency benchmark | `src/eval/benchmarks.py` | `measure_latency_ms` is model-agnostic — works as-is. |

**Stubs waiting for you:** `src/training/train_phishing.py`, `src/eval/eval_phishing.py`
(both raise `NotImplementedError` with notes). `select_threshold()` is already
stubbed in `eval_phishing.py`.

## 2. Decisions you MUST make first (do not skip — §11 open items)

These aren't coding choices; discuss with Insan and record in the config/PR.

1. **Base model (§11): IndoBERT vs multilingual BERT.**
   - The phishing dataset (`ealvaradob/phishing-dataset`) is **English-heavy**.
     IndoBERT is Indonesian-pretrained and may underperform on English phishing.
   - Options: `indobenchmark/indobert-base-p1` (consistency with priority) vs
     `bert-base-multilingual-cased` (handles English + Indonesian).
   - **Recommendation to evaluate:** try multilingual first given the data, but
     confirm with Insan — it diverges from the LOCKED priority base model.
2. **Negative (legit) class source.** The dataset is phishing-positive-heavy.
   §3 says use "public corporate email corpora" for legit examples. Decide which.
3. **Class imbalance handling (§11).** Weighted CE (as priority does) vs focal
   loss vs resampling. Phishing recall must be very high, so weight the phishing
   class heavily.

## 3. The §3 differences from priority (this is the important part)

The phishing model is NOT just "priority with 2 classes." Key differences:

| Aspect | Priority | Phishing (§3) |
|---|---|---|
| Classes | 4 | 2 (`legit`, `phishing`) |
| Input | `{subject} [SEP] {body}` | `FROM: {sender} [SEP] SUBJECT: … [SEP] BODY: …` |
| Decision rule | argmax | **threshold on phishing-prob, NOT 0.5** |
| Primary gate | macro F1 ≥ 0.80 | **recall ≥ 0.95** (false negatives are worst) |
| Secondary gate | per-class recall ≥ 0.65 | **precision ≥ 0.80** |
| Extra artifact | — | **`threshold.json`** shipped with checkpoint |
| Extra eval | confusion matrix | **adversarial test set** (Faiz owns) → `eval/results/adversarial/` |

**The threshold is the crux.** Pick the threshold on the VALIDATION set that
achieves recall ≥ 0.95 while maximizing precision (`select_threshold()` stub).
Persist it as `threshold.json` and APPLY it at inference — the backend reads it.

## 4. Step-by-step

1. **Branch:** `feat/phishing-model` (CLAUDE.md §10).
2. **Loader** — add a `load_phishing_dataset()` in `src/data/loaders.py` (the stub
   exists). Load `ealvaradob/phishing-dataset` + legit negatives; map to
   `PHISHING_LABEL2ID`. Inspect it first in a notebook (`notebooks/NN_phishing_overview.ipynb`)
   like we did for priority — check columns, label balance, language mix.
3. **Prepare** — either extend `src/data/prepare.py` or add `prepare_phishing()`.
   Build `model_input` via `build_phishing_input()`. Stratified split (reuse).
4. **Config** — `configs/phishing_v1.yaml`. Copy `priority_v1.yaml`'s shape; set
   `model_type: phishing`, `num_labels: 2`, the chosen base model, heavy phishing
   class weight. (See `docs/ML_PIPELINE.md` §3 training protocol.)
5. **Train** — fill `src/training/train_phishing.py`. Copy `train_priority.py`'s
   structure (seed, SHA stamp, wandb, weighted CE, early stopping). Monitor
   **recall** for early stopping, not macro F1.
6. **Threshold + Eval** — fill `select_threshold()` and `evaluate()` in
   `eval_phishing.py`. Gates: recall ≥ 0.95, precision ≥ 0.80, p95 < 500ms CPU.
   Save `threshold.json`. Run Faiz's adversarial set separately.
7. **Tests** — mirror `tests/test_eval.py`: pure gate logic + threshold selection,
   testable without a GPU. Keep `make test` green.
8. **Train on Colab** — same notebook flow as priority (see below). Needs a GPU.
9. **Eval locally**, write `model_cards/phishing_v1.0.md`, package + publish:
   `python -m src.exporter.export --checkpoint checkpoints/phishing_v1 --phishing --hf-org <org> --version v1.0`

## 5. Colab training (copy the priority flow)

Identical to the priority Colab run, just swap the config. Key gotchas already
solved (don't re-learn them the hard way):
- Set **Runtime → T4 GPU**; confirm `torch.cuda.is_available()`.
- `pip install` the libs directly (incl. **`sentencepiece`**); use `PYTHONPATH=.`
  (don't `pip install -e .` — the Python 3.11 pin conflicts with Colab's 3.12).
- **`%env WANDB_MODE=offline`** on its own line, NO trailing comment (a comment
  breaks it — cost us an hour).
- `git checkout <SHA>` to pin a reproducible commit (the run auto-records it).
- Download the checkpoint as a **zip** (`files.download`), not via Drive (flaky).
- HF upload: the uploader now sets `HF_HUB_DISABLE_XET=1` itself (Xet stalls on
  large files) — no action needed.

## 6. Definition of done (CLAUDE.md §12)
Config committed · wandb run · eval gates pass (recall ≥ 0.95, precision ≥ 0.80,
p95 < 500ms) · `threshold.json` saved · adversarial set checked · model card ·
uploaded + version-pinned · Syafiq notified · `ML_PIPELINE.md` §10 row added.

## 7. Known cross-cutting issue (affects phishing too)
**Free-tier Supabase caps uploads at 50MB; our checkpoints are ~475MB.** The
priority models are published to **HuggingFace Hub** instead (`hf://`), which
diverges from the §6 Supabase contract. This is unresolved — Syafiq must decide
(backend loads `hf://`, upgrade Supabase, or quantize). The phishing model will
hit the same wall. Use `--hf-org` to publish to HF for now.

---

*Reference implementation: the priority classifier (`*_priority.py`, `configs/priority_v1.yaml`,
`model_cards/priority_v1.1.md`). When in doubt, copy what it does and adapt per §3.*
