# prior-mail-model — AI Coding Guide

> This file is the primary context for any LLM coding assistant (Claude Code, Cursor, Copilot, etc.) working on **this ML repo**. Read it before writing code.

---

## 1. Repo Role

This is the **machine learning** repo of PriorMail. It owns:

- Fine-tuning **DistilBERT** for email priority classification (4 classes)
- Fine-tuning a **phishing detector** (binary)
- Dataset preparation, augmentation, and labeling tooling
- Evaluation pipelines and benchmark suites
- Checkpoint export to Supabase Storage (consumed by `prior-mail-backend`)

This repo **does not run in production**. It runs locally, on Colab/Kaggle for training, and produces artifacts.

### Sibling repos

| Repo | Role |
|---|---|
| [`prior-mail-backend`](https://github.com/PJK-GM095-PIJAK/prior-mail-backend) | Loads our trained checkpoints at inference time |
| [`prior-mail-frontend`](https://github.com/PJK-GM095-PIJAK/prior-mail-frontend) | No direct interaction |
| [`prior-mail-docs`](https://github.com/PJK-GM095-PIJAK/prior-mail-docs) | Shared specs — mounted as submodule at `./docs/` |

Team ID: **PJK-GM095**

---

## 2. Shared Specs (Submodule)

Cross-repo specs live in `./docs/` (git submodule pointing to `prior-mail-docs`).

**If `./docs/` is empty, run:**
```bash
git submodule update --init --recursive
```

**Files you must check before coding:**
- `./docs/ML_PIPELINE.md` — model architecture, training protocol, eval gates, checkpoint format
- `./docs/DATA_MODELS.md` — to know the exact priority enum + phishing field schema the backend expects
- `./docs/SECURITY.md` — for handling user-derived training data

> **Current status (2026-06-09):** `./docs/` only contains `README.md` — the spec files above are not yet created in `prior-mail-docs`. Use `./docs_local/` as the interim reference instead (see below).

**Local interim guides (`./docs_local/`):**
- `./docs_local/BACKEND_INTEGRATION_GUIDE.md` — how the backend loads the priority model from HuggingFace Hub
- `./docs_local/PHISHING_MODEL_GUIDE.md` — phishing detector build guide

**Updating the submodule:**
```bash
git submodule update --remote docs
git add docs && git commit -m "chore: bump docs submodule"
```

> **`./docs/ML_PIPELINE.md` is owned by this repo's team** (Insan + Faiz). If you change architecture, eval protocol, or checkpoint format, open the PR in `prior-mail-docs` first.

---

## 3. Quick Start for LLM Assistants

Before writing code:

1. Read this entire file.
2. Read `./docs/ML_PIPELINE.md` — the contract with backend.
3. For training runs: confirm `wandb` is configured and disk space is sufficient.

When uncertain — ask, don't assume:

- Hyperparameter choice not in `ML_PIPELINE.md` → propose, don't fire off a training run
- Dataset modification → discuss; reproducibility matters
- New model architecture → confirm with Insan
- Anything that costs GPU hours → confirm budget first

---

## 4. Tech Stack (LOCKED)

### Runtime
- **Python:** 3.11 — dependency manager: `uv`
- **CUDA:** 12.x (when on GPU)

### ML Core
- **PyTorch 2.x**
- **Hugging Face Transformers** — latest stable
- **Hugging Face Datasets** — for loading and processing
- **`accelerate`** — for multi-GPU / mixed-precision training
- **`evaluate`** — for standard metrics

### Models
- **`distilbert-base-uncased`** — base model for priority classifier (English)
- **`distilbert-base-uncased`** or multilingual DistilBERT — base for phishing detector (decision in `ML_PIPELINE.md`)
- Tokenizer: use the matching one from the base model, never mix

### Experiment Tracking
- **Weights & Biases (`wandb`)** — project: `priormail`
  - Run names: `<model>-<dataset>-<date>-<short-sha>`
  - Required tags: `model:priority|phishing`, `stage:exp|baseline|prod-candidate`

### Data Tooling
- **`pandas`** — tabular preprocessing
- **`scikit-learn`** — train/test splits, classical baselines, metrics
- **`emoji`, `regex`** — text cleaning

### Storage
- **HuggingFace Hub** — primary checkpoint store (`insanar/priormail-priority`); Supabase free tier caps at 50 MB, which blocks our ~475 MB checkpoints
- **Supabase Storage** — intended long-term home per the backend contract; blocked by size until a quantized model or paid tier is decided (see §9)
- Local checkpoints: `./checkpoints/` (gitignored)
- Datasets cache: `./data/` (gitignored, raw files), `./data/processed/` (gitignored)

### Dev Tooling
- **Lint + format:** `ruff`
- **Type check:** `mypy` — relaxed for ML code (research-y), strict for `src/exporter/`
- **Tests:** `pytest` for utilities and data pipelines
- **Notebooks:** Jupyter — only in `notebooks/`, never imported from production code

> **Do not** add a new dependency without proposing it first. **Do not** swap any of the above.

---

## 5. Repository Structure

```
prior-mail-model/
├── CLAUDE.md                  ← you are here
├── README.md
├── Makefile
├── pyproject.toml
├── docs/                      ← submodule → prior-mail-docs
├── src/
│   ├── data/                  ← loaders, cleaners, augmenters
│   │   ├── loaders.py
│   │   ├── preprocess.py
│   │   ├── augment.py
│   │   └── splits.py
│   ├── training/              ← training scripts (one per model)
│   │   ├── train_priority.py
│   │   └── train_phishing.py
│   ├── eval/                  ← evaluation harnesses
│   │   ├── eval_priority.py
│   │   ├── eval_phishing.py
│   │   └── benchmarks.py
│   ├── exporter/              ← checkpoint export + upload (production code)
│   │   ├── export.py
│   │   └── upload_supabase.py
│   └── utils/
├── notebooks/                 ← exploration ONLY, not executed in CI
│   ├── 01_dataset_overview.ipynb
│   ├── 02_error_analysis.ipynb
│   └── README.md              ← naming convention
├── configs/                   ← YAML / TOML training configs
│   ├── priority_baseline.yaml
│   ├── priority_v1.yaml
│   └── phishing_v1.yaml
├── data/                      ← gitignored
│   ├── raw/
│   ├── processed/
│   └── labeled/
├── checkpoints/               ← gitignored
└── tests/
```

---

## 6. Training Workflow

### Configs over code
- Every training run is driven by a **YAML config** in `configs/`
- Configs are versioned in git; they're how we reproduce a run
- Required fields: `model_name`, `dataset`, `hyperparameters`, `seed`, `output_dir`

### Reproducibility
- Set seeds: `torch`, `numpy`, `random`, `transformers.set_seed`
- Log the git SHA at run start (`subprocess` to get HEAD)
- Save the config alongside the checkpoint
- `wandb` run logs config + code diff automatically — do not skip

### Workflow
1. Branch: `exp/<short-desc>` for experiments
2. Edit or create a config in `configs/`
3. Run: `make train config=configs/priority_v2.yaml`
4. Check `wandb` for metrics + loss curves
5. If promising → run full eval (`make eval config=configs/priority_v2.yaml`)
6. If eval gates pass → run export (`make export checkpoint=path/to/ckpt`)

### Notebooks
- Exploration only. **Do not** import from notebooks into production code
- Naming: `NN_short_topic.ipynb` (numbered for ordering)
- Strip output cells before committing (`nbstripout`)

---

## 7. Datasets

| Dataset | Use | Source |
|---|---|---|
| `jason23322/high-accuracy-email-classifier` | Priority classifier — English source (baseline signal) | HuggingFace |
| `insanar/prior-mail-priority` | **Team-curated priority dataset** — English, priority-labelled; canonical training source | HuggingFace (published by Insan) |
| `ealvaradob/phishing-dataset` | Phishing detector primary training set | HuggingFace |

> **`insanar/prior-mail-priority`** is the canonical dataset for the priority classifier. Use `load_dataset("insanar/prior-mail-priority")` in `src/data/loaders.py`. It supersedes `jason23322/high-accuracy-email-classifier` for any new training run.

### Labeling protocol (internal set)
- Annotation guidelines in `./docs/ML_PIPELINE.md` (section: Labeling) — or `./docs_local/` while the submodule is sparse
- 2 annotators per email; resolve disagreements in a weekly sync
- Track inter-annotator agreement (Cohen's kappa target: ≥ 0.7)
- Store labels in `data/labeled/<batch>.jsonl` (gitignored); published version lives at `insanar/prior-mail-priority`

### Privacy
- **Never** commit raw user emails to git
- If using real user emails for training (with consent): redact PII first (names, addresses, phone, account numbers)
- Synthetic + public datasets are the default

---

## 8. Evaluation Gates

A checkpoint can be **promoted to production** (exported to Supabase, referenced by backend) only when **all** gates pass.

### Priority classifier
- Macro F1 ≥ **0.80** on held-out test set
- Per-class recall ≥ **0.65** (no class catastrophically missed)
- Inference latency p95 < **500 ms** per email on CPU (the backend runs CPU on Render)
- Confusion matrix logged to `wandb` + saved to `eval/results/`

### Phishing detector
- Recall ≥ **0.95** (false negatives are the worst case)
- Precision ≥ **0.80** (avoid flooding users with false alarms)
- Inference latency p95 < **500 ms** per email on CPU

### Required artifacts per promoted checkpoint
- `checkpoint.bin` (or `.safetensors`)
- `tokenizer/` directory
- `config.json` (HuggingFace standard)
- `eval_report.json` (metrics from the eval harness)
- `training_config.yaml` (the config that produced this run)
- `model_card.md` (short description, intended use, known limits, dataset breakdown)

---

## 9. Checkpoint Export & Backend Contract

> Full contract in `./docs/ML_PIPELINE.md` (pending creation) / `./docs_local/BACKEND_INTEGRATION_GUIDE.md` (current reference). Key points below.

- Versioning scheme: `v<MAJOR>.<MINOR>` per model (e.g. `v1.1`, `v1.2`)
- **Current storage:** HuggingFace Hub repo `insanar/priormail-priority`, with version as a subfolder (e.g. `v1.1/`)
  - Load via `huggingface_hub.snapshot_download("insanar/priormail-priority", allow_patterns="v1.1/*")`
- **Intended storage:** Supabase `models/<model_name>/<version>/` — blocked by 50 MB free-tier cap; unresolved (see §14)
- The backend pins to a specific version via env var `PRIORITY_MODEL_URI` — promotion to "production" means updating that env var, not overwriting old versions
- **Never delete or overwrite** an existing checkpoint version
- Notify Syafiq + Insan in the team channel when a new version is published

**Published versions:**

| Model | Version | HF path | Status |
|---|---|---|---|
| priority (English proxy) | v1.0 | `insanar/priormail-priority/v1.0` | Baseline — not promoted |
| priority (Indonesian proxy) | v1.1 | `insanar/priormail-priority/v1.1` | Baseline — not promoted |

---

## 10. Coding Conventions

### Style
- `ruff` for lint + format
- `mypy` strict for `src/exporter/` only (it talks to the backend contract); relaxed elsewhere
- Type hints encouraged everywhere but not enforced for research code
- Constants: `UPPER_SNAKE_CASE`
- Files: `snake_case.py`

### Determinism
- Always set seeds at script entry
- Document GPU model + CUDA version in `wandb` run config
- Note: full determinism on GPU is hard — aim for "same seed, same machine = same result"

### Logging
- `wandb` for everything quantitative (loss, metrics, learning rate)
- `logging` module for everything qualitative (script start/end, errors)
- No `print()` in non-notebook code

### Git
- Branches: `feat/<short-name>`, `exp/<short-name>`, `fix/<short-name>`, `chore/<short-name>`
- Conventional Commits
- Experiment branches can be long-lived; merge or close after eval
- Do **not** commit checkpoints, data, or `wandb/` folders

---

## 11. Do NOT (Anti-patterns)

- Do **not** add a dependency without proposing it
- Do **not** swap items in "Tech Stack (LOCKED)"
- Do **not** start a multi-hour training run without confirming
- Do **not** commit raw datasets, checkpoints, or `wandb/` artifacts
- Do **not** train on data that contains unredacted user PII
- Do **not** modify previously promoted checkpoint versions
- Do **not** import code from notebooks into `src/`
- Do **not** mix tokenizers (always pair tokenizer with its source model)
- Do **not** report metrics without specifying the test set + split
- Do **not** auto-bump the production model version — that's a deliberate decision

---

## 12. Definition of Done (model release)

A new model version is "released" when **all** apply:

- [ ] Config in `configs/` committed
- [ ] Training run logged in `wandb` with required tags
- [ ] Eval harness run; results saved to `eval/results/`
- [ ] All eval gates pass (see §8)
- [ ] Model card written
- [ ] Checkpoint + artifacts uploaded to `models/<name>/<version>/` in Supabase Storage
- [ ] Backend owner (Syafiq) notified with version string and changelog
- [ ] `./docs/ML_PIPELINE.md` updated with the new version row

---

## 13. Common Commands

```bash
make install                                   # uv pip install -e ".[dev]"
make data                                      # download + preprocess datasets
make train config=configs/priority_v1.yaml     # run training
make eval config=configs/priority_v1.yaml      # run eval suite
make export checkpoint=checkpoints/priority_v1 # package + upload to Supabase
make notebooks                                 # launch jupyter
make lint
make test

# Submodule
git submodule update --init --recursive        # after fresh clone
git submodule update --remote docs             # pull latest shared specs
```

---

## 14. Open Decisions

LLMs: do **not** assume an answer — ask.

- [ ] Phishing detector base model: `distilbert-base-uncased` vs `distilbert-base-multilingual-cased`?
- [ ] Class imbalance handling: focal loss vs class weights vs resampling?
- [ ] Summarizer: train our own (seq2seq) or rely on hosted LLM at inference time?
- [ ] Checkpoint storage: keep HuggingFace Hub as permanent home, or upgrade Supabase to Pro, or quantize under 50 MB?

---

## 15. Owners

- **Insan** — Lead, priority classifier, DistilBERT fine-tuning
- **Faiz** — Phishing detector, security-focused eval, adversarial testing
- **Syafiq** — Dataset preprocessing collab, integration handoff

---

*Last updated: 2026-06-09*
