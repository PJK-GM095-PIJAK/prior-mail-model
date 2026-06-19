---
language:
  - en
license: other
tags:
  - text-classification
  - phishing-detection
  - distilbert
base_model: distilbert-base-uncased
pipeline_tag: text-classification
---

# Model Card — PriorMail Phishing Detector v2.1

**Model ID:** `phishing/v2.1`
**Owner:** Faiz (PJK-GM095)
**Date:** 2026-06-19
**Git SHA:** `7cdeae7` (branch `feat/phishing-distilbert-v2`)
**Status:** In-distribution §8 gates **PASS**; real-world acceptance gate **FAILS on false-negative rate** (documented known-limitation below). Ship decision is deliberate, scoped, and pending sign-off from Insan + Syafiq — this is **not** an automatic promotion.

---

## What This Model Does

Binary email classifier: **legit (0)** vs **phishing (1)**.

Used by `prior-mail-backend` to flag potentially malicious emails. Intended as **one layer** of defense — see Known Limitations for why it should not be the sole line.

---

## Base Model

`distilbert-base-uncased` (~67M params).

Replaces v1.0's `bert-base-multilingual-cased`: the training data is English-only, and DistilBERT is ~3× smaller and faster on CPU (the backend runs CPU on Render), easing the 500 ms latency gate and the checkpoint-size constraint.

---

## Training Data

| Source | Role | Notes |
|---|---|---|
| `ealvaradob/phishing-dataset` (`texts`) | Phishing + benign legit | ~20,137 rows; ~62% benign |
| `insanar/prior-mail-priority` (`v4`) | Legit negatives | Product-distribution legit mail (transactional/banking) |
| Synthetic augmentation (`src/data/augment.py`) | Both classes | 600 rows/class, header-complete, seed 42 |

Enron (`emails.csv`) is **off** in v2 (it made v1 learn "non-Enron-style ⇒ phishing" → over-flagging). Legit pool downsampled to ~1.2× phishing; deduplicated on the first 200 chars of body before splitting to prevent leakage.

**After dedup:** ~16,669 rows (7,572 phishing / 9,097 legit)
**Split (stratified, seed 42):** train 11,667 / val 2,501 / test 2,501

The synthetic augmentation covers tactics the public corpora miss (BEC/CEO-fraud, fake invoice, brand- and IT-helpdesk credential harvest, delivery). It is **header-complete on both classes** so header *presence* cannot become a class proxy (the v1 leak).

---

## Input / Output Format

**Input** (`src/data/preprocess.build_phishing_input()`):
```
{sender display-name + domain} [SEP] {subject} [SEP] {body}
```
Text is cleaned with `keep_domains=True`: URL **host** and email **domain** are kept as signal (lookalike domains, odd TLDs) while the PII-bearing URL path/query and email local-part are masked. The `[SEP]` scaffolding is always present (header-less rows carry empty fields) — this is what makes the sender/subject reinstatement leak-safe.

**Output:**
- Softmax probability for class 1 (phishing), range [0, 1].
- Label `phishing` if probability ≥ threshold, else `legit`.
- **Threshold: 0.06**, stored in `threshold.json`. Selected on the validation set to maximize F-beta (β=2, recall-weighted — false negatives are the worst case) subject to precision ≥ 0.80; median of the optimal plateau.

---

## Training Config

See [`configs/phishing_v2.yaml`](../configs/phishing_v2.yaml). Key hyperparameters:

| Parameter | Value |
|---|---|
| Learning rate | 2e-5 (AdamW) |
| Batch size | 16 per device |
| Max sequence length | 512 |
| Epochs | 4 (early stopping, patience 2) |
| Early stopping metric | `f1_phishing` |
| Loss | Weighted cross-entropy, `phishing_class_multiplier=1.0` (threshold drives recall, not the loss) |
| Mixed precision | bf16 |
| Seed | 42 |

---

## Evaluation Results

### In-distribution test set (n=2,501) — §8 corpus gates

| Metric | Result | Gate | Status |
|---|---|---|---|
| Recall | **0.990** | ≥ 0.95 | PASS ✅ |
| Precision | **0.977** | ≥ 0.80 | PASS ✅ |
| Latency p95 (CPU) | **242 ms** | < 500 ms | PASS ✅ |
| F1 | 0.984 | — | — |

Confusion matrix (test):

|  | Predicted Legit | Predicted Phishing |
|---|---|---|
| **Actual Legit** | 1,339 (TN) | 26 (FP) |
| **Actual Phishing** | 11 (FN) | 1,125 (TP) |

> The test split is drawn from the **same corpora** as training, so it measures in-distribution fit, not real-world behaviour (v1.0 passed these yet over-flagged real mail). The acceptance set below is the real-world judge.

### Real-world acceptance set (36 hand-curated `.eml`, 18 legit / 18 phishing)

| Metric | Result | Gate | Status |
|---|---|---|---|
| False-positive rate | **0.00** (0/18) | ≤ 0.20 | PASS ✅ |
| False-negative rate | **0.33** (6/18) | ≤ 0.05 | **FAIL ❌** |
| Accuracy | 0.833 | — | — |

- **Zero false positives** — including deliberately "scary but legit" mail (a real new-sign-in security alert, bank statement, password-reset notice, CI/Slack/GitHub notifications). The v1/v2 over-flagging problem is resolved.
- **Caught (12/18):** credential harvest (PayPal, IT-helpdesk), BEC/CEO-fraud, fake invoice, bank/card suspension, lottery, package delivery, tax refund, crypto wallet, job-offer scam, voicemail lure.
- **Missed (6/18), all scored ≈0.00:** phishing that **mimics a routine SaaS/workflow notification** — DocuSign signature request, MFA-reset notice, payroll direct-deposit redirect, shared-document lure, subscription-renewal scare, and an Office 365 lookalike-domain login.

---

## Known Limitations

- **Workflow-mimicking phishing (primary gap).** The model learned "phishing = urgent threat + suspicious URL." Phishing that impersonates a *calm, routine* notification (DocuSign/MFA/payroll/shared-document/subscription) whose only tell is a lookalike sender domain is scored confidently legit. This is the **ceiling of synthetic-template augmentation** — the model reliably catches tactics it was trained on and misses novel ones. Closing it requires **diverse real phishing data** (e.g. Nazario / PhishTank email corpora), not more synthetic templates, and/or a **header/auth-signal layer (SPF/DKIM/DMARC, display-name↔domain mismatch) at the backend** — which is the right home for sender-spoofing detection.
- **Acceptance FN gate fails (0.33 > 0.05).** Per CLAUDE.md §8/§12 this is *not* a clean promotion. It is shipped as a scoped candidate with this limitation documented; do **not** treat the model as a sole line of defense.
- **English only.** Indonesian-language phishing is not represented and not validated.
- **Adversarial robustness** (obfuscated URLs, Unicode/homoglyph tricks beyond simple lookalikes) is not systematically evaluated.

---

## Artifacts

| File | Description |
|---|---|
| `model.safetensors` | Fine-tuned weights (~256 MB) |
| `config.json` | HuggingFace model config (`id2label`: 0=legit, 1=phishing) |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.txt` | Tokenizer (distilbert-base-uncased) |
| `threshold.json` | `{"threshold": 0.06}` |
| `training_config.yaml` | Full training config + git SHA + timestamp |
| `eval_report.json` | In-distribution test metrics + confusion matrix + latency |
| `acceptance_report.json` | 36-file real-world acceptance results (per-file table) |
| `val_metrics.json` | Validation metrics at best epoch |

---

## Changelog

| Version | Date | Notes |
|---|---|---|
| v1.0 | 2026-06-13 | Initial release. mBERT. Passed corpus gates; over-flagged real mail at inference. |
| v2.1 | 2026-06-19 | DistilBERT rework. Domain-aware cleaning (A1), header-complete synthetic augmentation + sender/subject input (B1/B2), FN-averse threshold selection, real-world `.eml` acceptance gate. Corpus gates pass, FP eliminated (0%); documented FN gap on workflow-mimicking phishing. |

> Reproduce: `make data-phishing && make train-phishing config=configs/phishing_v2.yaml && make eval-phishing config=configs/phishing_v2.yaml` then `python -m src.eval.acceptance_phishing --config configs/phishing_v2.yaml` (git `7cdeae7`, seed 42).
