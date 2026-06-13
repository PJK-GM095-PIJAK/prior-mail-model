# Model Card — PriorMail Phishing Detector v1.0

**Model ID:** `phishing/v1.0`
**Owner:** Faiz (PJK-GM095)
**Date:** 2026-06-13
**Status:** Eval gates passed — pending export approval from Insan + Syafiq

---

## What This Model Does

Binary email classifier: **legit (0)** vs **phishing (1)**.

Used by `prior-mail-backend` to flag potentially malicious emails before they reach the user's inbox. Runs at inference time on every incoming email.

---

## Base Model

`bert-base-multilingual-cased` (Google, 178M params)

Chosen over IndoBERT because the primary training data is English-dominant (~98% EN). mBERT handles English well and retains multilingual capability for future Indonesian phishing data.

---

## Training Data

| Source | Role | Rows (after dedup) |
|---|---|---|
| `ealvaradob/phishing-dataset` subset `texts` | Phishing + some legit | ~20,137 |
| Enron corpus (`emails.csv`) | Legit negative class | ~20,000 sampled |

**Total after deduplication:** ~38,000 rows  
**Train / Val / Test split:** 75 / 12.5 / 12.5 (stratified by label)

Deduplication applied on the first 200 chars of body before splitting to prevent leakage.

**Note:** Dataset is almost entirely English. Indonesian phishing emails are not well-represented. Performance on Indonesian-language phishing may be lower — to be addressed in v1.1 with internal labeled data.

---

## Training Config

See [`configs/phishing_v1.yaml`](../configs/phishing_v1.yaml) for the full reproducible config.

Key hyperparameters:

| Parameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 16 per device |
| Max sequence length | 512 |
| Epochs | 4 (early stopped at 3) |
| Warmup ratio | 0.10 |
| Loss | Weighted cross-entropy (phishing_class_multiplier=3.0) |
| Early stopping metric | `recall_phishing` (patience=2) |
| Mixed precision | bf16 |
| Seed | 42 |

---

## Evaluation Results

### Threshold Selection (Validation Set)

Threshold selected on val set to maximize precision while keeping recall ≥ 0.95.

**Threshold: 0.95**

| Metric | Val Set |
|---|---|
| Recall | 0.981 |
| Precision | 0.933 |

### Final Results (Test Set, n=5,722)

| Metric | Result | Gate | Status |
|---|---|---|---|
| Recall | **0.970** | ≥ 0.95 | PASS ✅ |
| Precision | **0.928** | ≥ 0.80 | PASS ✅ |
| Latency p95 (CPU) | **435 ms** | < 500 ms | PASS ✅ |
| F1 | 0.948 | — | — |

### Confusion Matrix (Test Set)

|  | Predicted Legit | Predicted Phishing |
|---|---|---|
| **Actual Legit** | 4,593 (TN) | 79 (FP) |
| **Actual Phishing** | 32 (FN) | 1,018 (TP) |

False negatives (missed phishing): **32 out of 1,050** — 3.0% miss rate.  
False positives (false alarms): **79 out of 4,672** — 1.7% false alarm rate.

---

## Input / Output Format

**Input** (assembled by `src/data/preprocess.build_phishing_input()`):
```
FROM: {sender_email} [SEP] SUBJECT: {subject} [SEP] BODY: {body}
```

If sender or subject are unavailable (e.g. plain-body-only sources), those fields are empty strings. The model still performs well on body-only input — the HF training data was ~98% body-only.

**Output:**
- Raw: softmax probability for class 1 (phishing), range [0, 1]
- Final label: `phishing` if probability ≥ 0.95, else `legit`
- Threshold stored in `threshold.json` alongside the checkpoint

---

## Known Limitations

- **Language:** Trained primarily on English data. Lower recall expected on Indonesian-language phishing. Mitigated by mBERT's multilingual pretraining, but not validated.
- **Sender signal weak:** ~98% of HF phishing training samples had no RFC 2822 headers — sender email is not a reliable feature for this version. Enron legit samples do have sender headers, which may introduce a small asymmetry.
- **Adversarial robustness:** Not evaluated against adversarial phishing (obfuscated URLs, lookalike domains, Unicode tricks). Planned for v1.1.
- **Domain shift:** Trained on public phishing corpora. Real-world phishing targeting Indonesian corporate users may have different patterns.

---

## Artifacts

| File | Description |
|---|---|
| `model.safetensors` | Fine-tuned weights (best checkpoint, step 835 / epoch 1) |
| `config.json` | HuggingFace model config |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer (bert-base-multilingual-cased) |
| `threshold.json` | `{"threshold": 0.95}` |
| `training_config.yaml` | Full training config + git SHA |
| `val_metrics.json` | Val set metrics at end of training |

Eval artifacts (in `eval/results/phishing/`):
| File | Description |
|---|---|
| `eval_report.json` | Full test set metrics + confusion matrix + latency |

---

## Changelog

| Version | Date | Notes |
|---|---|---|
| v1.0 | 2026-06-13 | Initial release. All eval gates passed. English-only training data. |
