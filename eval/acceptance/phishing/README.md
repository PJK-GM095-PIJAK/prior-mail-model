# Phishing acceptance set

Hand-curated, realistic `.eml` files for the **real-world** phishing gate — the
one the corpus test split can't give us (that split is in-distribution with
training, so it overstates real behaviour; v1.0 passed it yet over-flagged
ordinary mail).

## Convention
- **Label = filename prefix:** `legit_*.eml` (benign) or `phishing_*.eml`.
- Each file is a full RFC 822 message with headers (what the product receives).
  The model is body-only, so headers are stripped by the harness.
- Links use `*.example` placeholder domains — nothing resolvable or malicious.

## Coverage
- **legit/** weighted toward the categories v1 over-flagged: transactional
  (orders/receipts), 2FA codes, password resets you requested, newsletters,
  urgent-but-benign internal mail, personal mail.
- **phishing/** spans tactics: credential harvest, BEC/CEO fraud, fake
  invoice/delivery, account-suspension scare, lottery, IT-helpdesk quota scam.

## Run
```bash
python -m src.eval.acceptance_phishing --config configs/phishing_v2.yaml
```
Writes `eval/results/phishing/acceptance_report.json` (FP/FN rates + per-file
table). Informational — does not replace the §8 corpus gates, but a v2 candidate
should show a low false-positive rate here before promotion.

Extend freely: drop in more `legit_*.eml` / `phishing_*.eml` files; the harness
picks them up automatically.
