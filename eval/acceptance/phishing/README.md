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
36 files, balanced 18 legit / 18 phishing (expanded for v2.1 so the gate is a
meaningful *rate*, not an all-or-nothing call on 8 examples, and a true
out-of-sample judge — these tactics/wording are deliberately distinct from the
synthetic augmentation in `src/data/augment.py`).
- **legit/** weighted toward what v1/v2 over-flag: transactional (orders,
  receipts, statements, shipping), 2FA codes, password resets you requested,
  newsletters, urgent-but-benign internal mail, calendar invites, and
  service/dev notifications (GitHub, CI, Slack) — including deliberately
  "scary but legit" ones (a real new-sign-in security alert) to probe false
  positives.
- **phishing/** spans tactics: credential harvest (brand + IT-helpdesk), BEC/CEO
  fraud, fake invoice/delivery, account-suspension scare, lottery, plus tax
  refund, crypto wallet, shared-document lure, voicemail, payroll redirect,
  subscription scare, job-offer scam, DocuSign lure, MFA-reset scare, card block.

> Note: a single text content-model is structurally weak against adversarial
> phishing that mimics a legitimate notification and whose only tell is a
> lookalike sender domain (e.g. `office365_login`). Those cases are a documented
> known-limitation best closed with header/auth signals (SPF/DKIM/DMARC) at the
> backend, not by overfitting the augmentation to this holdout.

## Run
```bash
python -m src.eval.acceptance_phishing --config configs/phishing_v2.yaml
```
Writes `eval/results/phishing/acceptance_report.json` (FP/FN rates + per-file
table). Informational — does not replace the §8 corpus gates, but a v2 candidate
should show a low false-positive rate here before promotion.

Extend freely: drop in more `legit_*.eml` / `phishing_*.eml` files; the harness
picks them up automatically.
