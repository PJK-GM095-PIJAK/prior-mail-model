"""Real-world acceptance test for the phishing detector.

This is the gate the corpus test split CANNOT give us: the test split is drawn
from the same corpora as training (ealvaradob + priority), so high recall/
precision there measure in-distribution fit, not real-world behaviour. v1.0
passed those gates yet over-flagged ordinary mail at inference.

So we keep a small, hand-curated set of realistic ``.eml`` files in
``eval/acceptance/phishing/`` and measure the model on THEM:

  - legit_*.eml   : genuinely benign mail, weighted toward the categories v1
                    over-flagged (transactional, 2FA, password reset you asked
                    for, newsletters, urgent-but-benign internal mail).
  - phishing_*.eml: realistic phishing across tactics (credential harvest, BEC,
                    fake invoice/delivery, account-suspension scare, etc.).

The label is the filename prefix. Each file is a full RFC 822 message with
headers — exactly what the product receives — which also proves the body-only
pipeline strips headers correctly.

Reported metrics (informational; does NOT replace the §8 corpus gates):
  - false_positive_rate : legit flagged as phishing  (the v1 pain point)
  - false_negative_rate : phishing missed
  - per-file table with the phishing probability and decision.

Run via:  python -m src.eval.acceptance_phishing --config configs/phishing_v2.yaml
          (after training + eval_phishing has written threshold.json)
"""

from __future__ import annotations

import argparse
import json
import logging
from email import message_from_bytes
from email.policy import default as default_policy
from pathlib import Path

from src.data.preprocess import build_phishing_input
from src.utils.config import TrainingConfig
from src.utils.constants import MAX_SEQ_LENGTH

logger = logging.getLogger(__name__)

ACCEPTANCE_DIR = Path("eval/acceptance/phishing")
RESULTS_DIR = Path("eval/results/phishing")


def _eml_body(path: Path) -> str:
    """Extract the plain-text body from an ``.eml`` file (headers discarded)."""
    msg = message_from_bytes(path.read_bytes(), policy=default_policy)
    body_part = msg.get_body(preferencelist=("plain", "html"))
    if body_part is not None:
        return body_part.get_content()
    # Fallback: no structured body part — use the raw payload.
    payload = msg.get_payload(decode=True)
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="ignore")
    return str(msg.get_payload() or "")


def _label_from_name(path: Path) -> int:
    """1 if the filename starts with ``phishing``, 0 if ``legit``."""
    prefix = path.name.split("_", 1)[0].lower()
    if prefix == "phishing":
        return 1
    if prefix == "legit":
        return 0
    raise ValueError(
        f"{path.name}: filename must start with 'legit_' or 'phishing_' to encode its label."
    )


def evaluate_acceptance(config: TrainingConfig, eml_dir: Path = ACCEPTANCE_DIR) -> dict:
    """Run the checkpoint against the ``.eml`` acceptance set and report FP/FN.

    Loads the model, tokenizer, and the threshold persisted by ``eval_phishing``
    (falls back to 0.5 if ``threshold.json`` is absent). Writes
    ``acceptance_report.json`` to ``RESULTS_DIR``.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt = Path(config.output_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt} — train first.")

    files = sorted(eml_dir.glob("*.eml"))
    if not files:
        raise FileNotFoundError(f"No .eml files in {eml_dir}")

    threshold_path = ckpt / "threshold.json"
    if threshold_path.exists():
        threshold = float(json.loads(threshold_path.read_text())["threshold"])
    else:
        threshold = 0.5
        logger.warning("No threshold.json at %s — falling back to 0.5.", ckpt)

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    model.eval()
    model.to("cpu")
    max_len = int(config.hyperparameters.get("max_seq_length", MAX_SEQ_LENGTH))

    @torch.no_grad()
    def _prob(text: str) -> float:
        enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
        return float(torch.softmax(model(**enc).logits, dim=-1)[0, 1].item())

    rows = []
    fp = fn = n_legit = n_phishing = 0
    for path in files:
        label = _label_from_name(path)
        prob = _prob(build_phishing_input(None, None, _eml_body(path)))
        pred = 1 if prob >= threshold else 0
        correct = pred == label
        if label == 1:
            n_phishing += 1
            if not correct:
                fn += 1
        else:
            n_legit += 1
            if not correct:
                fp += 1
        rows.append({
            "file": path.name,
            "label": "phishing" if label else "legit",
            "phishing_prob": round(prob, 4),
            "predicted": "phishing" if pred else "legit",
            "correct": correct,
        })

    report = {
        "checkpoint": str(ckpt),
        "threshold": threshold,
        "n": len(files),
        "n_legit": n_legit,
        "n_phishing": n_phishing,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": round(fp / n_legit, 4) if n_legit else None,
        "false_negative_rate": round(fn / n_phishing, 4) if n_phishing else None,
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
        "results": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "acceptance_report.json").write_text(json.dumps(report, indent=2))

    logger.info(
        "Acceptance: acc=%.3f | FP=%d/%d (rate %.3f) | FN=%d/%d (rate %.3f) | threshold=%.4f",
        report["accuracy"], fp, n_legit, report["false_positive_rate"] or 0.0,
        fn, n_phishing, report["false_negative_rate"] or 0.0, threshold,
    )
    for r in rows:
        if not r["correct"]:
            logger.warning("  MISS %-32s label=%-8s prob=%.3f -> %s",
                           r["file"], r["label"], r["phishing_prob"], r["predicted"])
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Phishing acceptance test on real .eml files")
    parser.add_argument("--config", required=True, help="path to the phishing YAML config")
    parser.add_argument("--eml-dir", default=str(ACCEPTANCE_DIR), help="directory of labeled .eml files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TrainingConfig.from_yaml(args.config)
    evaluate_acceptance(cfg, Path(args.eml_dir))


if __name__ == "__main__":
    _main()
