"""Evaluation harness for the priority classifier.

Eval gates (ML_PIPELINE.md §2, CLAUDE.md §8) — ALL must pass for promotion:
  - Macro F1 >= 0.80 on the held-out test set
  - Per-class recall >= 0.65 (no class catastrophically missed)
  - Inference latency p95 < 500 ms per email on CPU (Render target)

Evaluates a trained checkpoint against the FIXED test split produced by
``make data`` (``data/processed/priority`` -> ``test``). Writes a confusion
matrix + ``eval_report.json`` to ``eval/results/priority/``. Run via
``make eval config=configs/priority_baseline.yaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.eval.benchmarks import measure_latency_ms
from src.utils.config import TrainingConfig
from src.utils.constants import PRIORITY_LABELS

logger = logging.getLogger(__name__)

# Promotion thresholds (do not relax without team sign-off).
MACRO_F1_GATE: float = 0.80
PER_CLASS_RECALL_GATE: float = 0.65
LATENCY_P95_MS_GATE: float = 500.0

PROCESSED_DIR = Path("data/processed/priority")
RESULTS_DIR = Path("eval/results/priority")
INPUT_COLUMN = "model_input"
LABEL_COLUMN = "labels"


def check_gates(metrics: dict) -> dict:
    """Pure gate logic: given computed metrics, decide pass/fail per gate.

    Separated from model code so it is unit-testable without a checkpoint.
    Expects keys: ``macro_f1``, ``recall_<label>`` for each label, ``p95_ms``.
    """
    per_class = {
        name: metrics[f"recall_{name}"] >= PER_CLASS_RECALL_GATE for name in PRIORITY_LABELS
    }
    gates = {
        "macro_f1": metrics["macro_f1"] >= MACRO_F1_GATE,
        "per_class_recall": all(per_class.values()),
        "latency_p95": metrics["p95_ms"] < LATENCY_P95_MS_GATE,
    }
    return {
        "gates": gates,
        "per_class_recall_detail": per_class,
        "all_passed": all(gates.values()),
    }


def compute_metrics(y_true, y_pred) -> dict:
    """Macro F1, per-class recall, and confusion matrix for the report."""
    from sklearn.metrics import confusion_matrix, f1_score, recall_score

    label_ids = list(range(len(PRIORITY_LABELS)))
    recalls = recall_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    metrics = {"macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}
    for i, name in enumerate(PRIORITY_LABELS):
        metrics[f"recall_{name}"] = float(recalls[i])
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=label_ids).tolist()
    return metrics


def evaluate(config: TrainingConfig) -> dict:
    """Evaluate the checkpoint at ``config.output_dir`` against the test split."""
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt = Path(config.output_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt} — train first (`make train ...`).")
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"{PROCESSED_DIR} not found — run `make data` first.")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    model.eval()
    model.to("cpu")  # gate is measured on CPU (Render target, §2)

    test = load_from_disk(str(PROCESSED_DIR))["test"]
    texts = test[INPUT_COLUMN]
    y_true = list(test[LABEL_COLUMN])
    max_len = int(config.hyperparameters.get("max_seq_length", 512))

    @torch.no_grad()
    def _predict_one(text: str) -> int:
        enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
        return int(model(**enc).logits.argmax(-1).item())

    y_pred = [_predict_one(t) for t in texts]

    metrics = compute_metrics(y_true, y_pred)
    # Latency on a capped sample (p95 is stable well before the full set).
    latency = measure_latency_ms(_predict_one, texts[: min(len(texts), 200)])
    metrics["p95_ms"] = latency["p95_ms"]
    metrics["latency"] = latency

    result = check_gates(metrics)
    report = {
        "checkpoint": str(ckpt),
        "test_split": str(PROCESSED_DIR / "test"),
        "n_test": len(y_true),
        "metrics": metrics,
        **result,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
    logger.info(
        "Gates %s | macro_f1=%.3f p95=%.0fms -> %s",
        "PASSED" if result["all_passed"] else "FAILED",
        metrics["macro_f1"],
        metrics["p95_ms"],
        report["gates"],
    )

    # Best-effort wandb logging (quantitative metrics, §4); never fail eval on it.
    try:
        import wandb

        if wandb.run is not None:
            wandb.log({k: v for k, v in metrics.items() if isinstance(v, int | float)})
    except Exception as e:  # pragma: no cover - logging only
        logger.debug("wandb logging skipped: %s", e)

    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the priority classifier")
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TrainingConfig.from_yaml(args.config)
    report = evaluate(cfg)
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    _main()
