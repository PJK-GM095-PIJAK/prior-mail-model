"""Evaluation harness for the phishing detector.

Eval gates (ML_PIPELINE.md §3, CLAUDE.md §8):
  - Recall >= 0.95 on test set (false negatives are the worst case)
  - Precision >= 0.80
  - Inference latency p95 < 500 ms per email on CPU

The decision threshold is chosen on validation to hit recall >= 0.95 while
maximizing precision, then persisted as ``threshold.json``. Adversarial checks
(Faiz) run separately into ``eval/results/adversarial/``. Wiring stub.
"""

from __future__ import annotations

import argparse
import logging

from src.utils.config import TrainingConfig

logger = logging.getLogger(__name__)

RECALL_GATE: float = 0.95
PRECISION_GATE: float = 0.80
LATENCY_P95_MS_GATE: float = 500.0


def select_threshold(val_scores, val_labels) -> float:
    """Pick the threshold achieving recall >= 0.95 with max precision (val set)."""
    raise NotImplementedError("select_threshold: not yet implemented (ML_PIPELINE.md §3).")


def evaluate(config: TrainingConfig) -> dict:
    """Run the phishing eval suite and return a report dict. Stub."""
    raise NotImplementedError("eval_phishing: not yet implemented.")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the phishing detector")
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TrainingConfig.from_yaml(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    _main()
