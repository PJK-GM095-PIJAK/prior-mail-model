"""Evaluation harness for the priority classifier.

Eval gates (ML_PIPELINE.md §2, CLAUDE.md §8) — ALL must pass for promotion:
  - Macro F1 >= 0.80 on the held-out test set
  - Per-class recall >= 0.65
  - Inference latency p95 < 500 ms per email on CPU

Writes a confusion matrix + ``eval_report.json`` to ``eval/results/`` and logs
to wandb. Wiring stub for now.
"""

from __future__ import annotations

import argparse
import logging

from src.utils.config import TrainingConfig

logger = logging.getLogger(__name__)

# Promotion thresholds (do not relax without team sign-off).
MACRO_F1_GATE: float = 0.80
PER_CLASS_RECALL_GATE: float = 0.65
LATENCY_P95_MS_GATE: float = 500.0


def evaluate(config: TrainingConfig) -> dict:
    """Run the eval suite and return a report dict. Stub."""
    raise NotImplementedError(
        "eval_priority: implement macro-F1 / per-class recall / CPU p95 latency "
        "against the fixed test split (ML_PIPELINE.md §2)."
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the priority classifier")
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TrainingConfig.from_yaml(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    _main()
