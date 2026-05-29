"""Fine-tune IndoBERT for binary phishing detection.

Same protocol as the priority classifier (ML_PIPELINE.md §3) with two
differences: class weights heavily favor the phishing class to push recall up,
and the inference threshold is selected on validation (not the default 0.5).

Wiring stub — Trainer loop not implemented yet.
"""

from __future__ import annotations

import argparse
import logging

from src.utils.config import TrainingConfig
from src.utils.repro import get_git_sha
from src.utils.seeding import set_global_seed

logger = logging.getLogger(__name__)


def train(config: TrainingConfig) -> None:
    """Run a phishing-detector training job. Stub."""
    set_global_seed(config.seed)
    logger.info(
        "Would start phishing training: model=%s dataset=%s seed=%d git=%s",
        config.model_name,
        config.dataset,
        config.seed,
        get_git_sha(),
    )
    raise NotImplementedError(
        "train_phishing: not implemented. Phishing base-model choice is an open decision "
        "(ML_PIPELINE.md §11)."
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train the phishing detector")
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TrainingConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    _main()
