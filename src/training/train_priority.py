"""Fine-tune IndoBERT for 4-class email priority classification.

Driven entirely by a YAML config (CLAUDE.md §6). Protocol locked in
ML_PIPELINE.md §2: AdamW lr 2e-5, wd 0.01, linear warmup 10% + decay,
batch 16, 3–5 epochs with early stopping on val macro F1, weighted
cross-entropy, bf16/fp16, global seed.

This is a wiring stub: it loads config, sets seeds, and logs run metadata.
The actual Trainer loop is intentionally not implemented yet.
"""

from __future__ import annotations

import argparse
import logging

from src.utils.config import TrainingConfig
from src.utils.repro import get_git_sha, is_working_tree_dirty
from src.utils.seeding import set_global_seed

logger = logging.getLogger(__name__)


def train(config: TrainingConfig) -> None:
    """Run a priority-classifier training job. Stub."""
    set_global_seed(config.seed)
    sha = get_git_sha()
    if is_working_tree_dirty():
        logger.warning("Working tree is dirty — run is not cleanly reproducible.")
    logger.info(
        "Would start priority training: model=%s dataset=%s seed=%d git=%s output=%s",
        config.model_name,
        config.dataset,
        config.seed,
        sha,
        config.output_dir,
    )
    raise NotImplementedError(
        "train_priority: Trainer loop not implemented. Confirm GPU budget before a real run."
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train the priority classifier")
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TrainingConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    _main()
