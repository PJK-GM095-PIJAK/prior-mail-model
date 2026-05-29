"""Package a trained checkpoint into the publishable artifact set.

Production code — held to mypy strict (CLAUDE.md §10) because it implements the
backend contract. Required files per ML_PIPELINE.md §5:

    checkpoint.bin / config.json / tokenizer.json / tokenizer_config.json /
    special_tokens_map.json / training_config.yaml / eval_report.json /
    model_card.md / threshold.json (phishing only)

This stub validates intent + arguments; packaging logic is not implemented yet.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Files every promoted checkpoint must contain (ML_PIPELINE.md §5).
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "checkpoint.bin",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "training_config.yaml",
    "eval_report.json",
    "model_card.md",
)
# Additional file required only for the phishing detector.
PHISHING_ONLY_ARTIFACTS: tuple[str, ...] = ("threshold.json",)


def validate_artifacts(checkpoint_dir: Path, *, is_phishing: bool) -> list[str]:
    """Return the list of required artifacts missing from ``checkpoint_dir``."""
    required = list(REQUIRED_ARTIFACTS)
    if is_phishing:
        required += list(PHISHING_ONLY_ARTIFACTS)
    return [name for name in required if not (checkpoint_dir / name).exists()]


def package_checkpoint(checkpoint_dir: Path, *, is_phishing: bool = False) -> Path:
    """Validate and package a checkpoint directory for upload. Stub."""
    missing = validate_artifacts(checkpoint_dir, is_phishing=is_phishing)
    if missing:
        raise FileNotFoundError(f"Checkpoint at {checkpoint_dir} missing artifacts: {missing}")
    raise NotImplementedError("package_checkpoint: packaging logic not yet implemented.")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Package + upload a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint directory")
    parser.add_argument("--phishing", action="store_true", help="checkpoint is the phishing model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    package_checkpoint(Path(args.checkpoint), is_phishing=bool(args.phishing))


if __name__ == "__main__":
    _main()
