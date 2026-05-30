"""Package a trained checkpoint into the publishable artifact set.

Production code — held to mypy strict (CLAUDE.md §10) because it implements the
backend contract. Required files per ML_PIPELINE.md §5:

    checkpoint weights (config.json + model.safetensors / checkpoint.bin),
    tokenizer files, training_config.yaml, eval_report.json, model_card.md,
    and threshold.json (phishing only).

``package_checkpoint`` gathers these into a clean ``dist/`` directory ready for
``upload_supabase.upload``. It pulls the eval report and model card in from their
canonical repo locations if they aren't already beside the checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Model weights: modern transformers writes safetensors; older runs wrote .bin.
# At least one must be present.
WEIGHT_FILES: tuple[str, ...] = ("model.safetensors", "checkpoint.bin", "pytorch_model.bin")

# HF config — always required (carries id2label, the backend reads it).
CONFIG_FILE: str = "config.json"

# Tokenizer files. tokenizer.json is the fast tokenizer and is sufficient on its
# own; the others are copied when present but not all are strictly required.
TOKENIZER_REQUIRED: tuple[str, ...] = ("tokenizer.json", "tokenizer_config.json")
TOKENIZER_OPTIONAL: tuple[str, ...] = ("special_tokens_map.json", "vocab.txt")

# Provenance / contract artifacts (ML_PIPELINE.md §5).
PROVENANCE_REQUIRED: tuple[str, ...] = ("training_config.yaml", "eval_report.json", "model_card.md")

# Phishing-only.
PHISHING_ONLY: tuple[str, ...] = ("threshold.json",)


def find_missing(checkpoint_dir: Path, *, is_phishing: bool) -> list[str]:
    """Return human-readable descriptions of any missing required artifacts."""
    missing: list[str] = []

    if not any((checkpoint_dir / w).exists() for w in WEIGHT_FILES):
        missing.append(f"weights (one of {WEIGHT_FILES})")
    if not (checkpoint_dir / CONFIG_FILE).exists():
        missing.append(CONFIG_FILE)
    missing += [t for t in TOKENIZER_REQUIRED if not (checkpoint_dir / t).exists()]
    missing += [p for p in PROVENANCE_REQUIRED if not (checkpoint_dir / p).exists()]
    if is_phishing:
        missing += [p for p in PHISHING_ONLY if not (checkpoint_dir / p).exists()]
    return missing


def package_checkpoint(
    checkpoint_dir: Path, *, is_phishing: bool = False, dist_dir: Path | None = None
) -> Path:
    """Assemble the canonical artifact set into ``dist_dir`` and return it.

    Validates that every required artifact is present, then copies the canonical
    file set (weights + config + tokenizer + provenance) into a clean directory.
    Intermediate ``checkpoint-*/`` and optimizer state are intentionally excluded.
    """
    missing = find_missing(checkpoint_dir, is_phishing=is_phishing)
    if missing:
        raise FileNotFoundError(
            f"Checkpoint at {checkpoint_dir} is missing required artifacts: {missing}"
        )

    out = dist_dir if dist_dir is not None else checkpoint_dir / "dist"
    out.mkdir(parents=True, exist_ok=True)

    to_copy: list[str] = [CONFIG_FILE, *TOKENIZER_REQUIRED, *PROVENANCE_REQUIRED]
    to_copy += [w for w in WEIGHT_FILES if (checkpoint_dir / w).exists()]
    to_copy += [t for t in TOKENIZER_OPTIONAL if (checkpoint_dir / t).exists()]
    if is_phishing:
        to_copy += list(PHISHING_ONLY)

    for name in to_copy:
        shutil.copy2(checkpoint_dir / name, out / name)
        logger.info("packaged %s", name)

    logger.info("Packaged %d artifacts into %s", len(to_copy), out)
    return out


def _main() -> None:
    parser = argparse.ArgumentParser(description="Package and optionally upload a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint directory")
    parser.add_argument("--phishing", action="store_true", help="checkpoint is the phishing model")
    parser.add_argument("--dist", default=None, help="output directory (default: <checkpoint>/dist)")
    parser.add_argument("--upload", action="store_true", help="upload to Supabase (§6 contract host)")
    parser.add_argument(
        "--hf-org", default=None,
        help="upload to HuggingFace Hub under this org/user instead of Supabase "
        "(interim path — free-tier Supabase can't hold large checkpoints)",
    )
    parser.add_argument("--version", default=None, help="version string, e.g. v1.0 (with upload)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    is_phishing = bool(args.phishing)
    dist = Path(args.dist) if args.dist else None
    packaged = package_checkpoint(Path(args.checkpoint), is_phishing=is_phishing, dist_dir=dist)
    model_name = "phishing" if is_phishing else "priority"

    if args.upload and args.hf_org:
        parser.error("choose one host: --upload (Supabase) OR --hf-org (HuggingFace), not both")

    if args.upload:
        if not args.version:
            parser.error("--upload requires --version (e.g. --version v1.0)")
        from src.exporter.upload_supabase import upload

        uri = upload(packaged, model_name, args.version)
        logger.info("Published. Backend pins to: %s", uri)
        logger.info("NOTE: uploading is NOT promotion — updating the backend env var is separate.")
    elif args.hf_org:
        if not args.version:
            parser.error("--hf-org requires --version (e.g. --version v1.0)")
        from src.exporter.upload_hf import upload as hf_upload

        uri = hf_upload(packaged, model_name, args.version, args.hf_org)
        logger.info("Published (interim host). Reference: %s", uri)
        logger.info("NOTE: hf:// diverges from the §6 Supabase contract — coordinate with Syafiq.")


if __name__ == "__main__":
    _main()
