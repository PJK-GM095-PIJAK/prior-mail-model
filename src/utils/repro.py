"""Reproducibility helpers: capture the git SHA at run start (ML_PIPELINE.md §8)."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def get_git_sha(short: bool = True) -> str:
    """Return the current HEAD commit SHA, or ``"unknown"`` if not in a repo.

    Logged at the start of every training run and embedded in the model card.
    """
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        sha = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not determine git SHA — recording 'unknown'")
        return "unknown"


def is_working_tree_dirty() -> bool:
    """True if there are uncommitted changes. A dirty tree hurts reproducibility."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
