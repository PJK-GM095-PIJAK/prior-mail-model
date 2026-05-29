"""Upload a packaged checkpoint to Supabase Storage.

Production code — mypy strict (CLAUDE.md §10).

Upload path (ML_PIPELINE.md §5): ``models/{model_name}/{version}/``
Versions are IMMUTABLE: never overwrite or delete a published version
(CLAUDE.md §9/§11). This module refuses to upload to a path that already exists.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

STORAGE_BUCKET: str = "models"


def remote_prefix(model_name: str, version: str) -> str:
    """Return the storage prefix ``models/{model_name}/{version}/`` for a checkpoint."""
    if model_name not in ("priority", "phishing"):
        raise ValueError(f"model_name must be 'priority' or 'phishing', got {model_name!r}")
    if not version.startswith("v"):
        raise ValueError(f"version must look like 'v1.0', got {version!r}")
    return f"{model_name}/{version}/"


def _get_client() -> object:
    """Build a Supabase client from environment credentials. Stub."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the environment.")
    raise NotImplementedError("_get_client: wire up supabase-py once credentials flow is settled.")


def version_exists(model_name: str, version: str) -> bool:
    """True if ``models/{model_name}/{version}/`` already has objects. Stub."""
    raise NotImplementedError("version_exists: not yet implemented.")


def upload(packaged_dir: Path, model_name: str, version: str) -> str:
    """Upload all files under ``packaged_dir`` to the version prefix. Stub.

    Refuses to proceed if the target version already exists (immutability).
    Returns the ``supabase://`` URI the backend will pin to.
    """
    if version_exists(model_name, version):
        raise FileExistsError(
            f"Version {model_name}/{version} already published — versions are immutable "
            "(CLAUDE.md §9). Bump the version instead of overwriting."
        )
    raise NotImplementedError("upload: not yet implemented.")
