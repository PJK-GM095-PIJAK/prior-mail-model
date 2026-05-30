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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supabase import Client

logger = logging.getLogger(__name__)

STORAGE_BUCKET: str = "models"


def remote_prefix(model_name: str, version: str) -> str:
    """Return the storage prefix ``models/{model_name}/{version}/`` for a checkpoint."""
    if model_name not in ("priority", "phishing"):
        raise ValueError(f"model_name must be 'priority' or 'phishing', got {model_name!r}")
    if not version.startswith("v"):
        raise ValueError(f"version must look like 'v1.0', got {version!r}")
    return f"{model_name}/{version}/"


def _get_client() -> Client:
    """Build a Supabase client from environment credentials.

    Reads ``SUPABASE_URL`` and ``SUPABASE_SERVICE_KEY`` (the service-role key is
    required to write to Storage). Never hard-code or commit these.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the environment.")
    from supabase import create_client

    return create_client(url, key)


def version_exists(model_name: str, version: str, client: Client | None = None) -> bool:
    """True if ``models/{model_name}/{version}/`` already has any objects."""
    client = client or _get_client()
    prefix = remote_prefix(model_name, version).rstrip("/")
    objects = client.storage.from_(STORAGE_BUCKET).list(path=prefix)
    return bool(objects)


def upload(
    packaged_dir: Path, model_name: str, version: str, client: Client | None = None
) -> str:
    """Upload every file in ``packaged_dir`` to ``models/{model_name}/{version}/``.

    Refuses to proceed if the target version already exists — published versions
    are IMMUTABLE (CLAUDE.md §9/§11). Returns the ``supabase://`` URI the backend
    pins to via ``PRIORITY_MODEL_URI`` / ``PHISHING_MODEL_URI``.

    NOTE: uploading the artifact is NOT promotion. Promotion = pointing the
    backend env var at this URI, which is a separate deliberate decision.
    """
    client = client or _get_client()

    if version_exists(model_name, version, client=client):
        raise FileExistsError(
            f"Version {model_name}/{version} already published — versions are immutable "
            "(CLAUDE.md §9). Bump the version instead of overwriting."
        )

    prefix = remote_prefix(model_name, version)
    files = sorted(p for p in packaged_dir.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files to upload in {packaged_dir}")

    bucket = client.storage.from_(STORAGE_BUCKET)
    for path in files:
        remote_path = f"{prefix}{path.name}"
        bucket.upload(remote_path, str(path))
        logger.info("uploaded %s", remote_path)

    uri = f"supabase://{STORAGE_BUCKET}/{prefix}"
    logger.info("Published %d files to %s", len(files), uri)
    return uri
