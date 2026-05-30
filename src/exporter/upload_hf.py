"""Upload a packaged checkpoint to the HuggingFace Hub.

Production code — mypy strict (CLAUDE.md §10).

⚠️ INTERIM / ALTERNATIVE PATH. The backend contract (ML_PIPELINE.md §6) specifies
Supabase Storage (``supabase://models/...``). Free-tier Supabase caps uploads at
50MB, but our checkpoints are ~475MB, so they cannot be published there. The HF
Hub handles large model files natively and is the standard host for models, so
this module provides an interim publishing path. Adopting it requires a backend
contract change (Syafiq) — until then, treat the returned ``hf://`` reference as
provisional, not the official ``PRIORITY_MODEL_URI`` value.

Layout on the Hub: one repo per model (``{org}/priormail-{model_name}``); each
version is a subfolder ``{version}/`` inside it, mirroring the Supabase layout.
Versions are IMMUTABLE — this refuses to upload to a version path that already
has files (CLAUDE.md §9/§11).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


def repo_id(model_name: str, org: str) -> str:
    """Return the Hub repo id ``{org}/priormail-{model_name}`` for a model."""
    if model_name not in ("priority", "phishing"):
        raise ValueError(f"model_name must be 'priority' or 'phishing', got {model_name!r}")
    if not org:
        raise ValueError("org (HF user or organization) is required")
    return f"{org}/priormail-{model_name}"


def _api(token: str | None = None) -> HfApi:
    """Build an authenticated HfApi client.

    Token comes from the ``token`` arg or the ``HF_TOKEN`` env var (a write token).
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be set (a HuggingFace WRITE token) to upload.")
    from huggingface_hub import HfApi

    return HfApi(token=token)


def version_exists(model_name: str, version: str, org: str, api: HfApi | None = None) -> bool:
    """True if ``{version}/`` already has files in the model's Hub repo."""
    from huggingface_hub.utils import RepositoryNotFoundError

    api = api or _api()
    repo = repo_id(model_name, org)
    prefix = f"{version}/"
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="model")
    except RepositoryNotFoundError:
        return False  # repo doesn't exist yet -> version can't exist
    return any(f.startswith(prefix) for f in files)


def upload(
    packaged_dir: Path,
    model_name: str,
    version: str,
    org: str,
    *,
    private: bool = True,
    api: HfApi | None = None,
) -> str:
    """Upload ``packaged_dir`` to ``{org}/priormail-{model_name}`` under ``{version}/``.

    Creates the repo if needed (private by default). Refuses to overwrite an
    existing version (immutability, §9). Returns a provisional ``hf://`` reference.
    """
    api = api or _api()
    repo = repo_id(model_name, org)

    if version_exists(model_name, version, org, api=api):
        raise FileExistsError(
            f"Version {repo}@{version} already published — versions are immutable "
            "(CLAUDE.md §9). Bump the version instead of overwriting."
        )

    files = [p for p in packaged_dir.iterdir() if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No files to upload in {packaged_dir}")

    api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(packaged_dir),
        path_in_repo=version,
        commit_message=f"Publish {model_name} {version}",
    )

    uri = f"hf://{repo}/{version}"
    logger.info("Published %d files to %s", len(files), uri)
    logger.info("NOTE: hf:// is a PROVISIONAL host — §6 contract still specifies supabase://.")
    return uri
