"""Tests for checkpoint packaging + the Supabase path/immutability logic."""

import pytest
from src.exporter.export import find_missing, package_checkpoint
from src.exporter.upload_supabase import remote_prefix


def _make_checkpoint(d, *, weights="model.safetensors", complete=True):
    """Create a fake checkpoint dir with the canonical artifact set."""
    (d / weights).write_bytes(b"fake-weights")
    (d / "config.json").write_text("{}")
    (d / "tokenizer.json").write_text("{}")
    (d / "tokenizer_config.json").write_text("{}")
    if complete:
        (d / "training_config.yaml").write_text("seed: 42")
        (d / "eval_report.json").write_text("{}")
        (d / "model_card.md").write_text("# card")
    return d


def test_find_missing_complete(tmp_path):
    _make_checkpoint(tmp_path)
    assert find_missing(tmp_path, is_phishing=False) == []


def test_find_missing_reports_gaps(tmp_path):
    _make_checkpoint(tmp_path, complete=False)
    missing = find_missing(tmp_path, is_phishing=False)
    assert "model_card.md" in missing
    assert "eval_report.json" in missing


def test_find_missing_phishing_needs_threshold(tmp_path):
    _make_checkpoint(tmp_path)
    assert "threshold.json" in find_missing(tmp_path, is_phishing=True)
    (tmp_path / "threshold.json").write_text('{"threshold": 0.3}')
    assert find_missing(tmp_path, is_phishing=True) == []


def test_package_excludes_training_state(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    ckpt = _make_checkpoint(ckpt_dir)
    # noise that must NOT be packaged
    (ckpt / "optimizer.pt").write_bytes(b"x")
    (ckpt / "checkpoint-100").mkdir()
    dist = package_checkpoint(ckpt, dist_dir=tmp_path / "dist")
    names = {p.name for p in dist.iterdir()}
    assert "model.safetensors" in names and "model_card.md" in names
    assert "optimizer.pt" not in names
    assert "checkpoint-100" not in names


def test_package_raises_on_incomplete(tmp_path):
    ckpt = _make_checkpoint(tmp_path, complete=False)
    with pytest.raises(FileNotFoundError):
        package_checkpoint(ckpt, dist_dir=tmp_path / "dist")


def test_remote_prefix_format():
    assert remote_prefix("priority", "v1.0") == "priority/v1.0/"


def test_remote_prefix_rejects_bad_input():
    with pytest.raises(ValueError):
        remote_prefix("summarizer", "v1.0")  # not a known model
    with pytest.raises(ValueError):
        remote_prefix("priority", "1.0")  # version must start with 'v'


class _FakeBucket:
    def __init__(self, existing):
        self._existing = existing
        self.uploaded = []

    def list(self, path):
        return [{"name": "x"}] if self._existing else []

    def upload(self, remote_path, local_path):
        self.uploaded.append(remote_path)


class _FakeStorage:
    def __init__(self, bucket):
        self._bucket = bucket

    def from_(self, _name):
        return self._bucket


class _FakeClient:
    def __init__(self, existing=False):
        self.storage = _FakeStorage(_FakeBucket(existing))


def test_upload_refuses_existing_version(tmp_path):
    from src.exporter.upload_supabase import upload

    _make_checkpoint(tmp_path)
    with pytest.raises(FileExistsError):  # immutability guard (§9)
        upload(tmp_path, "priority", "v1.0", client=_FakeClient(existing=True))


def test_upload_new_version_writes_all_files(tmp_path):
    from src.exporter.upload_supabase import upload

    _make_checkpoint(tmp_path)
    client = _FakeClient(existing=False)
    uri = upload(tmp_path, "priority", "v1.0", client=client)
    assert uri == "supabase://models/priority/v1.0/"
    # every file went under the right prefix
    assert all(p.startswith("priority/v1.0/") for p in client.storage._bucket.uploaded)
    assert "priority/v1.0/model.safetensors" in client.storage._bucket.uploaded


# --- HuggingFace Hub uploader (interim path) ------------------------------


def test_hf_repo_id():
    from src.exporter.upload_hf import repo_id

    assert repo_id("priority", "PJK-GM095") == "PJK-GM095/priormail-priority"


def test_hf_repo_id_rejects_bad():
    import pytest
    from src.exporter.upload_hf import repo_id

    with pytest.raises(ValueError):
        repo_id("summarizer", "org")
    with pytest.raises(ValueError):
        repo_id("priority", "")


class _FakeHfApi:
    def __init__(self, existing_files=None):
        self._files = existing_files or []
        self.created = None
        self.uploaded_folder = None

    def list_repo_files(self, repo_id, repo_type):
        return self._files

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        self.created = (repo_id, private)

    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, commit_message):
        self.uploaded_folder = (repo_id, path_in_repo)


def test_hf_upload_refuses_existing_version(tmp_path):
    import pytest
    from src.exporter.upload_hf import upload

    _make_checkpoint(tmp_path)
    api = _FakeHfApi(existing_files=["v1.0/model.safetensors"])
    with pytest.raises(FileExistsError):
        upload(tmp_path, "priority", "v1.0", "PJK-GM095", api=api)


def test_hf_upload_new_version(tmp_path):
    from src.exporter.upload_hf import upload

    _make_checkpoint(tmp_path)
    api = _FakeHfApi(existing_files=[])
    uri = upload(tmp_path, "priority", "v1.0", "PJK-GM095", api=api)
    assert uri == "hf://PJK-GM095/priormail-priority/v1.0"
    assert api.created == ("PJK-GM095/priormail-priority", True)
    assert api.uploaded_folder == ("PJK-GM095/priormail-priority", "v1.0")
