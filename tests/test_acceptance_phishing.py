"""Tests for the phishing acceptance harness — pure (non-model) parts."""

import pytest

from src.eval.acceptance_phishing import ACCEPTANCE_DIR, _eml_body, _label_from_name


def test_label_from_name() -> None:
    assert _label_from_name(ACCEPTANCE_DIR / "phishing_bank_suspension.eml") == 1
    assert _label_from_name(ACCEPTANCE_DIR / "legit_2fa_code.eml") == 0


def test_label_from_name_rejects_unknown_prefix(tmp_path) -> None:
    with pytest.raises(ValueError):
        _label_from_name(tmp_path / "unknown_sample.eml")


def test_acceptance_set_is_present_and_balanced() -> None:
    files = sorted(ACCEPTANCE_DIR.glob("*.eml"))
    assert len(files) >= 10, "acceptance set should be non-trivial"
    labels = [_label_from_name(f) for f in files]
    assert sum(labels) >= 4, "need several phishing examples"
    assert len(labels) - sum(labels) >= 4, "need several legit examples"


def test_eml_body_strips_headers() -> None:
    body = _eml_body(ACCEPTANCE_DIR / "legit_2fa_code.eml")
    # The body content is present...
    assert "verification code" in body.lower()
    # ...but RFC 822 headers are not.
    assert "Subject:" not in body
    assert "From:" not in body
    assert "Message-ID" not in body
