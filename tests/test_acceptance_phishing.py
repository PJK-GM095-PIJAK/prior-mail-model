"""Tests for the phishing acceptance harness — pure (non-model) parts."""

import pytest
from src.eval.acceptance_phishing import (
    ACCEPTANCE_DIR,
    ACCEPTANCE_FN_RATE_GATE,
    ACCEPTANCE_FP_RATE_GATE,
    _eml_body,
    _label_from_name,
    check_acceptance_gates,
)


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


def test_acceptance_gate_passes_when_within_bounds() -> None:
    report = {"false_negative_rate": 0.0, "false_positive_rate": 0.1}
    result = check_acceptance_gates(report)
    assert result["all_passed"] is True
    assert result["gates"] == {"false_negative_rate": True, "false_positive_rate": True}


def test_acceptance_gate_fails_on_missed_phishing() -> None:
    # The v2 failure mode: half the real phishing missed.
    report = {"false_negative_rate": 0.5, "false_positive_rate": 0.0}
    result = check_acceptance_gates(report)
    assert result["all_passed"] is False
    assert result["gates"]["false_negative_rate"] is False


def test_acceptance_gate_fails_on_over_flagging() -> None:
    # The v1 failure mode: too much legit mail flagged.
    report = {"false_negative_rate": 0.0, "false_positive_rate": 0.5}
    result = check_acceptance_gates(report)
    assert result["all_passed"] is False
    assert result["gates"]["false_positive_rate"] is False


def test_acceptance_gate_fails_when_a_class_absent() -> None:
    # A run that never exercised a class must not be reported as a pass.
    report = {"false_negative_rate": None, "false_positive_rate": 0.0}
    assert check_acceptance_gates(report)["all_passed"] is False


def test_acceptance_gate_boundaries_are_inclusive() -> None:
    report = {
        "false_negative_rate": ACCEPTANCE_FN_RATE_GATE,
        "false_positive_rate": ACCEPTANCE_FP_RATE_GATE,
    }
    assert check_acceptance_gates(report)["all_passed"] is True
