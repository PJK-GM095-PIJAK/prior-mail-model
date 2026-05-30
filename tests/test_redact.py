"""Tests for PII redaction (§7).

These verify the patterns we CLAIM to catch. They deliberately do NOT assert
the tool is exhaustive — names without honorifics are expected to slip through,
which is why §7 mandates human review.
"""

from src.data.redact import redact_record, redact_text
from src.utils.constants import (
    ADDRESS_TOKEN,
    NAME_TOKEN,
    NUMBER_TOKEN,
    PHONE_TOKEN,
)


def test_redacts_phone():
    out, repl = redact_text("Hubungi saya di 081234567890 ya")
    assert PHONE_TOKEN in out
    assert "081234567890" not in out
    assert any(r.rule == "phone" for r in repl)


def test_redacts_long_account_number():
    out, _ = redact_text("Transfer ke rekening 1234 5678 9012 3456 sekarang")
    assert NUMBER_TOKEN in out
    assert "1234 5678 9012 3456" not in out


def test_redacts_address():
    out, _ = redact_text("Alamat: Jl. Merdeka No. 17 Jakarta")
    assert ADDRESS_TOKEN in out
    assert "Merdeka" not in out


def test_redacts_name_after_honorific():
    out, repl = redact_text("Yth. Budi Santoso, terima kasih")
    assert NAME_TOKEN in out
    assert "Budi Santoso" not in out
    # honorific is preserved, only the name is masked
    assert "Yth." in out
    assert any(r.rule == "name" for r in repl)


def test_audit_trail_records_each_change():
    out, repl = redact_text("Pak Andi, no HP 081111111111")
    rules = {r.rule for r in repl}
    assert "name" in rules and "phone" in rules
    # every replacement carries the original + token
    assert all(r.original and r.token for r in repl)


def test_no_match_returns_empty_audit():
    out, repl = redact_text("Rapat tim berjalan lancar.")
    assert repl == []
    assert out == "Rapat tim berjalan lancar."


def test_empty_text():
    assert redact_text("") == ("", [])


def test_redact_record_only_touches_subject_body():
    rec = {
        "id": "e1",
        "subject": "Hubungi Pak Joko",
        "body": "Nomor: 082233445566",
        "label": "high",
        "annotator": "insan",
        "labeled_at": "2026-05-30T10:00:00",
    }
    out, repl = redact_record(rec)
    assert out["id"] == "e1"  # untouched
    assert out["label"] == "high"
    assert NAME_TOKEN in out["subject"]
    assert PHONE_TOKEN in out["body"]
    assert len(repl) >= 2
