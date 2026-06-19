"""Tests for the Nazario phishing CSV reader (v2.2) — pure, no network."""

from __future__ import annotations

import pandas as pd
from src.data.loaders import _read_nazario_phishing


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_reads_standard_naser_schema(tmp_path):
    # Naser's Kaggle cut: sender, receiver, date, subject, body, urls, label.
    csv = _write_csv(tmp_path / "Nazario.csv", [
        {"sender": "a@evil.tk", "receiver": "you@x.com", "date": "2005",
         "subject": "Verify your account", "body": "Click http://evil.tk now", "urls": 1, "label": 1},
        {"sender": "b@evil.tk", "receiver": "you@x.com", "date": "2005",
         "subject": "Account locked", "body": "Confirm details here", "urls": 1, "label": 1},
    ])
    recs = _read_nazario_phishing(csv)
    assert len(recs) == 2
    assert all(r["labels"] == 1 and r["phishing"] == "phishing" for r in recs)
    assert recs[0]["sender_email"] == "a@evil.tk"
    assert recs[0]["subject"] == "Verify your account"
    assert "Click" in recs[0]["body"]


def test_label_column_filters_non_phishing(tmp_path):
    # A combined mirror may carry both classes — keep only phishing rows.
    csv = _write_csv(tmp_path / "naz.csv", [
        {"subject": "phish", "body": "bad", "label": 1},
        {"subject": "ham", "body": "good", "label": 0},
        {"subject": "phish2", "body": "bad2", "label": "phishing"},
    ])
    recs = _read_nazario_phishing(csv)
    bodies = {r["body"] for r in recs}
    assert bodies == {"bad", "bad2"}  # the label=0 row is dropped


def test_no_label_column_treats_all_as_phishing(tmp_path):
    csv = _write_csv(tmp_path / "naz.csv", [
        {"subject": "s1", "body": "b1"},
        {"subject": "s2", "body": "b2"},
    ])
    recs = _read_nazario_phishing(csv)
    assert len(recs) == 2
    assert all(r["labels"] == 1 for r in recs)


def test_raw_email_body_is_parsed_into_headers(tmp_path):
    # A mirror with only a combined text column holding a raw RFC 822 message.
    raw = "From: attacker@evil.tk\nSubject: Reset your password\n\nClick the link to reset."
    csv = _write_csv(tmp_path / "naz.csv", [{"text": raw, "label": 1}])
    recs = _read_nazario_phishing(csv)
    assert len(recs) == 1
    r = recs[0]
    assert r["sender_email"] == "attacker@evil.tk"
    assert r["subject"] == "Reset your password"
    assert "Click the link" in r["body"]
    assert "Subject:" not in r["body"]  # headers stripped from the body


def test_empty_body_rows_dropped(tmp_path):
    csv = _write_csv(tmp_path / "naz.csv", [
        {"subject": "s1", "body": "real"},
        {"subject": "s2", "body": "   "},
        {"subject": "s3", "body": ""},
    ])
    recs = _read_nazario_phishing(csv)
    assert len(recs) == 1
    assert recs[0]["body"] == "real"


def test_sample_size_caps_and_is_deterministic(tmp_path):
    rows = [{"subject": f"s{i}", "body": f"body number {i}"} for i in range(50)]
    csv = _write_csv(tmp_path / "naz.csv", rows)
    a = _read_nazario_phishing(csv, sample_size=10, seed=7)
    b = _read_nazario_phishing(csv, sample_size=10, seed=7)
    assert len(a) == 10
    assert [r["body"] for r in a] == [r["body"] for r in b]  # seeded → reproducible


def test_missing_body_column_raises(tmp_path):
    csv = _write_csv(tmp_path / "naz.csv", [{"subject": "s", "urls": "u"}])
    try:
        _read_nazario_phishing(csv)
        raise AssertionError("expected ValueError for missing body column")
    except ValueError as e:
        assert "body" in str(e).lower()
