"""Tests for the internal labeled-set loader + §7 schema validation."""

import json

import pytest
from src.data.labeled import (
    LabelSchemaError,
    load_labeled_dataset,
    read_jsonl,
    validate_record,
)

VALID = {
    "id": "e1",
    "subject": "Rapat besok",
    "body": "Tolong hadir rapat jam 9.",
    "label": "high",
    "annotator": "insan",
    "labeled_at": "2026-05-30T10:00:00",
}


def test_validate_accepts_good_record():
    assert validate_record(dict(VALID)) == VALID


def test_validate_blank_subject_ok():
    r = dict(VALID, subject="")
    assert validate_record(r)["subject"] == ""


@pytest.mark.parametrize("field", ["id", "subject", "body", "label", "annotator", "labeled_at"])
def test_validate_missing_field_raises(field):
    r = dict(VALID)
    del r[field]
    with pytest.raises(LabelSchemaError, match="missing required field"):
        validate_record(r)


def test_validate_bad_label_raises():
    with pytest.raises(LabelSchemaError, match="not a valid priority"):
        validate_record(dict(VALID, label="spam"))


def test_validate_empty_body_raises():
    with pytest.raises(LabelSchemaError, match="empty body"):
        validate_record(dict(VALID, body="   "))


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_read_jsonl_skips_blank_lines(tmp_path):
    f = tmp_path / "batch.jsonl"
    f.write_text(json.dumps(VALID) + "\n\n" + json.dumps(dict(VALID, id="e2")) + "\n")
    assert len(read_jsonl(f)) == 2


def test_read_jsonl_bad_json_reports_line(tmp_path):
    f = tmp_path / "batch.jsonl"
    f.write_text(json.dumps(VALID) + "\n{not json}\n")
    with pytest.raises(LabelSchemaError, match=r"batch\.jsonl:2: invalid JSON"):
        read_jsonl(f)


def test_load_empty_dir_returns_none(tmp_path):
    assert load_labeled_dataset(tmp_path) is None


def test_load_adds_label_ids(tmp_path):
    _write_jsonl(tmp_path / "b.jsonl", [VALID, dict(VALID, id="e2", label="low")])
    ds = load_labeled_dataset(tmp_path)
    assert ds.num_rows == 2
    assert set(ds["labels"]) == {1, 3}  # high=1, low=3
    assert "priority" in ds.column_names


def test_load_rejects_duplicate_ids(tmp_path):
    _write_jsonl(tmp_path / "a.jsonl", [VALID])
    _write_jsonl(tmp_path / "b.jsonl", [VALID])  # same id "e1"
    with pytest.raises(LabelSchemaError, match="duplicate record id"):
        load_labeled_dataset(tmp_path)
