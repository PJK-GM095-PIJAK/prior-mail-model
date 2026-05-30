"""Tests for the synthetic Indonesian email generator (§7 bootstrap)."""

from collections import Counter

from src.data.labeled import load_labeled_dataset, validate_record
from src.data.synthetic import generate, write_jsonl
from src.utils.constants import PRIORITY_LABELS


def test_generate_is_deterministic():
    a = generate(40, seed=7)
    b = generate(40, seed=7)
    assert a == b


def test_different_seed_differs():
    a = generate(40, seed=1)
    b = generate(40, seed=2)
    # subjects/bodies should differ even if labels round-robin the same
    assert [r["body"] for r in a] != [r["body"] for r in b]


def test_balanced_across_classes():
    recs = generate(40, seed=42)
    counts = Counter(r["label"] for r in recs)
    assert set(counts) == set(PRIORITY_LABELS)
    # 40 / 4 = 10 each, exactly balanced
    assert all(c == 10 for c in counts.values())


def test_every_record_is_schema_valid():
    for r in generate(60, seed=3):
        validate_record(r)  # raises if invalid
    # ids are unique
    ids = [r["id"] for r in generate(60, seed=3)]
    assert len(set(ids)) == len(ids)


def test_no_unfilled_template_slots():
    # A leftover "{name}" etc. means a slot wasn't filled.
    for r in generate(40, seed=9):
        assert "{" not in r["subject"] and "}" not in r["subject"]
        assert "{" not in r["body"] and "}" not in r["body"]


def test_roundtrips_through_labeled_loader(tmp_path):
    write_jsonl(generate(20, seed=5), tmp_path / "synthetic.jsonl")
    ds = load_labeled_dataset(tmp_path)
    assert ds.num_rows == 20
    assert set(ds["priority"]) == set(PRIORITY_LABELS)
    assert all(r == "synthetic" for r in ds["annotator"])
