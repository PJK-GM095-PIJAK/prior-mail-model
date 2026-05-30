"""Leak-safety test: internal labeled records must land in TRAIN only.

Rows are tagged by their ``body`` text (which survives the SPLIT_COLUMNS
selection that append_internal_to_train applies — ``id`` is intentionally
dropped, mirroring the real training pipeline).
"""

from datasets import Dataset, DatasetDict
from src.data.prepare import append_internal_to_train


def _split(bodies, label=0):
    n = len(bodies)
    return Dataset.from_dict(
        {
            "id": [f"id{i}" for i in range(n)],
            "subject": ["s"] * n,
            "body": bodies,
            "priority": ["urgent"] * n,
            "labels": [label] * n,
        }
    )


def test_internal_goes_to_train_only():
    splits = DatasetDict(
        {
            "train": _split(["pub_tr1", "pub_tr2"]),
            "validation": _split(["pub_val1"]),
            "test": _split(["pub_test1"]),
        }
    )
    internal = _split(["int1", "int2", "int3"])

    out = append_internal_to_train(splits, internal)

    train = set(out["train"]["body"])
    val = set(out["validation"]["body"])
    test = set(out["test"]["body"])

    # internal rows present in train...
    assert {"int1", "int2", "int3"} <= train
    # ...and absent from val/test (no leak).
    assert not ({"int1", "int2", "int3"} & val)
    assert not ({"int1", "int2", "int3"} & test)
    # val/test untouched.
    assert val == {"pub_val1"}
    assert test == {"pub_test1"}


def test_none_internal_is_noop():
    splits = DatasetDict(
        {"train": _split(["a"]), "validation": _split(["b"]), "test": _split(["c"])}
    )
    out = append_internal_to_train(splits, None)
    assert out["train"]["body"] == ["a"]


def test_merge_aligns_mismatched_arrow_types():
    # Regression: public `subject` is large_string, synthetic is string —
    # concat must still succeed (we cast internal to the train features).
    from datasets import Features, Value

    large = Features(
        {
            "id": Value("string"),
            "subject": Value("large_string"),
            "body": Value("large_string"),
            "priority": Value("string"),
            "labels": Value("int64"),
        }
    )
    train = _split(["pub1"]).cast(large)
    splits = DatasetDict(
        {"train": train, "validation": _split(["v"]), "test": _split(["t"])}
    )
    internal = _split(["int1"])  # plain string features

    out = append_internal_to_train(splits, internal)  # must not raise
    assert set(out["train"]["body"]) == {"pub1", "int1"}
