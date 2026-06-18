"""Tests for the synthetic phishing augmentation generator (v2.1 fix B1)."""

from collections import Counter

from src.data.augment import generate_phishing_augmentation
from src.utils.constants import PHISHING_LABEL2ID


def test_augmentation_is_balanced_and_sized() -> None:
    ds = generate_phishing_augmentation(n_per_class=20, seed=42)
    assert ds.num_rows == 40
    counts = Counter(ds["labels"])
    assert counts[PHISHING_LABEL2ID["legit"]] == 20
    assert counts[PHISHING_LABEL2ID["phishing"]] == 20


def test_augmentation_has_full_schema_and_headers_on_both_classes() -> None:
    # The leak-safety contract: BOTH classes carry sender + subject + body, so
    # header presence cannot become a class proxy (the v1 trap).
    ds = generate_phishing_augmentation(n_per_class=30, seed=1)
    assert set(ds.column_names) == {"sender_email", "subject", "body", "phishing", "labels"}
    for row in ds:
        assert row["sender_email"].strip()
        assert row["subject"].strip()
        assert row["body"].strip()


def test_augmentation_is_deterministic() -> None:
    a = generate_phishing_augmentation(n_per_class=15, seed=7)
    b = generate_phishing_augmentation(n_per_class=15, seed=7)
    assert a["body"] == b["body"]
    assert a["sender_email"] == b["sender_email"]


def test_augmentation_seed_changes_output() -> None:
    a = generate_phishing_augmentation(n_per_class=15, seed=7)
    b = generate_phishing_augmentation(n_per_class=15, seed=8)
    assert a["body"] != b["body"]


def test_phishing_rows_carry_some_suspicious_signal() -> None:
    # Phishing rows should expose a suspicious host or a clearly spoofy sender
    # domain — the signal v2 blinded itself to. (BEC has no URL, so we check the
    # sender domain across the set rather than per-row.)
    ds = generate_phishing_augmentation(n_per_class=60, seed=3)
    phishing = ds.filter(lambda r: r["phishing"] == "phishing")
    suspicious_tld_hits = sum(
        any(row["sender_email"].lower().endswith(t) or t in row["body"].lower()
            for t in (".tk", ".xyz", ".online", ".info", ".icu", ".top", ".live", ".click"))
        for row in phishing
    )
    # The vast majority of phishing rows carry a suspicious TLD somewhere.
    assert suspicious_tld_hits >= int(0.8 * phishing.num_rows)
