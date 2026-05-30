"""Tests for the §2/§3 text-preprocessing pipeline."""

from src.data.preprocess import (
    SEP,
    build_phishing_input,
    build_priority_input,
    clean_text,
    strip_html,
)
from src.utils.constants import EMAIL_TOKEN, URL_TOKEN


def test_strip_html_drops_tags_and_script() -> None:
    raw = "<style>.a{}</style><p>Hi <b>there</b></p><script>evil()</script>"
    out = strip_html(raw)
    assert "Hi" in out and "there" in out
    assert "<" not in out and ">" not in out
    assert "evil" not in out  # script *content* removed, not just the tag
    assert ".a{}" not in out


def test_clean_text_unescapes_entities() -> None:
    assert "&" in clean_text("Tom &amp; Jerry")
    assert "&amp;" not in clean_text("Tom &amp; Jerry")


def test_clean_text_masks_url_and_email() -> None:
    out = clean_text("Mail me at a.b+x@mail.co.id or visit https://example.com/p?q=1")
    assert EMAIL_TOKEN in out
    assert URL_TOKEN in out
    assert "@mail.co.id" not in out
    assert "example.com" not in out


def test_clean_text_masks_bare_domain_url() -> None:
    # Regression: a bare two-part domain with a path must be masked (real data had these).
    out = clean_text("get $60 off: example.com/6058 code WELCOME20")
    assert "example.com/6058" not in out
    assert URL_TOKEN in out
    # A plain decimal number must NOT be mistaken for a URL.
    assert clean_text("total is 3.5 dollars") == "total is 3.5 dollars"


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("a\n\n  b\t c") == "a b c"


def test_clean_text_handles_none_and_empty() -> None:
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_email_masked_before_url_does_not_split_address() -> None:
    # An email must become exactly [EMAIL], not get partially eaten by the URL rule.
    out = clean_text("contact user@corp.com now")
    assert out == f"contact {EMAIL_TOKEN} now"


def test_build_priority_input_format() -> None:
    out = build_priority_input("Subject here", "Body here")
    assert out == f"Subject here {SEP} Body here"


def test_stratified_split_shapes_and_reproducibility() -> None:
    from datasets import Dataset
    from src.data.splits import stratified_split

    # 1000 rows, 4 imbalanced classes.
    labels = ([0] * 100) + ([1] * 200) + ([2] * 300) + ([3] * 400)
    ds = Dataset.from_dict({"labels": labels, "x": list(range(len(labels)))})

    s1 = stratified_split(ds, seed=42)
    assert set(s1) == {"train", "validation", "test"}
    assert s1["train"].num_rows == 700
    assert s1["validation"].num_rows == 150
    assert s1["test"].num_rows == 150

    # Same seed -> identical split (fixed split, reused across runs — §2).
    s2 = stratified_split(ds, seed=42)
    assert s1["test"]["x"] == s2["test"]["x"]


def test_build_phishing_input_keeps_sender_unmasked() -> None:
    out = build_phishing_input("attacker@evil.com", "Hi", "Body")
    # Sender carries signal — it must NOT be masked to [EMAIL].
    assert "attacker@evil.com" in out
    assert out.startswith("FROM: attacker@evil.com")
    assert out.count(SEP) == 2
