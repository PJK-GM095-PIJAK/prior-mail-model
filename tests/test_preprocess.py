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


def test_build_phishing_input_is_body_only() -> None:
    # v2: body-only. Sender/subject are ignored to avoid the header-presence
    # leak (Enron legit always has headers, phishing corpus rows do not).
    out = build_phishing_input("attacker@evil.com", "Hi", "Click http://evil.example.com now")
    # Sender + subject must NOT appear; no FROM/SUBJECT scaffolding.
    assert "attacker@evil.com" not in out
    assert "FROM:" not in out
    assert "SUBJECT:" not in out
    assert "Hi" not in out
    # Body is still cleaned (URL masked).
    assert URL_TOKEN in out
    assert out == clean_text("Click http://evil.example.com now")
