"""Tests for the §2/§3 text-preprocessing pipeline."""

from src.data.preprocess import (
    SEP,
    build_phishing_input,
    build_priority_input,
    clean_text,
    sender_signal,
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


def test_build_phishing_input_v21_format() -> None:
    # v2.1: {sender} [SEP] {subject} [SEP] {body}. Sender reduced to domain,
    # local-part dropped; subject + body present; two SEPs always present.
    out = build_phishing_input(
        "Boss <ceo@evil-corp.tk>", "Wire transfer", "Click http://evil.example.com now"
    )
    assert out.count(SEP) == 2
    assert "evil-corp.tk" in out          # sender domain kept as signal
    assert "ceo@evil-corp.tk" not in out  # local-part dropped
    assert "Wire transfer" in out         # subject present (case preserved by clean_text)
    assert "evil.example.com" in out      # body host kept (keep_domains)
    assert URL_TOKEN in out


def test_build_phishing_input_scaffolding_always_present() -> None:
    # Header-less row (ealvaradob style): empty sender + subject, body only,
    # but the [SEP] scaffolding is still there so presence never leaks the class.
    out = build_phishing_input(None, None, "hello there")
    assert out.count(SEP) == 2
    assert out.endswith("hello there")


def test_sender_signal_keeps_domain_drops_localpart() -> None:
    assert sender_signal("Michael Chen <ceo.mchen@acme-finance.tk>") == "michael chen acme-finance.tk"
    assert sender_signal("plain@bank.example") == "bank.example"
    assert sender_signal("") == ""
    assert sender_signal(None) == ""
    assert "ceo.mchen" not in sender_signal("Michael Chen <ceo.mchen@acme-finance.tk>")


def test_clean_text_keep_domains_preserves_url_host() -> None:
    # Fix A1: the host survives as signal; the PII-bearing path/query is dropped.
    out = clean_text(
        "verify at https://paypa1-secure.tk/login?token=SECRET now",
        keep_domains=True,
    )
    assert "paypa1-secure.tk" in out      # lookalike host kept
    assert URL_TOKEN in out               # path collapsed to the token
    assert "token=SECRET" not in out      # PII / secret dropped
    assert "/login" not in out


def test_clean_text_keep_domains_preserves_full_subdomain_chain() -> None:
    # A subdomain lookalike trick must stay fully visible.
    out = clean_text("go to http://paypal.com.secure-login.tk/x", keep_domains=True)
    assert "paypal.com.secure-login.tk" in out


def test_clean_text_keep_domains_keeps_email_domain_masks_localpart() -> None:
    out = clean_text("reply to ceo.name@evil-corp.tk asap", keep_domains=True)
    assert "@evil-corp.tk" in out         # domain kept as signal
    assert EMAIL_TOKEN in out             # local-part masked
    assert "ceo.name" not in out          # PII dropped


def test_clean_text_default_still_blanket_masks() -> None:
    # Priority pipeline (default) is unchanged: full opaque masking, no host kept.
    out = clean_text("visit https://paypal.com/login or mail a@b.com", keep_domains=False)
    assert "paypal.com" not in out
    assert "b.com" not in out
    assert URL_TOKEN in out
    assert EMAIL_TOKEN in out
