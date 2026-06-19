"""Text preprocessing for email inputs.

Implements the pipeline from ML_PIPELINE.md §2 (priority) and §3 (phishing):

    1. Strip HTML
    2. Collapse whitespace
    3. Replace URLs with ``[URL]``
    4. Replace emails with ``[EMAIL]``
    5. Truncation to the 512-token budget happens at tokenization time, not here.

Input format assembled here:
    priority: ``{subject} [SEP] {body}``
    phishing: ``{body}`` (body-only — see ``build_phishing_input`` for why)
"""

from __future__ import annotations

import html
import logging

import regex as re

from src.utils.constants import EMAIL_TOKEN, URL_TOKEN

logger = logging.getLogger(__name__)

# BERT's separator. The tokenizer maps this string to its real [SEP] id.
SEP = "[SEP]"

# Pulls the address out of a ``Display Name <local@domain>`` From header.
_ANGLE_ADDR_RE = re.compile(r"<([^>]+)>")

# --- Patterns (compiled once) ---------------------------------------------
# Drop <script>/<style> blocks *with their content* before stripping tags,
# so JS/CSS text doesn't leak into the cleaned body.
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
# Any remaining HTML/XML tag.
_TAG_RE = re.compile(r"<[^>]+>")
# Email addresses. Run BEFORE the URL pattern so an address isn't half-eaten.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# URLs: http(s)://… , www.… , and bare domains with a common TLD.
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    # bare domains: one-or-more dotted labels then a known TLD, e.g. example.com/x
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|id|co|io|gov|edu)\b\S*",
    re.IGNORECASE,
)
# Runs of any whitespace (incl. newlines/tabs) -> a single space.
_WS_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML: drop script/style blocks, strip tags, unescape entities."""
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return html.unescape(text)


def _url_host(url: str) -> str:
    """Return the lowercased host of a matched URL (scheme/path/query dropped).

    ``https://paypa1-secure.tk/login?token=abc`` -> ``paypa1-secure.tk``.
    The FULL host is kept (including subdomains) so lookalike tricks like
    ``paypal.com.secure-login.tk`` stay visible to the model.
    """
    s = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    s = re.sub(r"^www\.", "", s, flags=re.IGNORECASE)
    host = re.split(r"[/?#]", s, maxsplit=1)[0]
    return host.lower().rstrip(".")


def clean_text(text: str | None, *, keep_domains: bool = False) -> str:
    """Apply steps 1–4 of the ML_PIPELINE.md §2 pipeline to one raw field.

    Order matters: HTML strip -> emails -> URLs -> collapse whitespace.
    Emails are masked before URLs so an address is replaced whole rather than
    partially matched by the URL pattern. ``None`` is treated as empty string.

    Args:
        keep_domains: when ``True`` (phishing path, fix A1), preserve the host of
            each URL and the domain of each email as a signal, masking only the
            PII-bearing parts: the URL path/query and the email local-part.
            ``https://paypal.com/login?token=x`` -> ``paypal.com [URL]`` and
            ``ceo@evil-corp.tk`` -> ``[EMAIL]@evil-corp.tk``. Default ``False``
            (blanket ``[URL]``/``[EMAIL]`` masking) keeps the priority pipeline
            (ML_PIPELINE.md §2) unchanged.
    """
    if not text:
        return ""
    text = strip_html(text)
    if keep_domains:
        text = _EMAIL_RE.sub(
            lambda m: f"{EMAIL_TOKEN}@{m.group(0).split('@', 1)[1]}", text
        )
        text = _URL_RE.sub(lambda m: f"{_url_host(m.group(0))} {URL_TOKEN}", text)
    else:
        text = _EMAIL_RE.sub(EMAIL_TOKEN, text)
        text = _URL_RE.sub(URL_TOKEN, text)
    return _WS_RE.sub(" ", text).strip()


def build_priority_input(subject: str | None, body: str | None) -> str:
    """Assemble the priority model's single-string input: ``{subject} [SEP] {body}``.

    Both fields are cleaned first. Token-level truncation to the 512 budget
    (subject prepended, body truncated from the end) happens at tokenization
    time, not here (ML_PIPELINE.md §2).
    """
    return f"{clean_text(subject)} {SEP} {clean_text(body)}"


def sender_signal(sender_email: str | None) -> str:
    """Reduce a From header to its phishing-relevant signal: display name + domain.

    ``"Michael Chen (CEO)" <ceo.mchen@acmecorp-finance.tk>`` ->
    ``michael chen (ceo) acmecorp-finance.tk``. The local-part (PII, and the
    weakest signal) is dropped; the domain is kept because sender-domain
    mismatch is the core tell for BEC and brand-impersonation phishing. Returns
    ``""`` for an empty/header-less sender.
    """
    if not sender_email:
        return ""
    m = _ANGLE_ADDR_RE.search(sender_email)
    if m:
        display = sender_email[: m.start()].strip().strip('"')
        addr = m.group(1).strip()
    else:
        display, addr = "", sender_email.strip()
    domain = addr.split("@", 1)[1] if "@" in addr else ""
    return _WS_RE.sub(" ", f"{display} {domain}").strip().lower()


def build_phishing_input(
    sender_email: str | None, subject: str | None, body: str | None
) -> str:
    """Assemble the phishing model's input: ``{sender} [SEP] {subject} [SEP] {body}`` (v2.1).

    History:
      v1 used ``FROM: … [SEP] SUBJECT: … [SEP] BODY: …`` scaffolding. That leaked
        structure: phishing corpus rows were body-only (no headers) while Enron
        legit always carried From/Subject, so header *presence* became a class
        proxy ("empty FROM/SUBJECT -> phishing"). It broke at inference, where
        real ``.eml`` always has headers on BOTH classes.
      v2 over-corrected to body-only. That killed the leak but also discarded the
        sender — and with it the only signal for BEC/CEO-fraud (no URL, bland
        body). Real-world recall collapsed to 0.50.

    v2.1 reinstates sender + subject, made leak-safe by two changes:
      * The ``[SEP]`` scaffolding is ALWAYS present; header-less rows just carry
        empty sender/subject fields. The v1 leak was the *scaffolding* going
        missing — fixed structure with empty fields removes that proxy.
      * The synthetic augmentation (``augment.generate_phishing_augmentation``)
        supplies BOTH classes with realistic headers, so header *content* is not
        class-correlated. Inference always sees full headers, matching this regime.

    Body and subject are cleaned with ``keep_domains=True`` (fix A1) so lookalike
    URL hosts / email domains survive as signal instead of an opaque ``[URL]``.
    The sender is reduced to display-name + domain by ``sender_signal``.
    """
    sender = sender_signal(sender_email)
    subject_clean = clean_text(subject, keep_domains=True)
    body_clean = clean_text(body, keep_domains=True)
    return f"{sender} {SEP} {subject_clean} {SEP} {body_clean}"


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    demo_subject = "Verifikasi akun"
    demo_body = (
        "<p>Halo, klik <a href='https://evil.example.com/x'>di sini</a> "
        "atau email kami di support@bank.co.id.</p>"
    )
    logger.info("clean_text demo: %r", clean_text(demo_body))
    logger.info("priority input demo: %r", build_priority_input(demo_subject, demo_body))


if __name__ == "__main__":
    _main()
