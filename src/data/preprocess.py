"""Text preprocessing for email inputs.

Implements the pipeline from ML_PIPELINE.md §2 (priority) and §3 (phishing):

    1. Strip HTML
    2. Collapse whitespace
    3. Replace URLs with ``[URL]``
    4. Replace emails with ``[EMAIL]``
    5. Truncation to the 512-token budget happens at tokenization time, not here.

Input format assembled here:
    priority: ``{subject} [SEP] {body}``
    phishing: ``FROM: {sender} [SEP] SUBJECT: {subject} [SEP] BODY: {body}``
"""

from __future__ import annotations

import html
import logging

import regex as re

from src.utils.constants import EMAIL_TOKEN, URL_TOKEN

logger = logging.getLogger(__name__)

# BERT's separator. The tokenizer maps this string to its real [SEP] id.
SEP = "[SEP]"

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


def clean_text(text: str | None) -> str:
    """Apply steps 1–4 of the ML_PIPELINE.md §2 pipeline to one raw field.

    Order matters: HTML strip -> emails -> URLs -> collapse whitespace.
    Emails are masked before URLs so an address is replaced whole rather than
    partially matched by the URL pattern. ``None`` is treated as empty string.
    """
    if not text:
        return ""
    text = strip_html(text)
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


def build_phishing_input(
    sender_email: str | None, subject: str | None, body: str | None
) -> str:
    """Assemble the phishing model's input (ML_PIPELINE.md §3).

    ``FROM: {sender} [SEP] SUBJECT: {subject} [SEP] BODY: {body}`` — the sender
    is included intentionally; it carries strong phishing signal. The sender is
    NOT run through ``clean_text`` (we must not mask it to ``[EMAIL]``); it's
    only whitespace-normalized.
    """
    sender = _WS_RE.sub(" ", (sender_email or "")).strip()
    return (
        f"FROM: {sender} {SEP} "
        f"SUBJECT: {clean_text(subject)} {SEP} "
        f"BODY: {clean_text(body)}"
    )


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
