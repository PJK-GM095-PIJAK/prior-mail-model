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

import logging

from src.utils.constants import EMAIL_TOKEN, URL_TOKEN

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Apply steps 1–4 of the preprocessing pipeline to a raw email field."""
    raise NotImplementedError(
        "clean_text: implement HTML strip, whitespace collapse, "
        f"URL->{URL_TOKEN}, EMAIL->{EMAIL_TOKEN} (ML_PIPELINE.md §2)."
    )


def build_priority_input(subject: str, body: str) -> str:
    """Assemble the priority model's single-string input: ``{subject} [SEP] {body}``."""
    raise NotImplementedError("build_priority_input: not yet implemented.")


def build_phishing_input(sender_email: str, subject: str, body: str) -> str:
    """Assemble the phishing model's input string (sender carries strong signal)."""
    raise NotImplementedError("build_phishing_input: not yet implemented.")


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("preprocess.py invoked — stub, nothing to do yet.")


if __name__ == "__main__":
    _main()
