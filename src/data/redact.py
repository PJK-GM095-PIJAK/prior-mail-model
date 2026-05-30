"""PII redaction for real Indonesian emails (ML_PIPELINE.md §7, CLAUDE.md §11).

Redaction is **destructive** and replaces personal data with placeholder tokens:
names → ``[NAMA]``, phone → ``[TELEPON]``, addresses → ``[ALAMAT]``, account/ID
numbers → ``[NOMOR]``.

⚠️ THIS IS A REDACTION *AID*, NOT A GUARANTEE. Regex cannot reliably catch every
name in free text (Indonesian names have no fixed pattern). High-confidence
items (phone, account/NIK numbers) are redacted reliably; names are caught only
via conservative title cues (Bapak/Ibu/Yth.). The §7 mandatory second-reviewer
step exists precisely because this tool *will* miss things. Never commit a real
email on the strength of this tool alone — a human must review the diff.

``redact_text`` returns the redacted string AND an audit list of every
replacement so the reviewer can see what was caught (and reason about what wasn't).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import regex as re

from src.utils.constants import (
    ADDRESS_TOKEN,
    NAME_TOKEN,
    NUMBER_TOKEN,
    PHONE_TOKEN,
)

logger = logging.getLogger(__name__)


@dataclass
class Replacement:
    """One redaction: what was removed, the token it became, and the rule name."""

    original: str
    token: str
    rule: str


# Indonesian phone numbers: +62/0 followed by 8 then 7–12 more digits (with
# optional separators). Checked BEFORE the generic number rule so a phone number
# is tokenized as [TELEPON], not swallowed as a generic [NOMOR].
_PHONE_RE = re.compile(r"(?:\+62|62|0)8[\d\-\s]{7,12}\d")

# Account / ID numbers: long digit runs (NIK=16, NPWP, rekening) that the phone
# rule above didn't already claim.
_NUMBER_RE = re.compile(r"\b\d[\d.\- ]{9,}\d\b")

# Addresses: common Indonesian address markers, anchored on word boundaries so
# ordinary words like "berjalan" (contains "jalan") are not matched.
_ADDRESS_RE = re.compile(
    r"\b(?:Jl\.?|Jalan|Gang|Gg\.?|Perumahan|Komplek|Kompleks|RT\s?\d|RW\s?\d)\b[^\n,]{0,60}",
    re.IGNORECASE,
)

# Names after honorific cues. Conservative: only flags Capitalized word(s)
# directly after Bapak/Ibu/Pak/Bu/Sdr/Sdri/Yth. Misses bare names by design.
_NAME_RE = re.compile(
    r"(?:Bapak|Ibu|Pak|Bu|Sdr|Sdri|Yth\.?|Kepada Yth\.?)\s+"
    r"(\p{Lu}\p{Ll}+(?:\s+\p{Lu}\p{Ll}+){0,2})"
)

# (rule_name, compiled_pattern, token, uses_group1)
# Order matters: phone before the generic number rule; address/name last.
_RULES: tuple[tuple[str, re.Pattern, str, bool], ...] = (
    ("phone", _PHONE_RE, PHONE_TOKEN, False),
    ("number", _NUMBER_RE, NUMBER_TOKEN, False),
    ("address", _ADDRESS_RE, ADDRESS_TOKEN, False),
    ("name", _NAME_RE, NAME_TOKEN, True),
)


def redact_text(text: str) -> tuple[str, list[Replacement]]:
    """Redact PII in ``text``. Returns ``(redacted, replacements)``.

    ``replacements`` records every change for the mandatory human review (§7).
    An empty list does NOT mean the text is PII-free — only that no *pattern*
    matched. A reviewer must still read it.
    """
    if not text:
        return "", []

    replacements: list[Replacement] = []
    result = text

    for rule_name, pattern, token, uses_group1 in _RULES:
        def _sub(m: re.Match, _token=token, _rule=rule_name, _g1=uses_group1) -> str:
            captured = m.group(1) if _g1 else m.group(0)
            replacements.append(Replacement(original=captured, token=_token, rule=_rule))
            if _g1:
                # Replace only the captured name, keep the honorific prefix.
                return m.group(0).replace(captured, _token)
            return _token

        result = pattern.sub(_sub, result)

    return result, replacements


def redact_record(record: dict) -> tuple[dict, list[Replacement]]:
    """Redact the ``subject`` and ``body`` of a labeled record in place (copy).

    Other fields (id, label, annotator, labeled_at) are left untouched.
    """
    out = dict(record)
    all_repl: list[Replacement] = []
    for field in ("subject", "body"):
        if field in out and out[field]:
            redacted, repl = redact_text(str(out[field]))
            out[field] = redacted
            all_repl.extend(repl)
    return out, all_repl


def _main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Redact PII in a labeled JSONL (produces a redacted file + review report). "
        "A SECOND reviewer must check the report before committing (ML_PIPELINE.md §7)."
    )
    parser.add_argument("--in", dest="inp", required=True, type=Path, help="raw JSONL to redact")
    parser.add_argument("--out", required=True, type=Path, help="redacted JSONL output path")
    parser.add_argument("--report", type=Path, default=None, help="review report path (default: <out>.review.txt)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    report_path = args.report or args.out.with_suffix(args.out.suffix + ".review.txt")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    out_lines: list[str] = []
    report: list[str] = ["# PII redaction review (§7) — a second reviewer must verify this.\n"]
    total = 0
    for i, raw in enumerate(args.inp.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        rec = json.loads(raw)
        redacted, repl = redact_record(rec)
        out_lines.append(json.dumps(redacted, ensure_ascii=False))
        if repl:
            total += len(repl)
            report.append(f"record {rec.get('id', i)}:")
            report.extend(f"  [{r.rule}] {r.original!r} -> {r.token}" for r in repl)

    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.warning(
        "Redacted %d item(s) across the file -> %s. REVIEW %s before committing — "
        "regex misses names; a human must confirm no PII remains.",
        total,
        args.out,
        report_path,
    )


if __name__ == "__main__":
    _main()
