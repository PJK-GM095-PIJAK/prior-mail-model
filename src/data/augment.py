"""Synthetic, header-complete augmentation for the phishing detector (v2.1).

Why this exists (decision log, 2026-06-19):
    v2 (DistilBERT) passed the §8 in-distribution gates but missed half the
    real ``.eml`` acceptance phishing (FN rate 0.50): BEC/CEO-fraud, fake-
    invoice, O365 + PayPal credential harvesting all scored ~0.0. Root cause:
    those tactics live ONLY in the acceptance holdout, never in the training
    pool (ealvaradob + priority). A model cannot learn a tactic it never sees.

    This module fabricates a balanced, header-complete augmentation set covering
    exactly those tactics on the phishing side, plus matching modern legit mail
    (transactional / notification / internal) on the legit side. Two design
    rules make it leak-safe — the v1 trap was that header *presence* became a
    class proxy:
      1. BOTH classes are generated with full ``sender_email`` + ``subject`` +
         ``body``, so header presence never correlates with the label.
      2. Generation is template-based with a seeded RNG — fully reproducible
         (record ``augmentation_size`` + ``seed`` in the training config) and
         carries no real user PII.

    Wording is intentionally distinct from ``eval/acceptance/phishing/*.eml`` so
    the curated acceptance holdout stays a true out-of-sample judge.

The output schema matches ``loaders.load_phishing_dataset``:
    sender_email str | subject str | body str | phishing str | labels int
"""

from __future__ import annotations

import logging
import random

from src.utils.constants import PHISHING_LABEL2ID

logger = logging.getLogger(__name__)

# --- Shared placeholder pools ---------------------------------------------
_FIRST_NAMES = [
    "Michael", "Sarah", "David", "Priya", "James", "Anita", "Robert", "Lena",
    "Daniel", "Grace", "Thomas", "Mei", "Andre", "Putri", "Kevin", "Fatima",
]
_LAST_NAMES = [
    "Chen", "Patel", "Anderson", "Rodriguez", "Wijaya", "Okafor", "Tanaka",
    "Muller", "Santoso", "Halim", "Nguyen", "Brooks", "Ferreira", "Khan",
]
_COMPANIES = [
    "Northbridge", "Acme", "Vertex", "Meridian", "Lumen", "Orbit", "Crestline",
    "Bluewave", "Ironclad", "Summit", "Helios", "Granite", "Pinnacle", "Kirana",
]
# Brands a credential-harvest mail impersonates, with a legit reference domain.
# "Microsoft 365" carries the same reference (microsoft.com) but a distinct
# display name — productivity-suite credential phishing is a top real tactic and
# the bare "Microsoft" display alone did not generalize to it (v2.1 acceptance).
_BRANDS = [
    ("Microsoft", "microsoft.com"),
    ("Microsoft 365", "microsoft.com"),
    ("PayPal", "paypal.com"),
    ("Netflix", "netflix.com"),
    ("Google", "google.com"),
    ("Amazon", "amazon.com"),
    ("DHL", "dhl.com"),
    ("LinkedIn", "linkedin.com"),
]
_TLDS_SUSPICIOUS = ["tk", "xyz", "online", "info", "icu", "top", "live", "click"]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _lookalike_domain(rng: random.Random, base: str) -> str:
    """Mutate a brand/company name into a plausible phishing host.

    e.g. ``paypal.com`` -> ``paypal-secure.tk`` / ``paypaI.com`` / ``paypal-id.xyz``.
    """
    stem = base.split(".")[0].lower()
    trick = rng.choice([
        f"{stem}-secure",
        f"{stem}-verify",
        f"{stem}-support",
        f"{stem}-account",
        f"{stem}-account-team",   # mimics "<brand>-account-team" lookalike hosts
        f"{stem}-security",
        f"{stem}-services",
        f"secure-{stem}",
        f"account-{stem}",
        f"{stem}{rng.randint(0, 9)}",
        stem.replace("l", "I", 1) if "l" in stem else f"{stem}-id",
    ])
    tld = rng.choice(_TLDS_SUSPICIOUS)
    return f"{trick}.{tld}"


def _legit_domain(rng: random.Random, company: str) -> str:
    stem = company.lower()
    return f"{stem}.{rng.choice(['com', 'co', 'io', 'id'])}"


def _name(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"


# --- Phishing generators (label 1) ----------------------------------------
# Each returns (sender_email, subject, body). BEC carries NO URL (signal is the
# sender/display mismatch); the rest carry a lookalike-host URL.

def _gen_bec(rng: random.Random) -> tuple[str, str, str]:
    boss = _name(rng)
    company = rng.choice(_COMPANIES)
    domain = _lookalike_domain(rng, company)
    subject = rng.choice([
        "Quick task - are you around?",
        "Urgent - need your help",
        "Available now?",
        "Confidential request",
    ])
    body = (
        f"{rng.choice(['Hi', 'Hello', 'Hey'])}, are you at your desk? I need you to "
        f"{rng.choice(['process an urgent wire transfer', 'arrange a same-day payment', 'buy a few gift cards'])} "
        f"to a {rng.choice(['vendor', 'supplier', 'partner'])} before end of day. It is "
        f"time-sensitive and {rng.choice(['confidential', 'strictly between us'])} - "
        f"do not loop in anyone else yet. Reply and I'll send the details. "
        f"{rng.choice(['Sent from my iPhone', 'Thanks', 'Regards'])}, {boss.split()[0]}"
    )
    return f"{boss} <{boss.split()[0].lower()}.{boss.split()[1].lower()}@{domain}>", subject, body


def _gen_credential(rng: random.Random) -> tuple[str, str, str]:
    brand, ref = rng.choice(_BRANDS)
    domain = _lookalike_domain(rng, ref)
    url = f"http://{domain}/{rng.choice(['secure', 'verify', 'login', 'account'])}"
    subject = rng.choice([
        f"{brand}: unusual sign-in attempt",
        f"Your {brand} password expires today",
        f"Action required on your {brand} account",
        f"{brand} security alert",
    ])
    body = (
        f"We detected {rng.choice(['an unrecognized sign-in', 'unusual activity', 'a login from a new device'])} "
        f"on your {brand} account. To keep it secure, confirm your identity here: {url} . "
        f"If you do not verify within {rng.choice([24, 48, 72])} hours your access will be "
        f"{rng.choice(['limited', 'suspended', 'locked'])}."
    )
    return f"{brand} Service <{rng.choice(['service', 'security', 'no-reply'])}@{domain}>", subject, body


def _gen_fake_invoice(rng: random.Random) -> tuple[str, str, str]:
    company = rng.choice(_COMPANIES)
    domain = _lookalike_domain(rng, f"{company}-billing")
    inv = rng.randint(10000, 99999)
    url = f"http://{domain}/invoice{inv}.zip"
    amount = f"${rng.randint(1, 9)},{rng.randint(100, 999)}.00"
    subject = f"Outstanding invoice #{inv} - immediate payment required"
    body = (
        f"Please find your overdue invoice of {amount}. Payment is OVERDUE. "
        f"Open the document and follow the payment instructions immediately to avoid a "
        f"{rng.choice([5, 10, 15])}% late fee and service interruption. Download: {url}"
    )
    return f"Accounts <{rng.choice(['billing', 'invoicing', 'accounts'])}@{domain}>", subject, body


def _gen_delivery(rng: random.Random) -> tuple[str, str, str]:
    brand, ref = rng.choice([b for b in _BRANDS if b[0] in ("DHL", "Amazon")] or _BRANDS)
    domain = _lookalike_domain(rng, ref)
    url = f"http://{domain}/track/{rng.randint(100000, 999999)}"
    subject = rng.choice([
        "Your package is on hold",
        "Delivery failed - action needed",
        "Reschedule your delivery",
    ])
    body = (
        f"Your parcel could not be delivered because of an unpaid "
        f"{rng.choice(['customs fee', 'redelivery charge'])} of "
        f"${rng.randint(1, 4)}.{rng.randint(10, 99)}. Confirm your details and pay here "
        f"within 24 hours or the package will be returned: {url}"
    )
    return f"{brand} <{rng.choice(['delivery', 'parcel', 'no-reply'])}@{domain}>", subject, body


def _gen_it_credential(rng: random.Random) -> tuple[str, str, str]:
    """Generic IT-helpdesk / mailbox credential phishing — NO consumer brand.

    The 'your mailbox is over quota / will be deactivated, re-verify here' tactic
    impersonates an internal IT desk rather than a brand, so ``_gen_credential``
    (which always picks a ``_BRANDS`` brand) never produced it. v2 missed exactly
    this on the acceptance set. Wording is kept distinct from the holdout .eml.
    """
    host_stem = rng.choice([
        "mail-administrator", "webmail-team", "account-services",
        "mailbox-support", "email-team", "it-helpdesk", "secure-mailbox",
        "email-verification",
    ])
    tld = rng.choice(_TLDS_SUSPICIOUS)
    domain = f"{host_stem}.{tld}"
    url = f"http://{domain}/{rng.choice(['verify', 'revalidate', 'signin', 'renew', 'restore'])}"
    subject = rng.choice([
        "Mailbox storage limit reached",
        "Re-verify your email account",
        "Your account is scheduled for suspension",
        "Email account verification required",
        "Action needed to keep your mailbox active",
    ])
    body = (
        f"Your {rng.choice(['mailbox', 'email account', 'webmail account'])} has "
        f"{rng.choice(['reached its storage limit', 'pending security updates', 'unverified recent activity'])} "
        f"and will be {rng.choice(['suspended', 'locked', 'restricted'])} within "
        f"{rng.choice([12, 24, 48])} hours. To keep access, "
        f"{rng.choice(['re-verify your details', 'confirm your identity', 'update your information'])} "
        f"here: {url} . Failure to act will result in "
        f"{rng.choice(['loss of access', 'account closure', 'permanent deactivation'])}."
    )
    sender = (
        f"{rng.choice(['IT Service Desk', 'Account Security', 'Mail Administrator', 'Help Desk'])} "
        f"<{rng.choice(['support', 'admin', 'no-reply', 'security'])}@{domain}>"
    )
    return sender, subject, body


_PHISHING_GENERATORS = [
    _gen_bec, _gen_credential, _gen_credential, _gen_it_credential,
    _gen_it_credential, _gen_fake_invoice, _gen_delivery, _gen_fake_invoice,
]


# --- Legit generators (label 0) -------------------------------------------
# Modern transactional / internal mail — the distribution v1+v2 over-flagged.
# Some carry a URL, but on the brand's REAL domain (signal: legit host).

def _gen_order_confirmation(rng: random.Random) -> tuple[str, str, str]:
    company = rng.choice(_COMPANIES)
    domain = _legit_domain(rng, company)
    order = rng.randint(10000, 99999)
    subject = f"Your order #{order} is confirmed"
    body = (
        f"Thanks for shopping with {company}! Order #{order} has been confirmed and will "
        f"ship within {rng.choice([2, 3, 5])} business days. You can track it any time from "
        f"your account at https://{domain}/orders . No action is needed."
    )
    return f"{company} <no-reply@{domain}>", subject, body


def _gen_receipt(rng: random.Random) -> tuple[str, str, str]:
    company = rng.choice(_COMPANIES)
    domain = _legit_domain(rng, company)
    subject = rng.choice(["Your receipt", "Payment received", "Subscription renewed"])
    body = (
        f"This confirms your payment of ${rng.randint(5, 120)}.{rng.randint(10, 99)} to "
        f"{company}. A copy of your receipt is available in your account. "
        f"Thank you for your business."
    )
    return f"{company} Billing <billing@{domain}>", subject, body


def _gen_password_reset(rng: random.Random) -> tuple[str, str, str]:
    brand, ref = rng.choice(_BRANDS)
    subject = f"Reset your {brand} password"
    body = (
        f"You asked to reset your {brand} password. Click the link to choose a new one: "
        f"https://{ref}/account/reset . This link expires in 30 minutes. "
        f"If you didn't request this, you can safely ignore this email."
    )
    return f"{brand} <no-reply@{ref}>", subject, body


def _gen_notification(rng: random.Random) -> tuple[str, str, str]:
    company = rng.choice(_COMPANIES)
    domain = _legit_domain(rng, company)
    subject = rng.choice([
        "Your monthly statement is ready",
        "New sign-in to your account",
        "Your report is ready to view",
    ])
    body = (
        f"Hi, your {rng.choice(['monthly statement', 'usage report', 'account summary'])} "
        f"from {company} is ready. View it from your dashboard at https://{domain}/dashboard . "
        f"This is an automated message."
    )
    return f"{company} <notifications@{domain}>", subject, body


def _gen_internal_urgent(rng: random.Random) -> tuple[str, str, str]:
    sender = _name(rng)
    subject = rng.choice([
        "Need the deck before 3pm",
        "Can you review this today?",
        "Quick question on the report",
    ])
    body = (
        f"Hi, could you {rng.choice(['send over the latest deck', 'review the draft', 'check the numbers'])} "
        f"before {rng.choice(['the 3pm sync', 'end of day', 'the client call'])}? "
        f"{rng.choice(['Thanks!', 'Appreciate it.', 'No rush after that.'])} - {sender.split()[0]}"
    )
    return f"{sender} <{sender.split()[0].lower()}.{sender.split()[1].lower()}@yourcompany.example>", subject, body


_LEGIT_GENERATORS = [
    _gen_order_confirmation, _gen_receipt, _gen_password_reset,
    _gen_notification, _gen_internal_urgent, _gen_order_confirmation,
]


def generate_phishing_augmentation(n_per_class: int = 400, seed: int = 42):
    """Build a balanced, header-complete synthetic augmentation ``Dataset``.

    Args:
        n_per_class: how many phishing AND how many legit rows to generate
            (the result has ``2 * n_per_class`` rows, balanced 1:1).
        seed: RNG seed — record it in the training config for reproducibility.

    Returns:
        A HuggingFace ``Dataset`` with columns
        ``sender_email, subject, body, phishing, labels`` (same schema as
        ``loaders.load_phishing_dataset``).
    """
    from datasets import Dataset

    rng = _rng(seed)
    rows: list[dict] = []

    for label_str, generators in (("phishing", _PHISHING_GENERATORS), ("legit", _LEGIT_GENERATORS)):
        label_id = PHISHING_LABEL2ID[label_str]
        for _ in range(n_per_class):
            gen = rng.choice(generators)
            sender, subject, body = gen(rng)
            rows.append({
                "sender_email": sender,
                "subject": subject,
                "body": body,
                "phishing": label_str,
                "labels": label_id,
            })

    rng.shuffle(rows)
    logger.info(
        "Generated %d synthetic augmentation rows (%d phishing / %d legit, seed=%d)",
        len(rows), n_per_class, n_per_class, seed,
    )
    return Dataset.from_list(rows)


def augment(dataset, seed: int):  # noqa: ARG001 - kept for the priority-domain API
    """Priority-domain augmentation stub (unchanged). See module docstring for the
    phishing augmentation entry point, ``generate_phishing_augmentation``."""
    raise NotImplementedError("augment: no priority augmentation strategy decided yet.")
