"""CMP-1: Email Corpus Processor.

Interface contract (architecture.tex, CMP-1):
    generate_counterfactuals(legitimate_emails: List[EmailRecord],
                              phishing_emails: List[EmailRecord]
                              ) -> List[CounterfactualPair]

For each benign email, appends a high-risk action clause (Family A slot
templates) to build its phishing counterfactual, preserving the original
subject, greeting and body wording untouched.

For each phishing email that exhibits one of the five architecture-defined
action-risk categories (credential verification, payment redirection,
document download, MFA approval, reply-with-data), removes the sentence(s)
carrying that action and appends a matching neutral clause, to build its
benign counterfactual.

Revision note (see results/RESULTS.md, "FPR improvement pass"): earlier,
`_detect_category` defaulted to "credential_verification" for any
phishing email that matched none of the five categories. Measurement on
this corpus showed that fewer than 2% of its "Phishing Email" records
(mostly 419-style advance-fee scams, adult/pharma spam, and stock
pump-and-dump content, not BEC-style account phishing) actually match any
of the five categories -- so the fallback was firing on ~98% of inputs,
producing a "benign counterfactual" that differed from the original
phishing email in nothing this prototype's features can see. In
validation, pairs built this way were ranked correctly only 11% of the
time (worse than chance), actively teaching the ranker the wrong
direction. The fallback has been removed: a phishing email that matches
no category is now skipped rather than forced into a misleading pair.
This is a real limitation of this specific open dataset relative to the
architecture's five-category taxonomy, not a limitation of the mechanism
itself -- see ARCHITECTURE_SUMMARY.md, "Deviations".

Both directions are deterministic given a fixed random seed -- no ML
paraphraser is used (see ARCHITECTURE_SUMMARY.md, "Deviations").
"""

from __future__ import annotations

import random
import re
from typing import List, Optional, Tuple

from features import _ACTION_PATTERNS
from models import CounterfactualPair, EmailRecord
from templates import (
    FAMILY_A_RISKY,
    NEUTRAL_TEMPLATES,
    RISK_CATEGORIES,
    extract_entities,
    fill,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# features.py's _ACTION_PATTERNS keys (action_credential, ...) name the
# same five categories as templates.py's RISK_CATEGORIES
# (credential_verification, ...) but with different spellings, since one
# is a feature name and the other a template-dict key. Map between them
# rather than keeping the taxonomy in two different vocabularies.
_FEATURE_KEY_TO_RISK_CATEGORY = {
    "action_credential": "credential_verification",
    "action_payment": "payment_redirection",
    "action_document": "document_download",
    "action_mfa": "mfa_approval",
    "action_reply_data": "reply_with_data",
}
_RISK_CATEGORY_TO_FEATURE_KEY = {v: k for k, v in _FEATURE_KEY_TO_RISK_CATEGORY.items()}

GENERIC_PHISH_KEYWORDS_RE = re.compile(
    r"\b(click here|verify your account|suspended|urgent action|act now|limited time|"
    r"confirm your identity|unauthorized (login|access)|your account (will be|has been))\b",
    re.IGNORECASE,
)


def _sentences(text: str) -> List[str]:
    return [s for s in SENTENCE_SPLIT_RE.split(text.strip()) if s]


def _detect_category(text: str) -> Optional[str]:
    """Returns the first of the five architecture-defined action-risk
    categories whose pattern matches, or None if the email matches none
    of them (see module docstring -- this is common on this corpus, and
    is a case to skip, not to guess at)."""
    for feature_key, pattern in _ACTION_PATTERNS.items():
        if pattern.search(text):
            return _FEATURE_KEY_TO_RISK_CATEGORY[feature_key]
    return None


def build_phishing_counterfactual(
    benign: EmailRecord, category: str, rng: random.Random
) -> Tuple[EmailRecord, str]:
    entities = extract_entities(benign.body)
    template = rng.choice(FAMILY_A_RISKY[category])
    clause = fill(template, entities)
    new_body = benign.body.rstrip() + "\n\n" + clause
    counterfactual = EmailRecord(
        subject=benign.subject,
        body=new_body,
        sender=benign.sender,
        timestamp=benign.timestamp,
        label=1,
        subtype="counterfactual_phishing",
        source_id=f"{benign.source_id}::cf_phish::{category}",
    )
    return counterfactual, category


def build_benign_counterfactual(
    phishing: EmailRecord, rng: random.Random
) -> Optional[Tuple[EmailRecord, str]]:
    category = _detect_category(phishing.body)
    if category is None:
        return None
    feature_key = _RISK_CATEGORY_TO_FEATURE_KEY[category]
    sentences = _sentences(phishing.body)
    kept = [
        s
        for s in sentences
        if not _ACTION_PATTERNS[feature_key].search(s) and not GENERIC_PHISH_KEYWORDS_RE.search(s)
    ]
    if not kept:
        kept = sentences[:1]  # keep at least the opening line for length parity
    entities = extract_entities(phishing.body)
    template = rng.choice(NEUTRAL_TEMPLATES[category])
    clause = fill(template, entities)
    new_body = " ".join(kept).rstrip() + "\n\n" + clause
    counterfactual = EmailRecord(
        subject=phishing.subject,
        body=new_body,
        sender=phishing.sender,
        timestamp=phishing.timestamp,
        label=0,
        subtype="counterfactual_benign",
        source_id=f"{phishing.source_id}::cf_benign::{category}",
    )
    return counterfactual, category


def generate_counterfactuals(
    legitimate_emails: List[EmailRecord],
    phishing_emails: List[EmailRecord],
    seed: int = 42,
) -> List[CounterfactualPair]:
    pairs, _stats = generate_counterfactuals_with_stats(legitimate_emails, phishing_emails, seed)
    return pairs


def generate_counterfactuals_with_stats(
    legitimate_emails: List[EmailRecord],
    phishing_emails: List[EmailRecord],
    seed: int = 42,
) -> Tuple[List[CounterfactualPair], dict]:
    rng = random.Random(seed)
    pairs: List[CounterfactualPair] = []

    for i, benign in enumerate(legitimate_emails):
        category = RISK_CATEGORIES[i % len(RISK_CATEGORIES)]
        phishing_cf, used_category = build_phishing_counterfactual(benign, category, rng)
        pairs.append(
            CounterfactualPair(
                benign_email=benign,
                phishing_email=phishing_cf,
                changed_slot=used_category,
                origin="benign_to_phishing",
            )
        )

    n_phishing_skipped = 0
    for phishing in phishing_emails:
        result = build_benign_counterfactual(phishing, rng)
        if result is None:
            n_phishing_skipped += 1
            continue
        benign_cf, used_category = result
        pairs.append(
            CounterfactualPair(
                benign_email=benign_cf,
                phishing_email=phishing,
                changed_slot=used_category,
                origin="phishing_to_benign",
            )
        )

    stats = {
        "n_legitimate_input": len(legitimate_emails),
        "n_phishing_input": len(phishing_emails),
        "n_phishing_skipped_no_category": n_phishing_skipped,
        "n_phishing_used": len(phishing_emails) - n_phishing_skipped,
    }
    return pairs, stats
