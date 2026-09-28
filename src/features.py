"""Feature extraction shared by CMP-2 (ranker training) and CMP-4
(evaluation): "features for requested action, recipient behavior, channel,
urgency, artifact sufficiency, and contextual consistency, rather than
stylometric fingerprints" (architecture.tex, "Mechanism").

Scope note: this is a static-corpus prototype, not a live mail gateway, so
there is no real recipient-interaction log and only one channel (email).
"Channel" is therefore not a feature (constant across the whole corpus,
carries no information) and "recipient behavior" is approximated by
proxy features for what the message *asks the recipient to do*
(click / reply / approve / download), which is the only recipient-facing
signal available in a static text corpus. This narrowing is recorded in
ARCHITECTURE_SUMMARY.md.

Every regex here matches at the semantic/category level (verify+password,
transfer+account, ...), not the literal wording of any one template
family, so that a detector trained only on FAMILY_A phrasing is not
trivially memorizing FAMILY_B's held-out phrasing.

Revision note (see results/RESULTS.md, "FPR improvement pass"): an
earlier version of this module included a raw `log_word_count` feature.
It turned out to dominate the trained ranker's feature importances (by
far the single largest gain of any feature) purely because of how CMP-1
builds its training pairs -- benign-to-phishing counterfactuals APPEND a
clause (longer) and phishing-to-benign counterfactuals STRIP sentences
(shorter) -- a construction artifact, not a real signal about phishing
risk. It also runs backwards on this corpus: real legitimate email here
averages 686 words vs. 302 for real phishing, so a model that leans on
"short = risky" flags short, legitimate one-line business emails
("see attached file", "fyi") as phishing. The feature has been removed
entirely rather than reweighted, since nothing in the architecture's
mechanism calls for a length feature.

This corpus also mixes two text stylesa raw text and text that has been
tokenized with spaces inserted around punctuation (e.g. "$ 525 , 750",
"jan . 17 , 2001") -- roughly two-thirds of records, in both classes. The
regexes below are written to tolerate optional whitespace around such
punctuation so they fire consistently across both stylea a plain
substring/URL regex silently misses most of the tokenized records.
"""

from __future__ import annotations

import math
import re
from typing import List

FEATURE_NAMES = [
    "action_credential",
    "action_payment",
    "action_document",
    "action_mfa",
    "action_reply_data",
    "urgency_score",
    "artifact_sufficiency",
    "generic_greeting",
    "link_or_url_mention",
    "suspicious_link_wording",
    "click_request",
]

_ACTION_PATTERNS = {
    "action_credential": re.compile(
        r"\b(verify|re-?verify|confirm|re-?authenticat\w*|sign\s*in)\b.{0,60}"
        r"\b(password|login|credential\w*|account|profile)\b",
        re.IGNORECASE,
    ),
    "action_payment": re.compile(
        r"\b(payment|invoice|transfer|wire|settlement|billing|routing)\b.{0,60}"
        r"\b(account|bank|redirect\w*|updat\w*|rout\w*)\b",
        re.IGNORECASE,
    ),
    "action_document": re.compile(
        r"\b(download|open|retriev\w*|fetch)\b.{0,80}"
        r"\b(document|attachment|file|statement|contract)\b"
        r"|\b(document|attachment|file|statement|contract)\b.{0,80}"
        r"\b(download|open|retriev\w*|fetch)\b",
        re.IGNORECASE,
    ),
    "action_mfa": re.compile(
        r"\b(mfa|two-?factor|2fa|one-?time|verification)\b.{0,60}"
        r"\b(approve|approval|prompt|code)\b",
        re.IGNORECASE,
    ),
    "action_reply_data": re.compile(
        r"\breply\b.{0,80}\b(ssn|social\s*security|account\s*number|date\s*of\s*birth|"
        r"employee\s*id|badge\s*number|routing\s*number)\b",
        re.IGNORECASE,
    ),
}

_URGENCY_RE = re.compile(
    r"\b(urgent\w*|immediately|right\s*away|act\s*now|act\s*fast|"
    r"within\s*\d+\s*(minutes|hours|days)|"
    r"final\s*notice|expires?\s*(today|soon)|as\s*soon\s*as\s*possible|asap|"
    r"before\s*(the\s*)?end\s*of\s*(the\s*)?day|"
    r"will\s*be\s*(suspended|locked|closed|terminated))\b",
    re.IGNORECASE,
)

# Tolerates tokenized punctuation ("$ 525 , 750", "01 / 17 / 2001",
# "PO - 4821") as well as untokenized ("$525,750").
_ARTIFACT_RE = re.compile(
    r"\b[A-Z]{2,6}\s?-\s?\d{3,8}\b"
    r"|\$\s?\d[\d\s,]{0,12}(\.\s?\d{2})?"
    r"|\b\d{1,2}\s?/\s?\d{1,2}\s?/\s?\d{2,4}\b"
)

# Not anchored to the start of the text: greetings in this corpus can be
# preceded by forwarded-message headers or blank lines.
_GENERIC_GREETING_RE = re.compile(
    r"\b(dear|hi|hello)\s+(customer|user|member|valued\s*customer|account\s*holder|"
    r"sir\s*/\s*madam|sir|madam|applicant)\b",
    re.IGNORECASE,
)

# Matches both contiguous URLs and tokenized ones ("http : / / www . x . com").
_LINK_RE = re.compile(r"\bhttps?\s?:\s?/\s?/\s?\S+|\bwww\s?\.\s?\S+", re.IGNORECASE)

_SUSPICIOUS_LINK_WORDING_RE = re.compile(
    r"\bverify\b.{0,40}\b(link|here|below|now)\b"
    r"|\b(secure|account)[\s-]?(update|verify|confirm)\b"
    r"|\blogin[\s-]?confirm\b|\bsign[\s-]?in[\s-]?secure\b",
    re.IGNORECASE,
)

_CLICK_REQUEST_RE = re.compile(
    r"\bclick\b.{0,30}\b(here|link|below|now|button|this)\b",
    re.IGNORECASE,
)


def extract_features(text: str) -> List[float]:
    values = []
    for pattern in _ACTION_PATTERNS.values():
        values.append(float(len(pattern.findall(text))))

    values.append(float(len(_URGENCY_RE.findall(text))))
    values.append(float(len(_ARTIFACT_RE.findall(text))))
    values.append(1.0 if _GENERIC_GREETING_RE.search(text) else 0.0)
    values.append(float(len(_LINK_RE.findall(text))))
    values.append(1.0 if _SUSPICIOUS_LINK_WORDING_RE.search(text) else 0.0)
    values.append(1.0 if _CLICK_REQUEST_RE.search(text) else 0.0)

    return values
