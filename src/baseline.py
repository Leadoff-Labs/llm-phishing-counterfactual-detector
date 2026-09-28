"""Traditional rule-based/keyword phishing baseline, built for comparison
against the novel solution (task step 5). Same input/output interface as
the novel detector: takes raw email text, returns a 0/1 label.

This is the classic keyword-spotting approach the research brief describes
as losing effectiveness against LLM-generated phishing (review.tex,
abstract): flag anything containing one or more of a fixed list of
phishing-associated phrases. No training, no adaptation.
"""

from __future__ import annotations

import re
from typing import List

KEYWORDS = [
    "verify your account",
    "verify your password",
    "confirm your account",
    "confirm your identity",
    "click here",
    "click the link below",
    "account suspended",
    "account has been suspended",
    "account will be suspended",
    "unauthorized login",
    "unauthorized access",
    "urgent action required",
    "act now",
    "act immediately",
    "limited time",
    "wire transfer",
    "update your billing",
    "suspicious activity",
    "security alert",
    "your account will be closed",
    "social security number",
    "credit card number",
    "password expires",
    "winning notification",
    "claim your prize",
    "nigerian prince",
    "congratulations you have won",
    "dear valued customer",
    "dear customer",
    "irs",
    "lottery",
]

_KEYWORD_RE = re.compile(
    "|".join(re.escape(k) for k in KEYWORDS),
    re.IGNORECASE,
)


def predict_baseline(text: str) -> int:
    return 1 if _KEYWORD_RE.search(text) else 0


def predict_baseline_batch(texts: List[str]) -> List[int]:
    return [predict_baseline(t) for t in texts]
