"""Deterministic slot templates used by CMP-1 (Email Corpus Processor) to
build metamorphic counterfactual pairs, and by the synthetic LLM-phishing
test-set generator.

Two disjoint template families are kept on purpose:

- FAMILY_A is used to build the CMP-1 training counterfactual pairs
  (generate_counterfactuals in counterfactuals.py).
- FAMILY_B is used only to build the synthetic "LLM-generated phishing"
  held-out test set (synthetic_llm_phishing.py). It is never seen during
  training, so the held-out evaluation also functions as the
  leave-transformation-family-out check the architecture calls for
  (architecture.tex, "Key mechanisms", bullet 5).

Neither family is a real LLM. Both are hand-written string templates with
slot substitution -- see ARCHITECTURE_SUMMARY.md for why (T5-small
paraphrasing was scoped out as an optional extra, not a required
dependency).
"""

from __future__ import annotations

import re

# Five high-risk action categories named explicitly in architecture.tex,
# section "The Proposed Solution" -> "Mechanism".
RISK_CATEGORIES = [
    "credential_verification",
    "payment_redirection",
    "document_download",
    "mfa_approval",
    "reply_with_data",
]

FAMILY_A_RISKY = {
    "credential_verification": [
        "To keep {org} access active, please verify your login credentials now at {link}.",
        "Your {org} password will expire today -- confirm your username and password at {link} to avoid lockout.",
    ],
    "payment_redirection": [
        "Our payment details have changed; please redirect your next transfer of {amount} to the account listed at {link}.",
        "Please update the billing destination for {org} and resend the {amount} payment before {deadline}.",
    ],
    "document_download": [
        "A document requiring your signature is waiting -- download it from {link} before {deadline}.",
        "Please open and download the attached {org} statement at {link} to review the changes.",
    ],
    "mfa_approval": [
        "We sent a login approval request to your device; please approve the {org} MFA prompt within 10 minutes.",
        "Your account needs a one-time MFA approval -- tap approve on the {org} prompt sent to your phone.",
    ],
    "reply_with_data": [
        "Please reply to this email with your {org} employee ID and date of birth to confirm your identity.",
        "Kindly reply with your account number and the last four digits of your SSN so {org} can process this.",
    ],
}

FAMILY_B_RISKY = {
    "credential_verification": [
        "For continued {org} service, re-authenticate your account credentials through the secure portal at {link}.",
        "A credential mismatch was flagged on your {org} profile; sign in again at {link} to clear it.",
    ],
    "payment_redirection": [
        "{org} has switched banking providers -- route the outstanding {amount} to the new account referenced at {link}.",
        "Finance has issued a corrected invoice; send the {amount} settlement to the updated details at {link} by {deadline}.",
    ],
    "document_download": [
        "A revised {org} contract is ready for your review -- retrieve it from {link} prior to {deadline}.",
        "Please fetch the encrypted file {org} shared with you at {link} and open it on this device.",
    ],
    "mfa_approval": [
        "{org} security noticed an unrecognized sign-in; approve the verification prompt on your device to keep access.",
        "Confirm it was you -- approve the two-factor prompt {org} just sent to your registered device.",
    ],
    "reply_with_data": [
        "Reply directly to this message with your {org} badge number and home address for the compliance file.",
        "{org} payroll needs your bank routing number and account number by return email to avoid a delay.",
    ],
}

NEUTRAL_TEMPLATES = {
    "credential_verification": [
        "You can review your {org} account settings at your convenience by logging in normally through the usual site.",
        "No action is required on your {org} login; this is just a courtesy notice.",
    ],
    "payment_redirection": [
        "The {org} invoice has been recorded and no changes to payment details are needed on your side.",
        "Thanks for the {amount} payment already on file with {org}; nothing further is required.",
    ],
    "document_download": [
        "A summary of the {org} document is included below for your records; no download is necessary.",
        "The {org} statement will be available in your usual mailbox as normal, no action needed.",
    ],
    "mfa_approval": [
        "Your {org} account security settings are unchanged and no approval is needed at this time.",
        "This is a routine {org} security notice; there is nothing for you to approve.",
    ],
    "reply_with_data": [
        "There is no need to reply with any personal details; {org} already has what it needs on file.",
        "If you have questions about {org}, please use the normal support line rather than replying here.",
    ],
}

URGENCY_PHRASES = [
    "This requires your attention right away.",
    "Please handle this before the end of the day.",
]

GREETING_RE = re.compile(r"^\s*(dear|hi|hello|hey)\s+([a-z][a-z\.\-]{1,20})", re.IGNORECASE)
ORG_HINT_RE = re.compile(
    r"\b(bank|paypal|microsoft|amazon|apple|hr|it department|payroll|finance|accounting|"
    r"helpdesk|support team|admin(?:istration)?)\b",
    re.IGNORECASE,
)

DEFAULT_ORG = "the account services team"
DEFAULT_LINK = "http://account-verify-update.example-secure.com/login"
DEFAULT_AMOUNT = "$4,250.00"
DEFAULT_DEADLINE = "Friday"


def extract_entities(text: str) -> dict:
    """Very small heuristic entity extractor: greeting name and an org hint,
    used only to keep generated counterfactuals plausible. Falls back to
    generic placeholders when nothing is found -- this is a documented
    simplification, not a full NER pipeline."""
    org_match = ORG_HINT_RE.search(text)
    org = org_match.group(1).title() if org_match else DEFAULT_ORG
    return {
        "org": org,
        "link": DEFAULT_LINK,
        "amount": DEFAULT_AMOUNT,
        "deadline": DEFAULT_DEADLINE,
    }


def fill(template: str, entities: dict) -> str:
    try:
        return template.format(**entities)
    except KeyError:
        return template
