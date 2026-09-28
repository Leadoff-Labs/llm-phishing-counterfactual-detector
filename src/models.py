"""Data model classes matching the architecture's Data Models section
(architecture.tex, section "Data Models and Interface Contracts")."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class EmailRecord:
    subject: str
    body: str
    sender: str
    timestamp: datetime
    # Extra fields needed to run and evaluate the pipeline, not in the
    # architecture's minimal contract but required to track provenance.
    label: int = 0  # 0 = legitimate, 1 = phishing
    subtype: str = "legitimate"  # "legitimate" | "traditional_phishing" | "llm_generated_phishing"
    source_id: str = ""


@dataclass
class CounterfactualPair:
    benign_email: EmailRecord
    phishing_email: EmailRecord
    # Provenance fields (not in the architecture's minimal contract):
    # which semantic slot was changed to build this pair, and which side
    # was the original vs. the generated counterfactual.
    changed_slot: str = ""
    origin: str = ""  # "benign_to_phishing" | "phishing_to_benign"


@dataclass
class EvaluationReport:
    recall: float
    false_positive_rate: float
    comparison_to_baseline: str
    # Extra breakdown fields needed to check the success criterion, not in
    # the architecture's minimal contract.
    extra: dict = field(default_factory=dict)
