"""Builds the synthetic "LLM-generated phishing" test set.

No openly downloadable, credential-free dataset of real LLM-generated
phishing emails was found during this build (see ARCHITECTURE_SUMMARY.md,
"Datasets" -- the closest candidate, S1's balanced GPT-4.1/DeepSeek/Llama
corpus, was assembled by that paper's authors and is not released
publicly). Per the task brief, a clearly labeled synthetic set is
generated instead, documented in full here.

Method: take genuinely fluent, human-written business email bodies (from
the reserved "llm_source" slice of the Enron ham pool -- disjoint from the
emails used to build CMP-1's training counterfactuals) and graft a
high-risk action clause onto them using FAMILY_B_RISKY templates, which
are never used during CMP-1 training. The base text supplies the fluency,
personalization and grammatical cleanliness that the research brief
attributes to LLM-generated phishing (review.tex S1/S5); the grafted
clause supplies the malicious intent. This is a proxy for LLM-generated
phishing, not a captured LLM output, and every generated record's
`source_id` is tagged accordingly so it can never be mistaken for real
data downstream.
"""

from __future__ import annotations

import random
from typing import List

from models import EmailRecord
from templates import FAMILY_B_RISKY, RISK_CATEGORIES, extract_entities, fill


def generate_llm_phishing_synthetic(
    ham_source: List[EmailRecord], seed: int = 1337
) -> List[EmailRecord]:
    rng = random.Random(seed)
    out: List[EmailRecord] = []
    for i, ham in enumerate(ham_source):
        category = RISK_CATEGORIES[i % len(RISK_CATEGORIES)]
        entities = extract_entities(ham.body)
        template = rng.choice(FAMILY_B_RISKY[category])
        clause = fill(template, entities)
        new_body = ham.body.rstrip() + "\n\n" + clause
        out.append(
            EmailRecord(
                subject=ham.subject,
                body=new_body,
                sender=ham.sender,
                timestamp=ham.timestamp,
                label=1,
                subtype="llm_generated_phishing",
                source_id=f"{ham.source_id}::synth_llm_phish::{category}",
            )
        )
    return out
