"""Loads the open dataset and splits it into disjoint pools.

Dataset: "Phishing Email Detection" (Kaggle, user Cyber Cop / subhajournal),
mirrored openly on Hugging Face at
https://huggingface.co/datasets/zefang-liu/phishing-email-dataset under the
LGPL-3.0 license (see data/raw/README_dataset.md for the retrieved dataset
card). 18,650 rows: 11,322 "Safe Email" (legitimate) and 7,328
"Phishing Email" (traditional phishing). No LLM-generated-phishing split
exists in this or any other open, credential-free dataset found during this
build -- see synthetic_llm_phishing.py for how that class is constructed
instead.

All pools below are disjoint (no email appears in more than one pool), so
nothing used to build CMP-1's training counterfactuals, the synthetic
LLM-phishing set, or threshold calibration ever reappears in the held-out
test set.
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from models import EmailRecord

RAW_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "Phishing_Email.csv"
PLACEHOLDER_TIMESTAMP = datetime(2000, 1, 1)


def _to_records(df: pd.DataFrame, label: int, subtype: str, prefix: str) -> List[EmailRecord]:
    records = []
    for idx, row in df.iterrows():
        body = str(row["Email Text"]).strip()
        if not body or body.lower() == "nan":
            continue
        subject = body.split("\n", 1)[0][:80]
        records.append(
            EmailRecord(
                subject=subject,
                body=body,
                sender="unknown@corpus.invalid",
                timestamp=PLACEHOLDER_TIMESTAMP,
                label=label,
                subtype=subtype,
                source_id=f"{prefix}::{idx}",
            )
        )
    return records


def load_pools(seed: int = 42) -> dict:
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=["Email Text"])
    df = df.drop_duplicates(subset=["Email Text"])

    ham_df = df[df["Email Type"] == "Safe Email"]
    phish_df = df[df["Email Type"] == "Phishing Email"]

    ham_all = _to_records(ham_df, label=0, subtype="legitimate", prefix="kaggle_ham")
    phish_all = _to_records(phish_df, label=1, subtype="traditional_phishing", prefix="kaggle_phish")

    rng = random.Random(seed)
    rng.shuffle(ham_all)
    rng.shuffle(phish_all)

    # Calibration widened from 800 to 2000 (see results/RESULTS.md, "FPR
    # improvement pass"): with a coarse, regularized ranker's score
    # distribution having only ~40-50 distinct values, an 800-email
    # calibration split put most of its mass in 1-2 score buckets, making
    # the percentile-based threshold noisy against a differently-sized
    # test split. A larger calibration split reduces that noise.
    ham_cf_source = ham_all[0:2000]
    ham_calibration = ham_all[2000:4000]
    ham_llm_source = ham_all[4000:5500]
    ham_test = ham_all[5500:8500]

    # Widened vs. an earlier version (0:2000) because only ~2% of this
    # corpus's phishing emails match any of the five action-risk
    # categories CMP-1 needs (see counterfactuals.py); scanning more
    # candidates yields more usable phishing_to_benign pairs while
    # phish_test stays the same size and stays disjoint from this pool.
    phish_cf_source = phish_all[0:3557]
    phish_test = phish_all[3557:6557]

    return {
        "ham_cf_source": ham_cf_source,
        "ham_calibration": ham_calibration,
        "ham_llm_source": ham_llm_source,
        "ham_test": ham_test,
        "phish_cf_source": phish_cf_source,
        "phish_test": phish_test,
        "counts_total": {"ham_total": len(ham_all), "phish_total": len(phish_all)},
    }


if __name__ == "__main__":
    pools = load_pools()
    for name, pool in pools.items():
        if name == "counts_total":
            print("counts_total:", pool)
        else:
            print(f"{name}: {len(pool)}")
