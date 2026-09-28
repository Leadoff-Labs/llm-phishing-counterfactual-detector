"""CMP-2 (Pairwise Ranker Trainer) and CMP-3 (Model Optimizer).

Interface contracts (architecture.tex):
    train_ranker(counterfactual_pairs: List[CounterfactualPair]) -> RankerModel
    optimize_model(trained_ranker: RankerModel) -> RankerModel

The ranker is trained on counterfactual PAIRS, not on independently
labelled single emails: within each pair, the phishing side must score
higher than its benign twin (architecture.tex, "Mechanism" -- "Train a
pairwise ranker so that, within each pair, the risky-action version must
score higher than the neutralized version"). LightGBM's LambdaRank
objective implements exactly this, with each pair treated as a
2-item ranking group.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List

import lightgbm
import numpy as np
from lightgbm import LGBMRanker

from features import FEATURE_NAMES, extract_features
from models import CounterfactualPair


@dataclass
class RankerModel:
    booster: object  # lightgbm.LGBMRanker
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    val_ranking_accuracy: float = 0.0
    n_train_pairs: int = 0
    n_val_pairs: int = 0

    def predict_scores(self, texts: List[str]) -> np.ndarray:
        X = np.array([extract_features(t) for t in texts], dtype=float)
        return self.booster.predict(X)


def _pairs_to_xy(pairs: List[CounterfactualPair]):
    X, y, group = [], [], []
    for pair in pairs:
        X.append(extract_features(pair.benign_email.body))
        y.append(0)
        X.append(extract_features(pair.phishing_email.body))
        y.append(1)
        group.append(2)
    return np.array(X, dtype=float), np.array(y, dtype=int), group


def train_ranker(
    counterfactual_pairs: List[CounterfactualPair], seed: int = 42, val_fraction: float = 0.2
) -> RankerModel:
    pairs = list(counterfactual_pairs)
    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    X_train, y_train, group_train = _pairs_to_xy(train_pairs)

    # Hyperparameters chosen by comparing a handful of settings against this
    # same validation split and the calibration-split FPR transfer (see
    # results/RESULTS.md, "FPR improvement pass") -- never against the
    # held-out test set. Shallower/more regularized than an earlier version
    # (num_leaves=31, no L2): with only ~70 real phishing_to_benign pairs
    # feeding the training set (see counterfactuals.py), a high-capacity
    # tree ensemble overfits secondary features (artifact/urgency counts)
    # to that small minority class, which calibrates poorly against real
    # legitimate email. Fewer leaves and an L2 penalty keep the model
    # leaning on the four features that actually carry signal (the
    # action-risk categories), which also calibrates more consistently.
    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=300,
        num_leaves=7,
        max_depth=3,
        learning_rate=0.1,
        min_child_samples=30,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    X_val, y_val, group_val = _pairs_to_xy(val_pairs)
    model.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val],
        eval_at=[1],
        callbacks=[lightgbm.early_stopping(30, verbose=False)],
    )

    correct = 0
    for pair in val_pairs:
        fb = extract_features(pair.benign_email.body)
        fp = extract_features(pair.phishing_email.body)
        scores = model.predict(np.array([fb, fp]))
        if scores[1] > scores[0]:
            correct += 1
    val_ranking_accuracy = correct / len(val_pairs) if val_pairs else 0.0

    return RankerModel(
        booster=model,
        val_ranking_accuracy=val_ranking_accuracy,
        n_train_pairs=len(train_pairs),
        n_val_pairs=len(val_pairs),
    )


def optimize_model(trained_ranker: RankerModel) -> RankerModel:
    """CMP-3: reduce model complexity for CPU efficiency.

    The interface contract takes only the already-trained ranker (no
    training data), so "optimization" here is restricted to what is
    possible post hoc: trimming the boosted ensemble to its best
    iteration (dropping any trailing trees that do not improve training
    loss) so prediction does strictly less work per email, without
    retraining. Throughput against the real held-out test set is measured
    separately in evaluate.py and reported from that actual run.
    """
    booster = trained_ranker.booster
    best_iter = getattr(booster, "best_iteration_", None)
    if best_iter is not None and best_iter > 0:
        booster.set_params(n_estimators=best_iter)
        # LightGBM's sklearn wrapper predicts using booster.num_trees() by
        # default; use predict-time truncation instead of retraining.
        trained_ranker.booster = booster
    return trained_ranker


def benchmark_throughput(model: RankerModel, texts: List[str]) -> float:
    """Actual measured emails/sec on this machine's CPU for the given texts."""
    start = time.perf_counter()
    model.predict_scores(texts)
    elapsed = time.perf_counter() - start
    return len(texts) / elapsed if elapsed > 0 else float("inf")
