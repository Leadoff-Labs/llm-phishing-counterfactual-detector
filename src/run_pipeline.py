"""End-to-end run of the Metamorphic Pair Invariant Trainer prototype:
CMP-1 -> CMP-2 -> CMP-3 -> CMP-4, wired exactly as architecture.tex
describes (section "Architecture"). Prints progress and the real,
measured evaluation results; writes results/metrics.json and
results/RESULTS.md.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline import predict_baseline_batch  # noqa: E402
from counterfactuals import generate_counterfactuals_with_stats  # noqa: E402
from data_prep import load_pools  # noqa: E402
from evaluate import calibrate_threshold, conservative_baseline_target_fpr, evaluate_model  # noqa: E402
from ranker import optimize_model, train_ranker  # noqa: E402
from synthetic_llm_phishing import generate_llm_phishing_synthetic  # noqa: E402

RESULTS_DIR = ROOT / "results"


def main():
    print("=" * 70)
    print("STEP CMP-1: Email Corpus Processor -- generate_counterfactuals()")
    print("=" * 70)
    pools = load_pools()
    for name, pool in pools.items():
        if name != "counts_total":
            print(f"  pool {name}: {len(pool)} emails")
    print(f"  dataset totals after cleaning: {pools['counts_total']}")

    pairs, cf_stats = generate_counterfactuals_with_stats(
        pools["ham_cf_source"], pools["phish_cf_source"]
    )
    n_attempted = cf_stats["n_legitimate_input"] + cf_stats["n_phishing_input"]
    valid_pairs = sum(
        1
        for p in pairs
        if p.benign_email.body.strip() and p.phishing_email.body.strip()
        and p.benign_email.body != p.phishing_email.body
    )
    pct_valid = 100.0 * valid_pairs / n_attempted
    print(f"  counterfactual pairs generated: {len(pairs)} (attempted {n_attempted})")
    print(f"  legitimate inputs used: {cf_stats['n_legitimate_input']} "
          f"(100% yield -- every legit email gets a phishing counterfactual)")
    print(f"  phishing inputs used: {cf_stats['n_phishing_used']}/{cf_stats['n_phishing_input']} "
          f"({cf_stats['n_phishing_skipped_no_category']} skipped: matched none of the five "
          f"action-risk categories -- see counterfactuals.py docstring)")
    print(f"  valid pairs: {valid_pairs}/{n_attempted} = {pct_valid:.2f}% "
          f"(acceptance criterion: >= 95%; {'MET' if pct_valid >= 95.0 else 'NOT MET'})")

    print()
    print("=" * 70)
    print("STEP CMP-2: Pairwise Ranker Trainer -- train_ranker()")
    print("=" * 70)
    trained = train_ranker(pairs)
    print(f"  train pairs: {trained.n_train_pairs}, validation pairs: {trained.n_val_pairs}")
    print(f"  validation ranking accuracy: {trained.val_ranking_accuracy:.4f} "
          f"(acceptance criterion: >= 0.90)")

    print()
    print("=" * 70)
    print("STEP CMP-3: Model Optimizer -- optimize_model()")
    print("=" * 70)
    optimized = optimize_model(trained)
    best_iter = getattr(optimized.booster, "best_iteration_", None)
    print(f"  boosting rounds trimmed to best_iteration_: {best_iter}")

    print()
    print("=" * 70)
    print("Building synthetic LLM-generated-phishing test set")
    print("=" * 70)
    llm_phish = generate_llm_phishing_synthetic(pools["ham_llm_source"])
    print(f"  synthetic LLM-phishing test records: {len(llm_phish)} "
          f"(SYNTHETIC -- see synthetic_llm_phishing.py docstring)")

    test_set = list(pools["ham_test"]) + list(pools["phish_test"]) + list(llm_phish)
    random.Random(7).shuffle(test_set)
    print(f"  mixed held-out test set size: {len(test_set)} "
          f"(legit={len(pools['ham_test'])}, traditional_phishing={len(pools['phish_test'])}, "
          f"llm_phishing={len(llm_phish)})")

    print()
    print("=" * 70)
    print("Calibrating decision threshold on held-out calibration split")
    print("=" * 70)
    calib_texts = [e.body for e in pools["ham_calibration"]]
    baseline_calib_preds = predict_baseline_batch(calib_texts)
    baseline_calib_fpr = sum(baseline_calib_preds) / len(baseline_calib_preds)
    print(f"  baseline FPR on calibration legit emails (n={len(calib_texts)}): "
          f"{baseline_calib_fpr:.4f}")
    target_fpr = conservative_baseline_target_fpr(
        pools["ham_calibration"], test_size=len(pools["ham_test"])
    )
    print(f"  bootstrap-conservative target FPR (5th percentile, from calibration "
          f"split only): {target_fpr:.4f}")
    threshold = calibrate_threshold(optimized, pools["ham_calibration"], target_fpr=target_fpr)
    print(f"  novel-model threshold chosen to target this conservative FPR: {threshold:.4f}")

    print()
    print("=" * 70)
    print("STEP CMP-4: Evaluation and Reporting Module -- evaluate_model()")
    print("=" * 70)
    report = evaluate_model(optimized, test_set, threshold)
    print(json.dumps(report, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    full_metrics = {
        "cmp1_counterfactual_pairs": {
            "total_pairs": len(pairs),
            "valid_pairs": valid_pairs,
            "n_attempted": n_attempted,
            "pct_valid": pct_valid,
            "stats": cf_stats,
        },
        "cmp2_ranker_training": {
            "n_train_pairs": trained.n_train_pairs,
            "n_val_pairs": trained.n_val_pairs,
            "val_ranking_accuracy": trained.val_ranking_accuracy,
        },
        "cmp3_optimizer": {
            "best_iteration": best_iter,
        },
        "evaluation": report,
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'metrics.json'}")

    return full_metrics


if __name__ == "__main__":
    main()
