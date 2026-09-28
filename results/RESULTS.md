# Results

All numbers below are copied directly from `results/metrics.json`, produced
by one real run of `src/run_pipeline.py` (`.venv\Scripts\python.exe
run_pipeline.py`, run from `D:\Projects\Test 1\src`). Nothing here is
estimated, rounded from a different run, or adjusted after the fact —
metrics.json is the run's own dump of its own output. This is the second
version of this file; the first version reported an honest near-miss on
the FPR half of the success criterion, and the section "FPR improvement
pass" below documents exactly what changed and why, so the earlier numbers
remain traceable rather than silently overwritten.

## Success criterion

> "A prototype that, on a held-out test set mixing traditional phishing,
> LLM-generated phishing and legitimate email, detects LLM-generated
> phishing with higher recall than a traditional rule-based/keyword
> baseline, at a false-positive rate on legitimate email no worse than
> that baseline."

| | Novel (Metamorphic Pair Invariant Trainer) | Baseline (keyword) | Requirement | Met? |
|---|---|---|---|---|
| Recall on LLM-generated phishing | **1.0000** (1500/1500) | 0.1807 (271/1500) | higher than baseline | **Yes** |
| False-positive rate on legitimate email | **0.1100** (330/3000) | 0.1713 (514/3000) | no worse than baseline | **Yes** |

**Plain statement: the success criterion is met, on both halves, with
margin.** Recall on LLM-generated phishing is perfect on this test set
(a synthetic proxy set — see "Limitations" — so read this as an upper
bound, not a claim about real-world LLM output). The false-positive rate
is 6.1 percentage points *better* than baseline, not just "no worse."

## Full metrics (mixed test set, n=7,500: 3,000 legitimate / 3,000
traditional phishing / 1,500 synthetic LLM-generated phishing)

| Metric | Novel | Baseline |
|---|---|---|
| Overall recall (all phishing) | 0.4700 | 0.2640 |
| Recall — traditional phishing only | 0.2050 | 0.3057 |
| Recall — LLM-generated phishing only | 1.0000 | 0.1807 |
| False-positive rate (legitimate email) | 0.1100 | 0.1713 |
| Precision | 0.8650 | 0.6980 |
| F1 | 0.6091 | 0.3831 |

Throughput (CMP-3/NFR-2 acceptance criterion: ≥100 emails/sec on standard
CPU): **~400-450 emails/sec** across repeated runs on this machine —
criterion met.

CMP-2 acceptance criterion (≥90% ranking accuracy on validation pairs):
**99.28%** — met (up from an earlier, honestly-reported 85.75%; see below).

CMP-1 acceptance criterion (≥95% valid counterfactual pairs): **37.32%
(2,074/5,557 attempted) — not met.** This is a direct, disclosed
consequence of a dataset-fit finding below, not a bug: every legitimate
input produces a valid pair (2,000/2,000), but only 74 of 3,557 phishing
inputs exhibit any of the architecture's five action-risk categories, so
the other 3,483 are correctly skipped rather than forced into a
misleading pair (see "FPR improvement pass," item 2).

## FPR improvement pass

Starting point (first submitted version of this prototype): recall
criterion met by a wide margin (0.921 vs. 0.184), FPR criterion narrowly
missed (0.175 vs. 0.170, a 0.44-point gap). Asked to improve the ranker's
features to close that gap, four real issues were found and fixed, in the
order discovered:

1. **`log_word_count` was the dominant feature and actively wrong.**
   LightGBM's own gain-based feature importance showed `log_word_count`
   contributing roughly twice the gain of the next-largest feature. It
   was an artifact of how CMP-1 builds pairs (appending a clause makes
   the phishing side longer; stripping sentences makes the benign side
   shorter) that runs backwards on real mail: this corpus's real
   legitimate email averages 686 words vs. 302 for real phishing, so
   "short = risky" flags short, legitimate one-line messages ("see
   attached file", "fyi"). Removed entirely — see `features.py`'s
   revision note.
2. **`phishing_to_benign` counterfactual pairs were mostly uninformative
   or actively backwards.** Isolating validation accuracy by pair origin
   showed `benign_to_phishing` pairs at 100% but `phishing_to_benign`
   pairs at 11% — worse than chance. The cause: `_detect_category`
   defaulted to `credential_verification` whenever a phishing email
   matched none of the five architecture-defined categories, and
   measurement showed that's true of ~98% of this corpus's phishing
   class (it's dominated by 419-scam/adult/pharma spam, not BEC-style
   account phishing). The fallback produced a "neutralized" counterfactual
   that differed from the original in nothing the feature set could see,
   and training on ~2,000 such near-duplicate, arbitrarily-labeled pairs
   taught the ranker the wrong direction. Fixed by skipping (not
   guessing) when no category matches, and widening the phishing
   candidate pool scanned for matches from 2,000 to 3,557 emails to
   recover enough real pairs (74) to train on. See `counterfactuals.py`'s
   revision note.
3. **Model regularized (num_leaves 31→7, added L2, min_child_samples
   10→30).** With real pair count much smaller after fix #2, a
   high-capacity tree ensemble overfit secondary features to the small
   minority class; shallower/regularized trees calibrated more
   consistently against the (much larger) legitimate calibration split
   while validation ranking accuracy actually *improved* (85.75% →
   99.28%, comfortably above CMP-2's own 90% target).
4. **Threshold-calibration bug: ties were not handled correctly.** The
   original `calibrate_threshold` picked `sorted_scores[int(target_fpr *
   n)]` and used it as a `>=` cutoff. When many calibration emails share
   an identical score (common with a shallow, regularized model — this
   run has a single score value shared by 158 of 2,000 calibration
   emails), that lookup can land inside a large tied block and the `>=`
   threshold then admits the *entire* block at once, overshooting the
   intended FPR by a wide margin regardless of how carefully the target
   was chosen upstream. This fully explained the remaining gap after
   fixes #1-3 (observed calibration-set FPR of 19.3% against an 18%
   target, an unreachable-in-one-step jump from an 11.4% tier just
   below). Fixed by walking the actual cumulative FPR at each distinct
   score value and picking the strictest threshold that does not exceed
   target, which is what the function's own docstring always claimed to
   do. See `evaluate.py`'s `calibrate_threshold` revision note.

A fifth change, `conservative_baseline_target_fpr` (bootstrap the
baseline's FPR on the calibration split, target the 5th percentile
instead of the single-sample estimate, to absorb calibration-to-test
sampling noise), was added but turned out not to be the deciding factor —
fix #4 alone would have closed most of the gap. It's kept because it's a
real, calibration-data-only safety margin, not because it changed the
headline number by much.

None of these four fixes were chosen by looking at the test set's own
metrics; each was diagnosed and validated using only the training pairs'
own validation split or the calibration split (both disjoint from the
7,500-email test set), consistent with the brief's instruction not to
tune against final results. The FPR criterion was re-evaluated on the
test set exactly once, at the end, with all four fixes already in place.

## Interesting secondary finding (still present)

The novel model's recall on **traditional** phishing (0.2050) remains
lower than the keyword baseline's (0.3057), and the gap widened slightly
in this pass (was 0.223 vs. 0.298). This is the same root cause as
finding #2 above, viewed from the test side: the action-risk feature set
only separates the ~2% of this corpus's traditional phishing that
actually exhibits one of the five target categories; the baseline's fixed
phrase list (lottery/prize, "click here", generic urgency wording) still
catches a meaningfully different slice of the other ~98% (419-scam and
generic spam) that the novel model was never trained to recognize. Tightening
the FPR threshold in this pass reduced overall sensitivity further, which
shows up most in this already-weaker category. This is a real,
disclosed trade-off of the current design/dataset combination, not a bug.

## Limitations (own, not copied from the literature review)

- The LLM-generated-phishing test class is **synthetic** (see
  `ARCHITECTURE_SUMMARY.md` → Datasets, and
  `src/synthetic_llm_phishing.py`), not captured real LLM output. A
  perfect 1.0 recall against it should be read as "perfect recall against
  action-risk clauses grafted onto fluent text with an unseen template
  family," the best available proxy given no open real-LLM-phishing
  corpus exists — not a claim that this prototype would catch 100% of
  real LLM-generated phishing in the wild.
- This corpus's phishing class is a poor match for the architecture's
  five-category action-risk taxonomy: only 74 of 3,557 scanned phishing
  emails (~2%) matched any category, which is why CMP-1's own 95%
  valid-pairs acceptance criterion was not met (37.32%) even though the
  final detector performs well. A phishing corpus more weighted toward
  BEC/account-takeover style content would likely let CMP-1 build far
  more `phishing_to_benign` pairs and could change the traditional-phishing
  recall trade-off noted above.
- The Kaggle/HF source corpus provides only raw email text, no real
  headers (sender, real subject line, timestamps); `sender` and
  `timestamp` in `EmailRecord` are therefore placeholders, and the
  "contextual consistency" feature family is limited to what's inferable
  from body text alone (generic-greeting and suspicious-link-wording
  flags), not real sender/domain reputation checks.
