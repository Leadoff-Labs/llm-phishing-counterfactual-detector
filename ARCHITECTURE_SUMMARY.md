# Architecture Summary — Metamorphic Pair Invariant Trainer

Source design: `D:\Projects\mas-pipeline\reports\7ada19e1-5de9-448f-8ccd-d75a223f8d4e\architecture\architecture.tex`
(read in full before any code was written; nothing here deviates from its
stated mechanism, only from incidental implementation details it left
open — see "Deviations" below).

## Problem this prototype targets

LLM-generated phishing is fluent, personalized and free of the
grammatical tells that classical filters were built on, so detectors that
key on stylometry/grammar degrade as the generator changes (review.tex,
S1: cross-model F1 drops 28 points without recalibration). The brief asks
for a prototype that beats a keyword baseline's recall on LLM-generated
phishing without a worse false-positive rate on legitimate email, using
only open data, Python/CPU, and the standard NLP/ML stack.

## Core mechanism

Instead of collecting ever more generator-specific phishing samples and
extracting stylometric features from them (the approach the design
explicitly rejects, per its "Novelty" section vs. S1), the solution moves
the intervention into **how the training data itself is built**:

1. **CMP-1 — Email Corpus Processor.** For each legitimate email, append a
   high-risk action clause (credential verification, payment redirection,
   document download, MFA approval, or reply-with-data) to build its
   phishing counterfactual, keeping the rest of the email untouched. For
   each traditional-phishing email, strip the sentence(s) carrying the
   risky action and replace them with a neutral clause, to build its
   benign counterfactual. Every pair records which action slot changed.
   Interface: `generate_counterfactuals(legitimate_emails, phishing_emails)
   -> List[CounterfactualPair]`.
2. **CMP-2 — Pairwise Ranker Trainer.** Trains a LightGBM `LGBMRanker`
   (`objective="lambdarank"`) so that, within each pair, the risky-action
   side scores higher than its neutralized twin. Features describe the
   *requested action* and its context (see "Features" below) — not
   grammar, vocabulary richness, or any other stylometric signal — so the
   model has nothing generator-specific to key on.
   Interface: `train_ranker(counterfactual_pairs) -> RankerModel`.
3. **CMP-3 — Model Optimizer.** Trains with an early-stopping validation
   set (the same held-out counterfactual pairs used for CMP-2's ranking
   accuracy) and trims the boosted ensemble to its best iteration, so
   inference does strictly less work per email.
   Interface: `optimize_model(trained_ranker) -> RankerModel`.
4. **CMP-4 — Evaluation and Reporting Module.** Scores a mixed held-out
   test set (legitimate, traditional phishing, LLM-generated phishing —
   none seen during CMP-1/CMP-2) and compares recall/false-positive rate
   against the keyword baseline.
   Interface: `evaluate_model(optimized_model, test_set) -> EvaluationReport`.

## Inputs / outputs

| Stage | Input | Output |
|---|---|---|
| CMP-1 | `List[EmailRecord]` legitimate, `List[EmailRecord]` phishing | `List[CounterfactualPair]` |
| CMP-2 | `List[CounterfactualPair]` | `RankerModel` |
| CMP-3 | `RankerModel` | `RankerModel` (fewer boosting rounds) |
| CMP-4 | `RankerModel`, `List[EmailRecord]` test set | evaluation report (recall, FPR, precision, F1, vs. baseline) |

`EmailRecord`, `CounterfactualPair` and `EvaluationReport` match
architecture.tex's Data Models section (`src/models.py`), with a small
number of extra provenance fields (`label`, `subtype`, `source_id`) needed
to actually run and evaluate the pipeline, which the architecture's
minimal contract did not need to specify.

## Features (why no stylometry)

`src/features.py` extracts 12 numeric, regex-based features per email:
five requested-action categories (credential/payment/document/MFA/reply),
an urgency-phrase count, a count of "verifiable artifacts" (invoice/ticket
numbers, dates, dollar amounts — legitimate business mail tends to have
more of these than phishing), a generic-greeting flag, link count, a
suspicious-link-wording flag, a reply-with-data flag, and log word count.
None of these measure grammar, spelling, or word-choice style, by design —
that is precisely the signal LLM-generated phishing removes.

**Scope note on the architecture's stated feature list.** The mechanism
also names "recipient behavior" and "channel" as features. This prototype
runs on a static, header-stripped email-text corpus with no interaction
logs and a single channel (email, by the brief's own scope), so those two
are not present as live features: channel is a constant (carries no
information) and recipient behavior is approximated by the
requested-action proxy features above (what the message asks the
recipient to do), which is the only recipient-facing signal a static
corpus can supply. This is a scope reduction forced by the data
available, not a change to the ranking mechanism itself.

## Datasets

- **Legitimate + traditional phishing:** "Phishing Email Detection"
  (Kaggle, user `subhajournal`/"Cyber Cop"), mirrored openly on Hugging
  Face at `zefang-liu/phishing-email-dataset` under LGPL-3.0, no
  credentials required. 18,650 rows after download (11,322 "Safe Email",
  7,328 "Phishing Email"; 10,979/6,557 after de-duplication). Saved to
  `data/raw/Phishing_Email.csv`; the retrieved dataset card is saved to
  `data/raw/README_dataset.md`. The architecture's "Required data" list
  names the Enron Email Dataset as the open legitimate-email source; this
  corpus's "Safe Email" side is Enron-derived, and its "Phishing Email"
  side is a standard open phishing corpus (Nazario-family), so it
  satisfies both named requirements from a single, license-clear source.
  **Dataset-fit finding (added during the FPR improvement pass, see
  results/RESULTS.md):** this corpus's phishing class is mostly generic
  spam (419-style advance-fee scams, adult/pharma ads, stock
  pump-and-dump), not BEC-style account phishing. Only ~2% of it (74 of
  3,557 scanned emails) matches any of the architecture's five
  action-risk categories. This does not block the mechanism -- the
  benign-to-phishing direction of CMP-1 works regardless, since it
  constructs its own phishing side from templates -- but it does mean
  CMP-1's own 95% valid-pairs acceptance criterion is not met on this
  dataset (see RESULTS.md), and traditional-phishing recall is weaker
  than it would likely be with a more BEC-weighted phishing corpus.
- **LLM-generated phishing:** no openly downloadable, credential-free
  dataset of real LLM-generated phishing emails was found. The closest
  candidate in the reviewed literature, S1's balanced GPT-4.1/DeepSeek/
  Llama corpus, was assembled by that paper's own authors for their
  experiment and is not released publicly (review.tex confirms only that
  the *method* is described, not a public download link). Per the task
  brief, a clearly labeled **synthetic** set was generated instead:
  genuinely fluent, human-written Enron email bodies (a slice of the ham
  pool held disjoint from every other pool) with a high-risk action
  clause grafted on using a *second*, disjoint template family
  (`FAMILY_B_RISKY` in `src/templates.py`) never seen during CMP-1
  training. Every record's `source_id` is tagged
  `...::synth_llm_phish::<category>` so it can never be mistaken for
  captured data. See `src/synthetic_llm_phishing.py` for the full method.
  Using a second template family also means this test set doubles as the
  architecture's own "leave-transformation-family-out" check (Key
  mechanisms, bullet 5).

## Deviations from the written design

1. **T5-small paraphrasing skipped.** Architecture.tex lists T5-small as
   an *optional* CPU paraphrasing step on top of deterministic templates
   ("deterministic templates plus optional CPU-scale paraphrasing").
   Adding it means installing `transformers` + `torch`, multi-gigabyte
   dependencies with no confirmed wheels for the only Python available in
   this environment (3.14, released after most ML packages' usual release
   cadence). The prototype uses deterministic templates only, which is
   the required (non-optional) half of the mechanism.
2. **CMP-3 "optimization" is early-stopping-based trimming, not
   retraining.** The architecture's CMP-3 interface takes only the
   already-trained `RankerModel` (no data), so genuine hyperparameter
   retraining is out of scope for that function signature; CMP-2 trains
   with a validation split and early stopping, and CMP-3 trims the
   ensemble to `best_iteration_` — a real reduction in prediction work,
   measured (not assumed) in the throughput benchmark reported in
   `results/RESULTS.md`.
3. **Ranker hyperparameters were selected by comparing a handful of
   settings against CMP-2's own validation split** (never against the
   final mixed test set), which is ordinary model development, not
   test-set tuning.

## Mapping to goal / scope / constraints

- **Goal:** targets LLM-generated phishing specifically, via a mechanism
  designed to be generator-agnostic (action-risk features, not stylometry).
- **Scope:** email-only, textual+contextual features (behavioral scoped
  down as noted above), open corpora only — all satisfied.
- **Constraints:** open datasets only (Kaggle/HF, LGPL-3.0, no
  credentials); CPU only (no GPU used or required — LightGBM CPU training
  finishes in seconds); Python + scikit-learn/pandas/nltk/LightGBM only,
  no exotic dependencies; low FPR on legitimate email is the explicit
  target of the threshold-calibration step in CMP-4.
- **Success criterion:** evaluated exactly as stated — see
  `results/RESULTS.md` for the real, measured numbers and a plain
  statement of whether it was met.
