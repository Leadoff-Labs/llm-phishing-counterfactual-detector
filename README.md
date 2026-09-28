# Phishing Detection Prototype — Metamorphic Pair Invariant Trainer

A prototype implementation of a Metamorphic Pair Invariant Trainer: a
phishing detector trained on metamorphically constructed counterfactual
pairs so it keys on the requested action of a message rather than on
stylometric or generator-specific signal, aimed at generalizing to
LLM-generated phishing without per-generator recalibration.

See [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) for the mechanism,
[results/RESULTS.md](results/RESULTS.md) for the measured results, and
[paper.pdf](paper.pdf) for the accompanying conference paper.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Data

The raw dataset is not committed (see `.gitignore`). Fetch it per
[data/raw/README_dataset.md](data/raw/README_dataset.md) and place it at
`data/raw/Phishing_Email.csv` before running the pipeline.

## Run

```bash
cd src
..\.venv\Scripts\python.exe run_pipeline.py
```

This runs CMP-1 through CMP-4 end to end, printing progress and the final
evaluation report, and writing `results/metrics.json`.

## Layout

- `src/models.py` — `EmailRecord`, `CounterfactualPair`, `EvaluationReport`
- `src/templates.py` — deterministic slot templates (two disjoint families)
- `src/counterfactuals.py` — CMP-1: `generate_counterfactuals()`
- `src/synthetic_llm_phishing.py` — synthetic LLM-phishing test-set builder
- `src/features.py` — shared action/urgency/artifact/contextual features
- `src/ranker.py` — CMP-2 `train_ranker()`, CMP-3 `optimize_model()`
- `src/baseline.py` — rule-based/keyword baseline detector
- `src/evaluate.py` — CMP-4 `evaluate_model()`, threshold calibration
- `src/data_prep.py` — loads and splits the open dataset
- `src/run_pipeline.py` — wires CMP-1→CMP-4 end to end (the `__main__`)
- `paper.tex` / `references.bib` — the accompanying conference paper
- `diagrams/` — draw.io sources for the paper's figures
