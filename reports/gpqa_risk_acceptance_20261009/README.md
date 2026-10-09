# GPQA risk-controlled acceptance (2026-10-09)

Offline implementation and analysis of risk-controlled acceptance: maximize P(A) subject to P(W | A) <= alpha with confidence 0.95 (alpha = 0.05 primary, 0.10 sensitivity). No rule is certified; certification is pending independent data.

Built from branch `codex/gpqa-pilot` at 3263afb. PR #2 (confidence-conditioned likelihoods) is not merged into or modified by this work.

## Read first

| file | contents |
|---|---|
| `FINDINGS.md` | summary: measured results, guarantees, interpretation |
| `PROTOCOL.md` | frozen protocol, assumptions, limitations |
| `SAMPLE_SIZE_PLAN.md` | sample sizes, predeclaration, blockers and missing inputs |
| `REVIEWER_CHECKLIST.md` | what to check and where |
| `TABLES.md` | all tables (generated) |
| `figures/risk_coverage.png` | risk-coverage curves, both generators and cohorts |
| `figures/sample_size.png` | accepted examples needed vs. assumed true risk |

## Data and outputs

| path | contents |
|---|---|
| `data/observations.json.gz` | exported observations by question ID: validity, raw confidence, correctness, cached Flash record; frozen folds, fold fits and decisions; input hashes. No question text, options or credentials. |
| `phase1/` | exploratory outputs: `exploration.json`, `risk_coverage_curves.csv`, `item_scores.csv.gz` (per-item scores and decisions by question ID, method, seed), `fold_parameters.csv` (likelihood and cost per fold), `family_rules_exploratory.csv`, `low_risk_regions_exploratory.csv`, `matched_coverage.csv`, `ranking_auroc.csv`, `method_summary.csv`, `b_c_acceptance_overlap.csv`, `protocol.json` |
| `phase2/` | dry run: `partition.json` (role of every question ID), `selection_alpha_*.json` (sealed: family, alpha, delta, fitted tables and costs, selected rule), `calibration_alpha_*.json` (counts, bounds, fixed-sequence table, per-item calibration scores), `evaluation_alpha_*.json` (descriptive evaluation counts and per-item scores) |
| `validation/procedure_validation.json` | Monte Carlo check of both certification procedures |
| `plan/sample_size_plan.json`, `tables/sample_size_curve.csv` | sample-size tables |
| `manifest.json` | SHA-256 of every file here, the config, the code and the source package manifest; commands |
| `test_results.txt` | test output |

## Reproduce

From the repository root, with the project installed (`pip install -e ".[dev,report]"`, Python >= 3.12):

```
reports/gpqa_risk_acceptance_20261009/reproduce.sh OUT_DIR [PATH_TO_UNZIPPED_REVIEW_PACKAGE]
```

With the package path, the observation export is first rebuilt from the frozen source files and byte-compared with `data/observations.json.gz`. Without it, everything is rebuilt from the shipped export. The individual commands are in `reproduce.sh` and `manifest.json`. Tests: `PYTHONPATH=src python -m pytest tests/test_risk_control.py -q`.
