# Offline frozen comparison: implementation notes

Implements docs/offline-at-frozen-experiment-protocol-20261110.md. Code: `src/vgx/offline_frozen.py`,
`src/vgx/offline_frozen_report.py`; tests: `tests/test_offline_frozen.py`; config:
`configs/offline_at_frozen_20261111.json`; outputs: `results/offline_at_frozen_20261111/` (per-dataset JSON with every
seed and fold, `summary.csv`, per-decision logs `*_decisions.jsonl.gz`); page: `reports/offline_at_frozen_20261111/results.html`.

```
python -m vgx.offline_frozen config
python -m pytest tests/test_offline_frozen.py tests/test_selective_auditing.py
python -m vgx.offline_frozen run            # gpqa math500 qkatas; or name datasets
python -m vgx.offline_frozen_report
```

## Changes relative to the earlier code

- New Fit / Validation / Test split for every method (GPQA calibration 199 / 50 by seed 20261111; MATH-500 and
  QuantumKatas test fold k, validation fold (k+1) mod 5, three fit folds). Earlier Ours results used the whole
  training pool and are not compared.
- Ours plans with the fitted model on the Validation reports (no labels) and is then selected on Validation with
  labels, like the baselines. The earlier label-counted rule values are not used.
- The release-error constraint is set in code (`R.CONSTRAINT = 'fdr'`) and in the config (`risk_metric:
  selective_error`), not through an environment variable. A test checks the LP constrains sum w (e - alpha p).
- Verifier order: top 3 of 60 by label-free model-planned coverage, then candidates over alpha_plan x B_plan.
- AT and AT+CC: one pass over Fit in a seeded order (20 seeds), gamma_fit 0.1, labels only through the oracle, then
  frozen; gamma_test 0 in the main table, gamma_test = gamma_fit as a sensitivity row (expected release).
- Planning belief grid: 401 points instead of 2001 (run time).
- Budgets: B / c_ref in {0, 0.1, 0.25, 0.5, 0.75, 1}, c_ref = Fit mean cost of the full five-verifier panel; GPQA also
  the earlier dollar grid. A pooled test budget is the test-size-weighted mean of the fold budgets.
- Invalid candidates (no answer or no confidence, never label-based) are deferred by every method and counted in n.
- Mixtures: expected counts in the tables, 20 sampled executions stored per cell (`sampled_runs`).
- Paired bootstrap 2000, QuantumKatas by the split's 35 kata groups; AT-type methods use their seed-averaged release.
