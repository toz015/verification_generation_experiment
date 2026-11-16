# Offline multi-risk experiment: implementation notes (2026-11-16)

Implements `docs/offline-multirisk-agent-handbook-20261116.md` on top of audit baseline `4709be4`.
Code: `src/vgx/offline_multirisk.py` (runner), `offline_multirisk_verify.py` (independent rebuild and run comparison),
`offline_multirisk_report.py` with `offline_multirisk_template.html` (report). Config: `configs/offline_multirisk_20261116.json`.
Tests: `tests/test_offline_multirisk.py`. Results: `results/offline_multirisk_20261116/`; report: `reports/offline_multirisk_20261116/`.

## Commands

```bash
python -m vgx.offline_multirisk preflight --config configs/offline_multirisk_20261116.json
python -m vgx.offline_multirisk run --config configs/offline_multirisk_20261116.json [--only gpqa | gpqa:qwen3] [--out DIR]
python -m vgx.offline_multirisk verify --config configs/offline_multirisk_20261116.json [--out DIR]
python -m vgx.offline_multirisk report --config configs/offline_multirisk_20261116.json
python -m vgx.offline_multirisk_verify DIR1 DIR2      # compare two runs after normalization
python -m pytest tests/test_offline_multirisk.py tests/test_offline_frozen.py tests/test_selective_auditing.py
```

Every command patches `socket` so connections and DNS lookups raise; attempts are recorded in each result file
(`network_attempts`). `run` shards by dataset or dataset:generator and skips a shard only when its stored code hash
and config hash both match (resume never mixes versions). Python 3.12.13, numpy 2.2.6, scipy 1.18.0, pandas 2.3.3.

## Risk branches: one rule family

For all three risks the per-item Lagrangian of the LP is linear in the release decision a and the belief b, so the
best policy for any multipliers is a Bellman rule whose release value is a positive multiple of
`b - kappa (1 - b)` and whose cost weight is a multiple of mu. Writing the handbook's values:

| risk | LP (max form) | handbook g(b) | divide by | kappa | mu' |
| --- | --- | --- | --- | --- | --- |
| selective_error | p - lam (e - alpha p) - mu c | 1 - lam((1-b) - alpha) | 1 + lam alpha | (lam (1-alpha) - 1)/(1 + lam alpha) | mu/(1 + lam alpha) |
| fpr | p - lam e - mu c | 1 - lam (1-b) | 1 | lam - 1 | mu |
| fnr | -e - lam (e - p) - mu c (= -(e + lam m + mu c) + const) | lam b - (1-b) | lam | 1/lam | mu/lam |

Proof sketch for selective_error: `1 + lam alpha - lam (1 - b) = (1 + lam alpha) b + (1 + lam alpha - lam)(1 - b)`; dividing by
`1 + lam alpha > 0` gives `b - kappa (1 - b)`. The release boundary is `b >= kappa/(1 + kappa)`; for FNR that is `1/(1 + lam)`.
When kappa <= 0 (sel, fpr) the rule releases every eligible candidate, and with lam = 0 (fnr) it abstains on all: both are
the boundary rules every LP holds, so column generation stops there. `tests/test_offline_multirisk.py` checks the sign
agreement of the two forms on random (lam, alpha, b), the FNR boundary, the Bellman values against an exact grid-free
recursion, and that column generation leaves no dense-grid rule with positive reduced cost.

Grids: belief grid 401 points (logistic spacing, as in the previous offline run); kappa = 10^linspace(-2.5, 3, 22)
(release thresholds 0.003 to 0.999, wide enough for FNR's low thresholds; the old omega grid only reached 0.24);
mu = 0 and 10^linspace(-1, 3.5, 10) (unchanged). The LP keeps the release and abstain boundary rules; a solver
status other than optimal is recorded and the candidate dropped, never read as an all-zero policy.

## Choices made where the handbook leaves room

- **No answer**: every ineligible candidate in the cache has no answer; the official rule counts it wrong (Y = 0).
  `label_source` records `no_answer_official_rule`; no labels are missing. All methods defer these.
- **Validation risk not estimable** (FPR with no wrong, FNR with no correct candidate in the validation fold):
  status `risk_not_estimable`, then the fallback. It did not occur in this run (see the report's selection table).
- **Fallback**: selective error and FPR fall back to Abstain-all, FNR to Release-eligible (which is FNR-feasible when
  ineligible candidates are wrong). Selection never counts a zero-release candidate as a success for selective error
  or FPR. Fallbacks are flagged per fold and seed; the report shows the feasibility table with and without them.
- **Failed verifier call** (score missing): its cost is charged and the item is deferred, in Ours and Fixed alike.
  Fixed panels otherwise ask every member; there is no confidence-based early stop.
- **Fixed panels**: thresholds are the kappa grid; members in sorted order (order only matters for the failure path).
- **Bootstrap**: stratified by outer test fold (GPQA has one); units are questions, or merged kata groups for QuantumKatas.
- **Optional parts not run**: the GPQA fixed-USD budget appendix, Ours raw-prior / no-verifier / single-rule ablations,
  the `gamma_test = gamma_fit` AT sensitivity and the external audit-cost scenario. None is in the 702 main conditions.

## Audit fixes (handbook section 12)

| issue | where | fix | check |
| --- | --- | --- | --- |
| AT alpha_at = 0.9 dropped | `offline_frozen.at_candidates` | tolerance on `gamma_fit <= 1 - alpha_at` | 40 AT, 160 AT+Cal candidates (test) |
| mixture query rate | `offline_frozen.ours_apply/evaluate`; new `Runs.mixture` | sum_r w_r 1{N_ir > 0} | half-zero/half-one mix gives 0.5 (test) |
| Brier population and weights | `offline_frozen.run_dataset`, new `brier_summary` | sample-weighted, no fold skipped, AT+Cal without selection is `no_selected_calibrator` | unequal-fold test |
| pooled vs majority-seed risk exceedance | new report | two named columns | report section 5 |
| budget-overrun text | new report | counts generated from data | report |
| QK inner splits across groups | old release-error pipeline only | new pipeline has no inner splits; outer splits checked by group | preflight and test |
| budget reference from all data | old `fpr_compare.budgets_for` | new pipeline uses Fit only, per fold | price-perturbation test |
| empty-denominator bootstrap read as 0 | `offline_frozen` and new `bootstrap` | NA plus valid-replicate count | report fields |
| difference point = bootstrap mean | same | actual test difference | report fields |
| validity coupled to labels | new `eligible` / `truth` | eligibility from inputs only | label-perturbation test |

The old offline pipeline (`python -m vgx.offline_frozen run --out DIR`) keeps its entry point and now accepts an output
directory, so the historical `results/offline_at_frozen_20261111/` is not overwritten. It was not re-run for this round.

## Report scope

The report and its CSVs leave out FPR on MATH-500 (all three generators; `EXCLUDE_MAIN` in
`offline_multirisk_report.py`). FPR's denominator is the number of wrong answers, which on MATH-500 is 16 (Qwen3),
101 (gpt-oss, 84 of them no-answer) and 63 (Sonnet 5.5) over the five test folds. The runner still computes these
cells, so `results/offline_multirisk_20261116/math500_*.json` contains them.

## p_correct input (2026-11-16, later)

`confidence_input` in the config picks what every method reads (`fold_rows` in `offline_multirisk.py`):
`mixed` is the original run (only Ours and Fixed calibrate); `calibrated` gives every method the Fit-calibrated
p_correct (isotonic PAV fit on Fit, applied to all items; AT, AT+Cal and the threshold method included, so
Raw-threshold becomes Cal-threshold); `raw` gives every method the raw p_correct (Ours and Fixed plan with it as the
prior; likelihoods unchanged). Configs `configs/offline_multirisk_20261116_{calibrated,raw}.json`, results
`results/offline_multirisk_20261116_{calibrated,raw}/`. The calibrated report (main) includes the raw run as its
section 5. In the calibrated run Ours and Fixed match the mixed run exactly (2106 cells checked).
