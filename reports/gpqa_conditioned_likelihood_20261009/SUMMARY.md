# Confidence-conditioned verifier likelihoods: single Flash verifier (offline)

**Bottom line (observation).** Conditioning the Flash likelihood tables on the generator's original confidence group changed almost no release decision:
- **Gemini:** 0 of 20 runs (2 cohorts × 5 seeds × 2 losses) released or withheld a different answer.
- **Qwen:** 16 of 20 runs were identical; the 4 runs that changed moved in opposite directions.
- **Posterior risk underestimation is unchanged.** On the primary setting, the sum of (1 − posterior) over released answers stays at 4.05 vs 17 observed wrong releases for Gemini and 2.64 vs 20 for Qwen. The pooled baseline gives 3.83 and 2.65.
- **Posterior Brier after observing Flash improves slightly in the high-confidence group and worsens in the low group.**

No method is declared better on this development set.

Scope: development calibration partition only (249-question cohort, plus the overlapping 199-question cohort); cached responses only. New API spend $0, no candidates regenerated, no Diamond answer keys read. Raw generator confidence is the unmodified initial prior.

## 1. Design (as declared in `configs/gpqa_conditioned_likelihood.json` before results were computed)

**Confidence groups.** From the original confidence `c`: `low` is `c < 0.95` and `high` is `c ≥ 0.95`. The group selects one table per item before any query. It is never switched after a Bayes update.

**Bins.** The existing probability bins are unchanged: `[0, ⅓)`, `[⅓, ⅔)`, `[⅔, 1]`. The high-score bin is score ≥ 2/3.

**Pooled table (B).** One table per training fold, refit exactly as the frozen raw-prior analysis (Laplace 1). In every fold it reproduces the frozen table to 1e-12.

**Conditioned table (C).** `g[group, y, k] = (n[group, y, k] + τ · g_pooled[y, k]) / (N[group, y] + τ)`, using raw training counts `n` and the same fold's pooled table.
- **What τ means.** τ is a pseudo-count, the number of pooled-table "observations" mixed into each group's distribution. τ → 0 uses the group's raw frequencies; τ → ∞ recovers the pooled table.
- **Primary τ.** Chosen inside each training fold only, from the grid {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}. It maximizes the mean held-out log P(bin | Y, group) over 5 inner splits stratified by outcome × group. Ties go to the larger τ. Held-out outcomes, signals and utilities never enter this choice.
- **Declared fixed-τ sensitivity set.** {5, 20, 100}.

**Fallbacks.** An empty (group, outcome) training cell uses that fold's pooled distribution, which the formula yields exactly when N = 0. If inner strata are too small, stratification falls back to outcome only, then to the largest τ. No fallback was triggered in any of the 60 fold fits (2 generators × 2 cohorts × 5 seeds × 3 folds); every cell had at least one training example.

**Held fixed.** Value-table construction, Bayes updates and stopping rules (`planner.py` and `raw_prior_analysis.decide_public`, hashes unchanged); R = 1; L = 19 (primary) and L = 99 (sensitivity); 10 utility per USD. Expected cost is the frozen training-fold Flash cost, identical for B and C.

**Policies compared on identical items, folds and costs.**
- A: raw confidence only.
- B: Flash with the pooled table.
- C: Flash with the conditioned table.

## 2. Primary results (249 questions, seed 20261005, L = 19)

Utility and coverage use all 249 items as the denominator. Accuracy is computed over released answers. "Predicted" is the model's mean J₀ value. "Σ(1−p)" is the expected number of wrong releases implied by the posteriors.

| Generator | Policy | Released | Wrong | Accuracy | Coverage | Queries | Cost (USD) | Mean utility [95% CI] | Predicted | Mean posterior (released) | Σ(1−p) vs wrong |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| Gemini | A | 190 | 19 | 0.900 | 0.763 | 0 | 0 | −0.763 [−1.430, −0.140] | +0.345 | 0.9726 | 5.20 vs 19 |
| Gemini | B | 187 | 17 | 0.909 | 0.751 | 188 | 0.313 | −0.627 [−1.266, −0.040] | +0.421 | 0.9795 | 3.83 vs 17 |
| Gemini | C | 187 | 17 | 0.909 | 0.751 | 188 | 0.313 | −0.627 [−1.266, −0.040] | +0.413 | 0.9783 | 4.05 vs 17 |
| Qwen | A | 154 | 38 | 0.753 | 0.618 | 0 | 0 | −2.434 [−3.345, −1.578] | +0.109 | 0.9588 | 6.34 vs 38 |
| Qwen | B | 158 | 20 | 0.873 | 0.635 | 207 | 0.398 | −0.988 [−1.639, −0.349] | +0.472 | 0.9832 | 2.65 vs 20 |
| Qwen | C | 158 | 20 | 0.873 | 0.635 | 207 | 0.398 | −0.988 [−1.639, −0.349] | +0.478 | 0.9833 | 2.64 vs 20 |

- **C vs B decisions.** Paired C − B differences are exactly zero for both generators on the primary setting: no change in releases, wrong releases, queries or utility.
- **Selected τ.** Gemini: 50, 10, 50 by fold. Qwen: 100, 200, 200.
- **Fixed τ for Qwen.** With τ = 5 or 20, C released 8 more answers (3 of them wrong) and made 14 more queries. All 14 changed items had c = 0.85 and moved from "abstain without query" to "query"; the 8 released landed at posterior 0.9527. Paired Δ utility was −0.210 [−0.515, +0.019].
- **Predicted vs realized value.** Predicted value stays positive for every Flash policy while realized utility stays negative. The gap is essentially unchanged by conditioning: Gemini +0.421 → +0.413 predicted, Qwen +0.472 → +0.478.

**Posterior Brier after observing Flash.** Computed on matched eligible items (valid answer and a parsed Flash score), regardless of whether the policy queried.

| Generator | Scope | n | Accuracy | Mean raw c | B Brier | C Brier | C − B [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|
| Gemini | all | 246 | | | 0.1188 | 0.1180 | −0.0008 [−0.0054, +0.0031] |
| Gemini | high | 189 | 0.899 | 0.973 | 0.0893 | 0.0858 | −0.0035 [−0.0088, −0.0001] |
| Gemini | low | 57 | 0.702 | 0.807 | 0.2166 | 0.2246 | +0.0081 [−0.0009, +0.0184] |
| Qwen | all | 248 | | | 0.1388 | 0.1351 | −0.0036 [−0.0061, −0.0013] |
| Qwen | high | 154 | 0.753 | 0.959 | 0.1129 | 0.1049 | −0.0081 [−0.0117, −0.0047] |
| Qwen | low | 94 | 0.426 | 0.856 | 0.1810 | 0.1847 | +0.0036 [+0.0021, +0.0052] |

The raw-prior Brier on the same items is 0.1260 for Gemini and 0.3026 for Qwen. The decision-time posterior Brier difference (C − B, matched) is −0.0027 [−0.0067, −0.0001] for Gemini and −0.0038 [−0.0062, −0.0015] for Qwen. The released/abstained decisions themselves did not change.

## 3. Stability across seeds, cohorts and losses (C with inner-CV τ, minus B)

- **Gemini.** No change in released or wrong answers in any of the 20 runs. Query counts changed in 7 of the 10 L = 19 runs, by −35 to +30 queries. Utility changed by at most 0.003 per item, cost only.
- **Qwen.** 16 of 20 runs identical. The 4 runs with release changes:
  - 249 questions, seed 20261007, L = 19: +8 released, +6 wrong, −0.450 [−0.905, −0.068]
  - 199 questions, seed 20261007, L = 19: −7 released, −3 wrong, +0.268 [−0.019, +0.645]
  - 199 questions, seed 20261009, L = 19: −8 released, −3 wrong, +0.262 [−0.024, +0.640]
  - 199 questions, seed 20261008, L = 99: +18 released, +2 wrong, −0.917 [−2.430, +0.098]
- **Signal-posterior Brier.** Qwen's C − B is negative in all 10 cohort × seed runs, but its 95% CI excludes zero in only 2. Gemini's ranges from −0.0018 to +0.0042, and every CI includes zero.
- **Fixed τ, summed over 5 seeds.** For Qwen at 249 questions and L = 19, τ = 5 or 20 gives +29 released / +16 wrong, and τ = 100 gives +14 / +9. For Qwen at 199 questions and L = 19, τ = 5 or 20 gives −22 released / −10 wrong, and τ = 100 gives 0. The sign of the effect depends on cohort and τ.
- **τ selection is itself unstable.** For Qwen it picks 10 to 1000 on the 249-question cohort, but τ = 1 (the grid's lower edge) in 11 of 15 folds on the 199-question cohort. For Gemini it picks 5 to 50 at 249 questions and 1 to 20 at 199.

**Training cells per fold** (incorrect / correct, range over seeds and folds):

| Generator | Cohort | Low group | High group |
|---|---|---|---|
| Gemini | 249 | 10–13 / 21–32 | 11–14 / 108–119 |
| Gemini | 199 | 7–12 / 18–24 | 4–9 / 90–96 |
| Qwen | 249 | 34–38 / 23–32 | 23–28 / 73–80 |
| Qwen | 199 | 25–32 / 16–26 | 15–22 / 60–69 |

## 4. Interpretation (mine, from arithmetic on the observations above)

**Why release decisions barely move.** At L = 19, a release needs posterior ≥ 0.95.
- For a high-group item (c ≥ 0.95), a high-bin Flash score keeps the posterior ≥ c whenever that bin's likelihood ratio is ≥ 1. To block such a release, the ratio would have to fall below 19 · (1 − c)/c: 1.00 at c = 0.95, 0.39 at c = 0.98, 0.19 at c = 0.99.
- Descriptively (all 249 items, existing ≥ 2/3 bin, not used for fitting), the high-bin rates are:
  - Gemini high group: 170/170 correct vs 17/19 wrong, ratio ≈ 1.12. Low group: 37/40 vs 11/17.
  - Qwen high group: 114/116 vs 12/38, ratio ≈ 3.1. Low group: 38/40 vs 14/54.
- Conditioning lowers the fitted high-bin ratio for Gemini's high group only slightly (fold 0: 1.24 pooled → 1.21 conditioned at τ = 50). Even the unshrunk descriptive ratio (≈ 1.12) is above 1, so no table these data can support pushes it below 1. So every high-group wrong answer with a high-bin Flash score is released under B and C alike. That accounts for all 17 Gemini and 12 of 20 Qwen wrong releases in the primary run.
- Gemini's low group (c ≤ 0.90) needs a ratio ≥ 2.1 to release and never gets one.
- In the primary run, Qwen's low-group changes under fixed τ are threshold effects on c = 0.85 items, whose posterior lands at 0.9527 against a 0.95 bar.

**What this means for the research question.**
- Conditioning moves posteriors in the direction the group-specific evidence indicates, and helps the post-signal Brier score in the high group.
- It does not correct the overconfident starting point. Mean released posterior is 0.978 vs 0.909 accuracy (Gemini) and 0.983 vs 0.873 (Qwen).
- Weaker positive evidence did not translate into more abstention. This matches the caution in the request.
- The group differences are consistent with signal dependence on generator confidence. They could also come from the confidence groups differing in question difficulty within Y, or from sampling noise with 11–14 high-group wrong answers per fold. They do not by themselves establish a violation of conditional independence given (Y, w).

## 5. Assumptions and limitations

- Development-set, out-of-fold analysis after earlier screening on the same questions. The 199-question cohort and the five seeds re-split overlapping data; they are not independent replications.
- Bootstrap intervals are paired item resamples with fixed out-of-fold predictions. They exclude refitting and τ-selection uncertainty and are not multiplicity corrected.
- The inner-CV criterion (bin log-likelihood) is not the decision objective. A different training-only criterion could select different τ.
- Signals were taken from the frozen nested-combination traces in the review package. The run checks every one against the frozen pooled fits and decisions, and every input file against the package manifest. Re-parsing from the original request caches (`--check-cache`) needs the full project and was not run here.
- Costs are counterfactual cached-call estimates, not billing.

## 6. Does the evidence support a three-verifier extension?

**Not yet, in my reading.**
- **Decisions.** The single-verifier comparison shows no decision change for Gemini and seed- and cohort-dependent, sign-inconsistent changes for Qwen.
- **Sample size.** Each verifier added to a conditioned three-layer plan needs 2 groups × 2 outcomes × 3 bins of estimates from the same 4–14 high-group wrong answers per fold (Gemini).
- **Same constraint.** In the frozen pooled fits (fold 0), the other probability arms' high-bin ratios are 1.08–1.32 for Gemini and 0.79–1.62 for Qwen. Only one arm is below 1 (Mistral Medium for Qwen, 0.79), and that is still above the 0.39 needed to block a release at c = 0.98. So the same arithmetic applies at every layer: a high-bin endorsement cannot block a high-confidence release, conditioned or not.

No API collection was started.

If you want to go further offline first, two checks would be informative:
1. Repeat this single-verifier comparison for the Haiku and Flash-Lite probability arms. This is the same code with a different `verifier` tag, but it needs those signals for every item. The review package only partly contains them, so it would run in the full project.
2. Pre-register a separate analysis of the joint high-bin endorsement pattern for wrong high-confidence answers.

## 7. Validation

- **Pooled reproduction.** The refit pooled table, training item lists and expected cost match the frozen raw-prior fit in all 60 folds. B reproduces every frozen `sequential:gemini_flash:probability` decision (action, posterior, query count, both cost fields): 8,960 of 8,960 item × loss checks.
- **Executor vs replay.** 44,600 executor/replay agreements. No signal is read outside a selected query. Query cost equals the frozen fold cost and the cached usage estimate.
- **Input integrity.** 81 input files match the package `FILE_MANIFEST.json`, and their SHA-256 values are identical before and after the run. Implementation hashes of `planner.py`, `raw_prior_analysis.py`, `score.py` and `report.py` equal those recorded by the frozen source analysis.
- **Tests.** New file `tests/test_conditioned_likelihood.py` (11 tests). Full suite: 303 passed, 15 skipped. The new tests cover:
  - Fitting and τ selection receive training rows only, and are invariant to held-out labels or signals.
  - Shrinkage yields valid distributions; empty cells fall back to pooled.
  - The table is fixed by original confidence even when the posterior crosses 0.95.
  - No signal read before a query decision, and no label input.
  - Executor/replay and cost accounting; τ → ∞ reproduces pooled.
  - Tampered signals are rejected.
  - Frozen inputs are unchanged, with manifest and mid-run mutation detection.
- **Determinism.** Two independent runs gave the same analysis id: `c800a1b07a4b93607f648209e178a04df144bf3a49ef0bd669c419dc38a004b6`.

## 8. Reproduction

From a checkout of this branch, with either the full project or the unzipped review package as `ROOT`:

```bash
PYTHONPATH=src python -m vgx.gpqa.conditioned_likelihood --root "$ROOT" \
  --output results/gpqa_conditioned_likelihood_20261009          # add --check-cache in the full project
PYTHONPATH=src python -m vgx.gpqa.conditioned_likelihood_report \
  --input results/gpqa_conditioned_likelihood_20261009 --output reports/gpqa_conditioned_likelihood_20261009
PYTHONPATH=src python -m pytest -q tests/test_conditioned_likelihood.py
```

Protocol and hashes:
- Config SHA-256: `609d07c7…7950756`
- `conditioned_likelihood.py` SHA-256: `a60a44dd…5defa1cf`
- Signal tables: Gemini `7e530d48…ff414a758`, Qwen `fbe9885c…2fd2f4`
- The full protocol is in `manifest.json`.

## 9. Files in this directory

| File | Contents |
|---|---|
| `TABLES.md` | All primary tables, per-group tables, stopping transitions, Brier and stability by seed |
| `policy_results.csv`, `group_results.csv` | Every generator × cohort × seed × loss × policy, overall and by confidence group |
| `paired_vs_pooled.csv` | Paired differences against B with bootstrap intervals (utility, releases, wrong releases, queries, Brier) |
| `stopping_transitions.csv` | Item-level stopping changes against B |
| `signal_posterior_brier.csv` | Post-signal posterior Brier, by group |
| `fitted_likelihoods.csv` | Every fitted pooled and conditioned table (per fold, group, bin, with likelihood ratios) |
| `tau_selection.csv`, `training_group_sizes.csv` | Inner-CV scores per τ, selected τ, training cell sizes and bin counts |
| `item_decisions.jsonl.gz` | Per-item decision traces for every run, loss and policy (stage values, posterior, costs, group) |
| `manifest.json` | Sealed protocol (config, input and implementation hashes), run report, and SHA-256 of these files |
