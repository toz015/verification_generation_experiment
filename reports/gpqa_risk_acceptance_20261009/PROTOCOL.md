# Risk-controlled acceptance: frozen protocol

Status: frozen for the software dry run on 2026-10-09. Certification is **pending**: it needs calibration and evaluation data that did not inform this design, and none exist in this project (see "Blockers" in `SAMPLE_SIZE_PLAN.md`).

Configuration: `configs/gpqa_risk_acceptance.json` (its SHA-256 is in `manifest.json`). Code: `src/vgx/gpqa/risk_control.py`, `risk_exploration.py`, `risk_certification.py`, `risk_report.py`.

## 1. Target

Maximize the acceptance rate P(A) subject to the conditional error among accepted answers P(W | A) <= alpha, certified with confidence 1 - delta.

- Primary alpha = 0.05. Secondary sensitivity alpha = 0.10. delta = 0.05.
- The target is conditional error among released answers. It is not the unconditional P(W and A), a stopped FDR, mean posterior error or expected utility. Utility and posterior diagnostics are reported as secondary only.

## 2. Inputs (unchanged)

- Generators: Qwen (primary) and Gemini (comparator). Their candidate answers, original confidence values, prompts, model settings and cached responses are used as frozen; nothing is regenerated or retried.
- Verifier: Gemini Flash, probability format (`gemini_flash:probability`). No third verifier is added.
- Generator confidence is used as given: no calibration, replacement or endpoint clipping.
- Source files: the review package `GPQA_experiment_and_diagnosis_review_20261009` (`results/gpqa_raw_prior_completed_signals_20261006` for confidences, labels, folds and frozen decisions; `results/gpqa_nested_combinations_20261008` traces for the cached Flash records). Every input file is checked against the package `FILE_MANIFEST.json` before use.
- The compact export `data/observations.json.gz` holds, per generator and question ID: validity, raw confidence, correctness, and the cached Flash record (score, failure, usage estimate, request key); plus the frozen folds, fold fits and decisions. It contains no question text, answer options or credentials.
- No GPQA Diamond answer key is read. Diamond was used in historical experiments and is not described as untouched.

## 3. Methods

All three methods produce, for every question, an eligibility flag and a selection score. A rule is "accept iff eligible and score >= threshold". Scores are treated as selection scores, not as calibrated probabilities.

- **A, generator confidence.** Eligible iff the candidate is valid. Score = raw generator confidence. No verifier query.
- **B, always-query Flash.** Every valid candidate is sent to Flash. Score = Bayes belief from the raw confidence and a pooled likelihood P(score bin | correct), 3 equal-width bins, Laplace 1, fitted on fitting data only. A malformed Flash response is paid for and makes the candidate ineligible.
- **C, sequential Flash plus a gate.** The frozen nested sequential policy (R = 1, L = 19, 10 utility units per USD, grid 1001, pooled likelihood, expected cost = mean fitting-data usage) decides whether to query Flash and whether to release. Only candidates it releases are eligible; their score is the terminal belief. The acceptance gate is a threshold on that belief; abstentions remain abstentions. This is a fixed query policy plus an acceptance gate. It is not optimal stopping for the risk-constrained objective, and its query rule is not re-optimized for that objective.

## 4. Failures and denominators

- Invalid candidate (no parsable A-D answer or invalid confidence): counted in N, never accepted, never sent to Flash.
- Returned malformed Flash response: counted in N, paid for (usage estimate counted), never accepted under B; under C the policy abstains.
- No returned response is retried. N always includes every question of the partition.

## 5. Certification procedure (reference)

1. **Partition.** Question-level, shared by both generators. Roles: fit, calibration, evaluation. The dry-run partition orders question IDs by SHA-256 of `"20261010|<question_id>"` and cuts 34% / 33% / 33%. A future study should use separately collected calibration and evaluation sets of predeclared size.
2. **Fit** (fit role only): pooled likelihood and expected cost per generator, and the sequential planner built from them.
3. **Score** calibration questions without reading labels; the Flash record of a question is reachable only through a counted query.
4. **Count**, for every rule of the declared family, m = accepted and k = accepted and incorrect.
5. **Bound.** Exact one-sided binomial upper bound U = BetaQuantile(1 - delta/M; k + 1, m - k), U = 1 if k = m; rules with m = 0 are ineligible.
6. **Family.** M = 26, every rule eligible for joint selection, declared before any calibration label:
   - generators {Qwen, Gemini} x
   - A thresholds {0.95, 0.98, 0.99}; B thresholds {0.95, 0.97, 0.98, 0.99, 0.995}; C thresholds {0.95, 0.97, 0.98, 0.99, 0.995}.
   - Declared order: generator, then method A/B/C, then threshold ascending.
7. **Eligible** iff m > 0 and U <= alpha. No monotonicity of risk in the threshold is assumed; every rule is bounded separately.
8. **Select** the eligible rule with the highest calibration acceptance rate m/N. Ties: lower expected verifier cost per question (fit-role cost model), then declared order. Labels never enter the tie-break.
9. If no rule is eligible, report exactly "No positive-coverage rule certified." Abstaining on everything is not reported as a certified 0% rule.
10. **Seal** the selection (fitted tables, family, alpha, delta, partition hash) with a SHA-256 checksum before evaluation labels are opened. `evaluate` refuses a selection whose checksum or partition hash does not match.

**Guarantee (when its assumptions hold).** With probability at least 1 - delta over the draw of the calibration set, every rule the procedure marks eligible, and therefore the selected rule, has P(W | A) <= alpha on the distribution the calibration questions were drawn from. It is a statement about the population rate. It does not promise that any particular deployment batch has at most an alpha fraction of errors among its accepted answers.

## 6. Alternative procedure (secondary)

Fixed-sequence testing inside each (generator, method) chain, ordered from the strictest to the most lenient threshold, with Bonferroni over the 6 chains (each test at delta/6). Testing in a chain stops at its first failure. Reported separately; the reference selection always comes from section 5. It is validated by Monte Carlo against the reference (familywise error at most delta under all-at-alpha and non-monotone scenarios). With few accepted examples at the strict end of each chain it usually stops at the first test, so it is less powerful than the reference here.

## 7. Evaluation (frozen rule)

Apply the sealed rule to evaluation questions. Report N, valid candidates, m, k, k/m with an exact 95% interval, m/N with an exact 95% interval, queries, expected and usage-estimate costs, cost per question and per accepted answer. Observed evaluation risk, certification status and evaluation results are reported separately. If no rule was certified, only descriptive counts for each family rule are shown.

## 8. Assumptions

- Calibration questions and future questions are independent draws from the same distribution (same benchmark population, same generator and verifier models, prompts, decoding settings and parsing).
- Model behavior is stable between calibration and use: the same cached or re-issued requests would produce answers and Flash scores from the same distribution. Provider model updates break this.
- Labels are correct.
- The family, thresholds, fitting procedure, tie-breakers and calibration size are fixed before calibration labels are read; calibration data are used once; there is no optional stopping or topping up.
- Fit-role data are independent of calibration and evaluation data (the partition is by question ID).

## 9. What the development data can and cannot show

- Phase I uses the 249-question cohort (primary) and the 199-question cohort (pilot excluded). The cohorts overlap; the 5 fold seeds re-split the same questions. Results are exploratory and are not independent confirmation.
- Thresholds or regions read from Phase I curves were chosen on the same data and are not certified.
- The Phase II dry run splits the same, repeatedly inspected development data. It exercises the software; it is not a certification and not a prospective confirmation.

## 10. Limitations

- Costs are counterfactual usage estimates from cached calls, not confirmed billing.
- The probability likelihood has 3 bins, so B and C scores take few distinct values; the threshold grid is coarse at the top.
- C can only accept beliefs at or above 0.95 (its L = 19 release bar), which caps its coverage.
- The 26-rule family was declared after the development data were examined; that is allowed for a future independent calibration set but makes any development-data bound optimistic.
- Certification on GPQA would only describe GPQA-like questions. Other benchmarks need their own calibration; distributions are not pooled.
