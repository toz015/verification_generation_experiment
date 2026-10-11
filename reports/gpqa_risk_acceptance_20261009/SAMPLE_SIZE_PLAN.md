# Sample-size plan and blockers

Planning inputs only. Nothing here is a result about the GPQA questions. Full tables: `TABLES.md` section 9, `plan/sample_size_plan.json`, `tables/sample_size_curve.csv`, `figures/sample_size.png`.

## How the numbers are computed

- A rule certifies when its exact one-sided bound U(k, m) = BetaQuantile(1 - delta/M; k + 1, m - k) is at most alpha.
- **Minimum accepted examples.** For an observed error count k, the smallest m with U(k, m) <= alpha (`risk_control.min_accepted`).
- **Power.** For a rule whose true conditional error is r, the pass probability with m accepted examples is P(Binomial(m, r) <= k_max(m)), where k_max(m) is the largest k that still certifies. Because this is a sawtooth in m, the plan reports the smallest m at which it is at least 0.8 and stays there for the next 25 values (`risk_control.accepted_for_power`).
- **Questions.** Calibration questions = accepted examples needed / acceptance rate. The acceptance rates come from out-of-fold development scores (249-question cohort, median over 5 seeds). They are optimistic, because the family was declared after those data were inspected.

## Reference checks

| check | computed | expected |
|---|---|---|
| zero errors, alpha = delta = 0.05, M = 1 | 59 | 59 |
| zero errors, alpha = delta = 0.05, M = 20 | 117 | 117 |

## Accepted calibration examples needed

| | M = 1 | M = 26 |
|---|---|---|
| alpha 0.05, zero errors observed | 59 | 122 |
| alpha 0.10, zero errors observed | 29 | 60 |
| alpha 0.05, 80% power, true risk 0.01 | 124 | 274 |
| alpha 0.05, 80% power, true risk 0.02 | 286 | 593 |
| alpha 0.05, 80% power, true risk 0.03 | 694 | 1,470 |
| alpha 0.10, 80% power, true risk 0.02 | 61 | 135 |
| alpha 0.10, 80% power, true risk 0.04 | 142 | 280 |
| alpha 0.10, 80% power, true risk 0.06 | 333 | 693 |

The margin between the true risk and alpha drives the requirement much more than the family size does.

## Implied calibration questions for candidate rules

Development acceptance rate and k/m are out-of-fold medians on the 249-question cohort. "Questions" assumes the stated true risk.

| rule | dev m/N | dev k/m | alpha | assumed risk | M = 1 questions | M = 26 questions |
|---|---|---|---|---|---|---|
| gemini:A:0.98 | 0.486 | 0.041 | 0.10 | 0.041 (dev estimate) | 293 | 603 |
| gemini:A:0.98 | 0.486 | 0.041 | 0.05 | 0.020 | 589 | 1,221 |
| gemini:A:0.98 | 0.486 | 0.041 | 0.05 | 0.041 (dev estimate) | 7,837 | 17,247 |
| qwen:B:0.98 | 0.506 | 0.095 | 0.10 | 0.060 | 659 | 1,370 |
| qwen:B:0.98 | 0.506 | 0.095 | 0.10 | 0.095 (dev estimate) | 48,348 | 108,126 |
| qwen:B:0.98 | 0.506 | 0.095 | 0.05 | 0.095 (dev estimate) | not certifiable | not certifiable |

Read with the development evidence:

- **Qwen with Flash (B or C).** Development k/m is about 0.09 to 0.15 at the declared thresholds with non-trivial acceptance, and no lower than about 0.08 anywhere on the 249-question curves. If that is close to the truth, alpha = 0.05 cannot be certified at any sample size, and alpha = 0.10 needs tens of thousands of questions. Certifying Qwen needs a better signal, not more data.
- **Gemini, generator confidence alone (A at 0.98).** Development k/m is about 0.04 at about 49% acceptance. alpha = 0.10 needs roughly 300 to 600 independent questions (M = 1 to 26). alpha = 0.05 needs well over a thousand unless the true risk is near 0.02 or below.
- Questions are shared by both generators, so a study certifying several rules needs the largest of their requirements, not the sum.

## Predeclaration

Before any calibration label is read, fix and record (with hashes): the configuration, the family and its order, the fitting procedure and fit data, the tie-breakers, the calibration and evaluation sizes, and the data source. Run calibration once on the full predeclared set. Do not add questions after seeing calibration results, and do not re-run on a new split if the first run fails.

Two design choices could reduce the requirement and are allowed if declared beforehand: a smaller family chosen from development evidence (for example only the Gemini A rules, M = 3), and alpha = 0.10 as the operating target. Neither changes the conclusion for Qwen with the current Flash signal.

## Blockers (missing inputs)

1. **Independent calibration and evaluation data.** None exist in this project. The 249-question development cohort (GPQA Main outside Diamond) has been used for all design work; the 199-question cohort is a subset of it; Diamond was used in earlier experiments and its answer keys are not read here. Inferred from the public GPQA subset sizes (not re-verified here): no unused GPQA Main questions remain, and the roughly 100 Extended-only questions were excluded from Main by GPQA's own quality filters, so they are a different distribution and too few. Certification therefore needs a new same-distribution question source of the size above, or a decision to certify on another benchmark separately (never pooled).
2. **Original request caches.** The review package has no raw request caches, so cached responses could not be re-parsed from source. The export takes Flash scores, failures and usage estimates exactly as recorded in the frozen nested-combination traces, which were checked against the package manifest and are consistent across every trace file.
3. **Candidate answer letters and generator raw text** are not in the package. Validity and correctness are taken from the frozen raw-prior analysis files.
4. **Confirmed billing.** Costs are usage estimates; billed amounts were not available.
