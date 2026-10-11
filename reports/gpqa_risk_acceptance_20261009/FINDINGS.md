# Risk-controlled acceptance on GPQA: findings

**Status: no rule is certified, and certification is pending.** The offline software, the exploratory analysis and a sample-size plan are complete. A certificate needs calibration and evaluation questions that did not inform this design, and none exist in this project.

Target: maximize acceptance P(A) subject to conditional error among accepted answers P(W | A) <= alpha, with confidence 1 - delta = 0.95. Primary alpha = 0.05; sensitivity alpha = 0.10. Methods: A = generator confidence, B = always-query Gemini Flash (pooled likelihood), C = the frozen sequential Flash policy plus an acceptance gate (a fixed query policy with a gate, not optimal stopping). Details: `PROTOCOL.md`. All numbers: `TABLES.md`. Plot: `figures/risk_coverage.png`.

No paid calls, no regenerated candidates, no retried responses, no Diamond answer keys. Generator confidence is used raw.

## 1. Measured on development data (exploratory)

Out-of-fold scores on the 249-question cohort (primary) and the 199-question cohort without the pilot, 5 frozen fold seeds each. The cohorts overlap and the seeds re-split the same questions, so these are development results, not independent confirmation. Ranges are over seeds.

1. **Flash improves Qwen's ranking a lot.** AUROC of correct over incorrect rises from 0.701 (A) to 0.879-0.889 (B) on 249 questions, and from 0.710 to 0.898-0.906 on 199. The per-seed paired bootstrap 95% interval of the gain stays above 0.12 in every seed.
2. **For Qwen, Flash does not create an alpha = 0.05 region.** No threshold of any method reached development k/m <= 0.05 in any seed or cohort. With at least 10 accepted answers, the lowest k/m for B and C was 0.078-0.092 (249 questions) and 0.057-0.067 (199 questions). At alpha = 0.10, B and C reach k/m 0.095-0.098 at 57-62% acceptance (249) and 0.074-0.100 at 61-65% (199), right at the edge, with single-rule 95% upper bounds of 0.13-0.15. A alone never gets below 0.156 on 249 questions or 0.135 on 199 (both at about 18% acceptance).
3. **For Qwen, a higher score is not always lower risk.** Among answers with a high Flash score (seed 20261005), Qwen's 0.98-confidence answers were wrong 4 of 42 times and its 0.95-confidence answers 3 of 59 times. The counts are small, so this is an observation, not a finding; it is the reason the procedure bounds every rule separately instead of assuming risk falls with the threshold.
4. **For Gemini, its own confidence already gives the low-risk region; Flash adds little.** A at threshold 0.98 accepts 48.6% with k/m 0.041 (121 accepted, 5 wrong); B and C at 0.97-0.98 accept 48.2% with k/m 0.042. Flash moves AUROC only from 0.730 to 0.750-0.761, and the gain's 95% interval includes zero in 4 of 5 seeds (249) and 5 of 5 (199).
5. **B and C accept the same answers; C asks Flash less often.** At the declared thresholds (all >= 0.95), C's accepted set equals B's in 99 of 100 threshold x seed x cohort comparisons. C skips 8-18% of B's Flash queries for Qwen and 24-36% for Gemini. C cannot accept anything below belief 0.95, so it cannot reach B's higher-coverage operating points.
6. **The scores are not calibrated probabilities of being correct.** Read as probabilities, Qwen B scores at threshold 0.98 imply 1.2% error among accepted answers; observed k/m is 9.4-9.5%. Gemini A at 0.98 implies 1.4%; observed 4.1%. This repeats the posterior risk underestimation seen earlier. The confidence-conditioned likelihoods in PR #2 barely changed decisions and are cited here only as a historical reference; no new shrinkage search was run.
7. Secondary only: mean utility at L = 19 when accepting at threshold 0.95 is negative for every method (for example Qwen B -1.36 to -0.61, Gemini A -0.76), consistent with point 6.

Taken together: Flash improves ranking enough to produce an approximate alpha = 0.10 region for Qwen that raw confidence does not have, but not an alpha = 0.05 region. For Gemini the region comes from raw confidence, and Flash neither creates nor clearly extends it. No method is declared the winner on development coverage.

## 2. Phase II software dry run (not a certification)

The certification workflow (fit, then risk calibration, then evaluation, with the selection sealed before evaluation labels are opened) was run on a question-level split of the same development data: 85 fit, 82 calibration, 82 evaluation questions, shared by both generators.

- alpha = 0.05 and alpha = 0.10: **No positive-coverage rule certified.** The smallest bound among the 26 rules was 0.261 (Gemini A at 0.98: 42 accepted, 3 wrong).
- The fixed-sequence alternative also certified nothing.
- This outcome was expected from the sample size alone: at M = 26, even a rule with zero errors needs at least 122 accepted calibration answers for alpha = 0.05, or 60 for alpha = 0.10, and the calibration partition has 82 questions.
- Evaluation-partition counts are reported in `TABLES.md` section 8 as description only. There is no certified rule to evaluate.

## 3. Statistical guarantees

- **None is established for GPQA.** The development data were used to design everything, and the dry run split those same data.
- **What the procedure would guarantee on independent data:** under the assumptions in `PROTOCOL.md` section 8, with probability at least 0.95 over the calibration sample, the selected rule's population P(W | A) is at most alpha. It does not promise an error fraction at most alpha in any particular deployment batch.
- **Procedure check (synthetic):** in 12 Monte Carlo settings with 5,000 repetitions each (6 chains of 5 nested rules, including non-monotone risk), the share of runs that certified any rule with true risk >= alpha was at most 0.027 for the reference Bonferroni procedure and at most 0.038 for fixed-sequence, both within delta = 0.05. Fixed-sequence was much less powerful here; it never selected a rule when the strictest thresholds carried the most errors.

## 4. Interpretation and next steps

- **Qwen with Flash:** the residual error among its best-scored answers (about 8-10% on development data) is above both targets. If that holds, alpha = 0.05 is not certifiable at any sample size, and alpha = 0.10 would need tens of thousands of questions. Certifying Qwen needs a stronger signal, not just more calibration data. That is a design decision for you; this study did not add a verifier.
- **Gemini's own confidence** is the most plausible candidate. At alpha = 0.10 a certificate looks feasible with roughly 300 to 600 independent questions (family of 1 to 26 rules). At alpha = 0.05 it needs over 1,000 questions unless the true error of the accepted set is about 2% or lower.
- **Blocker:** no independent same-distribution question set of that size exists in this project. See `SAMPLE_SIZE_PLAN.md`.

## 5. What was verified

- The observation export was rebuilt from the review package and is byte-identical to the shipped `data/observations.json.gz`; all 81 input files match the package manifest.
- Every frozen fold fit was refitted from its training questions (60 of 60), and every frozen always-query and sequential decision was reproduced, with an independent replay of the sequential policy (4,480 of 4,480 each).
- 28 new tests pass. Full suite: 320 passed, 15 skipped (SciFact and PubMedQA data not downloaded). See `test_results.txt`.
- Original request caches were not in the package, so cached responses were not re-parsed; Flash records are taken as recorded in the frozen traces. See the blockers list.
