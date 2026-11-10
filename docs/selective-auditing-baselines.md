# Selective-auditing baselines: AT and AT+CC on QuantumKatas

An add-on experiment on existing cached data, not part of the pre-registered main experiment.
Paper: Gibbs, Tibshirani, Han, Zrnic, *Reliable AI-Assisted Decisions via Selective Auditing*,
arXiv:2610.31662v1 (Sec. 2, 3.1, 3.2 with Algorithm 1, Sec. 6, App. C). Algorithm 2 is out of scope.

## Commands

```
python -m pytest tests/test_selective_auditing.py
python -m vgx.selective_auditing_runs tune        # development only (GPQA calibration); writes the frozen config
python -m vgx.selective_auditing_runs simulate    # synthetic streams
python -m vgx.selective_auditing_runs smoke       # 10 seeds, alpha 0.1, primary order
python -m vgx.selective_auditing_runs run         # full QuantumKatas replay + end-to-end + sensitivity
python -m vgx.selective_auditing_report           # reports/selective_auditing_qk_20261110/results.html
```

Code: `src/vgx/selective_auditing.py` (oracle, AT, AT+CC, random audit, metrics),
`src/vgx/selective_auditing_runs.py` (streams, tuning, replays), `src/vgx/selective_auditing_report.py`.
Config: `configs/selective_auditing_qk_20261110.json`. Outputs: `results/selective_auditing_qk_20261110/`.
No model API is called; the cached candidates, raw `p_correct` and official checker labels are fixed.

## Review: issues that shape the protocol

1. **The paper reports no hyperparameter values.** App. C says each method's hyperparameters were chosen
   once and held fixed across datasets, without listing eta, gamma, beta or the grid. Every value used
   here is this implementation's choice, selected on development data and frozen (below).
2. **Short streams make the guarantee loose.** Theorem 1 bounds |Err(T) - alpha| by (1/eta + 1/gamma)/E.
   QuantumKatas has E = 134 (Qwen3), 113 (gpt-oss) and 35 (Sonnet) errors. Keeping the bound small needs
   large eta and gamma, and gamma is the spot-check rate on every high-confidence round, so it caps the
   audit savings: with gamma = 0.5 at least half of the confidently accepted rounds are still audited.
3. **The paper's risk is not ours.** Err(T) = E[M]/E[E] (unaudited errors over all errors) is the leak
   rate; it equals our FPR when "audited" corresponds to "not released". Our release-error target is
   M/A (selective risk). The two are reported separately and never treated as the same constraint.
4. **The feedback differs from our method.** AT audits before the final decision and the auditor corrects
   the error; the learner sees a label only when it audits. Our method abstains (no correction), uses about
   258 labelled training tasks per fold to fit likelihoods and thresholds, and pays for verifier calls.
   Results against our method are therefore a cross-protocol reference, not a ranking.
5. **Coarse confidences.** Raw QuantumKatas `p_correct` takes 18 to 25 distinct values per generator, so
   the AT+CC grid only matters where values fall; ties are handled by the strict `C_t < tau_t` rule.
6. **End-to-end failures.** 12 (Qwen3) and 14 (gpt-oss) tasks have no extractable code and no confidence.
   They are excluded from the main stream (counted) and entered as confidence 0, error 1 in a sensitivity run.
7. **Expectation, not per run.** The guarantee is about E[M]/E[E]. Per-run leak rates are reported as a
   distribution and can sit well above alpha for a single run, especially with few errors.
8. **Relation to SSV.** AT is a simplification of Kiyani et al. Algorithm 1, which is our existing SSV
   baseline; they are kept separate here.

## Protocol

- **Semantics.** H_t = 0: accept the raw candidate, never read its label. H_t = 1: hand it to the auditor
  and reveal the official checker verdict to the learner. Candidates and their original labels never change;
  no corrected answer is claimed. An independent evaluator reads all labels and never feeds the learner.
- **Oracle.** `Oracle.audit(t, mask)` is the only label path; it records every reveal and refuses a second
  reveal of the same round. The audit decision is fixed (u_t < pi_t) before the call.
- **AT.** tau_1 = 0.5; pi_t = 1 if C_t < tau_t else gamma; H_t ~ Bern(pi_t);
  tau_{t+1} = tau_t + eta (1 - pi_t - alpha) e_t H_t / pi_t. No clipping, no other update.
- **AT+CC (Algorithm 1).** Grid 0 = theta_1 < ... < theta_k <= 1 plus theta_{k+1} = +inf; w_1 uniform over
  k+1; S_t = sum_j w_j 1{C_t >= theta_j}; pi_t from S_t; on audit, g_j = (S_t - y_t) 1{C_t >= theta_j} / pi_t,
  log-weights minus beta g_j then log-sum-exp normalisation; the same tau update. S_t, pi_t and g_t use the
  state before the round's update. Unaudited rounds change nothing.
- **Random audit.** pi_t = 1 - alpha (valid by construction) as a sanity reference; the hindsight oracle
  threshold (all labels) is shown as an unattainable reference, as in the paper.
- **Streams.** Evaluable rounds only (code, confidence in [0, 1], official label). Orders: primary
  (seeded permutation, seed 20261110), plan order (grouped by kata), and 20 seeded permutations.
- **Randomness.** 100 audit seeds per order; run r uses default_rng(1000 + r) for all methods (common random
  numbers, not common audit decisions). Seeds and orders are not new independent questions.
- **Metrics.** M = unaudited errors, E = errors, A = auto-accepted, T = rounds. leak_error_rate = M/E (the
  paper's Err), selective_risk = M/A, total_unaudited_error_rate = M/T, plus A/T, audits/T, E/T and the
  mandatory / spot-check split. Across seeds: ratio of mean counts (estimates the paper's E[M]/E[E]) and the
  per-run distribution, reported separately. E = 0 or A = 0 gives NA. Brier is computed by the evaluator over
  all rounds, using the pre-update score (raw C_t for AT, S_t for AT+CC).
- **Cost.** Logical audits = the number of H_t = 1, whatever the cache did; confidence comes with the original
  generator call (not recharged); local update time and state size are recorded; new API spend is zero. No
  dollar price is put on an audit. AT has no budget B; it is not stopped when audits run high.
- **Hyperparameters.** Chosen on development data only: the 249 GPQA calibration questions of the same
  three generators (raw `p_correct`, answer-key labels), 50 seeds. Stage 1: eta and gamma shared by AT and
  AT+CC (beta 1, grid step 0.1). Stage 2: beta and grid for AT+CC. Feasible if, for each method and alpha,
  the mean over generators of (leak - alpha) <= 0.01; among feasible, least mean audit rate. The first search
  (eta <= 0.2, gamma <= 0.2) had no feasible point, so it was widened once on development data. Frozen:
  eta 1.0, gamma 0.5, beta 1.0, grid step 0.05 (21 points). Two alternative sets (gamma 0.2; eta 0.2 with
  gamma 0.1) were declared before the QuantumKatas runs and are reported only as sensitivity.

## Differences from the paper

- Hyperparameters are ours (the paper gives none).
- The paper's experiments have thousands of rounds; ours have 308 to 322.
- Confidence is the generator's own `p_correct`, not a separate verifier or self-consistency vote.
- The auditor is the official checker label read from cache; no correction is generated.
- alpha is evaluated at 0.05, 0.10, 0.20, 0.30 (the paper plots 0.1 to 0.3).
