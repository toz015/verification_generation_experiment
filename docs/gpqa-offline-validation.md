# Offline validation of forecasts and nested stopping

This analysis reuses the 2026-10-04 Vertex and Jev responses. It makes no provider
requests, loads no credentials, and changes no candidate, prompt, response,
parser or sealed live policy. The existing Diamond results have informed
development, so **every new real-data comparison is exploratory**.

## Run

The private analysis records and sealed cost policies must already exist at the
paths in `configs/gpqa_offline_validation.json`.

```bash
.venv/bin/python -m vgx.gpqa.offline_validation \
  --config configs/gpqa_offline_validation.json \
  --output results/gpqa_offline_validation_20261004

.venv/bin/python -m pytest tests/test_offline_validation.py -q

MPLCONFIGDIR=/tmp/vgx-matplotlib .venv/bin/python -m vgx.gpqa.offline_plots \
  --report results/gpqa_offline_validation_20261004/report.json \
  --output reports/gpqa_offline_validation_20261004
```

The output protocol records input and implementation hashes, package versions,
configuration and seed. Reusing an output directory with different inputs,
implementation or configuration fails. Choose a new directory for a deliberate
protocol revision. Reports contain aggregate results; individual decisions stay
under the ignored `results/` directory.

## 1. Do verifier signals add information?

Keep the generator answer fixed and forecast its correctness using:

- Raw elicited generator confidence and a calibration-set constant base rate.
- A generator-only logistic calibration of its clipped log-odds.
- The same generator feature plus each verifier separately, or all three
  primary verifiers together: Jev Choice, Llama and Gemini.
- The existing three-bin Bayesian update, starting with either the raw or
  calibrated generator prior, for each verifier separately.

Logistic regularization is fixed at `C=1`; generator log-odds are clipped at
`1e-4` for this diagnostic fit. Verifier features are raw probabilities plus
missingness indicators. Missing scores are imputed with **training-fold medians**.
All scaling, imputation, coefficients and likelihood tables use calibration
records only. Three-fold stratified calibration predictions are out of fold;
evaluation predictions use models fitted on all eligible calibration records.

Compare Brier score and log loss with paired intervals. ECE, AUROC and reliability
bins are diagnostics. A score's apparent calibration does not establish an LLM's
internal belief. The joint logistic forecasts test additional predictive
information without multiplying independent verifier likelihoods; they are not
substituted into the planner without a compatible observation model.

Risk–coverage curves rank fixed forecasts. When the boundary contains tied
scores, report expected errors under uniform random selection from that tie.
Its hypergeometric interval describes **tie-breaking variation on these items**,
not population uncertainty. Evaluation labels score the curve but never choose a
deployable threshold. Joint forecasts use all their verifier signals; their
curves do not represent cost-free adaptive execution.

## 2. Does nested stopping outperform simpler policies?

Use the same candidate, prior variant, likelihood tables, stage order and
expected per-query costs for each comparison:

| Rule | Query decision |
|---|---|
| None | Release or abstain immediately |
| Always | Query every remaining stage, then release or abstain |
| Uncertainty gate | If the initial prior is close enough to the release threshold, query every stage |
| Myopic | Query the next stage only if that one signal improves expected terminal utility after its cost |
| Nested | Query when the value of the entire remaining sequence exceeds stopping now |

The primary order is fixed from cheap to expensive: **Jev Choice → Llama →
Gemini**. Secondary nested order selection evaluates all six permutations using
calibration out-of-fold utility. Gate width is also chosen using calibration
folds only. The selected calibration score is optimistic because it is the best
of several candidates; it is not independent validation.

The offline decision boundary accepts no labels and exposes a response only
after its stage is selected. A missing requested response causes abstention,
costing one logical query. All 196 evaluation items remain in the policy
denominator; the invalid generator response abstains without querying.

The nested rule is Algorithm 1's precomputed `J`/`Q` tables followed by Algorithm
2's sequential decisions. Both rules operate on singleton verifier layers in a
fixed order. They cannot skip an earlier verifier to reach a later one. This is
the implemented scope, not optimal selection over arbitrary verifier subsets.

### Utility and money

For a correctness belief `b`, the value of releasing is `A(b) = R*b - L*(1-b)`.
Stopping is `max(0, A(b))`; the release threshold is `L/(R+L)`. Query cost enters
`Q_k(b) = -c_k + E[J_(k+1)(b')]`. Primary values are `R=1`, `L=19`, and
`c_k = 10 * E[USD per calibration request for verifier k]`.

The conversion factor is a chosen value of money relative to wrong answers;
it is not learned from API billing or added to model prompts. Replaying cached
signals incurs **zero new API spending**. Policy costs describe the expected
deployment cost of the logical requests, not a new invoice. The original source
prices and usage estimates remain unchanged and are not confirmed billing.
Generator cost is fixed and common across policies, so the comparison includes
incremental verification cost only.

Prespecified sensitivity axes are query cost multipliers `0.1, 1, 10, 100` and
wrong-release losses `1, 4, 19, 99`. These are descriptive comparisons, not a
search for a favorable evaluation setting.

### Uncertainty

- Paired evaluation bootstrap: 1,000 resamples, conditional on fitted models.
- Two-sample refit bootstrap: 200 independent calibration/evaluation resamples;
  refit generator calibration, joint forecasts and verifier likelihoods.
- Refit intervals hold the primary order, feature set, regularization, binning,
  prices and utility scale fixed. They do not include uncertainty from choosing
  those design parameters or provider model training.
- Released accuracy uses Wilson intervals. Only 15 valid evaluation answers are
  incorrect; apparent high precision and bootstrap tails remain fragile.

Main-minus-Diamond calibration and Diamond evaluation have different selection
rules and subject mixes. Resampling does not resolve this distribution shift.
Verifier likelihood updates still assume independence conditional on correctness,
including from the generator's confidence. Different model families do not
guarantee that assumption.

## 3. Does the algorithm work when the data-generating process is known?

Synthetic worlds specify correctness priors, binary verifier sensitivities and
specificities, and costs. A mixture of independent and shared-uniform signals
creates conditional dependence while preserving each verifier's marginal
quality. Overconfidence shifts the reported prior's log-odds by `+1.2`.

The exact oracle knows the true joint distribution and follows the same fixed
order. It does **not** observe the current hidden correctness or future signal
realizations. Recursion enumerates the finite tree; evaluation integrates over
all outcomes exactly. Comparisons include:

- Correct independent model, conditional dependence, prior overconfidence, and
  both forms of misspecification together.
- Uninformative positive-cost verifiers, where querying should stop.
- An uninformative first stage followed by a useful verifier, where lookahead
  has value under the fixed-order constraint.
- Grid sizes 101, 1,001 and 10,001 versus exact value recursion.
- Likelihood estimation from 50, 250 and 1,000 calibration examples, with 40
  replicates per size. Evaluation remains exact; only calibration is sampled.

Passing these simulations establishes behavior in the tested finite worlds.
It does not certify assumptions, empirical gains or general mechanism-design
claims for LLMs. Generator incentives, strategic reporting and generator updates
remain outside this validation.

## Decision after the analysis

Use this study to decide whether the next bottleneck is signal quality, prior
calibration, observation-model assumptions or stopping logic. Freeze any revised
design on development data before collecting an independent confirmation set.
Do not repeatedly use these Diamond outcomes as fresh confirmatory evidence.
