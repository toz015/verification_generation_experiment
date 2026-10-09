# Verification-generation experiment

This repository explores confidence and verifier signals for selective scientific
QA. The GPQA pilot is motivated by Green–Laffont sequential verification; frozen
model forecasts do not test strategic truthfulness or incentive compatibility.

## Current experiment decision (2026-10-06)

Use Gemini and Qwen as generators, with their frozen answers and **raw,
self-reported `p_correct` as the initial belief `b0`**. The user has deferred
label-based generator-confidence calibration to future work. Do not apply the
logistic mappings, replace the prior with an empirical base rate, or make
confidence calibration a prerequisite for the main algorithm experiment.
The existing live executor already initializes belief from `candidate.p_correct`.

The offline confidence-calibration code and results below are retained as a
historical diagnostic only; they are not part of the active experiment. A raw
report is the probability elicited from the generator, not a demonstrated
faithful internal belief or a guaranteed empirical accuracy. Estimating verifier
conditional likelihoods from labeled reference questions is a separate existing
step; this decision does not change that step or allow access to evaluation
answer keys during sequential decisions.

## Fresh GPQA experiment

The new Vertex-only experiment calibrates on Main minus Diamond and evaluates
on disjoint Diamond questions: **249 calibration + 196 evaluation** after
documented duplicate-choice exclusions. See the
[fresh experiment guide](docs/gpqa-fresh-vertex.md). It downloads authorized
Hugging Face data and freezes new candidates; historical VM artifacts are not
required or used as evidence. Jev is deferred for this run.

The run is complete. See the [results and interpretation](reports/gpqa_fresh_vertex_20261004.md):
the live sequential policy released 158 answers, 155 correct, using 266 verifier
calls versus 390 for querying both. Total token-based estimated spending was
USD 1.912709 on `llm-applications-490420`; confirmed billing is not yet available.
Offline resume issued no API requests and reproduced the report exactly.

The subsequent [Jev comparison](reports/gpqa_jev_comparison_20261004.md) reuses
those candidates. Jev Choice sequential verification released 131 answers,
130 correct, with 45 queries. The complete Choice/Noul collection cost an
estimated USD 0.0224. See the [Jev protocol](docs/gpqa-jev-comparison.md) for
the separate USD 10 budget, strict parsing and offline rounding sensitivity.

## Historical GPQA pilot

The completed GPQA Main experiment uses **200 questions: 120 calibration and 80
held-out evaluation**. This expanded the original 50-question pilot design;
the 50/30/20 plan is retained as a synthetic test fixture, not the current
experiment size. Sampling and answer-choice shuffling are deterministic with
seed `20260928`.
See [the design](docs/superpowers/specs/2026-09-28-gpqa-verification-pilot.md) and
[configuration](configs/gpqa_experiment.json).

The current hosted comparison uses Gemini 3.8 Flash as generator, and Llama 3.3
70B plus Gemini 3.7 Flash as verifiers. The generator reports an explicit JSON
confidence. Each verifier estimates correctness for the fixed candidate without
seeing generator confidence or the answer key. These are elicited numbers, not
next-token probabilities. The historical experiment collected both verifiers
before replaying stopping decisions. Its reported adaptive savings remain
counterfactual.

The new [frozen-candidate workflow](docs/gpqa-frozen-workflow.md) separates recovery,
shared response caching, calibration-only planning, live sequential execution,
and offline scoring. **It has no generator collection path.** All commands are
offline by default; model access requires an explicit `--allow-api` opt-in.
The original prompts and hosted generator configuration are unchanged.

The original private run artifacts are unavailable locally, and its VM is
stopped. Resuming that historical run requires recovery and validation of its
original candidates. The user authorized the separate fresh run above; it does
not depend on those artifacts. The [earlier implementation status](docs/reviews/2026-10-04-implementation-status.md)
documents the historical recovery work.

## Offline validation

Use Python 3.12+ and CPU/dev dependencies (GPU extras are unnecessary):

```sh
uv sync --extra dev
uv run pytest
```

`tests/test_gpqa.py` uses invented questions, responses, and an injected fake
inference engine. It tests the original 50/30/20 design split, parsing, exact known scoring cases,
calibration/evaluation separation, stopping, failure costs and denominators,
reporting, and cache invalidation. **Synthetic results validate software only;
they do not demonstrate real verifier discrimination or calibration.** No tests
require model weights or GPQA data. Existing dataset-dependent tests skip when
those datasets are absent.

Score saved records without starting inference:

```sh
uv run python -m vgx.gpqa.report \
  --records /path/to/pilot_records.jsonl \
  --output /path/to/pilot_metrics.json
```

Add `--synthetic` for fixture records. The command uses the adjacent run
manifest's config when available, falling back to the repository config. An
explicit `--config` override is allowed and any difference from collection is
flagged. The report records the source-record hash and analysis-config hash.

## Reporting conventions

- Report the full sample, calibration, and evaluation partitions separately
  (fresh run: 445/249/196; historical run: 200/120/80). All/calibration
  diagnostics include fitted-on observations and are not held-out evidence.
- Answer accuracy counts a correct parsed answer even if its confidence is
  malformed. Invalid/missing answers count as incorrect. Forecast metrics use
  valid answer/confidence pairs and disclose missing counts and parse failures.
- Brier score, log loss, AUROC, five equal-width reliability bins and ECE compare
  explicit confidence with a constant base rate fitted only on valid calibration
  forecasts. Reliability tables and risk–coverage points are saved in JSON.
- Verifier likelihoods use three equal-width bins and Laplace smoothing of 1.
  A regularized score-only logistic comparison also fits calibration data only.
  Paired Brier/log-loss differences quantify added forecast value over generator
  confidence on matched items; negative differences favor the verifier update.
- Fixed-seed, 500-resample percentile bootstrap intervals cover Brier score,
  log loss, AUROC, policy utility, and paired utility/cost differences. Accuracy, coverage, and
  reliability-bin accuracy use Wilson intervals to avoid zero-width intervals
  for all-success/all-failure samples. A separate calibration bootstrap reports likelihood-bin stability and failed
  fits. Forecast intervals condition on the fitted likelihood; they do not fully
  propagate calibration uncertainty. One-class AUROC resamples are excluded and
  counted. Tiny/degenerate samples can yield misleadingly narrow intervals.
- Risk–coverage curves release score ties together, use the full partition as
  the coverage denominator, and integrate only over observed coverage. Missing
  scores are never extrapolated to full coverage.
- Dependence diagnostics include verifier/verifier and generator/verifier
  residual correlations, error agreement/correlation at 0.5, score correlations
  conditional on correctness, and the distance between empirical joint score
  bins and product marginals. Sparse cells make these descriptive checks noisy;
  they cannot certify conditional independence.
- Routing compares six policies on **every evaluation item**, with per-policy
  failure counts. Missing required inputs cause abstention; attempted queries
  still cost a call. A missing signal does not stop a sequential policy that
  already chose to release/abstain before requesting it. Confidence-based
  policies abstain without querying when generator confidence is invalid.
  An unestimable verifier likelihood marks its policy unavailable and reports
  the abstention fallback explicitly. Always-answer releases any valid answer.
- Query counts and normalized verification costs in the historical report are
  counterfactual policy costs. Records alone do not establish paid executions;
  actual observed executions are reported in the separate usage ledger. The
  configured sensitivity costs are not measured money or per-item latency.
- Vertex call logs preserve provider-reported prompt and completion token usage.
  The offline accounting command reconciles inclusive/separate reasoning counts
  and cached-input rates before estimating costs. Ambiguous usage remains
  unpriced. A provider execution is distinct from a reusable logical request;
  unknown outcomes and rejected attempts are disclosed. Imported billing
  evidence is separate from usage-based estimates. Live policies can explicitly
  map calibration mean USD to utility via a frozen `utility_per_usd` factor;
  realized evaluation usage never determines the preceding query decision.
- Bayesian updates use raw generator confidence and assume conditional
  independence, including from generator confidence given correctness. They
  are **not certified calibrated**. Exact 0/1 priors remain fixed. This patch
  does not silently clip or recalibrate them or tune thresholds on evaluation.
  Routing thresholds remain the preregistered `L / (R + L)` utility thresholds.

## Frozen artifacts, caching, and execution

The [offline validation report](reports/gpqa_offline_validation_20261004.md)
compares verifier information, simpler stopping rules, calibration uncertainty,
and exact synthetic oracles using the frozen Vertex/Jev responses. It makes no
new API requests. See the [protocol and commands](docs/gpqa-offline-validation.md).
The tested value-table implementation agrees with the oracle under correct
assumptions; these exploratory GPQA results do not establish a reliable advantage
over simpler verification rules.

Follow [the workflow guide](docs/gpqa-frozen-workflow.md) for concrete commands.
The old `vgx.gpqa.run_pilot` CLI now delegates to this frozen workflow; the former
unrestricted generator collection invocation is unavailable.

- Recovery validates the original source checksum, sample manifest, split,
  shuffled choices, response keys, prompts/settings, candidate records, and core
  reported generator metrics. Missing or mismatched inputs fail closed.
- Public candidate records exclude answer keys and historical verifier scores.
  Calibration records and evaluation labels are separate private artifacts.
- Exact provider requests share a cache under `results/request_cache/`, across
  collection and analysis runs. Request identity includes provider/model,
  endpoint/project scope, actual settings, prompt and system message. It excludes
  pricing, analysis code, role/item labels, and verifier-list bookkeeping.
- Successful responses are durable before progress is checkpointed. A request
  whose response was lost has an unresolved attempt and cannot automatically be
  repeated. HTTP rejections are logged and can be explicitly retried by resuming.
- Algorithm 1 writes reusable value tables. Algorithm 2 compares current stop
  and continuation values, requests only the selected next verifier, and logs
  the belief, values, observations, cost and stop reason. Ties stop; impossible
  observations and malformed required scores cause abstention.
- Hosted Gemini model names are still mutable aliases. A fixed cache freezes
  existing responses, not the provider's future model weights. The configured
  temperature/top-p are omitted from Gemini 3 request payloads; actual submitted
  settings and returned metadata are logged.

Vertex uses Application Default Credentials on the laptop, set up with
`gcloud auth application-default login`. This is separate from `gcloud auth login`
for the CLI. No GPU or running VM is needed for API experiments. Credentials
are used in memory only and are never stored in experiment logs.

The prepared [Jev comparison specifications](configs/gpqa_frozen_verifiers.json)
keep the original generator answers and prompts fixed. Jev Choice and Noul are
separate arms. Choice uses the probability assigned to the generator's option;
`confidence` is not treated as a correctness probability. Jev requests use an
explicit model version and separately versioned structured questions.

Question text, shuffled answer keys, manifests, prompts, call logs and records
must stay under gitignored `data/` and `results/`. Historical recovery remains
available when old artifacts are supplied; the fresh experiment uses the
separate workflow described above.

## Phase one: five-verifier signal screening

`configs/gpqa_signal_study.json` defines a calibration-only pool of five candidate
verifiers, each with a binary and an explicit probability prompt. This is a
screening pool, not a five-stage stopping policy. The original generator,
candidate answers, prompts, responses, and sealed policies remain frozen.
There are no Jev requests in this study.

The candidates are Gemini 3.7 Flash, Gemini 3.5 Flash-Lite, Mistral Small 3.1,
Mistral Medium 3, and Claude Haiku 4.5. They span three model families; different
families do not establish conditional independence. Google lists the Mistral
and Claude native managed endpoints, but project access must be checked before
collection. Unavailable models are reported without automatic substitution.
The previous Llama response set remains a historical baseline; its MaaS endpoint
is scheduled for retirement on 2026-10-21.

```bash
.venv/bin/python -m vgx.gpqa.signal_study prepare \
  --config configs/gpqa_signal_study.json \
  --output results/gpqa_signal_study_20261005_v2

# Offline report; never loads evaluation labels or calls a model.
.venv/bin/python -m vgx.gpqa.signal_study report \
  --output results/gpqa_signal_study_20261005_v2

# Exploratory calibration-fold forecasts on complete paired responses.
.venv/bin/python -m vgx.gpqa.signal_analysis \
  --output results/gpqa_signal_study_20261005_v2

# Paid collection: explicit opt-in, authorized project checked before requests.
.venv/bin/python -m vgx.gpqa.signal_study collect \
  --output results/gpqa_signal_study_20261005_v2 \
  --limit 10 --workers 3 --allow-api

.venv/bin/python -m pytest tests/test_signal_study.py -q
```

Review the first ten calibration candidates for access and format failures before
extending the same run to `--limit 50`, then `--limit 247`. Candidate order is
deterministic and label-blind. Successful exact requests are reused on resume;
timeouts with unknown provider outcomes remain blocked. Models and prompt
formats are compared on the same candidates. Failed responses are retained.

`plan.json` fixes candidate identities, prompt text, generation settings, prices,
and collection implementation hashes. Changing the study requires a new output
directory. `signal_report.json` reports score distributions, parsing failures,
confusion counts, descriptive likelihood fits, and usage-estimated costs.
`signal_records.json` remains private under ignored `results/`. A binary judgment
is not evaluated as though it were a calibrated probability. The diagnostic 0.5
score threshold is separate from the later stopping policy's release threshold.

This collection/report step does not select a winner or change live priors or
value tables. `signal_analysis` compares raw confidence, a training-fold base
rate, generator-only logistic calibration, and each signal added to that same
generator feature. It also reports binned Bayesian updates with binary-specific
two-bin likelihoods. Every fit uses training folds only. Within-model format
comparisons use complete pairs; the all-model comparison uses a common complete
cohort. Failures excluded from those forecast fits remain in the collection
report. Fewer than three errors or correct answers blocks three-fold fitting.
The initial generator diagnostic uses all eligible calibration candidates.

`calibration_comparison.json` stores fold membership, predictions and descriptive
paired differences. Small partial cohorts cannot establish which signal or model
is better. The ten candidate arms introduce selection uncertainty that needs to
be disclosed; full calibration collection and independent confirmation follow.

The stage's local usage-estimate cap is $20 within the previously authorized
$200 experiment total. It is not a provider billing limit; unknown outcomes keep
reservations and unpriced usage stops collection. Observed prior Vertex spend
was approximately $1.91, not a complete account billing audit. Gemini reasoning
tokens are reconciled separately from visible text; token-based estimates remain
distinct from confirmed billing. Generation uses no tools or prompt caching;
Gemini low reasoning and partner non-extended-thinking settings are recorded,
and do not imply equal internal computation across providers.

The initial run paused when a length-limited Gemini reply omitted
`completion_tokens`. Version 2 reconciles billable output from reported total
minus input tokens, checking consistency with reasoning tokens. The empty reply
remains a signal failure with nonzero cost. `migration.json` records copying the
160 existing responses into version 2 without new calls or candidate changes;
the original run is retained. Copied executions must be deduplicated by execution
ID when aggregating costs across both directories.

### First 50 calibration candidates (2026-10-05)

Completed 500 verifier requests (five models × two formats × 50 questions),
estimated at **$0.5440552**, with 490 parseable signals. The candidate answers
contain 43 correct and seven incorrect answers. No generator or evaluation
requests were made. The accounting migration preserved all 160 preceding
responses verbatim; the final cache contains 500 unique provider executions.

| Model | Binary: errors detected / correct answers rejected / invalid | Probability: errors detected / correct answers rejected / invalid |
|---|---:|---:|
| Gemini 3.7 Flash | 3 / 3 / 0 | 4 / 2 / 0 |
| Gemini 3.5 Flash-Lite | 1 / 7 / 0 | 1 / 7 / 1 |
| Mistral Small 3.1 | 5 / 29 / 0 | 4 / 18 / 0 |
| Mistral Medium 3 | 4 / 23 / 0 | 5 / 20 / 0 |
| Claude Haiku 4.5 | 7 / 19 / 3 | 4 / 9 / 6 |

Probability rejection in this diagnostic means score < 0.5, not a policy
decision. Invalid signals are listed separately and do not count as successful
error detection. The common complete cohort for all ten arms has 43 questions,
including seven errors; excluded failures here occurred on correct candidates.
Out-of-fold forecast comparisons therefore describe a selected subset and must
be read alongside the failure counts. No model or format has been selected as
the winner. Review the 1024-token truncation failures before collecting the
remaining calibration questions; any generation-setting revision is a separate
condition, not a silent repair of the failed outputs.

Private detailed results are in
`results/gpqa_signal_study_20261005_v2/{signal_report,calibration_comparison,audit}.json`.

### Separate 4096-token condition

`configs/gpqa_signal_study_4096.json` keeps the same five models, prompts, seed,
and frozen calibration candidates, and raises each model's `max_tokens` from
1024 to 4096. These are new verifier requests; failures in the previous condition
are not replaced. Provider-internal reasoning budgets still differ. The local
phase-one allowance remaining after the first screen is $19.4559448.

```bash
.venv/bin/python -m vgx.gpqa.signal_study prepare \
  --config configs/gpqa_signal_study_4096.json \
  --output results/gpqa_signal_study_4096_20261005
.venv/bin/python -m vgx.gpqa.signal_collect \
  --output results/gpqa_signal_study_4096_20261005 --limit 10 --allow-api
# After checking the ten-item format gate, continue the same study:
.venv/bin/python -m vgx.gpqa.signal_collect \
  --output results/gpqa_signal_study_4096_20261005 --limit 247 --allow-api
.venv/bin/python -m vgx.gpqa.signal_study report \
  --output results/gpqa_signal_study_4096_20261005
.venv/bin/python -m vgx.gpqa.signal_analysis \
  --output results/gpqa_signal_study_4096_20261005
MPLCONFIGDIR=/tmp/vgx-matplotlib .venv/bin/python -m vgx.gpqa.signal_plots \
  --root results/gpqa_signal_study_4096_20261005 \
  --output reports/gpqa_signal_study_4096_20261005
```

The bounded collector paces each model and stops submission on transport or
accounting failure; already-running requests can finish. It keeps a durable
progress snapshot and records its own implementation hash. This parallelism is
only for independent screening requests and does not alter the live sequential
policy. Resuming the same condition reuses exact requests, including empty or
invalid responses, without regeneration. The ten-item gate permits at most 2%
invalid signals per model (with 20 replies this requires all replies valid).

Analysis additionally reports high-confidence tails at 0.95 and 0.99, five
prespecified fold seeds, and paired resampling of fixed out-of-fold predictions.
Those resampling intervals are descriptive: they omit fitting and selection
uncertainty and do not account for overlapping training folds. No seed is chosen
for better results, no winner is automatically selected, and no live prior or
stopping table is replaced by this analysis.

Binary and probability formats are also compared directly within each model.
Conditional likelihood uncertainty is reported using 200 resamples within the
correct/incorrect calibration classes, keeping class counts, binning and Laplace
smoothing fixed. A supplementary full-cohort joint forecast uses training-fold
medians and missingness indicators, to expose sensitivity to dropping malformed
signals; it does not change live abstention on missing responses. Aggregate
plots show class-conditional score distributions, including invalid replies,
and the five-fold-seed range of incremental forecast performance.

### Full calibration screen: 4096-token condition (2026-10-05)

Completed all **247 frozen calibration candidates × five models × two formats**:
2,470 unique provider executions, 2,466 parseable signals. Candidates contain
211 correct and 36 incorrect answers. The four failures were two Flash-Lite
probability replies, one Flash probability reply, and one Mistral Medium binary
reply; all were retained. All 494 Claude replies parsed. No generator or
evaluation requests were made. Resource and quota projects both matched
`llm-applications-490420` for every request.

Provider-token-based estimated cost was **$2.6260761**, or **$3.1701313** including
the original 1024-token screen. These are not confirmed billing amounts. The
focused caching, recovery, accounting, sequential execution and analysis suite
passed **95 tests**.

The primary comparison uses 243 candidates with all ten signals available
(208 correct, 35 incorrect). Three-fold out-of-fold Brier score is lower when
forecasts are better; calibrated generator alone scores **0.117794**. Each cell
below fits a separate regularized joint forecast using generator confidence and
one verifier signal, with fitting restricted to training folds. These forecasts
are diagnostics, not replacement live observation models.

| Verifier | Binary signal: Brier | Probability signal: Brier |
|---|---:|---:|
| Gemini 3.7 Flash | 0.123773 | **0.107698** |
| Gemini 3.5 Flash-Lite | 0.117459 | 0.122166 |
| Mistral Small 3.1 | 0.118417 | 0.118294 |
| Mistral Medium 3 | 0.117737 | 0.117925 |
| Claude Haiku 4.5 | 0.119226 | 0.118411 |

Flash probability is the most promising signal in this exploratory screen.
Its incremental Brier score remains negative across five prespecified fold
seeds (-0.010635 to -0.008089), and in the all-candidate missingness diagnostic
(-0.008461). The binned Bayesian comparator also improves (-0.008324).
Nevertheless, its paired resampling interval against calibrated generator alone
includes zero; neither the seed range nor this selected comparison establishes
general superiority. Other arms provide little consistent incremental benefit.

Two issues must be resolved before freezing the probability model and value
tables:

- On all 247 candidates, raw generator confidence >=0.95 includes 190 answers
  with 19 errors (90% accuracy); confidence >=0.99 includes 65 answers with
  three errors. Calibration improves overall Brier from 0.125491 to 0.115871,
  but leaves only two out-of-fold predictions >=0.99. Their observed success
  does not establish 99% reliability.
- Conditional on correct candidates, Flash probability and generator confidence
  have Pearson correlation approximately 0.676 in the common cohort. A pooled
  likelihood update assuming independence given correctness can double-count
  evidence. A conditional observation model, with uncertainty appropriate to
  only 36 incorrect candidates, needs assessment before policy deployment.

The five candidates span three model families. Model aliases and provider-returned
labels are recorded, but not all labels identify immutable weights. No live
prior, penalty, verifier order or stopping table was changed. The findings do
not justify forcing three verifiers into the policy.

Detailed local artifacts:
`results/gpqa_signal_study_4096_20261005/{signal_report,calibration_comparison,audit}.json`.
Aggregate figures (no question text) are in
`reports/gpqa_signal_study_4096_20261005/`.

### Qwen and Llama generator comparison

`configs/gpqa_generator_comparison_v2.json` defines a separate experiment on
the original 50-question, subject-stratified calibration pilot selected before
any generator responses. Historical Gemini candidates remain frozen. The new
generators are Vertex Qwen3-235B-A22B-Instruct-2507 (`us-south1`) and
Llama-3.3-70B-Instruct (`us-central1`), with the same generator prompt and option
order, temperature 0 and max_tokens 4096. Gemini's earlier 1024-token/low-reasoning
setting is retained and disclosed as a model-plus-settings difference.

Gemini 3.7 Flash and Mistral Medium 3 supply probability verifier signals using
the existing 4096-token protocol. Identical candidate answers produce identical
verifier requests and reuse one cached execution across generator arms; these
are not independent repeated verifier samples. Collection does not read answer
keys or call evaluation items. Analysis reads calibration labels only and refits
calibration/likelihoods separately for each generator in training folds.

```bash
.venv/bin/python -m vgx.gpqa.generator_study prepare \
  --config configs/gpqa_generator_comparison_v2.json \
  --output results/gpqa_generator_comparison_20261005_v2
.venv/bin/python -m vgx.gpqa.generator_study generators \
  --output results/gpqa_generator_comparison_20261005_v2 --allow-api
.venv/bin/python -m vgx.gpqa.generator_study verifiers \
  --output results/gpqa_generator_comparison_20261005_v2 --allow-api
.venv/bin/python -m vgx.gpqa.generator_study report \
  --output results/gpqa_generator_comparison_20261005_v2
```

The initial preflight returned one successful Qwen response with implicit cached
tokens and a Llama HTTP 429 rejection. Missing cache pricing stopped collection.
The original cache and plan remain in `results/gpqa_generator_comparison_20261005`.
Its `pricing_reconciliation.json` values the Qwen execution at $0.00006204, using
the published ordinary input rate for implicit cached tokens as a conservative
estimate; no unverified cache discount is claimed. Version 2 imports that exact
response without regeneration and reserves the remaining $9.99993796 of this
stage's $10 cap. Imported historical costs are separate from new ledger spend.
All requests use `llm-applications-490420` within the existing $200 authorization.
Provider-token estimates are not confirmed billing.

Google lists both generator MaaS endpoints for retirement on 2026-10-21.
Provider labels and raw responses are retained; this does not guarantee immutable
provider weights or future endpoint availability. The pilot does not replace
the original probability model, penalty, or sequential stopping policy.

#### Completed 50-question generator comparison (2026-10-05)

| Generator | Correct / all 50 | Mean reported confidence | Mean confidence on wrong answers | Valid confidence |
|---|---:|---:|---:|---:|
| Frozen Gemini 3.8 Flash | 39 / 50 | 0.9358 | 0.9291 | 50 |
| Qwen3-235B-A22B-Instruct-2507 | 19 / 50 | 0.9235 | 0.9103 | 49 |
| Llama-3.3-70B-Instruct | 25 / 50 | 0.8460 | 0.8200 | 50 |

Qwen's one malformed confidence key is retained. Its valid answer still counts
toward accuracy and receives verification. On the common 49 confidence-valid
questions, raw Brier is 0.20286 (Gemini), 0.52743 (Qwen), and 0.35245 (Llama).
Lower confidence from Llama did not establish better calibration. This is the
original pre-answer pilot; it differs from the later signal-study first-50
ordering, so its Gemini baseline is 39/50 rather than that screen's 43/50.

Three-fold OOF joint forecasts using generator confidence plus one verifier:

| Generator (confidence n) | Calibrated generator only | + Flash probability | + Mistral Medium probability |
|---|---:|---:|---:|
| Gemini (50) | 0.18280 | 0.16935 | 0.17793 |
| Qwen (49) | 0.20107 | 0.11572 | 0.20348 |
| Llama (50) | 0.23931 | 0.15746 | 0.24778 |

These within-generator Brier comparisons are exploratory, with small training
folds. They are not a ranking of final system utility or certified release risk.
The training-fold base-rate comparator scores 0.17220 for Gemini, better than
generator-only calibration in this small pilot, illustrating fitting variance.
At the descriptive verifier threshold 0.5, Flash rejects 26/31 Qwen errors and
18/25 Llama errors, while rejecting 3/19 and 4/25 correct answers respectively.
This threshold is not the sequential policy's release threshold.

All 300 logical verification results are available and parseable. There were
100 new generator executions and 80 new verifier executions, with 220 verifier
cache hits across historical reuse and equal answers. The revised ledger records
179 new executions; adding the recovered preflight Qwen execution gives 180.
Incremental estimated cost is **$0.11416017**, including that preflight, billed
via resource/quota project `llm-applications-490420`; confirmed billing remains
unavailable. The cache's $0.19941062 usage valuation includes historical imported
verifier executions and must not be added to old experiment costs. There are
no unresolved attempts in the completed cache. **100 focused tests passed.**

These results do not show that switching generator families solves overconfidence.
The larger Flash benefit on Qwen/Llama may reflect stronger verification of
weaker generated answers; it does not isolate family dependence or demonstrate
weak-verifier/strong-generator performance. The JSON-only generator prompt,
non-thinking endpoints, historical Gemini reasoning setting, and small sample
limit comparisons with published model benchmarks. Reasoning-enabled protocols
would require a separate, prospectively specified condition.

Offline matched-cohort analysis and plotting:

```bash
.venv/bin/python -m vgx.gpqa.generator_analysis \
  --root results/gpqa_generator_comparison_20261005_v2
MPLCONFIGDIR=/tmp/vgx-matplotlib .venv/bin/python -m vgx.gpqa.generator_plots \
  --root results/gpqa_generator_comparison_20261005_v2 \
  --output reports/gpqa_generator_comparison_20261005
```

Detailed reports are `comparison.json`, `matched_comparison.json`, `audit.json`,
and `cost_reconciliation.json` in the new results directory. Aggregate figures
are in `reports/gpqa_generator_comparison_20261005/`.

### Rationale-before-answer prompt comparison

`configs/gpqa_reasoning_comparison.json` defines a prospective follow-up on the
same original 50 calibration questions for Qwen and Llama. It keeps each model,
endpoint, temperature, top_p and 4096-token output cap identical to its direct
condition. The new prompt asks for a worked explanation followed by exactly one
terminal `<final>{"answer": ..., "p_correct": ...}</final>` block. The parser
ignores JSON objects inside the explanation, requires a unique complete final
block, and retains malformed/truncated outputs without regeneration.

This is explicit rationale prompting on instruction models, **not native
thinking mode**. Qwen's Instruct-2507 model card states it is non-thinking-only.
The original direct generator prompts, candidates and reports are preserved.
Verifier prompts still receive only question/options/final answer: no rationale,
generator identity or prior confidence. Exact verifier requests reuse cached
executions across prompt conditions whenever the final answer is unchanged.

The six-response format preflight (three items per model) passed without reading
answer keys. All six had a nonempty explanation and valid final block, with no
length termination. The resumed paid run is under
`results/gpqa_reasoning_comparison_20261005_v3`; the earlier unsuffixed directory
contains only a superseded preparation plan, with no API collection. Version 2
retains the first 53 received responses. One of them was an HTTP 200 body
reporting concurrency throttling, without answer or usage. Its $0.00408694
reservation remains an unknown-cost allowance, not a zero-cost claim. Version 3
imports all 53 responses without retries, reduces collection workers from four
to two, and reserves $9.90570802 after deducting the earlier $0.09020504 known
usage estimate and that unknown-cost allowance. Model settings and prompts are
unchanged. The original `recovery.json` records this accounting. The new
stage has a $10 local cap within the existing $200 authorization, with all calls
bound to `llm-applications-490420`. Pricing is based on provider token usage,
including conservative ordinary-input pricing for Qwen/Llama implicit cache hits;
these estimates are not confirmed billing.

```bash
.venv/bin/python -m vgx.gpqa.reasoning_study prepare \
  --config configs/gpqa_reasoning_comparison_resume.json \
  --output results/gpqa_reasoning_comparison_20261005_v3
.venv/bin/python -m vgx.gpqa.reasoning_study generators \
  --output results/gpqa_reasoning_comparison_20261005_v3 --allow-api
.venv/bin/python -m vgx.gpqa.reasoning_study verifiers \
  --output results/gpqa_reasoning_comparison_20261005_v3 --allow-api
.venv/bin/python -m vgx.gpqa.reasoning_analysis \
  --root results/gpqa_reasoning_comparison_20261005_v3
MPLCONFIGDIR=/tmp/vgx-matplotlib .venv/bin/python -m vgx.gpqa.reasoning_plots \
  --root results/gpqa_reasoning_comparison_20261005_v3 \
  --output reports/gpqa_reasoning_comparison_20261005
```

Analysis compares accuracy changes on all paired questions, including parse
failures, and raw Brier changes on paired valid-confidence answers. Each prompt
condition uses its own answer correctness, rather than inheriting the direct
condition's labels. Calibration and verifier likelihood fitting stay inside
training folds; five prespecified fold seeds assess sensitivity. This follow-up
was motivated by inspecting the earlier pilot and is exploratory, not independent
held-out confirmation. No final probability model or stopping policy is selected.

#### Completed rationale-first comparison (2026-10-05)

| Model / prompt | Correct / requested | Completed answers | Mean confidence | Mean confidence on wrong completed answers |
|---|---:|---:|---:|---:|
| Qwen direct | 19 / 50 | 50 | 0.9235 (49 valid scores) | 0.9103 |
| Qwen rationale-first | 23 / 50 | 29 | 0.9607 (29 scores) | 0.9500 |
| Llama direct | 25 / 50 | 50 | 0.8460 | 0.8200 |
| Llama rationale-first | 29 / 50 | 50 | 0.7814 | 0.7424 |

Frozen Gemini remains 39/50 in this same pilot. Qwen's 21 incomplete responses
comprise 20 length terminations at 4096 tokens and the one preserved HTTP 200
throttle envelope. Its six wrong completed answers are distinct from those 21
incomplete responses. Reporting 23/29 as an overall success rate would select
only completed cases; all-question correct delivery is 23/50.

Qwen corrected ten previous wrong answers, changed none of the previously
correct answers to a wrong completed answer, but failed to finish six previously
correct questions. Llama corrected eight previous errors and introduced four.
Both therefore gained four correct deliveries (+8 percentage points). Paired
question bootstrap 95% intervals include zero: Qwen [-8, +24] points and Llama
[-6, +22] points. These exploratory intervals do not include model-selection
or repeated-inference uncertainty.

On questions with valid confidence in both conditions, raw Brier changed from
0.48457 to 0.19684 for Qwen (n=28) and from 0.36160 to 0.29711 for Llama (n=50).
The Qwen subset excludes incomplete responses and the direct condition's one
invalid confidence. Improvement in Brier can also reflect changed answer
correctness; it does not establish faithful confidence or calibration at 99%.

For rationale-first answers, three-fold OOF Brier scores are:

| Generator / usable n | Training-fold base rate | Calibrated generator | + Flash probability | + Mistral Medium probability |
|---|---:|---:|---:|---:|
| Qwen / 29 | 0.16421 | 0.17935 | 0.09036 | 0.18058 |
| Llama / 50 | 0.24377 | 0.26397 | 0.15092 | 0.27881 |

Flash's joint forecast improves on generator-only calibration across all five
prespecified fold seeds for both models. Mistral's joint forecast worsens in all
five. However, generator-only calibration itself loses to the base-rate forecast
in this small pilot; the probability model is not ready to freeze. Qwen has only
six completed wrong answers for likelihood estimation. Verifier utility under
a fixed sequential policy has not been established by these forecasting metrics.

The rationale prompt increased mean billed output from approximately 17 to 2877
tokens for Qwen (49 replies with usage) and from 16 to 851 for Llama. Known
generator usage estimates were $0.12758856 and $0.043038 respectively. There
were 100 new generator requests and 22 new verifier requests in this condition;
458 logical verification results were available, with 436 cache hits and no
verifier parse failures. All earlier replies and candidate answers remain intact.

Combined known incremental usage estimate is **$0.20003921**, plus **$0.00408694**
reserved for the single missing-usage response (total including reservation:
**$0.20412615**). This is not a confirmed invoice. All resource/quota projects
match `llm-applications-490420`; no evaluation requests were made. Recovery
checks confirm all 53 imported records match their source records exactly.
**113 focused tests passed.**

The current rationale-first Qwen configuration needs a completion-rate remedy
before expansion: any higher output cap or shorter-rationale prompt should be
a separate condition, preserving this run's failures. Llama provides a usable
cross-family baseline, but these results do not justify replacing the frozen
Gemini main experiment or directly trusting any model's reported confidence.

### Qwen completion controls and full generator expansion (2026-10-05)

`completion_study.py` adds two independent conditions on the original 50
calibration questions. Old collectors and their implementation hashes remain
unchanged:

- `short_4096`: the same Qwen endpoint/settings, with a concise explanation
  (requested maximum 200 words) followed by the unique final answer/confidence
  block. The output cap stays 4096.
- `original_8192`: exactly the previous rationale prompt and parser, with only
  the output cap raised to 8192.

The first three questions per condition passed the output-format check. The
pilot has a separate USD 2 usage-estimate allowance inside the existing USD 200
Vertex authorization. Resource and quota project remain
`llm-applications-490420`. A requested word limit is a prompt instruction, not
an enforced token reservation. No malformed, truncated, or received error
response is silently repaired or regenerated.

Before examining correctness, the completion gate requires every pilot item to
have a valid final answer and confidence, with reported output usage. Among
qualifying conditions it selects the lowest mean output-token count, breaking
ties by condition ID. If neither condition qualifies, expansion is blocked.
This is an operational pilot gate, not a statistical guarantee of zero failures
on unseen questions.

```sh
.venv/bin/python -m vgx.gpqa.completion_study prepare \
  --config configs/gpqa_qwen_completion.json \
  --root results/gpqa_qwen_completion_20261005
.venv/bin/python -m vgx.gpqa.completion_study collect \
  --root results/gpqa_qwen_completion_20261005 --allow-api
.venv/bin/python -m vgx.gpqa.completion_study gate \
  --root results/gpqa_qwen_completion_20261005
```

The user clarified that expansion means **both Qwen and Llama answer every
question previously answered by Gemini**, not merely the calibration subset.
The pinned dataset contains 445 valid questions: 249 Main-minus-Diamond
calibration items and 196 Diamond evaluation items. `generator_expansion.py`
checks exact membership, question text, choice order, source hashes, and the
completion-selected Qwen condition before preparing requests. Llama retains
its prior rationale-first prompt, endpoint and 4096-token settings. Identical
pilot requests are imported without another API execution. Each model has at
most one outstanding request during expansion.

Evaluation answers may be collected and frozen now; evaluation labels stay
closed during collection and calibration-only reporting. `expansion_analysis.py`
reports the 50-item pilot and remaining 199 calibration questions separately.
This stage does not collect new verifier signals, fit likelihood tables, or
change the sequential stopping policy. Prompt differences from the historical
Gemini condition are retained and must be stated in model comparisons.

#### Two-stage fallback and automatic continuation

During collection, both single-call controls produced length-truncated replies
(the concise arm at 4096 and the unchanged-prompt arm at 8192). Raising the cap
or requesting a word limit therefore does not ensure final-answer delivery.
`staged_generator.py` defines a separate condition: reuse the concise draft,
then give the same Qwen endpoint a dedicated 512-token request that must output
only the final answer/confidence JSON. Every question gets this second request,
including those whose drafts already contain valid answers. Drafts are quoted
as data; correctness labels and verifier responses are never included. The
finalizer may revise the draft answer. This is part of candidate generation,
not the paper's verifier feedback or stopping algorithm.

The pilot imports all existing draft responses without regeneration. Its USD 1
local allowance covers finalization calls only. A strict JSON parser and a
50/50 completion gate precede the full study, whose separate local allowance is
USD 5. Both draft and finalization usage are charged to the new generator
protocol; imported draft usage is historical rather than new spend. No claim
of confirmed provider billing is made. If the gate or accounting fails, the
pipeline stops instead of rewriting candidates or loosening the gate.

The authorized continuation is running via:

```sh
.venv/bin/python -m vgx.gpqa.completion_pipeline \
  --root results/gpqa_generator_expansion_pipeline_20261005 --allow-api
```

It waits for the original controls to finish, records their completion gate,
then selects the appropriate frozen protocol. If necessary it runs the staged
50-item pilot before expanding Qwen and Llama to all 445 questions. Status is
stored in `results/gpqa_generator_expansion_pipeline_20261005/status.json`.
The full staged run, if its pilot passes, is stored under
`results/gpqa_staged_generator_expansion_20261005`. Evaluation responses are
collected but not scored. Any blocked status requires review of the recorded
error and budget ledger; do not blindly restart or replace received failures.

**Local validation: 125 focused tests passed**, covering strict final parsing,
unchanged model settings, exact cache reuse, source/membership checks, absence
of label access during collection, separate finalization for every question,
and completion-gated expansion. This validates software behavior, not model
calibration or a zero-failure guarantee.

### Llama-only recovery with lower request rate (2026-10-06)

The full staged run retained all 445 Qwen final answers and 183 Llama answers
before repeated HTTP 429 responses stopped Llama. The user authorized resuming
only missing Llama requests. `llama_resume.py` keeps the original study, prompts,
model parameters, request identities, and USD 5 budget ledger unchanged. It
requires every Qwen draft/final to be available offline, fingerprints all
existing provider replies, and checks those fingerprints again after collection.
Existing invalid HTTP-200 replies are preserved; they are not retried.

Llama requests are serial and spaced at least 10 seconds apart. Only explicit
HTTP 429 rejections receive up to two retries, with 60- and 120-second waits.
Historical pilot requests are imported when identical. The prior interrupted
collection snapshot is preserved under the recovery output directory. After
collection, the original frozen bundle is assembled offline and all three
generators are summarized on calibration labels only. No Qwen API fallback,
new verifier requests, or evaluation scoring is permitted in this recovery.

```sh
.venv/bin/python -m vgx.gpqa.llama_resume \
  --root results/gpqa_staged_generator_expansion_20261005 \
  --output results/gpqa_staged_generator_expansion_20261005/resume_20261006 \
  --interval 10 --allow-api
```

Read `resume_20261006/status.json` for the recovery status; the earlier pipeline's
blocked status remains as history. Its manifest records the original response
hashes, pacing/retry rules, source hash, and budget before recovery. The final
`calibration_summary.json` reports accuracy, valid-confidence counts, mean
confidence, confidence on wrong answers, raw Brier, and paired differences.
The pilot and remaining calibration questions are separated. These are
comparisons of different generator protocols, not isolated model-weight effects.

### Gemini/Qwen initial-belief calibration (2026-10-06, offline; deferred)

**Superseded for the active experiment by the user decision above.** Preserve
this analysis for possible future follow-up; do not use its fitted mappings as
the main algorithm's initial belief.

The next algorithm experiment uses frozen Gemini and two-stage Qwen candidates;
Llama results remain a baseline. `generator_calibration.py` fits each generator
separately with the existing fixed specification: standardized logit of reported
confidence, logistic regression C=1, probability clipping 1e-4, three stratified
folds, and five prespecified seeds (20261005–20261009). All transformations are
trained inside the training fold. The 50-item pilot is also excluded in a
separate remaining-calibration sensitivity analysis. No evaluation labels,
provider calls, active prior changes, or stopping-table changes occur.

```sh
.venv/bin/python -m vgx.gpqa.generator_calibration \
  --root results/gpqa_staged_generator_expansion_20261005 \
  --output results/gpqa_generator_calibration_20261006 \
  --screen-config configs/gpqa_signal_study_4096.json \
  --cache results/gpqa_signal_study_4096_20261005/cache \
  --cache results/gpqa_generator_comparison_20261005_v2/cache \
  --cache results/gpqa_reasoning_comparison_20261005_v3/cache
```

Primary-fold OOF Brier (lower is better):

| Generator / valid n | Raw confidence | Training-fold base rate | Calibrated confidence |
|---|---:|---:|---:|
| Gemini / 247 | 0.12549 | 0.12451 | 0.11624 |
| Qwen / 249 | 0.30145 | 0.23300 | 0.20449 |

Across all five seeds, calibrated Brier is 0.11624–0.11981 for Gemini and
0.20427–0.20550 for Qwen. Excluding the original pilot, primary-fold Brier changes
from 0.10688 to 0.10326 (Gemini, n=197) and from 0.28643 to 0.19992 (Qwen, n=199).
Gemini's Brier advantage over the base-rate comparator has a fixed-OOF bootstrap
interval including zero; Qwen's does not. These exploratory intervals do not
include model fitting or selection uncertainty. Coarse-bin ECE alone is not a
suitable selection criterion: a constant base-rate predictor can obtain zero
ECE while failing to distinguish correct from wrong answers.

Full-calibration coefficients and example mappings are saved **for review only**,
separately from OOF performance. In that full-data fit, raw confidence 0.95 maps
to approximately 0.851 for Gemini and 0.696 for Qwen. These are fitted estimates,
not ground truth or a release guarantee. In the primary OOF run, calibrated
confidence >=0.95 selects 25 Gemini answers (23 correct) and zero Qwen answers.
At >=0.99 Gemini selects only two answers. High-confidence tail evidence is
therefore insufficient to certify a 95%/99% release policy.

The exact-cache audit covers the existing five verifiers and both binary and
probability formats; it does not choose a final pool/order. Gemini's eligible
247 candidates have all requested responses cached (four invalid signals remain
preserved). Qwen lacks 758 distinct verifier requests across those ten arms.
Flash and Mistral Medium probability each lack 63; each other arm lacks 79.
Existing cached Qwen signals disproportionately cover correct/shared answers
(e.g. binary arms cover 152 correct and 18 wrong candidates out of Qwen's total
157 correct and 92 wrong). Fitting likelihoods only on that cached subset would
introduce selection bias. No final likelihood table is fitted until this
coverage issue is addressed; cached malformed replies are not regenerated.

Results, fold memberships, predictions, source hashes, review coefficients, and
cache coverage are under `results/gpqa_generator_calibration_20261006`.
**Validation: 10 focused tests passed**, including fold disjointness, refusal to
fit on evaluation rows, and separate handling of missing, invalid, and valid
zero-valued verifier signals. No paid API requests were made in this step.

### Raw-confidence verifier experiment (2026-10-06, active offline analysis)

The active comparison preserves generator `p_correct` exactly, including 0/1
endpoints. It compares raw confidence with a Bayes update using verifier
likelihoods fitted only on training folds. It does not fit or apply generator
confidence calibration. Each of five verifiers has separate binary and
probability arms; this experiment does not select a three-verifier sequence.

```sh
.venv/bin/python -m vgx.gpqa.raw_prior_analysis \
  --config configs/gpqa_raw_prior_analysis.json \
  --output results/gpqa_raw_prior_analysis_20261006
```

The protocol freezes three-fold analysis, five split seeds, signal bins,
Laplace smoothing, and training-fold expected costs. It evaluates both all 249
development questions and the 199 questions outside the original pilot.
Release/abstain utility uses reward 1, error penalty 19 (99 as a sensitivity
condition), and 10 utility units per estimated USD. Historical token-based
costs are estimates, not confirmed billing; this offline run spends no API
budget and reads no Diamond answer keys.

Gemini analysis is complete. On the common 243 valid signal records, Flash
probability updates reduce Brier from 0.124579 to 0.117108, but the paired
descriptive interval includes zero. Every single-verifier sequential policy
has negative mean utility on all 249 questions under the primary setting,
despite several improving on the raw-confidence baseline. Qwen is blocked by
758 missing verifier requests; its selectively cached subset is not used to
fit or rank complete policies. Existing malformed responses are preserved.

The run checked fold separation, unchanged source/cache hashes, and 44,400
lazy sequential/replay decision agreements. These are offline consistency
checks, not validation of real API execution or a final held-out evaluation.
See the [Chinese experiment report](reports/gpqa_raw_prior_20261006.md) for
results, uncertainty, costs, and limitations. Machine-readable fits and
decisions are under `results/gpqa_raw_prior_analysis_20261006`.

### Matched coverage and Qwen signal completion (2026-10-06)

The additional confidence-only comparator matches each verifier policy's number
of releases within each validation fold, ranking unchanged raw confidence.
Boundary ties use uniform-selection expectations; labels never break ties.
This is a retrospective risk–coverage comparison, not an operational threshold
fitted on training data. See the [matched-coverage report](reports/gpqa_matched_coverage_20261006.md).

```sh
.venv/bin/python -m vgx.gpqa.matched_coverage \
  --root results/gpqa_raw_prior_analysis_20261006 \
  --output results/gpqa_matched_coverage_20261006
```

`qwen_signal_completion.py` fills only the 758 requests absent from the frozen
Qwen development cohort's verifier cache. It preserves prompts, models,
candidates, successful replies (including invalid-format replies), and all
historical artifacts. A nonblocking collection lock prevents a duplicate
collector; uncertain provider attempts block automatic resubmission. Five
serial model queues each have a minimum three-second request interval. A local
USD5 usage-estimate sublimit stays within the existing USD20 signal-study and
USD200 experiment allowances. Historical unpriced usage retains reservations;
neither reservations nor token estimates are confirmed billing.

```sh
# Prepare offline first; add --allow-api only for authorized collection.
.venv/bin/python -m vgx.gpqa.qwen_signal_completion \
  --output results/gpqa_qwen_verifier_completion_20261006
```

Progress and accounting are in that directory's `status.json`, `plan.json`,
and `cache/budget.json`. After completion, the unchanged raw-prior analysis can
use `configs/gpqa_raw_prior_completed_signals.json` with a new output directory.
No generator requests or Diamond evaluation requests are part of this phase.

The [dependence diagnostic](reports/gpqa_signal_dependence_20261006.md) checks
raw-confidence/signal and between-verifier associations separately within
correct and incorrect development candidates. It does not change the prior,
likelihood fit, or policy. Zero correlation does not establish independence.

**Completion:** all 758 missing Qwen verifier requests returned, with one new
invalid-format response retained. New provider-usage cost estimate: USD
1.0451689 on `llm-applications-490420`; no outstanding new reservations or
confirmed billing. The [combined report](reports/gpqa_completed_verifier_comparison_20261006.md)
contains the completed Gemini/Qwen raw-prior, matched-coverage, and dependence
analyses. Qwen Flash probability updates improve common-sample Brier from
0.3029 to 0.1400, and reduce errors relative to matched-confidence coverage;
primary single-verifier policy utilities remain negative. This is development
evidence, not a final held-out result or selection of three verifiers.

After the completion status is `complete`, reproduce the combined analyses:

```sh
.venv/bin/python -m vgx.gpqa.raw_prior_analysis \
  --config configs/gpqa_raw_prior_completed_signals.json \
  --output results/gpqa_raw_prior_completed_signals_20261006
.venv/bin/python -m vgx.gpqa.matched_coverage \
  --root results/gpqa_raw_prior_completed_signals_20261006 \
  --output results/gpqa_matched_coverage_completed_signals_20261006
.venv/bin/python -m vgx.gpqa.signal_dependence \
  --config configs/gpqa_raw_prior_completed_signals.json \
  --output results/gpqa_signal_dependence_completed_signals_20261006
```
