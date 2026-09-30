# Verification-generation experiment

This repository explores confidence and verifier signals for selective scientific
QA. The GPQA pilot is motivated by Green–Laffont sequential verification; frozen
model forecasts do not test strategic truthfulness or incentive compatibility.

## GPQA pilot

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
next-token probabilities. Both verifier signals are collected for every valid
candidate, including candidates whose generator confidence failed parsing.
Adaptive query savings are counterfactual; collection does not save those calls.

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
  (currently 200/120/80). All/calibration
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
- Query counts and normalized verification costs are counterfactual policy
  costs. Actual collection counts are separately reported. The configured
  costs are sensitivity assumptions, not measured money or per-item latency.
- Bayesian updates use raw generator confidence and assume conditional
  independence, including from generator confidence given correctness. They
  are **not certified calibrated**. Exact 0/1 priors remain fixed. This patch
  does not silently clip or recalibrate them or tune thresholds on evaluation.
  Routing thresholds remain the preregistered `L / (R + L)` utility thresholds.

## Run identity and remaining inputs

Collection uses Vertex AI's OpenAI-compatible Chat Completions endpoint with
the user-managed Application Default Credentials file; no Hugging Face token or local
model weights are needed. The configured model IDs and locations are pinned in
`configs/gpqa_experiment.json`. Gemini 3 models use low reasoning effort and
omit temperature/top-p because those parameters are unsupported for that API;
Llama uses temperature 0 and top-p 1.

Each run writes under `results/gpqa/<run_id>/`. Its manifest fingerprints the
config, dataset content, pinned split, code, relevant package versions, and a
hash of the active Google Cloud project identifier.
Each response cache key additionally hashes the rendered prompt, system prompt,
metadata (including candidate), cloud project hash, model, region, and inference
settings. Legacy role/item-only keys cannot be reused. Prompt, source, settings,
or model changes isolate runs; identical requests in an unchanged run resume.
Operational `--limit` collections have separate identities and are explicitly
marked as smoke-only, not full evaluation.

Before collection, the VM needs ADC credentials, a configured project, and
Vertex AI API/model access. The collector obtains short-lived ADC access tokens
in memory and never writes credentials to the run files. It deliberately uses
ADC rather than the active gcloud account, which can be a differently scoped VM
service account.

1. An authorized local GPQA Main CSV at `data/gpqa/gpqa_main.csv` (or configure
   a local CSV/Parquet path). The loader never downloads GPQA.
2. Dataset provenance: source revision, access date, and access-terms acceptance,
   recorded privately alongside the local source. The code records its SHA-256.
3. A frozen review of prompts, verifier order, cost/loss scenarios, and handling
   of endpoint confidences before inspecting evaluation outcomes.

Question text, shuffled answer keys, manifests, prompts, call logs and records
remain under gitignored `data/` and `results/`.
Scale only after real held-out diagnostics show useful additional verifier
signal, tolerable uncertainty/dependence, and a promising utility/coverage tradeoff.
