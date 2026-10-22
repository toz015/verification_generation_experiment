# Fresh GPQA experiment on Vertex

The new experiment uses the official Hugging Face data and newly generated
candidates. Historical VM outputs are not inputs or evidence. Jev is deferred.

## Frozen protocol

- Dataset: `Idavidrein/gpqa`, revision `83022cefff930aea54f654c0b282e74b9eeda5c6`.
- Calibration: Main excluding every Diamond question. Evaluation: Diamond.
- Three Main questions have duplicate answer choices, including two Diamond
  questions. The existing exclusion rule leaves **249 calibration + 196
  evaluation** questions. This is a filtered Diamond evaluation, not the full
  official 198-question score. Exclusions, normalized IDs, source checksums,
  subjects, split and choice permutations are recorded locally.
- Seed: `20261004`. A subject-stratified 50-question calibration batch checks
  software and costs before expansion. Its responses are reused.
- Generator: `google/gemini-3.8-flash`, global endpoint.
- Verifier 1: `meta/llama-3.3-70b-instruct-maas`, `us-central1`.
- Verifier 2: `google/gemini-3.7-flash`, global endpoint.
- Generator/verifier prompts and generation settings remain those in the prior
  configuration. New candidates are frozen once; no replacement sampling.
- Model API names are pinned, but managed aliases are not immutable weight
  revisions. Raw provider responses, returned model names/versions, timestamps,
  actual request settings and token usage are retained where supplied.

The primary policy uses reward 1, incorrect-release loss 19, and abstention 0.
Its nominal terminal assertion threshold is 0.95. Secondary settings use losses
4 and 1 (thresholds 0.8 and 0.5). These are predeclared sensitivity settings,
not thresholds optimized using evaluation answers. A nominal threshold is not
an empirical correctness guarantee.

The monetary conversion is **10 utility units per USD**, corresponding to a
correct release being worth USD 0.10 in this illustrative decision problem.
Each expected verifier cost is its calibration mean token-based USD estimate
times 10. This normative conversion must be reconsidered for a real application.
Generator cost is sunk at verification time but included in experiment spending.

Three-bin Laplace-smoothed likelihoods use calibration outcomes only. The raw
generator confidence remains the prior. Conditional independence and transfer
from non-Diamond to Diamond are assumptions; the report diagnoses rather than
certifies them. Endpoint priors of exactly 0 or 1 cannot be changed by Bayes.

## Collection order and accounting

1. Verify authorized Hugging Face access and Google ADC; pin/download data.
2. Validate source membership, separate public inputs from answer keys.
3. Collect the 50 calibration candidates and both verifier responses.
4. Collect remaining generator responses once; freeze the candidate bundle.
5. Finish calibration verifier collection; fit and seal all policies.
6. Run the primary sequential policy on evaluation candidates. Only selected
   verifier responses are requested or read; evaluation labels remain separate.
7. Collect additional evaluation verifier responses for paired baselines.
8. Open evaluation labels for offline scoring, dependence/calibration reports,
   uncertainty intervals, and item-by-item live/replay consistency checks.

The maximum complete successful collection is 445 generator calls plus 890
verifier calls. Calibration, live execution, and additional baseline collection
have distinct operation IDs. Logical policy queries and actual new paid calls
are reported separately. The full comparison collection is more expensive than
deploying the sequential policy alone.

Operational timing: Llama intermittently rejected requests with HTTP 429.
After the first 50 generator responses passed parsing, remaining generator
collection continued while four pilot verifier pairs awaited capacity. All
100 pilot verifier responses subsequently parsed. Full calibration verifier
collection overlapped with evaluation *candidate generation* using already
saved calibration candidates. Every likelihood, cost and policy was frozen
before evaluation verifier requests or label-based evaluation scoring.
The overall frozen bundle contains 442 valid generator responses and three
format failures. Those failures remain in the cohort and are not regenerated.

The user authorized **USD 200 total**, conditional on the project being
`llm-applications-490420`. Both URL resource project and the explicit
`x-goog-user-project` header target that project. The local preflight records a
successful Cloud Billing `billingInfo` read with `billingEnabled=true` and an
attached account. The Cloud Billing management API was enabled to perform this
inspection; the account association was not changed.

Prices were checked on 2026-10-04 against Google's official pricing page.
The budget ledger reserves a conservative input allowance and, for Gemini,
131072 output/reasoning tokens per pending request. Actual reported usage then
replaces the reservation. Unpriced responses pause collection. Unknown transport
outcomes remain reserved and are not automatically retried. Explicit HTTP 429
rejections may be retried up to three times with 5/10/20-second backoff; every
attempt is recorded. Llama calls are serialized after the endpoint rejected
the initial concurrent burst. This controls local
usage estimates, not Google billing; confirmed billing requires separate
provider billing evidence. Other workloads on the project are outside this run.

## Commands

Install the API-only project environment (no GPU dependencies):

```sh
uv pip install --python .venv/bin/python -e '.[dev]'
```

Prepare a pinned dataset snapshot:

```sh
.venv/bin/python -m vgx.gpqa.fresh prepare \
  --root results/gpqa_vertex_20261004 \
  --config configs/gpqa_vertex_fresh.json \
  --revision 83022cefff930aea54f654c0b282e74b9eeda5c6
```

Collect the engineering pilot, then resume the full experiment:

```sh
.venv/bin/python -m vgx.gpqa.experiment \
  --root results/gpqa_vertex_20261004 \
  --config configs/gpqa_vertex_fresh.json --phase pilot --allow-api

.venv/bin/python -m vgx.gpqa.experiment \
  --root results/gpqa_vertex_20261004 \
  --config configs/gpqa_vertex_fresh.json --phase complete --allow-api
```

Omitting `--allow-api` runs entirely from cache and fails on any cache miss.
Use the same root for resume. Completed responses and candidates must not be
deleted to force retries. The source, prompts, raw responses, keys and detailed
artifacts remain under ignored `data/` and `results/` paths. Publish aggregate
results only, respecting the dataset's benchmark-text restrictions.

Artifacts: `dataset_manifest.json`, `generation_manifest.json`, `frozen/`,
`analysis_plan.json`, `policies/`, `live/`, `cache/budget.json`, `report.json`,
`primary_live_usage.json`, and `usage_estimate.json`.

To export aggregate PNG/SVG figures after scoring, install the optional
`report` extra and run:

```sh
uv pip install --python .venv/bin/python -e '.[report]'
.venv/bin/python -m vgx.gpqa.plots \
  --policy results/gpqa_vertex_20261004/policies/primary_95.json \
  --report results/gpqa_vertex_20261004/report.json \
  --output-dir results/gpqa_vertex_20261004/figures
```
