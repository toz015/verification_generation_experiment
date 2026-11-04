# Frozen GPQA workflow

This workflow implements fixed-candidate verification. It never generates a new candidate, changes an existing option selection, changes the original generator/verifier prompts, or trains a model. All commands below are **offline**. API collection remains disabled until an explicitly authorized invocation adds `--allow-api` to `collect` or `execute`.

## 1. Recover and validate the original run

The required original run is `a5d92086584910dbe156297b5c1083bb8bf47765670b46a0dfc1b7ddc49cbce7`.

Required private inputs:

- The exact `gpqa_main.csv` source used for collection.
- `sample_200.json`, including the source hash, exclusions, shuffled correct positions, and 120/80 membership.
- `run_manifest.json`, `pilot_records.jsonl`, `pilot_metrics.json`.
- `generator.jsonl`, `verifier_1.jsonl`, `verifier_2.jsonl` from that run.
- Dataset provenance/access records should remain alongside the source privately.

Copy existing files from an available archive or the original disk into ignored `data/` and `results/`; do not replace them with newly collected answers. The importer checks run and source identities, sample content, partition membership, prompts, generation settings, responses, parsed candidates, verifier scores, and core reported generator metrics. It refuses missing files, mixed runs, conflicting responses, or tampering.

```sh
python -m vgx.gpqa.artifacts \
  --source data/gpqa/gpqa_main.csv \
  --sample-manifest data/gpqa/sample_200.json \
  --run-dir results/gpqa/a5d92086584910dbe156297b5c1083bb8bf47765670b46a0dfc1b7ddc49cbce7 \
  --output results/frozen/gpqa-original \
  --cache results/request_cache
```

The importer writes an immutable bundle and migrates validated successful responses into the shared cache. It is idempotent and does not overwrite different contents. Original files remain untouched. The bundle contains:

| File | Reader |
|---|---|
| `candidates.jsonl` | Collector and live executor; question/options, frozen answer/confidence, original prompt and provenance only |
| `calibration_records.jsonl` | Offline fitter only |
| `evaluation_labels.jsonl` | Offline scorer only |
| `bundle_manifest.json` | Hashes, split IDs and collection provenance |

Raw questions are not committed. The actual 200-item recovery is currently blocked because these files are not on the laptop and the original VM is stopped. Synthetic recovery tests demonstrate importer behavior, not validation of those real files.

## 2. Shared requests and accounting

`results/request_cache/calls/` contains one JSONL response log per exact provider request. Model, endpoint/project, actual generation settings, prompt and system message determine identity. Prices, analysis settings, verifier ordering, role labels, and code-report revisions do not. Changing a provider input creates a cache miss. Adding a verifier never invokes a generator.

The historical logs' original keys are checked before migration. A migration preserves the original response and provenance; it does not trust role/item-only legacy logs. Distinct historical billed executions should be accounted from the full original logs, even if only one frozen response is needed for future inference reuse.

Reprice original usage without model access:

```sh
python -m vgx.common.billing \
  --log results/gpqa/a5d92086584910dbe156297b5c1083bb8bf47765670b46a0dfc1b7ddc49cbce7/generator.jsonl \
  --log results/gpqa/a5d92086584910dbe156297b5c1083bb8bf47765670b46a0dfc1b7ddc49cbce7/verifier_1.jsonl \
  --log results/gpqa/a5d92086584910dbe156297b5c1083bb8bf47765670b46a0dfc1b7ddc49cbce7/verifier_2.jsonl \
  --pricing configs/gpqa_experiment.json \
  --output results/original_usage_estimate.json
```

Token reconciliation rules:

- Native Google `candidatesTokenCount` and `thoughtsTokenCount` are summed, with `totalTokenCount` checked when present.
- For compatible Chat Completions usage, total/input/completion/reasoning counts determine whether reasoning is already included. It is never added twice.
- Separate reasoning without a reconcilable total remains unpriced unless an explicit, documented per-model `usage_semantics` override supplies `completion_includes_reasoning` or `completion_excludes_reasoning`. A conflicting total still fails.
- Gemini 3 usage lacking both total and reasoning semantics remains unresolved.
- Invalid, fractional, negative, conflicting or nonfinite token counts are rejected. Cached input requires its own rate if any cached tokens are reported.
- The accounting unit is an execution ID or provider response ID, not a request cache key. Exact copied legacy rows are deduplicated conservatively; missing historical retries cannot be reconstructed.
- Only successfully observed usage is priced. Failed HTTP attempts and uncertain/lost outcomes are disclosed separately. An unpriced total is not zero spend.

`estimated_usd_for_priced_calls` is a usage-based estimate, **not confirmed billing**. Optional `--confirmed-billing` imports external evidence with `currency`, `usd`, `source`, `reference`, and `scope`. That evidence is not inferred from tokens and is not independently checked against a billing service. Account terms, unlogged executions, discounts, credits and extra fees can prevent reconciliation to an invoice.

Each new collection/execution has a persistent `operation_id`. Incremental usage estimates include only responses actually created for that operation. Cached historical responses cost zero new model executions, although they still count as logical policy queries. Incremental ledgers inspect only request keys observed by the run, not future verifier responses.

## 3. Freeze verifier arms and prepare the Jev comparison

The versioned specification is `configs/gpqa_frozen_verifiers.json`. It retains the original Vertex models, settings and prompts, and adds two separate Jev arms:

- `jev_choice_v1`: four probabilities for the original option order; extract the probability of the frozen generator answer, even when Jev favors another option.
- `jev_noul_v1`: probability that the fixed candidate is correct.

Never use Jev's `confidence` field as `p_correct`. Choice and Noul are not independent layers and are not chained together. The primary proposed Jev arm is Choice-only; Noul and Choice→Gemini are secondary. Original one-verifier baselines and both original orders are included. Confidence-only, always-release/abstain, and fixed-budget comparisons remain analysis baselines.

```sh
python -m vgx.gpqa.workflow prepare-comparison \
  --bundle results/frozen/gpqa-original \
  --specs configs/gpqa_frozen_verifiers.json \
  --output results/jev_comparison_plan.json
```

This creates a candidate-bound comparison plan, request-cache inventory and zero-generator-call declaration. It performs no inference. Missing original artifacts currently prevent creating the real candidate-bound plan; the checked-in spec and mocked adapter are ready.

The existing generator and verifier prompt file is unchanged. Jev requires its own structured request shape; its new question templates are explicitly versioned as `gpqa_jev_v1`. An exact Jev model version is required and returned model mismatches are treated as invalid signals. The adapter implements the [TypeSafe HTTP API](https://docs.typesafe.ai/api), [Choice](https://docs.typesafe.ai/primitives/choice) and [Noul](https://docs.typesafe.ai/primitives/noul) semantics. On 2026-10-04 the [official model page](https://docs.typesafe.ai/models) confirmed $0.042 per million input tokens and free output tokens. No API key is needed for offline preparation/tests. The separately authorized [fresh-candidate Jev comparison](gpqa-jev-comparison.md) uses `TYPESAFE_API_KEY` from the local ignored `.env` or process environment and a separate USD 10 budget.

A collection invocation selects an explicit partition and verifier subset:

```sh
python -m vgx.gpqa.workflow collect \
  --bundle results/frozen/gpqa-original \
  --specs configs/gpqa_frozen_verifiers.json \
  --verifiers verifier_1 verifier_2 \
  --partition calibration \
  --output results/original_calibration_signals.json
```

Without `--allow-api`, a missing response stops collection. It never falls back to generation. Existing complete responses survive partial collection; repeating the command resumes from the shared cache. A different verifier list should use a new output artifact while sharing the same response cache.

## 4. Fit calibration likelihoods and construct value tables

Only calibration candidate labels and calibration verifier responses are used. Evaluation labels or scores are not opened. The fitter uses the existing three-bin Laplace-smoothed observation model; it does not silently recalibrate or clip the generator's reports. Likelihood fits must contain both outcome classes.

A sensitivity-policy example reproduces the previous `R=1`, `L=4`, and cost assumptions:

```sh
python -m vgx.gpqa.workflow prepare-policy \
  --bundle results/frozen/gpqa-original \
  --specs configs/gpqa_frozen_verifiers.json \
  --verifiers verifier_1 verifier_2 \
  --reward 1 --loss 4 --normalized-costs 0.02 0.02 \
  --grid-size 1001 \
  --output results/policy_original_order.json
```

This is **not a monetary policy**. For monetary costs, replace `--normalized-costs` with `--pricing CONFIG --utility-per-usd ALPHA`, choosing and freezing `ALPHA` before evaluation. For each verifier:

```text
expected_usd_j = mean reconciled calibration request estimate
cost_utility_j = ALPHA * expected_usd_j
```

`ALPHA` has units of utility per USD, while `R` and `L` are utility per released correct/incorrect answer. For example, if one utility unit is valued at $0.10, then `ALPHA=10 utility/USD`. The value is an experimental preference, not determined by the provider. The fitter rejects missing/ambiguous calibration prices. Invalid verifier answers still incurred a call and are included in expected cost observations when their usage is known.

Realized evaluation output length, reasoning tokens, outcomes and future signals never enter the preceding stop decision. Record realized usage afterward. Generator collection and calibration are sunk for a within-item decision but belong in total experimental spend. No audit cost is added because this implementation has no audit. Costs here are fixed per verifier layer; prompt-length-conditioned cost models and latency utility are future extensions.

Algorithm 1 computes `J_K(b)=max(0,(R+L)b-L)` and backward `Q_k(b)=-cost_(k+1)+E[J_(k+1)(b')]`, then `J_k=max(stop,Q_k)`. It saves the grid, J/Q tables, likelihoods, costs, model/prompt identities, calibration hashes and cost-mapping provenance. Linear interpolation is approximate; grid size is part of policy identity. Table consistency is checked on load. Layers currently contain one verifier each in a fixed order; the planner does not optimize model selection or arbitrary subsets.

New offline `vgx.gpqa.report` replays also use this shared planner with 1,001 grid points and identify the numerical method in their output. The historical run used memoized recursion, so do not overwrite its original metrics with a newly computed replay. Raw forecast scoring and frozen responses are unaffected. Reports no longer infer an “actual API call count” from the number of candidate records; that count requires provider execution logs.

## 5. Execute and resume Algorithm 2

```sh
python -m vgx.gpqa.workflow execute \
  --bundle results/frozen/gpqa-original \
  --policy results/policy_original_order.json \
  --partition evaluation \
  --pricing configs/gpqa_experiment.json \
  --output results/sequential/original_order
```

This example uses existing cached responses and is an execution-consistency/replay check, not evidence of fresh API savings. A later authorized live run adds `--allow-api`; the policy must already be frozen. The old `vgx.gpqa.run_pilot` module delegates to this same CLI and has no generator-collection path.

At each stage the executor:

1. Reads only the current candidate, policy and already-observed scores.
2. Computes stop and continuation values. Stops on a tie.
3. If continuing, checkpoints the selected stage, then obtains only that verifier's response.
4. Durably caches the response before checkpointing the observation.
5. Updates belief and repeats, or abstains on an invalid required score or zero predictive likelihood.

When stopping, release requires `b >= L/(R+L)`; passing this threshold does not by itself force a stop. Invalid generator answers/confidence cause abstention without querying. Endpoint reports zero/one retain their usual Bayes behavior.

Checkpoints bind the candidate hash, policy hash, ordered observations and stop trace. A policy/candidate mismatch refuses resume. New prices can reprice the ledger freely; changes to the policy's cost assumptions require preparing a new policy/output directory, with exact provider requests still reusable.

**Interruption semantics:** local file locks prevent concurrent sends of the same request by these runners. Started attempts are durable before sending. A cached successful response is reused after a crash even if the final progress event is missing. A timeout, lost response or crash after sending leaves an unresolved attempt and blocks automatic retry. Exactly-once execution across an external API cannot be guaranteed without provider support. Inspect the provider outcome and recover the original response; do not delete the attempt record to force a repeat. There is intentionally no silent retry of an ambiguous potentially billed request. HTTP rejections are recorded; explicitly resuming retries those requests. An HTTP/transport error pauses the run rather than inventing a verifier score.

The finished run writes `decisions.json`, per-item checkpoints, and `incremental_usage_estimate.json`. Logical queries and newly observed API executions are different measures. Existing cached responses can validate sequencing but cannot demonstrate newly achieved live savings.

## 6. Score separately and retain experimental limits

```sh
python -m vgx.gpqa.workflow score \
  --bundle results/frozen/gpqa-original \
  --policy results/policy_original_order.json \
  --decisions results/sequential/original_order/decisions.json \
  --output results/sequential/original_order/scored.json
```

Only this separate command opens evaluation labels. It checks full evaluation coverage and candidate/policy identity, then reports coverage, accuracy among releases, wrong-release rates, queries and configured-cost utility. Monetary usage remains in its separate ledger. A crashed partial run is not silently scored as a complete cohort.

For the controlled Jev study, compare signals on the same frozen candidates using paired forecast errors, policy utility/cost and risk–coverage tradeoffs. Fit/select using calibration only. The old 80 evaluation items have already informed design discussion, so additional comparisons there are exploratory. Reserve unused, nonoverlapping questions for a later confirmatory study. Changing Main to Diamond does not remove overlap.

Remaining scientific assumptions are the raw generator prior, pooled conditional-independent verifier likelihoods, sparse incorrect calibration examples, and hosted model drift. Software tests and grid agreement do not establish calibration, independence, incentive compatibility or real cost savings. The audit/payment mechanism and reward-driven generator updates remain outside this implementation.
