# Implementation status: frozen collection, accounting and sequential execution

## Result

The independent software work is implemented locally. The generator model, original `src/vgx/gpqa/prompt.py`, and existing answers have not been changed or regenerated. No model APIs were called, no VM was started, and no paid collection was launched. Existing user edits and the unrelated `asqa_prompt.json` remain intact. No changes have been committed or pushed.

The actual original-run recovery is **blocked**, not completed. Consequently no real fitted policy or candidate-bound Jev plan has been produced yet. The checked-in comparison specification, adapters and importer are prepared and tested using synthetic artifacts.

## Implemented

| Area | Changes |
|---|---|
| Recovery | Strict source/run/split/choice-order validation; original request identity, prompts, settings, raw-response/candidate agreement, and core reported generator metrics checked; idempotent content-based frozen bundle |
| Frozen inputs | Public candidate data separated from calibration labels and evaluation labels; original prompts stored verbatim; no generator-collection CLI |
| Caching | Shared cache by actual provider request, independent of prices, analysis identity and verifier lists; validated original responses can be imported; stale settings/prompts cannot silently reuse responses |
| Crash recovery | Durable response and progress writes; advisory process locks; truncated CallLog tail no longer consumes the next successful row; unknown provider outcomes block automatic repeat |
| Cost accounting | Native Google and compatible Chat Completions token reconciliation; explicit reasoning inclusion checks; cached-input pricing; execution-based deduplication; failed/unknown attempts disclosed; estimates separate from external billing evidence |
| Cost-to-utility mapping | Explicit frozen utility-per-USD conversion using calibration mean request cost, or explicitly labeled normalized sensitivity costs; no future evaluation usage in decisions |
| Algorithm 1 | Reusable belief-grid J/Q value tables with likelihood/cost validation, linear interpolation, and saved policy identity |
| Algorithm 2 | Resumable stage-by-stage execution; selected verifier only; stop ties; release threshold; malformed/impossible observation abstention; full decision trace; no label/future-output reads |
| Accounting scope | Per-operation incremental ledger includes only selected request keys; historical cached responses are logical queries but not new model executions |
| Offline scoring | Separate evaluation-label reader with full-cohort and identity checks, coverage/risk/accuracy/query/utility reporting |
| Jev preparation | Pinned-version HTTP adapter, versioned Choice/Noul requests, fixed-generator-option probability extraction, separate arms, original verifier controls, candidate-bound plan/cache inventory command |

The offline report now identifies its shared grid solver and stops labeling inferred record counts as actual API executions. The historical experiment reports are preserved.

## Real-artifact blockers

The original run ID is:

`a5d92086584910dbe156297b5c1083bb8bf47765670b46a0dfc1b7ddc49cbce7`

Missing locally:

1. `data/gpqa/gpqa_main.csv` — exact source bytes used in the run.
2. `data/gpqa/sample_200.json` — pinned 120/80 split and shuffled choices.
3. `results/gpqa/<run_id>/run_manifest.json`.
4. `results/gpqa/<run_id>/pilot_records.jsonl`.
5. `results/gpqa/<run_id>/pilot_metrics.json`.
6. `results/gpqa/<run_id>/generator.jsonl`.
7. `results/gpqa/<run_id>/verifier_1.jsonl`.
8. `results/gpqa/<run_id>/verifier_2.jsonl`.

A targeted search of the local workspace area did not find a copy. SSH to the saved VM address timed out. Google Cloud CLI could not refresh its credentials for a read-only instance lookup. The user then confirmed the VM was closed; it was left off. An existing archive or a later authorized file transfer from the original disk is needed. API authentication alone does not recover those files. If the files no longer exist, this experiment cannot be reconstructed by silently regenerating candidates.

Until recovery succeeds, the real-run candidate hashes, raw-token semantics, original metrics and monetary estimates remain unverified. No claimed dollar total has been substituted for missing evidence.

## Validation

Local validation uses only invented GPQA questions, recorded synthetic provider responses, mocked transports and existing dataset fixtures. It covers:

- Original-artifact import, relocation and idempotency; rejection of missing, mismatched and tampered inputs.
- No generator calls during verifier addition, repricing or resume; prompt/settings changes produce cache misses.
- Truncated logs, response-before-checkpoint crashes, unresolved transport outcomes, explicit HTTP-error resume, and duplicate execution accounting.
- Inclusive/separate reasoning tokens, inconsistent totals, malformed usage, cached-input rates, incomplete pricing and independent billing evidence.
- Live/replay agreement, threshold and tie cases, grid approximation, impossible/missing signals, frozen-policy/candidate validation and zero-query invalid candidates.
- Guards against opening calibration/evaluation labels or unselected future verifier outputs during execution and incremental accounting.
- Jev Choice/Noul semantics, model-version checks, structured inputs, mocked transport and secret-free logs.
- An offline CLI round trip: recovery → comparison preparation → calibration collection from cache → policy preparation → execution from cache → separate scoring.

Final validation: **188 passed, 4 skipped, 1 warning** using `.venv/bin/python -m pytest -q -rs` (15.52 seconds). This adds 70 passing tests to the previously reviewed 118. `git diff --check` and Python compilation checks also passed. Four existing PubMedQA tests skip because that dataset is absent. The existing SciFact warning concerns the official evaluator truncating rationales beyond three sentences.

A final comparison against Git HEAD confirmed that the generator/verifier model configuration, generation settings, dataset/sample settings, and original prompt file are unchanged. Pricing metadata gained the published cached-input rates; these do not affect provider-request cache identity.

## Remaining limitations before paid collection

- Recover and validate the eight private files above. No replacement candidates will be generated.
- Confirm laptop ADC credentials and provider access separately; no inference-based access check was performed.
- Review the prepared arm specification and freeze the cost/loss scale. Jev's output-billing terms/account rate remain to be confirmed before monetary policy fitting.
- Hosted Gemini aliases do not provide immutable future weight snapshots. Existing cached responses are frozen; new calls can still experience provider drift.
- The executor currently supports a fixed order of singleton verifier layers. The planner uses approximate grid interpolation; arbitrary subset selection and parallel multi-verifier layers are not implemented.
- Likelihood independence, raw generator calibration, small error counts and held-out quality remain scientific limitations. Existing evaluation comparisons are exploratory.
- Unknown API outcomes require reconciliation; exactly-once behavior across a network failure is not guaranteed by these providers. Such requests fail closed rather than being automatically billed twice.
- No real API latency, live cost saving, Jev quality, truthful reporting or audit/payment mechanism has been validated by these software tests.

See `docs/gpqa-frozen-workflow.md` for commands and detailed semantics. Commands default to offline execution and must not be treated as authorization for paid API calls.
