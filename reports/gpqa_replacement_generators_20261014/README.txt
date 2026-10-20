Replacement-generator format screen: 2026-10-14
===============================================

Status: BLOCKED at the initial 5-item-per-model format gate.
The 50-item expansion was not started. No process remains running.

Model                Responses  Strictly valid  HTTP rejections  Estimated USD
gpt-oss-120b                  5               5                0     0.00085959
DeepSeek-R1-0528              4               3          1 (429)     0.08774865
Total                        9               8                1     0.08860824

All nine responses ended normally (finish_reason=stop), with provider usage
reconciled. Returned model labels match the requested models. Resource and
quota project bindings match llm-applications-490420. These are token-price
estimates, not confirmed billing. The local stage cap was USD 10.

R1 diagnosis
------------
One reply has explanatory text after its complete thinking section, then a
trailing answer/confidence JSON object. The frozen parser requires JSON-only
final content, so it rejects this wrapper. This is not an answer-free or
token-truncated response. Offline inspection confirms the trailing object;
the primary parser, parsed result, and provider response were not changed.
The fifth R1 request was rejected with HTTP 429 RESOURCE_EXHAUSTED. No retry
or regeneration was attempted. All received responses remain cached.

Interpretation boundary
-----------------------
This only establishes initial endpoint/format/accounting behavior. No labels
were opened, no accuracy or Brier score was computed, and confidence was not
calibrated. These few items cannot establish which generator is better.
No new verifier or Diamond requests were sent. Historical data is unchanged.

Next proposed stage
-------------------
Continue gpt-oss on the frozen 50 development items under a separately recorded
model-specific gate, reusing its five replies. R1 needs an explicitly versioned
final-content parsing protocol (applied offline to the existing raw replies)
and a conservative schedule for missing requests after the capacity error.
Do not change or override this run's frozen joint gate silently.
R1's Vertex endpoint retires on 2026-10-21, so it is only a short-term research
comparison. Current gpt-oss-120b is not listed in the checked retirement table.

Files
-----
format_summary.json: aggregate format, termination, usage and cost audit.
status.json: final collector status.
plan.json: frozen model settings, input identities, request hashes and budget.
prior_budget_audit.json: deduplicated old Vertex estimates with uncertainty.
sha256.txt: hashes of the above files and this README.

No question text, answer keys, per-item answers, provider responses or
credentials are included in this report directory.

Implementation/configuration/protocol (new local files; not committed/pushed):
src/vgx/gpqa/replacement_generators.py
configs/gpqa_replacement_generators_20261014.json
docs/gpqa-replacement-generators-20261014.txt

Validation performed: offline preparation checked original pilot membership,
public input hashes, baseline identities and question/option-order equality;
the live format gate and token/project/model audit ran as recorded. No new
unit test suite was added or run in this task.
