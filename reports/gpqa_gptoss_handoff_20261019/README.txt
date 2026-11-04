GPT-OSS checkpoint and agent handoff — 2026-10-19

CURRENT STATE
291 / 445 replies have been received; 286 parse as valid. Development:
249 / 249 received, 244 valid. Diamond: 42 / 196 received, 42 valid.
Three invalid_final_json and two finish_reason:abort replies are preserved.
Collection is BLOCKED at question 292 after an uncertain request lasting
300.187 seconds. No response or known token usage was returned for that attempt.
Questions 293–445 have not been attempted. No collector is running; monitoring
is paused. Do not mistake older blocked status files for the current state.

Current local status:
results/gpqa_gptoss_continuation_20261017/retry179_ledgerfix_20261018/status.json
Current code: src/vgx/gpqa/gptoss_retry179_ledgerfix.py
Frozen analysis: src/vgx/gpqa/gptoss_retry179_ledgerfix_score.py
Collection and analysis identifiers are in audit.json and the copied plans.

WHAT HAS BEEN SCORED
The previously completed 50-question development comparison is published in
../gpqa_gptoss_completed50_20261017/result.json and README.txt. It compares
GPT-OSS, Gemini, Qwen and Sonnet on the same 50 questions. It is an exploratory
pilot, not the full 249-development / 196-Diamond comparison.
The full 445-question comparison has NOT been run. Raw confidence has not been
calibrated. The partition confidence means in status.json are descriptive only;
they do not report correctness. No labels were read in this checkpoint export.
Historical reports and documentation are preserved byte-for-byte, including
their historical statements that they had not yet been pushed.

CHECKS PERFORMED FOR THIS HANDOFF
Read-only checkpoint audit passed: frozen source/plan seals, original 178 reply
and candidate hashes, all 291 received request identities and priced usage,
partition/format/confidence counts, ledger reconciliation and request intervals.
Minimum measured interval after completion in the latest leg: 30.532 seconds.
Latest leg reused 178 replies and added 113; no HTTP rejections in this leg.
This is not the full-completion audit: the unresolved question292 still blocks
that audit. No model API requests or unit tests were run for this export.
Reproduce the partial audit in the original checkout with its local artifacts:
  .venv/bin/python scripts/export_gpqa_gptoss_handoff_20261019.py

COSTS (TOKEN ESTIMATES, NOT CONFIRMED BILLING)
Project: llm-applications-490420. Shared GPT-OSS cap remains USD1.
All 291 received replies: USD0.13021686, including the first five imported
replies (USD0.00085959) outside the shared ledger. Shared ledger estimated
settled cost: USD0.12935727. Latest leg increment: USD0.05463450.
Seven unresolved historical attempts retain USD0.08399538 total reservation;
question292 contributes USD0.01197954. Actual total billing is unknown.
Do not release these reserves or infer that unknown attempts were free.
budget_inventory_at_launch.json is a historical inventory, not a current bill.

CONTINUATION BOUNDARY
This publication authorizes no new paid retry. Question292 has uncertain usage;
obtain the user's authorization before retrying it. Preserve every received
reply, invalid reply, prior plan/status, cache, source seal and budget entry.
If authorized later, implement a separate sealed continuation rather than edit
the old blocked state. Keep model/prompt/parser/prices, medium reasoning,
32768 max_tokens, temperature0, top_p1, raw confidence and USD1 cap unchanged.
Use 30 seconds after each request completes. Only explicit HTTP429 permits
60/120-second bounded retries (maximum three total attempts per item, including
historical attempts). A new unknown attempt blocks again.
Only after all445 replies are frozen may the four-model offline analysis run.
A new continuation requires a corresponding frozen analysis-source update;
retain the scoring metrics and baselines without refitting or calibration.

LOCAL DATA REQUIRED BY THE NEXT AGENT
Git contains code, aggregate results, plans, audit metadata and hashes only.
It does NOT contain the private results/ cache, question text, answer keys,
raw replies, candidates, per-item correctness or credentials. A fresh clone
alone cannot resume collection. On this same laptop use the original checkout:
/Users/wanghd/Documents/workplace/verification-generation-experiment
For a different machine, arrange a separate authorized transfer of required
private artifacts; do not upload them to Git. See local_artifacts_manifest.json
and the sealed plans for exact paths and checksums. Required sources include
the GPT-OSS continuation/replacement roots and frozen Gemini/Qwen/Sonnet inputs.
Labels are only needed for the eventual offline scoring stage.

The latest sealed loader also pins the exact zero-cost QuantumKatas offline
mock budget and plan described in docs/gpqa-gptoss-retry179-ledgerfix-20261018.txt.
Those two local metadata files are not included here. Their absence or changes
must be reported rather than bypassing the ledger guard. All real project
ledgers remain counted. The original checkout's unrelated QuantumKatas branch
and local work were preserved; this package was prepared on a separate worktree
based on origin/diamond-verifier-signals.

FILES
status.json: copied blocked checkpoint, no answers.
collection_plan.json / analysis_plan.json: frozen protocol metadata and seals.
billing_check.json / budget_inventory_at_launch.json: historical cost checks.
audit.json / budget_summary.json: partial audit and current cost reconciliation.
local_artifacts_manifest.json: private local file paths and hashes, not contents.
code_sha256.json / sha256.txt: code and handoff file integrity checks.
