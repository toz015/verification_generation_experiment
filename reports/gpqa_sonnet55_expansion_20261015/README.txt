Sonnet 5.5: original 445-question expansion, 2026-10-15 (America/Los_Angeles)
============================================================================

Collection finished: 445/445 provider responses, comprising 50 reused pilot
responses and 395 new responses. Every request has one successful response.
All 445 stop reasons are end_turn: no output-token truncation was reported.
Current protocol, prompts, raw confidence and prior results are preserved.

Partition             Responses   Valid answer + confidence   Mean raw confidence
Development           249         248                         0.797741935483871
Diamond evaluation    196         196                         0.837295918367347

One development response failed the unchanged parser with
nonterminal_json_or_unbalanced_fence. It was retained and was not repaired,
regenerated or retried. Mean confidence excludes this unparsed response.
Development valid confidence range: 0.30 to 0.97.
Diamond valid confidence range: 0.28 to 0.98.
Confidence distributions are in completion_report.json.

Usage and accounting
--------------------
                         Input tokens   Output tokens including thinking   USD estimate
Original 50 pilot        18,331         29,572                             0.332382
Additional 395           157,530        266,838                            2.983440
All 445                  175,861        296,410                            3.315822

Provider reported thinking tokens across all responses: 117,500 (included in
output tokens, not an additional charge). The shared USD 20 budget was unchanged.
No outstanding reservations or unknown costs remain. Resource and quota project:
llm-applications-490420. Estimates use provider token usage and the frozen price
table; they are not confirmed Cloud billing. The historic rejected access attempt
is preserved in its original log; it does not represent a successful generation.

Offline validation and repeat-report issue
-----------------------------------------
The collector's finalization succeeded and saved the sealed candidate bundle,
completion report and stage=complete status. The requested later offline report
command exited 1:
ValueError: existing artifact differs: results/gpqa_sonnet55_20261015/expansion_445/frozen_candidates.json

A read-only audit identified a Python tuple/list comparison issue in the
existing immutable-file helper: candidate choices are tuples in memory but lists
after reading JSON. All 445 reconstructed records match the existing bundle
after JSON normalization; canonical digests and sealed bundle IDs match exactly.
The source seals, original 50 response files/attempt logs, and old report hashes
passed. Provider token counts and costs reconcile with the saved report and
shared budget ledger. This separate audit passed, but the report command itself
did NOT pass. No source code, settings, cached reply or candidate file was edited.
No new unit tests were added or run.

Scope
-----
No answer keys were read in this expansion or completion audit. No correctness
scoring, confidence calibration, likelihood fitting, verifier requests or Jev
requests were performed. Confidence means are not accuracy estimates. Evaluation
answers are collected and frozen only. No commit or push was performed.

Contents
--------
plan.json, status.json, completion_report.json and billing_check.json are exact
copies of the aggregate run artifacts. read_only_audit.json documents the audit;
report_command_error.txt records the failed offline command. sha256.txt covers
all report files. No question text, candidate answers, answer keys, raw provider
responses or credentials are included. The frozen candidate bundle remains in
the local results directory and is deliberately excluded from this report.
