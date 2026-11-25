Sonnet 5.5: completed 50-question development pilot (2026-10-15)
================================================================

50/50 answers and confidence values accepted under the user-approved v2 parser.
5 original replies reused byte-for-byte; 45 new requests. No duplicate generation.
No truncation, no unknown usage, no confidence calibration. No verifier or Diamond calls.
The frozen strict-JSON parser accepts 0/50: all replies add text around the final JSON.
The original strict-format failure and 404 reports are preserved in parent directories.

Same original 50 development questions (historical answers are reused):

Model | Correct | Mean confidence | Confidence if correct | Confidence if wrong | Raw Brier | AUROC
Gemini 3.8 Flash (historical) | 39/50 (78.0%) | 93.58% | 93.77% | 92.91% | 0.198802 | 0.6329
Qwen3-235B Instruct (historical, two-step) | 28/50 (56.0%) | 91.56% | 92.93% | 89.82% | 0.361220 | 0.6623
Sonnet 5.5 (new) | 39/50 (78.0%) | 78.72% | 81.44% | 69.09% | 0.149252 | 0.7774

Sonnet raw confidence value counts:
0.4: 2
0.5: 1
0.55: 1
0.6: 5
0.68: 3
0.7: 1
0.72: 2
0.75: 1
0.8: 8
0.85: 10
0.88: 6
0.9: 2
0.93: 3
0.95: 1
0.96: 2
0.97: 2

Interpretation
--------------
In this 50-item screen, Sonnet and historical Gemini both answer 39 correctly.
Sonnet has lower average confidence and lower confidence on wrong answers.
Sonnet raw Brier is lower, and its confidence ranking separates correct/wrong
answers better by AUROC in this sample. These are descriptive point estimates;
the sample is small and historical generation protocols/serving dates differ.
The close average confidence and accuracy do not establish calibration.
Labels were used only for offline scoring after the new answers were frozen.
No confidence was fitted, calibrated or replaced using labels.
This screen provides no evidence yet of verifier benefit for Sonnet.

Format amendment and thinking
-----------------------------
The user approved accepting explanatory text plus one complete final JSON after
seeing the first five formats, before any correctness scoring. Multiple JSON
objects, malformed fields or nonterminal JSON remain rejected. Original model
settings and prompts stayed frozen; no structured-output option was added.
Requests use adaptive thinking / effort high, max_tokens 32768, default sampling.
Provider-reported thinking tokens: 10506 across 15 responses with positive counts.
All 50 terminated with end_turn. The final text is the only source of answers.

Accounting
----------
Input tokens: 18331; output tokens including thinking: 29572.
First-five estimated cost: USD 0.021174.
Additional 45 estimated cost: USD 0.311208.
Total 50 estimated cost: USD 0.332382; shared stage cap: USD 20.
Resource/quota project: llm-applications-490420. These are token-usage estimates,
not confirmed billing. The budget status is per request, so its zero rejected
requests does not erase the one historical 404 preserved in the attempt log.

Artifacts
---------
comparison.json: aggregate metrics, bootstrap/Wilson intervals, raw confidence
distributions, risk-coverage, and format metadata (no per-item correctness).
plan.json: parser amendment, hashes and unchanged request identity.
completion_audit.json: cache preservation, request counts, token reconciliation.
status.json and billing_check.json: completion and project metadata.
protocol.txt: exact procedure and limitations.
Raw replies and frozen candidates remain local in ignored results/; no question
text, answer keys, per-item outcomes or credentials are in this report folder.
No git commit or push was performed for this run.
