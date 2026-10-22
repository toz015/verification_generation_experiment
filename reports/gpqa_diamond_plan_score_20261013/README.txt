Diamond plan scoring: aggregate review artifacts

Source scoring commit: d037ab61907a1b6e668b217e22d2b18c5b5bab31
Branch: diamond-verifier-signals
Scoring instructions: docs/gpqa-diamond-plan-score.md
Original result: results/gpqa_diamond_plan_score_20261013/result.json

Contents
- result.json: exact aggregate scoring output, including all prior variants, Brier comparisons and policy summaries.
- sha256.txt: the two input hashes checked before scoring (frozen plan and frozen verifier signals).
- step2_stdout.txt: requested pytest run; 2 passed.
- step3_stdout.txt: computed input SHA256 values.
- step4_stdout.txt: outcomes command stdout (artifact ID only).
- step5_stdout.txt: score command stdout (artifact ID only).
- step6_stdout.txt: requested six-row numeric results table.
- manifest.json: SHA256 checksums for the files in this review directory.

These are byte-for-byte copies of the existing result and captured stdout; no fitting, tuning, regeneration or API collection was performed for publication.
The captured stderr for steps 2, 4 and 5 was empty.
The result and stdout contain aggregate metrics and artifact IDs. No outcomes.json, question text, answer-key file, per-item outcomes or credentials are included.
The data and other local work remain outside this commit. The full local inputs are required to rerun scoring.
