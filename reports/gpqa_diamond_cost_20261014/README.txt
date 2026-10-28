Diamond cost comparison: aggregate review artifacts

Source scoring commit: bb0c2ffd23dbbc8015a7a25aa5b743b519d415ce
Branch: diamond-verifier-signals
Instructions: docs/gpqa-diamond-plan-score.md, Cost comparison section
Source result: results/gpqa_diamond_cost_20261014/result.json

Contents
- result.json: exact aggregate output of the diamond_cost score command.
- step2_stdout.txt: tests/test_diamond_cost.py and tests/test_diamond_score.py; 4 passed.
- step3_stdout.txt: label-free decisions preview (aggregate calls, costs and release counts).
- step4_stdout.txt: score command stdout (result ID only).
- sha256.txt: unchanged plan, signals and existing outcomes file hashes.
- manifest.json: source commit, artifact identities, preview checks and report file hashes.
Stderr for steps 2-4 was empty.

This comparison uses raw confidence, the independence model and the unchanged development plan.
The adaptive preview matched the published raw/triple_independence policy exactly:
qwen L=19: 122 released / 342 calls; qwen L=99: 78 / 302.
gemini L=19: 147 released / 213 calls; gemini L=99: 56 / 131.

No model API requests, refitting, retuning, candidate/signal regeneration or outcomes regeneration occurred.
Existing outcomes.json was reused. No question text, answer keys, per-item outcomes or credentials are included here.
The three protected inputs were hash-checked again after scoring; all were unchanged.

Utility uses the plan's expected verifier cost, reward 1, loss 19 or 99, and 10 utility units per USD.
measured_usd_per_question sums recorded request usage cost estimates along the selected query path and divides by all 196 questions.
It is not confirmed billing or new spend. Costs treat each generator as run alone; shared requests were paid once during collection.
The Brier block uses raw confidence and the independence model, with the L=19 selected triple always queried on the common complete-signal subset.
Bootstrap intervals are descriptive paired intervals from 10,000 resamples, seeded from the plan ID. This cost comparison was added after the earlier analysis plan.
