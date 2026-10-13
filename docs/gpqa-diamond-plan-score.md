# Diamond: score the pre-registered plan

`configs/gpqa_diamond_analysis_plan_20261012.json` (plan_id `e8674593b22690bd…`,
file sha256 `56d5e4f8…`) was fitted on the 249 development questions only and
sealed at 2026-10-12 20:53 UTC (commit f7890d8), before the Diamond verifier
collection ran. Nothing in it is refitted on Diamond.

Pre-declared comparisons (Brier of the always-query posterior, paired bootstrap
over questions, 10,000 resamples seeded from the plan_id):

- P1: Flash vs generator confidence alone (raw confidence).
- P2: the declared triple (correlation-aware) vs Flash, raw and monotone-calibrated confidence.

Secondary: utility at L = 19 (L = 99 sensitivity) of abstain-all, confidence
alone, Flash, and the frozen-order triple under both likelihood models.

## Run (local; no API requests)

```sh
.venv/bin/python -m pytest -q tests/test_diamond_score.py
OUT=results/gpqa_diamond_plan_score_20261013
.venv/bin/python -m vgx.gpqa.diamond_score outcomes \
  --dataset-root results/gpqa_vertex_20261004 \
  --signals results/gpqa_diamond_signals_20261012/signals.json \
  --output $OUT/outcomes.json
.venv/bin/python -m vgx.gpqa.diamond_score score \
  --plan configs/gpqa_diamond_analysis_plan_20261012.json \
  --signals results/gpqa_diamond_signals_20261012/signals.json \
  --outcomes $OUT/outcomes.json \
  --output $OUT/result.json
```

`outcomes` is the only step that opens Diamond answers; it writes one
correct/incorrect flag per (generator, question). `result.json` holds only
aggregate metrics and intervals and is the file to share; keep `outcomes.json`
local.
