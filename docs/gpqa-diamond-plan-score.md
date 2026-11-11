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

## Cost comparison (added 2026-10-14; not pre-declared)

`vgx.gpqa.diamond_cost` compares, with raw confidence and the independence
model, four ways to use the same frozen fits at each loss: confidence alone,
the selected triple asked one at a time (the adaptive policy above), the same
triple asked for every question, and all five asked for every question. It
reports calls, the plan's expected cost and the measured cost of the Diamond
requests per question, outputs, wrong outputs and utility. Intervals are
descriptive paired bootstraps. Measured cost counts each generator's requests
as if run alone; requests shared by both generators were paid once.

```sh
.venv/bin/python -m pytest -q tests/test_diamond_cost.py
# label-free preview (no answers opened)
.venv/bin/python -m vgx.gpqa.diamond_cost decisions \
  --plan configs/gpqa_diamond_analysis_plan_20261012.json \
  --signals results/gpqa_diamond_signals_20261012/signals.json
.venv/bin/python -m vgx.gpqa.diamond_cost score \
  --plan configs/gpqa_diamond_analysis_plan_20261012.json \
  --signals results/gpqa_diamond_signals_20261012/signals.json \
  --outcomes results/gpqa_diamond_plan_score_20261013/outcomes.json \
  --output results/gpqa_diamond_cost_20261014/result.json
```

It reuses the existing `outcomes.json`; `result.json` holds aggregates only.
