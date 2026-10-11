#!/usr/bin/env bash
# Rebuild every result in this folder offline. No model or network calls are made.
#   reproduce.sh [OUT] [PACKAGE]
# OUT defaults to a fresh temporary directory. With PACKAGE (the unzipped
# GPQA_experiment_and_diagnosis_review_20261009 folder) the observation export is
# rebuilt from the frozen source files and compared with the shipped export first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$(mktemp -d)}"
PACKAGE="${2:-}"
PY="${PYTHON:-python}"
CONFIG="$ROOT/configs/gpqa_risk_acceptance.json"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
cd "$ROOT"

gunzip -c "$HERE/data/observations.json.gz" > "$OUT/observations.json"
if [ -n "$PACKAGE" ]; then
  "$PY" -m vgx.gpqa.risk_exploration export --config "$CONFIG" --root "$PACKAGE" --output "$OUT/observations_rebuilt.json"
  cmp "$OUT/observations.json" "$OUT/observations_rebuilt.json" && echo "export rebuilt from package: identical"
fi
"$PY" -m vgx.gpqa.risk_exploration run --config "$CONFIG" --observations "$OUT/observations.json" --output "$OUT/phase1"
"$PY" -m vgx.gpqa.risk_certification calibrate --config "$CONFIG" --observations "$OUT/observations.json" \
  --output "$OUT/phase2" --dry-run-on-development-data
for alpha in 0.05 0.1; do
  "$PY" -m vgx.gpqa.risk_certification evaluate --config "$CONFIG" --observations "$OUT/observations.json" \
    --output "$OUT/phase2" --dry-run-on-development-data --selection "$OUT/phase2/selection_alpha_$alpha.json"
done
"$PY" -m vgx.gpqa.risk_certification validate --config "$CONFIG" --repeats 5000 --output "$OUT/validation"
"$PY" -m vgx.gpqa.risk_certification plan --config "$CONFIG" --exploration "$OUT/phase1/exploration.json" --output "$OUT/plan"
"$PY" -m vgx.gpqa.risk_report --config "$CONFIG" --phase1 "$OUT/phase1" --phase2 "$OUT/phase2" --plan "$OUT/plan" \
  --validation "$OUT/validation/procedure_validation.json" --output "$OUT/report"
echo "outputs in $OUT"
