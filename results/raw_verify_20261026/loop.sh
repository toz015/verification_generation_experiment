#!/bin/zsh
# usage: loop.sh DATASET LIMIT  — resume after 429/502/503 or a transient unresolved entry.
cd /Users/wanghd/Documents/workplace/verification-generation-experiment
D=$1; L=$2; O=results/raw_verify_20261026/$D; mkdir -p $O
for i in $(seq 1 300); do
  .venv/bin/python -m vgx.raw_verify $D --limit-usd $L --allow-api --workers 10 > $O/last.json 2>>$O/err.log
  python3 - "$O/last.json" <<'PY' || exit 0
import json,sys
d=json.load(open(sys.argv[1])); print(d['collected'], d['jobs'], [e.get('http_status') or e.get('message','')[:40] for e in d['errors']], round(d['budget']['estimated_usd'],3), flush=True)
if d['collected']>=d['jobs']: sys.exit(1)
if not all(e.get('http_status') in (429,502,503) or 'unresolved' in (e.get('message') or '') for e in d['errors']): sys.exit(1)
PY
  sleep 60
done
