#!/bin/zsh
cd /Users/wanghd/Documents/workplace/verification-generation-experiment
O=results/qwen_raw_20261021/stage2b
for i in $(seq 1 200); do
  .venv/bin/python $O/run.py > $O/last.json 2>>$O/err.log
  python3 - "$O/last.json" <<'PY' || exit 0
import json,sys
d=json.load(open(sys.argv[1])); print(d['collected'], d['jobs'], [e.get('http_status') or e.get('message','')[:40] for e in d['errors']], flush=True)
if d['collected']>=d['jobs']: sys.exit(1)
if not all(e.get('http_status') in (429,502,503) or 'unresolved' in (e.get('message') or '') for e in d['errors']): sys.exit(1)
PY
  sleep 60
done
