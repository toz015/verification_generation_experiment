#!/bin/zsh
# usage: loop_any.sh OUTDIR LIMIT_USD [extra qwen_raw args...]
# Resume after recoverable halts (HTTP 429/502/503, or a transient 'unresolved' ledger entry that the next
# start settles as a service failure); stop when complete or on any other error.
cd /Users/wanghd/Documents/workplace/verification-generation-experiment
O=$1; L=$2; shift 2
for i in $(seq 1 300); do
  .venv/bin/python -m vgx.qwen_raw --out $O --limit-usd $L "$@" --allow-api > $O/last.json 2>>$O/err.log
  python3 - "$O/last.json" <<'PY' || exit 0
import json,sys
d=json.load(open(sys.argv[1]))
print(d['collected'], d['jobs'], [e.get('http_status') or e.get('message','')[:40] for e in d['errors']], round(d['budget']['estimated_usd'],3), flush=True)
if d['collected']>=d['jobs']: sys.exit(1)
ok=lambda e: e.get('http_status') in (429,502,503) or 'unresolved' in (e.get('message') or '')
if not all(ok(e) for e in d['errors']): sys.exit(1)
PY
  sleep 60
done
