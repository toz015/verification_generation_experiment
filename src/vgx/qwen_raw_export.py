"""Export every raw Qwen3 job (results/qwen_raw_20261021/**) to compact gzip JSONL, one file per kind.

Each row: kind, item_id, sample (seed / candidate index / option index), attempt, status, full response
text, parsed fields, finish_reason, usage and, where requested, token logprobs (top-k per position).
Rows marked status=service_failure have no response. Candidate lists for the judge jobs are copied
alongside (stage2*/candidates). Nothing is re-requested.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
import shutil

from vgx.common.api import RequestCache

ROOT = Path('results/qwen_raw_20261021')
OUT = Path('reports/qwen_raw_export_20261021')
DIRS = ['.', 'peg_gpqa', 'stage2_gpqa', 'samples', 'samples_nolp', 'retry1', 'retry2', 'stage2', 'stage2b']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by = defaultdict(list); seen = set()
    for d in DIRS:
        base = ROOT/d; path = base/'collected.jsonl'
        if not path.exists():
            continue
        cache = RequestCache(base/'cache')
        for line in open(path):
            row = json.loads(line)
            key = (row['kind'], row['item_id'], row['sample'], row['request_key'])
            if key in seen:
                continue
            seen.add(key)
            out = dict(row, source_dir=d, attempt=2 if d.startswith('retry') else 1)
            if row.get('status') == 'ok':
                recs = [r for r in cache.log(row['request_key']).records() if r.get('response') is not None and not r.get('error')]
                if recs:
                    r = recs[0]; pr = (r.get('meta') or {}).get('provider_response') or {}
                    ch = (pr.get('choices') or [{}])[0]
                    out.update(response=r['response'], usage=pr.get('usage'), model=pr.get('model'),
                               prompt=r['prompt'], logprobs=(ch.get('logprobs') or {}).get('content'))
            by[row['kind']].append(out)
    summary = {}
    for kind, rows in by.items():
        with gzip.open(OUT/f'{kind}.jsonl.gz', 'wt') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        summary[kind] = {'rows': len(rows), 'ok': sum(r.get('status') == 'ok' for r in rows),
                         'service_failure': sum(r.get('status') == 'service_failure' for r in rows)}
    for d in ('stage2', 'stage2b'):
        src = ROOT/d/'candidates'
        if src.exists():
            shutil.copytree(src, OUT/f'{d}_candidates', dirs_exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
