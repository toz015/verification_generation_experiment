"""Tidy export of the five verifiers' raw GPQA scores for sharing (no API calls, no question text).

Source: results/raw_verify_20261026/gpqa/signals.jsonl (python -m vgx.raw_verify gpqa). Every option
A-D of every one of the 445 questions was scored by each verifier with the main-experiment probability
prompt, so any generator answer (letter) can be joined on (item_id, candidate). Output:
reports/raw_verifier_export_20261027/gpqa_verifier_scores.jsonl.gz, one row per (item_id, candidate, verifier).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from vgx.common.storage import read_jsonl
from vgx.raw_generator_export import GPQA_BUNDLE, LABELS

SIGNALS = Path('results/raw_verify_20261026/gpqa/signals.jsonl')
SCREEN = Path('configs/gpqa_signal_study_4096.json')
OUT = Path('reports/raw_verifier_export_20261027')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    key = json.loads((LABELS/'gpqa_key.json').read_text())
    part = {r['item_id']: r['partition'] for r in json.loads(Path(GPQA_BUNDLE).read_text())['groups']['qwen']}
    model = {m['id']: m['model'] for m in json.loads(SCREEN.read_text())['models']}
    rows = [{'item_id': r['item_id'], 'partition': part[r['item_id']], 'candidate': r['candidate'],
             'verifier': r['verifier'], 'model': model[r['verifier']], 'score': r['score'], 'failure': r['failure'],
             'candidate_is_key': r['candidate'] == key[r['item_id']]}
            for r in read_jsonl(SIGNALS, tolerate_tail=True)]
    rows.sort(key=lambda r: (r['item_id'], r['candidate'], r['verifier']))
    with gzip.open(OUT/'gpqa_verifier_scores.jsonl.gz', 'wt') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(len(rows), 'rows;', sum(r['score'] is None for r in rows), 'without a score')


if __name__ == '__main__':
    main()
