"""Grade frozen math-trial answers with the pinned official grader and summarize accuracy.

Reference answers are read only here, after frozen_candidates.json is sealed. Missing,
unparsable or service-failed replies count as incorrect. The grader runs in its own venv
(config grader.interpreter) via grade_worker.py.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import numpy as np

from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa.sequential import seal, unseal
from vgx.math_trial import data

WORKER = Path(__file__).with_name('grade_worker.py')


def grade(out, interpreter):
    out = Path(out)
    plan = json.loads((out/'plan.json').read_text()); unseal(plan, 'plan_id')
    bundle = json.loads((out/'frozen_candidates.json').read_text()); unseal(bundle, 'bundle_id')
    refs = {}
    for name in plan['sample']:
        items, meta = data.load(name)
        if meta['file_sha256'] != plan['sources'][name]['file_sha256']:
            raise ValueError('dataset file changed since prepare')
        refs.update({it['item_id']: it['answer'] for it in items})
    todo = [r for r in bundle['rows'] if r.get('answer') is not None]
    payload = ''.join(json.dumps({'item_id': r['item_id'], 'answer': r['answer'], 'reference': refs[r['item_id']]})+'\n' for r in todo)
    grader_dir = str(Path(data.SOURCES['omnimath_rule']['path']).parent/'evaluation')
    proc = subprocess.run([interpreter, '-I', str(WORKER), grader_dir], input=payload, capture_output=True, text=True,
                          timeout=60+10*len(todo))
    if proc.returncode:
        raise RuntimeError(proc.stderr[-2000:])
    graded = {g['item_id']: g for g in map(json.loads, proc.stdout.splitlines())}
    meta = {it['item_id']: it for rows in plan['sample'].values() for it in rows}
    rows = []
    for r in bundle['rows']:
        g = graded.get(r['item_id'], {'correct': False, 'status': 'no_answer'})
        rows.append({'item_id': r['item_id'], 'dataset': r['item_id'].split(':')[0], 'stratum': meta[r['item_id']]['stratum'],
                     'topic': meta[r['item_id']]['topic'], 'p_correct': r.get('p_correct'), 'failure': r.get('failure'),
                     'correct': bool(g['correct']), 'grade_status': g['status'], 'pred': g.get('pred'), 'gt': g.get('gt')})
    summary = {}
    for name in plan['sample']:
        rs = [r for r in rows if r['dataset'] == name]
        conf = [r for r in rs if r['p_correct'] is not None]
        y = np.array([r['correct'] for r in conf], float); p = np.array([r['p_correct'] for r in conf])
        by = defaultdict(list)
        for r in rs:
            by[r['stratum']].append(r['correct'])
        summary[name] = {'n': len(rs), 'correct': sum(r['correct'] for r in rs), 'accuracy': float(np.mean([r['correct'] for r in rs])),
                         'format_failures': sum(r['failure'] is not None for r in rs),
                         'grade_status': {s: sum(r['grade_status'] == s for r in rs) for s in sorted({r['grade_status'] for r in rs})},
                         'mean_p_correct': float(p.mean()) if len(p) else None, 'brier': float(((p-y)**2).mean()) if len(p) else None,
                         'share_p_ge_0.95': float((p >= .95).mean()) if len(p) else None,
                         'accuracy_when_p_ge_0.95': float(y[p >= .95].mean()) if (p >= .95).any() else None,
                         'accuracy_by_stratum': {k: [int(sum(v)), len(v)] for k, v in sorted(by.items())}}
    result = seal({'plan_id': plan['plan_id'], 'bundle_id': bundle['bundle_id'], 'grader_commit': data.OMNI_COMMIT,
                   'grader_worker_sha256': file_digest(WORKER), 'grade_sha256': file_digest(__file__),
                   'summary': summary, 'rows': rows}, 'grade_id')
    atomic_json(out/'graded.json', result)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--interpreter', required=True)
    a = p.parse_args()
    print(json.dumps(grade(a.out, a.interpreter), indent=2))


if __name__ == '__main__':
    main()
