"""Labels for every raw sample/candidate (no API calls): MATH with the pinned Omni-MATH-Rule grader,
GPQA against the dataset key, QuantumKatas with the pinned official checker in the macOS sandbox.

Each distinct answer/code is graded once and shared across generators. Outputs go to
results/raw_labels_20261026/{math500,gpqa,qkatas}_labels.json (keyed by item and normalized answer or
code sha256) plus the QuantumKatas official_results.jsonl store.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import uuid

from vgx.common.storage import atomic_json, read_jsonl
from vgx.math_trial import data
from vgx.quantum_katas import oracle
from vgx.quantum_katas.load import sha256_text
from vgx.quantum_katas.run_pilot import Run, official_checker

OUT = Path('results/raw_labels_20261026')
COLLECTED = ['results/qwen_raw_20261021', 'results/raw_gptoss_20261022', 'results/raw_sonnet55_20261026']
QK_RUN = 'results/quantum_katas_full_qwen3_20261019'


def rows(kind_prefix):
    out = []
    for root in COLLECTED:
        for f in Path(root).glob('**/collected.jsonl'):
            out += [r for r in read_jsonl(f, tolerate_tail=True) if r['kind'].startswith(kind_prefix) and r.get('status') == 'ok']
    return out


def grade_math(pairs, interpreter):
    """{item_id: {answer: {correct, status}}} for (item_id, answer) pairs, via the pinned grader."""
    refs = {it['item_id']: it['answer'] for it in data.load('math500')[0]}
    payload = ''.join(json.dumps({'item_id': f'{i}\t{a}', 'answer': a, 'reference': refs[i]}) + '\n' for i, a in sorted(pairs))
    grader_dir = str(Path(data.SOURCES['omnimath_rule']['path']).parent/'evaluation')
    proc = subprocess.run([interpreter, '-I', 'src/vgx/math_trial/grade_worker.py', grader_dir], input=payload,
                          capture_output=True, text=True, timeout=36000)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-2000:])
    out = {}
    for g in map(json.loads, proc.stdout.splitlines()):
        item, answer = g['item_id'].split('\t', 1)
        out.setdefault(item, {})[answer] = {'correct': g['correct'], 'status': g['status']}
    return out


def math_labels(interpreter):
    pairs = {(r['item_id'], r['answer']) for r in rows('math500_samples') if isinstance(r.get('answer'), str) and r['answer'].strip()}
    for d in ('qwen3', 'gptoss', 'sonnet55'):
        for r in json.loads(Path(f'results/math500_full_{d}_20261021/frozen_candidates.json').read_text())['rows']:
            if r.get('answer') is not None:
                pairs.add((r['item_id'], r['answer']))
    out = grade_math(pairs, interpreter)
    atomic_json(OUT/'math500_labels.json', out)
    return {'pairs': len(pairs), 'correct': sum(v['correct'] for d in out.values() for v in d.values())}


def gpqa_labels():
    b = json.loads(Path('results/gpqa_staged_generator_expansion_20261005/frozen_candidates.json').read_text())
    root = json.loads(Path('results/gpqa_gptoss_verification_20261020/protocol.json').read_text())['sources']['dataset_root']
    key = {r['item_id']: 'ABCD'[r['correct_index']] for r in read_jsonl(Path(root)/'labels.jsonl')}
    items = {r['item_id'] for r in b['groups']['qwen']}
    out = {i: key[i] for i in items}
    atomic_json(OUT/'gpqa_key.json', out)
    return {'items': len(out)}


@dataclass(frozen=True)
class _Cand:
    task_id: str
    candidate_id: str
    raw_code: str
    code_sha256: str

    @property
    def has_code(self):
        return True


def qk_labels():
    run = Run(QK_RUN)
    checker, tasks = official_checker(run)
    codes = {}
    for r in rows('qkatas_samples'):
        if r.get('code'):
            codes[(r['item_id'], sha256_text(r['code']))] = r['code']
    for d in ('qwen3', 'gptoss', 'sonnet55'):
        for c in Run(f'results/quantum_katas_full_{d}_20261019').candidates.all():
            if c.has_code:
                codes[(c.task_id, c.code_sha256)] = c.raw_code
    cands = [_Cand(t, f'raw|{t}|{h}', code, h) for (t, h), code in sorted(codes.items())]
    store = oracle.OfficialStore(OUT/'qkatas_official_results.jsonl')
    summary = oracle.score_candidates(cands, tasks, checker, store, operation_id=str(uuid.uuid4()))
    res = store.results()
    out = {}
    for c in cands:
        r = res.get(c.candidate_id)
        out.setdefault(c.task_id, {})[c.code_sha256] = None if r is None else int(r['label'])
    atomic_json(OUT/'qkatas_labels.json', out)
    return {**{k: v for k, v in summary.items() if k != 'errors'}, 'distinct_codes': len(cands)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('what', choices=['math', 'gpqa', 'qk'])
    p.add_argument('--interpreter', default='tmp/math_grader_venv_20261020/bin/python')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps(math_labels(a.interpreter) if a.what == 'math' else gpqa_labels() if a.what == 'gpqa' else qk_labels(), default=str)[:2000])


if __name__ == '__main__':
    main()
