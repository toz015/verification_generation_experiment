"""Export cached GPQA outputs into learner / evaluator files for the online replay (no API calls).

Sources (committed, no question text):
  reports/raw_generator_export_20261026/gpqa.jsonl.gz      10 samples per (generator, item): answer letter,
                                                           stated p_correct, offline label vs the key
  reports/raw_verifier_export_20261027/gpqa_verifier_scores.jsonl.gz
                                                           each of the 5 verifiers' probability for every
                                                           option A-D of all 445 items
  reports/gpqa_gptoss_verification_20261020/policy.json    mean development cost per verifier call (USD)

A generator sample's verifier panel is the five verifiers' scores for the option it chose.
Writes, per generator and sample s:
  learner_<gen>_s<s>.jsonl.gz  task id, fold, group (= item; GPQA has no task groups), raw r = p_correct,
                               has_code (= an answer letter was returned), signals {v: score, failure,
                               estimated_usd}. No label and no label-availability flag.
  labels_<gen>_s<s>.jsonl.gz   offline label, for the evaluator / D_init / arm D only.
Folds: 5 folds over the 445 items from a fixed seeded permutation (labels not used), shared by all
generators and samples. The calibration/evaluation partition of the main GPQA study is ignored here.

  PYTHONPATH=src python scripts/export_gpqa_online_rows_20261110.py --out reports/gpqa_online_20261110/rows
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from vgx.common.storage import atomic_json, file_digest

GEN = Path('reports/raw_generator_export_20261026/gpqa.jsonl.gz')
VER = Path('reports/raw_verifier_export_20261027/gpqa_verifier_scores.jsonl.gz')
COST = Path('reports/gpqa_gptoss_verification_20261020/policy.json')
VERIFIERS = ('gemini_flash', 'gemini_lite', 'mistral_small', 'mistral_medium', 'claude_haiku')
FOLD_SEED = 20261110


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    cost = json.loads(COST.read_text())['expected_cost_usd']
    usd = {v: float(cost[f'{v}:probability']) for v in VERIFIERS}
    scores = {}
    with gzip.open(VER, 'rt') as f:
        for line in f:
            r = json.loads(line)
            scores[(r['item_id'], r['candidate'], r['verifier'])] = (r['score'], r['failure'])
    with gzip.open(GEN, 'rt') as f:
        gen_rows = [json.loads(line) for line in f]
    items = sorted({r['item_id'] for r in gen_rows})
    perm = np.random.default_rng(FOLD_SEED).permutation(len(items))
    fold = {items[i]: int(k % 5) for k, i in enumerate(perm)}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'sources': {str(p): file_digest(p) for p in (GEN, VER, COST)}, 'verifiers': list(VERIFIERS),
                'cost_usd_per_call': usd, 'fold_seed': FOLD_SEED, 'folds': {str(k): sum(v == k for v in fold.values())
                                                                          for k in range(5)}, 'files': {}}
    by = {}
    for r in gen_rows:
        by.setdefault((r['generator'], r['sample']), []).append(r)
    for (gen, s), rows in sorted(by.items()):
        name = f'{gen}_s{s}'
        learner, labels = args.out/f'learner_{name}.jsonl.gz', args.out/f'labels_{name}.jsonl.gz'
        n_ok = n_correct = 0
        with gzip.GzipFile(learner, 'wb', mtime=0) as fl, gzip.GzipFile(labels, 'wb', mtime=0) as fy:
            for r in sorted(rows, key=lambda x: x['item_id']):
                ans = r['answer']
                sig = {}
                for v in VERIFIERS:
                    sc, fail = scores.get((r['item_id'], ans, v), (None, 'no_answer')) if ans else (None, 'no_answer')
                    sig[v] = {'score': sc, 'failure': fail, 'estimated_usd': usd[v] if ans else 0.}
                fl.write((json.dumps({'task_id': r['item_id'], 'fold': fold[r['item_id']], 'group': r['item_id'],
                                      'raw_r': r['p_correct'], 'has_code': ans is not None, 'signals': sig},
                                     sort_keys=True) + '\n').encode())
                lab = r['correct'] is not None
                fy.write((json.dumps({'task_id': r['item_id'], 'labeled': lab,
                                      'outcome': int(r['correct']) if lab else None}, sort_keys=True) + '\n').encode())
                n_ok += ans is not None
                n_correct += bool(r['correct'])
        manifest['files'][name] = {'learner_sha256': file_digest(learner), 'labels_sha256': file_digest(labels),
                                   'tasks': len(rows), 'answered': n_ok, 'correct': n_correct}
    atomic_json(args.out/'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=1))
    for k in ('qwen3_s0', 'gptoss_s0', 'sonnet55_s0'):
        print(k, manifest['files'][k])


if __name__ == '__main__':
    main()
