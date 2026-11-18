"""Export the cached QuantumKatas runs into separate learner and evaluator files for the online replay.

Reads the three full runs through vgx.quantum_katas.crossfit.load_rows (the same loader the
grouped cross-fit used), then writes
  learner_<gen>.jsonl.gz   public fields only: task id, fold, group, raw r, code present, and the
                           cached verifier outputs (score, failure, estimated USD). No label and no
                           label-availability flag.
  labels_<gen>.jsonl.gz    official checker label per task, for the evaluator only.
No model calls are made.

  PYTHONPATH=src python scripts/export_quantum_katas_online_rows_20261109.py \
      --results-root results --out reports/quantum_katas_online_20261109/rows
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from vgx.common.storage import atomic_json, file_digest
from vgx.quantum_katas import crossfit

CONFIG = 'configs/quantum_katas_crossfit_3gen_20261019.json'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-root', type=Path, default=Path('results'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = json.loads(Path(CONFIG).read_text())
    split_value = json.loads(Path(cfg['split']).read_text())
    split = {'value': split_value, 'sha256': file_digest(cfg['split'])}
    group_of = {t: g for g, ts in split_value['groups'].items() for t in ts}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'config': CONFIG, 'config_sha256': file_digest(CONFIG), 'split_sha256': split['sha256'],
                'verifiers': cfg['verifiers'], 'files': {}}
    for gen, root in cfg['runs'].items():
        root = args.results_root / Path(root).relative_to('results')
        rows = crossfit.load_rows(root, split, cfg['verifiers'])
        learner = args.out / f'learner_{gen}.jsonl.gz'
        labels = args.out / f'labels_{gen}.jsonl.gz'
        with gzip.open(learner, 'wt') as fl, gzip.open(labels, 'wt') as fy:
            for r in rows:
                fl.write(json.dumps({'task_id': r['task_id'], 'fold': r['fold'], 'group': group_of[r['task_id']],
                                     'raw_r': r['prior'], 'has_code': r['has_code'],
                                     'signals': r['signals']}, sort_keys=True) + '\n')
                fy.write(json.dumps({'task_id': r['task_id'], 'labeled': r['labeled'],
                                     'outcome': int(r['outcome']) if r['labeled'] else None}, sort_keys=True) + '\n')
        manifest['files'][gen] = {'run_root': cfg['runs'][gen], 'learner': learner.name, 'labels': labels.name,
                                  'learner_sha256': file_digest(learner), 'labels_sha256': file_digest(labels),
                                  'tasks': len(rows)}
    atomic_json(args.out / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
