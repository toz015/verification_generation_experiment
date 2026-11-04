"""Pre-registered grouped K-fold split of the 322 non-development QuantumKatas tasks.

Rules (fixed before any model output is collected):
  1. The 28 pilot tasks are development only and excluded.
  2. Base group = top-level kata directory; `tutorials/` and `examples/` are
     collections of unrelated topics, so they group by their second-level entry.
  3. Two groups are merged (union-find) when any cross-group task pair has
     token-set Jaccard >= THRESHOLD between reference solutions. Prompt overlap is
     recorded but not used: stubs share Qiskit imports and signatures, so short
     prompts overlap heavily without being the same task.
  4. Groups are assigned to K folds with a fixed seed: shuffle, stable-sort by
     size (largest first), then place each group in the currently smallest fold
     (lowest index on ties). Every candidate of a task inherits its task's fold.
Only upstream task text is read; no model output or label is involved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

from vgx.common.storage import atomic_json, digest, file_digest
from vgx.quantum_katas import load

THRESHOLD = 0.8
FOLDS = 5
SEED = 20261019
COLLECTIONS = ('tutorials', 'examples')
_TOKEN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]')


def base_group(task_id: str) -> str:
    parts = task_id.split('/')
    return '/'.join(parts[:2]) if parts[0] in COLLECTIONS and len(parts) >= 2 else parts[0]


def tokens(text: str) -> frozenset:
    lines = [line for line in text.splitlines() if not line.lstrip().startswith('#')]
    return frozenset(_TOKEN.findall('\n'.join(lines)))


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def build(source=load.DEFAULT_SOURCE, pilot=load.DEFAULT_PILOT, *, folds=FOLDS, seed=SEED, threshold=THRESHOLD) -> dict:
    load.verify_sources(source)
    dev, _ = load.load_pilot_ids(pilot)
    rows = [json.loads(line) for line in (Path(source)/'dataset/qiskit_quantumkatas.jsonl').read_text().splitlines()
            if line.strip()]
    rows = [r for r in rows if r['task_id'] not in set(dev)]
    groups = {r['task_id']: base_group(r['task_id']) for r in rows}
    parent = {g: g for g in set(groups.values())}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    features = {r['task_id']: (tokens(r['canonical_solution']), tokens(r['prompt'])) for r in rows}
    merges, prompt_only = [], []
    ids = sorted(features)
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            if groups[a] == groups[b]:
                continue
            sol = jaccard(features[a][0], features[b][0])
            prm = jaccard(features[a][1], features[b][1])
            if sol < threshold and prm >= threshold:
                prompt_only.append({'task_a': a, 'task_b': b, 'solution_jaccard': round(sol, 4),
                                    'prompt_jaccard': round(prm, 4)})
            if sol >= threshold:
                merges.append({'task_a': a, 'task_b': b, 'group_a': groups[a], 'group_b': groups[b],
                               'solution_jaccard': round(sol, 4), 'prompt_jaccard': round(prm, 4)})
                ra, rb = find(groups[a]), find(groups[b])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    final = {}
    for tid, g in groups.items():
        final.setdefault(find(g), []).append(tid)
    names = sorted(final)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda n: -len(final[n]))  # stable: seed only breaks size ties
    sizes, fold_of_group = [0]*folds, {}
    for name in names:
        fold = min(range(folds), key=lambda k: (sizes[k], k))
        fold_of_group[name] = fold
        sizes[fold] += len(final[name])
    group_members = {n: sorted(final[n]) for n in sorted(final)}
    task_fold = {tid: fold_of_group[n] for n, members in group_members.items() for tid in members}
    summary = [{'fold': k, 'tasks': sizes[k],
                'groups': sorted(n for n in fold_of_group if fold_of_group[n] == k),
                'top_level_katas': dict(sorted(Counter(t.split('/')[0] for t, f in task_fold.items() if f == k).items()))}
               for k in range(folds)]
    value = {'schema': 1, 'dataset': load.DATASET, 'revision': load.REVISION,
             'dataset_sha256': load.SOURCE_SHA256['dataset/qiskit_quantumkatas.jsonl'],
             'pilot_config_sha256': file_digest(pilot), 'excluded_development_task_ids': dev,
             'rules': {'base_group': 'top-level kata; tutorials/ and examples/ by second-level folder',
                       'merge': f'union groups when any cross-group task pair has reference-solution token-set '
                                f'Jaccard >= {threshold} (comment lines removed); prompt overlap recorded only',
                       'assignment': 'seeded shuffle, stable sort by group size descending, greedy to smallest fold',
                       'folds': folds, 'seed': seed, 'threshold': threshold,
                       'candidates': 'every candidate inherits its task fold; no task spans folds'},
             'tasks': len(task_fold), 'base_groups': len(set(groups.values())), 'final_groups': len(group_members),
             'merges': merges, 'prompt_only_similar_pairs_not_merged': prompt_only, 'groups': group_members, 'folds': summary, 'task_fold': dict(sorted(task_fold.items())),
             'labels_or_model_outputs_used': False}
    return {**value, 'split_id': digest(value)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit('split output exists; refusing to overwrite a pre-registered split')
    value = build()
    atomic_json(args.output, value)
    print(json.dumps({'split_id': value['split_id'], 'tasks': value['tasks'], 'base_groups': value['base_groups'],
                      'final_groups': value['final_groups'], 'merges': len(value['merges']),
                      'fold_sizes': [f['tasks'] for f in value['folds']]}, indent=2))


if __name__ == '__main__':
    main()
