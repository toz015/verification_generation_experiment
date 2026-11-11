"""Fixed 5-fold cross-fitting split for MATH-500 (made from metadata only, before grading).

MATH problems are independent items (no shared starter code as in QuantumKatas), so folds are
stratified rather than grouped: within each subject x level cell, items are shuffled with a fixed
seed and dealt round-robin, giving every fold the same subject and difficulty mix.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import sys

from vgx.common.storage import atomic_json
from vgx.math_trial import data


def make(seed=20261021, k=5):
    items, meta = data.load('math500')
    cells = defaultdict(list)
    for it in items:
        cells[(it['topic'], it['stratum'])].append(it['item_id'])
    rng, fold, offset = random.Random(seed), {}, 0
    for key in sorted(cells):
        ids = sorted(cells[key]); rng.shuffle(ids)
        for i, x in enumerate(ids):
            fold[x] = (offset+i) % k
        offset += len(ids)
    return {'dataset': meta, 'seed': seed, 'k': k, 'method': 'stratified by subject x level, round-robin within cell',
            'labels_used': False, 'item_fold': fold}


if __name__ == '__main__':
    atomic_json(Path(sys.argv[1]), make())
