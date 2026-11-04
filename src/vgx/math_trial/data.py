"""Pinned MATH-500 and Omni-MATH-Rule problems and the fixed stratified trial sample.

Sources (fixed 2026-10-20 before any model call):
- MATH-500: Hugging Face HuggingFaceH4/MATH-500 test.jsonl at 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be.
- Omni-MATH-Rule: github.com/KbsdJames/omni-math-rule omni_math_rule.jsonl at
  4793415ef37d31c9cdb4e5b82dbe172f76f8cf08, the authors' rule-gradable subset of Omni-MATH.
The sample is drawn from problem metadata only (level / difficulty), never from answers.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random

from vgx.common.storage import file_digest

OMNI_COMMIT = '4793415ef37d31c9cdb4e5b82dbe172f76f8cf08'
SOURCES = {
    'math500': {'path': 'data/math_trial_20261020/math500_test.jsonl',
                'revision': 'HuggingFaceH4/MATH-500@6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be',
                'sha256': None},
    'omnimath_rule': {'path': f'third_party/omni_math_rule_{OMNI_COMMIT}/omni_math_rule.jsonl',
                      'revision': f'KbsdJames/omni-math-rule@{OMNI_COMMIT}', 'sha256': None},
}


def load(name, root='.'):
    src = SOURCES[name]
    path = Path(root)/src['path']
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = []
    for i, r in enumerate(rows):
        if name == 'math500':
            out.append({'item_id': f'math500:{r["unique_id"]}', 'problem': r['problem'], 'answer': r['answer'],
                        'stratum': f'level{r["level"]}', 'topic': r['subject']})
        else:
            domain = r['domain'][0].split(' -> ')[1] if r.get('domain') else 'none'
            out.append({'item_id': f'omnimath_rule:{i}', 'problem': r['problem'], 'answer': r['answer'],
                        'stratum': f'difficulty{int(math.floor(float(r["difficulty"])))}', 'topic': domain,
                        'source': r.get('source')})
    return out, {'revision': src['revision'], 'file_sha256': file_digest(path), 'n': len(out)}


def stratified_sample(items, n, seed):
    """Proportional allocation over strata (largest remainder), then a seeded draw inside each stratum."""
    groups = defaultdict(list)
    for it in items:
        groups[it['stratum']].append(it)
    keys = sorted(groups)
    quota = {k: n*len(groups[k])/len(items) for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(quota[k]-alloc[k]), k))[:n-sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(seed)
    chosen = []
    for k in keys:
        chosen += rng.sample(sorted(groups[k], key=lambda x: x['item_id']), alloc[k])
    return sorted(chosen, key=lambda x: x['item_id']), alloc
