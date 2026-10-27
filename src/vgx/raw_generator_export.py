"""Tidy export of every raw generator sample (answer, stated confidence, offline label) for sharing.

One gzip JSONL per dataset under reports/raw_generator_export_20261026/, one row per
(generator, item_id, sample): the generator's final answer (GPQA letter, MATH answer string or
QuantumKatas code), its stated probability of being correct (p_correct, as written by the model),
and the offline label (correct: true/false, null while ungraded). GPQA per-option probabilities
(the model's probability for each of A-D, plus Sonnet's "probability each option is incorrect")
go in gpqa_option_probs.jsonl.gz. No question text or prompts are written. Nothing is re-requested.

gpt-oss often wraps its JSON reply ({"final": {...}}, {"final": "<json>"}, {"final": "8"},
{"name": "final", "content": "<json>"}). When the collector's parser found no answer, the answer is
recovered here from the stored response by unwrapping those forms (answer_parse="recovered");
recovered MATH answers are graded with the same pinned grader. Rows parsed directly have
answer_parse="direct".
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
import re

from vgx.common.api import RequestCache
from vgx.common.storage import read_jsonl
from vgx.quantum_katas.load import sha256_text
from vgx.raw_labels import grade_math

GENERATORS = {'qwen3': 'results/qwen_raw_20261021', 'gptoss': 'results/raw_gptoss_20261022',
              'sonnet55': 'results/raw_sonnet55_20261026', 'gemini38': 'results/raw_gemini38_20261027'}
LABELS = Path('results/raw_labels_20261026')
OUT = Path('reports/raw_generator_export_20261026')
GPQA_BUNDLE = 'results/gpqa_staged_generator_expansion_20261005/frozen_candidates.json'


def collected(root):
    out = []
    for f in sorted(Path(root).glob('**/collected.jsonl')):
        out += [dict(r, _dir=f.parent) for r in read_jsonl(f, tolerate_tail=True)]
    return out


def response(row, caches={}):
    cache = caches.setdefault(row['_dir'], RequestCache(row['_dir']/'cache'))
    recs = [r for r in cache.log(row['request_key']).records() if r.get('response') is not None and not r.get('error')]
    return recs[0]['response'] if recs else None


def _unwrap(value):
    for _ in range(4):
        if isinstance(value, dict) and value.get('name') == 'final' and 'content' in value:
            value = value['content']
        elif isinstance(value, dict) and 'final' in value:
            inner, rest = value['final'], {k: v for k, v in value.items() if k != 'final'}
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except ValueError:
                    pass
            value = {**rest, **inner} if isinstance(inner, dict) else {**rest, 'answer': inner}
        elif isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return {}
        else:
            break
    return value if isinstance(value, dict) else {}


def recover(text, ds):
    """Best-effort (answer or code, p_correct) from a reply the collector could not parse."""
    try:
        value = _unwrap(json.loads(text))
    except ValueError:
        value = {}
    p = value.get('p_correct')
    if p is None and (m := re.search(r'"p_correct"\s*:\s*([0-9.]+)', text)):
        p = float(m.group(1))
    if ds == 'qkatas':
        code = value.get('code')
        return (code if isinstance(code, str) and code.strip() else None), p
    ans = value.get('answer')
    if ans is None and ds == 'gpqa' and (m := re.search(r':\s*"([ABCD])"', text)):
        ans = m.group(1)
    if ds == 'gpqa' and ans not in tuple('ABCD'):
        ans = None
    return (str(ans) if ans is not None and str(ans).strip() else None), p


def qk_labels():
    p = LABELS/'qkatas_labels.json'
    if p.exists():
        return {(t, h): v for t, d in json.loads(p.read_text()).items() for h, v in d.items()}
    out = {}
    for r in read_jsonl(LABELS/'qkatas_official_results.jsonl', tolerate_tail=True):
        if r.get('event') != 'intent' and 'label' in r:
            out[(r['task_id'], r['code_sha256'])] = int(r['label'])
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--interpreter', default='tmp/math_grader_venv_20261020/bin/python')
    args = a.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gpqa_key = json.loads((LABELS/'gpqa_key.json').read_text())
    # GPQA split of the main experiments: 249 calibration (Main minus Diamond) + 196 evaluation (Diamond).
    gpqa_part = {r['item_id']: r['partition'] for r in json.loads(Path(GPQA_BUNDLE).read_text())['groups']['qwen']}
    math = json.loads((LABELS/'math500_labels.json').read_text())
    qk = qk_labels()
    rows = {'gpqa': {}, 'math500': {}, 'qkatas': {}}; options = []; summary = Counter()
    for gen, root in GENERATORS.items():
        for r in collected(root):
            kind = r['kind']
            if r.get('status') != 'ok':
                continue
            if kind.startswith('gpqa_option_probs'):
                probs, how = r.get('probs'), 'direct'
                if not probs or all(v is None for v in probs.values()):
                    try:
                        v = _unwrap(json.loads(response(r) or 'null'))
                        probs, how = {l: v.get(l) for l in 'ABCD'}, 'recovered'
                    except ValueError:
                        probs = None
                options.append({'generator': gen, 'item_id': r['item_id'], 'partition': gpqa_part[r['item_id']],
                                'question': 'incorrect' if kind.endswith('incorrect') else 'correct',
                                'probs': probs, 'answer_parse': how, 'correct_letter': gpqa_key.get(r['item_id'])})
                continue
            ds = kind.split('_', 1)[0]
            if ds not in rows or '_samples' not in kind:
                continue
            key = (gen, r['item_id'], int(r['sample']))
            # Prefer the no-logprob variant (the complete set); keep the first ok row otherwise.
            if key in rows[ds] and not kind.endswith('_nolp'):
                continue
            row = {'generator': gen, 'item_id': r['item_id'], 'sample': int(r['sample']),
                   'p_correct': r.get('p_correct'), 'finish_reason': r.get('finish_reason'), 'answer_parse': 'direct'}
            broken = r.get('failure') == 'unexpected_fields' if ds == 'qkatas' else r.get('answer') is None
            if broken and (text := response(r)):
                got, p = recover(text, ds)
                if got is not None:
                    r = dict(r, **({'code': got, 'failure': None} if ds == 'qkatas' else {'answer': got}))
                    row.update(p_correct=p, answer_parse='recovered')
            if ds == 'gpqa':
                row['partition'] = gpqa_part[r['item_id']]
                row['answer'] = r.get('answer')
                row['correct'] = None if row['answer'] is None else row['answer'] == gpqa_key[r['item_id']]
            elif ds == 'math500':
                row['answer'] = r.get('answer')
                lab = math.get(r['item_id'], {}).get(row['answer']) if isinstance(row['answer'], str) else None
                row['correct'] = False if not (isinstance(row['answer'], str) and row['answer'].strip()) \
                    else (lab or {}).get('correct')
            else:
                row['code'] = r.get('code'); row['parse_failure'] = r.get('failure')
                if r.get('code'):
                    row['code_sha256'] = sha256_text(r['code'])
                    v = qk.get((r['item_id'], row['code_sha256']))
                    row['correct'] = None if v is None else bool(v)
                else:
                    row['correct'] = False
            rows[ds][key] = row
    missing = {(row['item_id'], row['answer']) for row in rows['math500'].values()
               if row['correct'] is None and isinstance(row['answer'], str)}
    if missing:
        graded = grade_math(missing, args.interpreter)
        for row in rows['math500'].values():
            if row['correct'] is None and isinstance(row['answer'], str):
                row['correct'] = graded.get(row['item_id'], {}).get(row['answer'], {}).get('correct')
    for ds, d in rows.items():
        with gzip.open(OUT/f'{ds}.jsonl.gz', 'wt') as f:
            for k in sorted(d):
                f.write(json.dumps(d[k], ensure_ascii=False) + '\n')
        for (gen, *_), row in d.items():
            summary[f'{ds}|{gen}|rows'] += 1
            summary[f'{ds}|{gen}|labelled'] += row['correct'] is not None
            summary[f'{ds}|{gen}|correct'] += bool(row['correct'])
            summary[f'{ds}|{gen}|recovered'] += row['answer_parse'] == 'recovered'
    with gzip.open(OUT/'gpqa_option_probs.jsonl.gz', 'wt') as f:
        for r in sorted(options, key=lambda r: (r['generator'], r['item_id'], r['question'])):
            f.write(json.dumps(r) + '\n')
    table = {}
    for k, v in summary.items():
        ds, gen, what = k.split('|')
        table.setdefault(ds, {}).setdefault(gen, {})[what] = v
    for ds in table:
        for gen in table[ds]:
            items = {i for g, i, _ in rows[ds] if g == gen}
            table[ds][gen]['items'] = len(items)
    (OUT/'summary.json').write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()
