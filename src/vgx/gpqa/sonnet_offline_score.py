"""User-authorized offline scoring of frozen Gemini/Qwen/Sonnet raw forecasts.

No collection, re-parsing, repairs, fitting, or confidence transformations.
Writes aggregate results only; never writes question text or per-item outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import fmean

from vgx.common.storage import file_digest, read_jsonl, digest
from vgx.gpqa import sonnet_expansion as expansion
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.fresh import inputs
from vgx.gpqa.score import correctness_outcome
from vgx.gpqa.sequential import seal, unseal


METHODS = {
    'accuracy_all': 'Correct valid answers / every assigned question; unparsed answers are non-correct. A valid answer is scored even if its confidence is missing.',
    'accuracy_valid_answers': 'Correct valid answers / valid answers only.',
    'confidence_and_brier': 'Use only frozen candidates with both a valid answer and a finite numeric raw p_correct in [0,1]. No imputation, clipping or calibration.',
    'mean_confidence_if_wrong': 'Mean raw confidence among valid answer/confidence pairs whose answer is wrong; excludes unparsed answers.',
    'brier': 'Mean (raw p_correct - 1[answer correct])^2, one binary correctness forecast per item, with no factor of 2.',
    'common_valid': 'For each partition, use the intersection of valid answer/confidence item IDs across all three models; membership fixed without labels.',
    'interpretation': 'Descriptive comparison of historical frozen model-plus-generation protocols. Qwen used two-step generation; settings and serving dates differ.',
}


def mean(values):
    return fmean(values) if values else None


def usable(c):
    return c['answer'] in ('A', 'B', 'C', 'D') and c['p_correct'] is not None


def summarize(rows, labels):
    y = [correctness_outcome(c['answer'], labels[c['item_id']]) for c in rows]
    pairs = [(c['p_correct'], t) for c, t in zip(rows, y) if usable(c)]
    wrong = [p for p, t in pairs if t == 0]
    valid_answers = sum(c['answer'] is not None for c in rows)
    return {'n': len(rows), 'correct': sum(y), 'accuracy_all': mean(y),
        'valid_answer_n': valid_answers, 'invalid_answer_n': len(rows)-valid_answers,
        'accuracy_valid_answers': sum(y)/valid_answers if valid_answers else None,
        'numeric_confidence_n': sum(c['p_correct'] is not None for c in rows),
        'confidence_n': len(pairs), 'brier_n': len(pairs),
        'mean_confidence': mean([p for p, _ in pairs]),
        'wrong_with_confidence_n': len(wrong), 'mean_confidence_if_wrong': mean(wrong),
        'mean_confidence_if_correct': mean([p for p, t in pairs if t]),
        'raw_brier': mean([(p-t)**2 for p, t in pairs]),
        'failures': dict(Counter(c['generator_failure'] for c in rows if c['generator_failure']))}


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('existing scoring output differs: '+str(path))
    else:
        with path.open('xb') as f:
            f.write(data)


def run(root, out):
    # Validate all prediction artifacts and determine comparison cohorts before
    # opening labels. Imported helpers have no network fallback on this path.
    plan, cfg, _ = expansion.load(root)
    manifest, items = inputs(Path(cfg['dataset_root']))
    gm, gemini = load_candidates(Path(cfg['baseline_bundle']))
    qpath = Path(cfg['qwen_baseline']); qwen = json.loads(qpath.read_text()); unseal(qwen, 'bundle_id')
    spath = expansion.output(root)/'frozen_candidates.json'
    sonnet = json.loads(spath.read_text()); unseal(sonnet, 'bundle_id')
    completion = json.loads((expansion.output(root)/'completion_report.json').read_text())
    audit_path = Path('reports/gpqa_sonnet55_expansion_20261015/read_only_audit.json')
    audit = json.loads(audit_path.read_text())
    if (sonnet['expansion_id'] != plan['expansion_id'] or sonnet['dataset_id'] != manifest['dataset_id']
        or sonnet['bundle_id'] != completion['bundle_id']
        or file_digest(spath) != audit['frozen_candidates_file_sha256']):
        raise ValueError('Sonnet frozen bundle identity mismatch')
    groups = {'gemini': [asdict(c) for c in gemini], 'qwen': qwen['groups']['qwen'],
              'sonnet_5_5': sonnet['groups']['sonnet_5_5']}
    by_id = {i.item_id: i for i in items}
    for name, rows in groups.items():
        if len(rows) != len(items) or {c['item_id'] for c in rows} != set(by_id):
            raise ValueError(name+' membership differs')
        for c in rows:
            i = by_id[c['item_id']]
            if (c['partition'], c['subject'], c['question'], tuple(c['choices'])) != (
                i.partition, i.subject, i.question, i.choices):
                raise ValueError(name+' frozen input differs')
            if c['answer'] is not None and c['answer'] not in ('A', 'B', 'C', 'D'):
                raise ValueError(name+' malformed stored answer')
            p = c['p_correct']
            if p is not None and (type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1):
                raise ValueError(name+' malformed stored confidence')
    partitions = {part: [i.item_id for i in items if i.partition == part]
                  for part in ('calibration', 'evaluation')}
    if {p: len(ids) for p, ids in partitions.items()} != {'calibration': 249, 'evaluation': 196}:
        raise ValueError('partition counts differ')
    if set(partitions['calibration']) & set(partitions['evaluation']):
        raise ValueError('development/evaluation overlap')
    common = {part: set.intersection(*[{c['item_id'] for c in rows if c['partition'] == part and usable(c)}
                                      for rows in groups.values()]) for part in partitions}
    paths = [Path(__file__), Path(cfg['dataset_root'])/'dataset_manifest.json',
        Path(cfg['dataset_root'])/'inputs.jsonl', Path(cfg['dataset_root'])/'labels.jsonl',
        Path(cfg['baseline_bundle'])/'bundle_manifest.json', Path(cfg['baseline_bundle'])/'candidates.jsonl',
        qpath, spath, expansion.output(root)/'plan.json', expansion.output(root)/'completion_report.json',
        root/'parser_v2/comparison.json', root/'cache/budget.json', audit_path]
    hashes = {str(p): file_digest(p) for p in paths}
    lpath = Path(cfg['dataset_root'])/'labels.jsonl'
    if hashes[str(lpath)] != manifest['files']['labels.jsonl']:
        raise ValueError('answer key checksum differs')
    records = read_jsonl(lpath)
    if len(records) != len(items) or {r['item_id'] for r in records} != set(by_id):
        raise ValueError('answer key membership differs')
    for r in records:
        if (r['partition'] != by_id[r['item_id']].partition or type(r['correct_index']) is not int
                or r['correct_index'] not in range(4)):
            raise ValueError('answer key partition/index invalid')
    labels = {r['item_id']: r['correct_index'] for r in records}
    full = {part: {g: summarize([c for c in rows if c['partition'] == part], labels)
                   for g, rows in groups.items()} for part in partitions}
    matched = {part: {g: summarize([c for c in rows if c['item_id'] in common[part]], labels)
                      for g, rows in groups.items()} for part in partitions}
    # Cross-check the already published fifty-item screen without changing it.
    pilot_ids = set(manifest['pilot_calibration_ids'])
    pilot = json.loads((root/'parser_v2/comparison.json').read_text())
    old_names = {'gemini': 'gemini_baseline', 'qwen': 'qwen_baseline', 'sonnet_5_5': 'sonnet_5_5'}
    for name, rows in groups.items():
        fresh = summarize([c for c in rows if c['item_id'] in pilot_ids], labels)
        old = pilot['generators'][old_names[name]]
        if fresh['correct'] != old['correct'] or fresh['brier_n'] != old['confidence_n']:
            raise ValueError('pilot counts do not reproduce')
        for a, b in [(fresh['mean_confidence'], old['mean_confidence']),
                     (fresh['mean_confidence_if_wrong'], old['mean_confidence_if_incorrect']),
                     (fresh['raw_brier'], old['raw_metrics']['brier'])]:
            if not math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError('pilot metric does not reproduce')
    if any(file_digest(p) != sha for p, sha in hashes.items()):
        raise ValueError('an input changed during offline scoring')
    expansion.load(root)  # Preserve prior source seals, replies, and reports.
    result = seal({'schema': 1, 'analysis': 'frozen raw generator correctness scoring',
        'dataset_id': manifest['dataset_id'], 'expansion_id': plan['expansion_id'],
        'bundle_ids': {'gemini': gm['bundle_id'], 'qwen': qwen['bundle_id'], 'sonnet_5_5': sonnet['bundle_id']},
        'models': {g: sorted({c['generator_model'] for c in rows}) for g, rows in groups.items()},
        'methods': METHODS, 'all_assigned_questions': full, 'common_valid_questions': matched,
        'common_membership_sha256': {p: digest(sorted(ids)) for p, ids in common.items()},
        'source_sha256': hashes, 'validation': {'identical_question_and_choice_order': True,
            'development_evaluation_overlap_n': 0, 'source_and_bundle_checksums_valid': True,
            'original_50_pilot_metrics_reproduced': True, 'input_files_unchanged': True},
        'confidence_calibrated': False, 'refit_performed': False, 'new_api_requests': 0,
        'candidate_or_parser_changes': False, 'answer_keys_read': True,
        'per_item_outcomes_exported': False, 'unit_tests_run': False}, 'result_id')
    write_new(out/'result.json', (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/gpqa_sonnet55_20261015'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(args.root, args.output)
    print(json.dumps({k: result[k] for k in ('result_id', 'all_assigned_questions',
        'common_valid_questions', 'validation', 'confidence_calibrated', 'new_api_requests')}, indent=2))
