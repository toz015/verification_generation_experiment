"""Offline comparison of four frozen generators after all 445 replies arrive.

Prepare is label-free. Score requires a complete, audited GPT-OSS expansion,
freezes all prediction/cohort identities before opening the existing labels,
and exports aggregate metrics only. No API calls, fitting or calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from vgx.common.storage import digest, file_digest, read_jsonl
from vgx.gpqa import gptoss_retry96_expansion as expansion
from vgx.gpqa import gptoss_continuation as base
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.fresh import inputs
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa import sonnet_offline_score as scoring

ROOT = expansion.ROOT
DIRECTORY = 'offline_score_445_retry96_20261018'
REPORT = Path('reports/gpqa_gptoss_offline_score_retry96_20261018')
NAMES = {'gpt_oss_120b': 'GPT-OSS-120B', 'gemini': 'Gemini',
         'qwen': 'Qwen（两步）', 'sonnet_5_5': 'Sonnet 5.5'}
METHODS = {**scoring.METHODS,
    'common_valid': 'For each partition, the intersection of valid answer/confidence item IDs across all four models; membership fixed before opening labels.',
    'interpretation': 'Historical model-plus-generation protocols differ: Qwen two-step, Sonnet adaptive high effort, GPT-OSS medium reasoning. No causal attribution to the base model alone.',
}


def prediction_sources(root):
    plan, parent, _, _, _ = expansion.load(root)
    cfg = parent['config']
    manifest, items = inputs(Path(cfg['dataset_root']))
    gm, gemini = load_candidates(Path(cfg['baseline_bundle']))
    qpath = Path(cfg['qwen_baseline']); qwen = json.loads(qpath.read_text()); unseal(qwen, 'bundle_id')
    sonnet = json.loads(expansion.SONNET.read_text()); unseal(sonnet, 'bundle_id')
    groups = {'gemini': [asdict(c) for c in gemini], 'qwen': qwen['groups']['qwen'],
              'sonnet_5_5': sonnet['groups']['sonnet_5_5']}
    paths = [Path(__file__), Path(scoring.__file__), Path(cfg['dataset_root'])/'dataset_manifest.json',
        Path(cfg['dataset_root'])/'inputs.jsonl', Path(cfg['baseline_bundle'])/'bundle_manifest.json',
        Path(cfg['baseline_bundle'])/'candidates.jsonl', qpath, expansion.SONNET,
        root/expansion.DIRECTORY/'plan.json', root/expansion.pilot.DIRECTORY/'result.json']
    ids = {'gemini': gm['bundle_id'], 'qwen': qwen['bundle_id'], 'sonnet_5_5': sonnet['bundle_id']}
    return plan, cfg, manifest, items, groups, ids, paths


def prepare(root):
    expansion_plan, _, manifest, items, _, ids, paths = prediction_sources(root)
    previous_path = root/'offline_score_445_retry65_20261018/plan.json'
    previous = json.loads(previous_path.read_text()); unseal(previous, 'analysis_id')
    if (previous['methods'] != METHODS or previous['baseline_bundle_ids'] != ids
            or previous['dataset_id'] != manifest['dataset_id']
            or previous['ordered_input_hash'] != digest([asdict(i) for i in items])):
        raise ValueError('previous frozen analysis methods or inputs differ')
    paths += [previous_path]
    plan = seal({'previous_analysis_id': previous['analysis_id'], 'authorization': '2026-10-18 user renewed authorization after retry96 for offline comparison of GPT-OSS, Gemini, Qwen and Sonnet on all445, development and Diamond separately; raw confidence unchanged.',
        'expansion_id': expansion_plan['expansion_id'], 'dataset_id': manifest['dataset_id'],
        'partition_counts': {'calibration': 249, 'evaluation': 196}, 'baseline_bundle_ids': ids,
        'ordered_input_hash': digest([asdict(i) for i in items]), 'methods': METHODS,
        'source_sha256': {str(p): file_digest(p) for p in paths},
        'completion_gate': 'All445 received and frozen, collection status complete, expansion audit passes; malformed replies retained.',
        'confidence_calibrated': False, 'new_api_requests': 0,
        'answer_keys_read_during_prepare': False}, 'analysis_id')
    base.immutable(root/DIRECTORY/'plan.json', plan)
    return {'stage': 'prepared', 'analysis_id': plan['analysis_id'],
            'completion_required': 445, 'answer_keys_read': False, 'new_api_requests': 0}


def validate_groups(groups, items):
    by_id = {i.item_id: i for i in items}
    for name, rows in groups.items():
        if len(rows) != 445 or {c['item_id'] for c in rows} != set(by_id):
            raise ValueError(name+' membership differs')
        for c in rows:
            i = by_id[c['item_id']]
            if (c['partition'], c['subject'], c['question'], tuple(c['choices'])) != (
                    i.partition, i.subject, i.question, tuple(i.choices)):
                raise ValueError(name+' question or choice order differs')
            if c['answer'] is not None and c['answer'] not in ('A', 'B', 'C', 'D'):
                raise ValueError(name+' malformed stored answer')
            p = c['p_correct']
            if p is not None and (type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1):
                raise ValueError(name+' malformed stored confidence')


def independent_check(rows, labels, metric):
    correct = sum(c['answer'] == 'ABCD'[labels[c['item_id']]] for c in rows)
    pairs = [(c['p_correct'], int(c['answer'] == 'ABCD'[labels[c['item_id']]]))
             for c in rows if scoring.usable(c)]
    wrong = [p for p, y in pairs if not y]
    if (metric['correct'] != correct or metric['n'] != len(rows)
            or metric['brier_n'] != len(pairs) or metric['wrong_with_confidence_n'] != len(wrong)):
        raise ValueError('independent metric count mismatch')
    values = {'accuracy_all': correct/len(rows) if rows else None,
        'mean_confidence': sum(p for p, _ in pairs)/len(pairs) if pairs else None,
        'raw_brier': sum((p-y)**2 for p, y in pairs)/len(pairs) if pairs else None,
        'mean_confidence_if_wrong': sum(wrong)/len(wrong) if wrong else None}
    for key, expected in values.items():
        actual = metric[key]
        if (actual is None) != (expected is None) or (actual is not None and
                not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12)):
            raise ValueError('independent metric arithmetic mismatch: '+key)


def score(root):
    out = root/DIRECTORY
    analysis = json.loads((out/'plan.json').read_text()); unseal(analysis, 'analysis_id')
    for path, sha in analysis['source_sha256'].items():
        if file_digest(path) != sha:
            raise ValueError('frozen scoring source changed: '+path)
    status_path = root/expansion.DIRECTORY/'status.json'
    status = json.loads(status_path.read_text())
    if status.get('stage') != 'complete' or status.get('responses') != 445:
        raise ValueError('445 replies are not complete; labels will not be opened')
    collection_audit = expansion.audit(root)
    if collection_audit['stage'] != 'complete':
        raise ValueError('collection audit is not complete; labels will not be opened')
    expansion_plan, cfg, manifest, items, groups, bundle_ids, paths = prediction_sources(root)
    if (analysis['expansion_id'] != expansion_plan['expansion_id'] or analysis['dataset_id'] != manifest['dataset_id']
            or analysis['ordered_input_hash'] != digest([asdict(i) for i in items])
            or analysis['baseline_bundle_ids'] != bundle_ids):
        raise ValueError('analysis input identities changed')
    gpath = root/expansion.DIRECTORY/'frozen_candidates.json'
    gptoss = json.loads(gpath.read_text()); unseal(gptoss, 'bundle_id')
    cpath = root/expansion.DIRECTORY/'completion_report.json'
    complete = json.loads(cpath.read_text())
    if (gptoss['expansion_id'] != analysis['expansion_id'] or gptoss['dataset_id'] != manifest['dataset_id']
            or complete['bundle_id'] != gptoss['bundle_id'] or complete['responses'] != 445):
        raise ValueError('frozen GPT-OSS completion bundle mismatch')
    groups['gpt_oss_120b'] = gptoss['groups']['gpt_oss_120b']; bundle_ids['gpt_oss_120b'] = gptoss['bundle_id']
    validate_groups(groups, items)
    partitions = {part: {i.item_id for i in items if i.partition == part}
                  for part in ('calibration', 'evaluation')}
    if {p: len(ids) for p, ids in partitions.items()} != analysis['partition_counts'] or set.intersection(*partitions.values()):
        raise ValueError('development/Diamond partition mismatch or overlap')
    common = {part: set.intersection(*[{c['item_id'] for c in rows if c['partition'] == part and scoring.usable(c)}
                                      for rows in groups.values()]) for part in partitions}
    paths += [gpath, cpath, status_path, out/'plan.json', root/'cache/budget.json']
    hashes = {str(p): file_digest(p) for p in paths}
    # Freeze prediction and cohort hashes before accessing the answer key.
    cohort = seal({'analysis_id': analysis['analysis_id'], 'bundle_ids': bundle_ids,
        'prediction_hashes': {g: digest(rows) for g, rows in groups.items()},
        'assigned_membership_hashes': {p: digest(sorted(ids)) for p, ids in partitions.items()},
        'common_valid_membership_hashes': {p: digest(sorted(ids)) for p, ids in common.items()},
        'common_valid_n': {p: len(ids) for p, ids in common.items()},
        'source_sha256': hashes, 'answer_keys_read': False}, 'cohort_id')
    base.immutable(out/'frozen_cohort.json', cohort)
    label_path = Path(cfg['dataset_root'])/'labels.jsonl'
    label_hash = file_digest(label_path)
    if label_hash != manifest['files']['labels.jsonl']:
        raise ValueError('answer key checksum differs')
    records = read_jsonl(label_path); by_id = {i.item_id: i for i in items}
    if len(records) != 445 or {r['item_id'] for r in records} != set(by_id):
        raise ValueError('answer key membership differs')
    if any(r['partition'] != by_id[r['item_id']].partition or type(r['correct_index']) is not int
           or r['correct_index'] not in range(4) for r in records):
        raise ValueError('answer key partition or index invalid')
    labels = {r['item_id']: r['correct_index'] for r in records}
    full = {}; matched = {}
    for part in partitions:
        full[part] = {}; matched[part] = {}
        for name, rows in groups.items():
            for membership, target in ((partitions[part], full), (common[part], matched)):
                selected = [c for c in rows if c['item_id'] in membership]
                metric = scoring.summarize(selected, labels); independent_check(selected, labels, metric)
                target[part][name] = metric
    old = json.loads((root/expansion.pilot.DIRECTORY/'result.json').read_text()); unseal(old, 'result_id')
    pilot_ids = set(manifest['pilot_calibration_ids'])
    old_names = {'gemini': 'gemini_baseline', 'qwen': 'qwen_baseline', 'sonnet_5_5': 'sonnet_5_5', 'gpt_oss_120b': 'gpt_oss_120b'}
    for name, rows in groups.items():
        fresh = scoring.summarize([c for c in rows if c['item_id'] in pilot_ids], labels)
        prior = old['generators'][old_names[name]]
        for key, value in fresh.items():
            expected = prior[key]
            if isinstance(value, float):
                if not math.isclose(value, expected, abs_tol=1e-12, rel_tol=1e-12):
                    raise ValueError('original50 pilot metric differs: '+name+'/'+key)
            elif value != expected:
                raise ValueError('original50 pilot metric differs: '+name+'/'+key)
    if file_digest(label_path) != label_hash or any(file_digest(path) != sha for path, sha in hashes.items()):
        raise ValueError('an input changed during scoring')
    expansion.load(root)
    result = seal({'analysis_id': analysis['analysis_id'], 'cohort_id': cohort['cohort_id'],
        'dataset_id': manifest['dataset_id'], 'expansion_id': analysis['expansion_id'],
        'bundle_ids': bundle_ids, 'models': {g: sorted({c['generator_model'] for c in rows}) for g, rows in groups.items()},
        'methods': METHODS, 'all_assigned_questions': full, 'common_valid_questions': matched,
        'common_valid_n': cohort['common_valid_n'], 'source_sha256': hashes, 'label_sha256': label_hash,
        'validation': {'identical_questions_and_choice_order': True, 'development_evaluation_overlap_n': 0,
            'predictions_and_cohorts_frozen_before_labels': True, 'original50_metrics_reproduced': True,
            'independent_arithmetic_matches': True, 'source_hashes_unchanged': True},
        'collection_audit': collection_audit, 'confidence_calibrated': False,
        'new_api_requests_in_scoring': 0, 'refit_performed': False, 'answer_keys_read': True,
        'candidate_or_parser_changes': False, 'per_item_outcomes_exported': False,
        'unit_tests_run': False}, 'result_id')
    base.immutable(out/'result.json', result)
    REPORT.mkdir(parents=True, exist_ok=True)
    for name, value in (('result.json', result), ('plan.json', analysis), ('audit.json', result['validation'])):
        base.immutable(REPORT/name, value)
    lines = ['四模型同一445题：原始confidence离线比较', '',
        '开发集249题、Diamond196题分别报告；所有445条GPT-OSS回复收齐且审计通过后才读取标签。',
        '正确率分母为各组全部题目；无法解析的答案算未答对。confidence及Brier只使用有效答案/概率，不插补。',
        'Brier = mean((raw confidence - 是否答对)^2)，越低越好。全部confidence未校准。', '']
    def pct(value): return '—' if value is None else f'{value:.2%}'
    for part, title in (('calibration', '开发集（249题）'), ('evaluation', 'Diamond（196题）')):
        lines += [title, '模型 | 答对/全部 | 正确率 | 平均confidence (n) | 答错confidence (n) | Brier (n)']
        for g, label in NAMES.items():
            m = full[part][g]
            brier = '—' if m['raw_brier'] is None else f"{m['raw_brier']:.6f}"
            lines.append(f"{label} | {m['correct']}/{m['n']} | {pct(m['accuracy_all'])} | {pct(m['mean_confidence'])} ({m['confidence_n']}) | {pct(m['mean_confidence_if_wrong'])} ({m['wrong_with_confidence_n']}) | {brier} ({m['brier_n']})")
        lines += [f"四模型共同有效子集：{len(common[part])}题，配套指标见result.json/common_valid_questions。", '']
    lines += ['原50题四模型全部指标复现；新增评分没有API请求、再生成、解析修复、参数拟合或confidence变换。',
        'Qwen使用两步生成，Sonnet采用adaptive/high effort，GPT-OSS采用medium reasoning；协议和服务日期不同，不能将差异全归因于基础模型。',
        '开发题经过多轮探索，Diamond也已用于先前模型比较；本表是描述性结果，不作为全新盲测或风险保证。',
        '费用详见collection_audit；均为冻结价格下的token用量估算，并非确认账单。历史未知请求费用预留未释放。',
        '本目录仅含汇总、协议、审计及哈希，不含题干、答案键、逐题对错、原始回复或凭证。']
    scoring.write_new(REPORT/'README.txt', ('\n'.join(lines)+'\n').encode())
    banned = {'question', 'choices', 'answer', 'correct_index', 'response', 'provider_response',
              'prompt', 'system', 'access_token', 'refresh_token', 'private_key', 'client_secret'}
    def safe(value):
        if isinstance(value, dict):
            if banned.intersection(value): raise ValueError('unsafe exported report field')
            for v in value.values(): safe(v)
        elif isinstance(value, list):
            for v in value: safe(v)
    for path in REPORT.glob('*.json'): safe(json.loads(path.read_text()))
    manifest_text = ''.join(f'{file_digest(p)}  {p.name}\n' for p in sorted(REPORT.iterdir())
                            if p.is_file() and p.name != 'sha256.txt')
    scoring.write_new(REPORT/'sha256.txt', manifest_text.encode())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'score'])
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    result = prepare(args.root) if args.command == 'prepare' else score(args.root)
    fields = ('result_id', 'all_assigned_questions', 'common_valid_questions', 'validation')
    print(json.dumps(result if args.command == 'prepare' else {k: result[k] for k in fields}, indent=2))
