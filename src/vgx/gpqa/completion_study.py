"""Frozen Qwen completion controls; calibration only, no automatic repairs.

Keep the previous collectors unchanged so their implementation pins remain valid.
Selection for expansion uses completion and output length, never answer keys.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import threading

from vgx.common.concurrency import parallel_map
from vgx.common.billing import estimate_call
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.gpqa import generator_study as direct, reasoning_study as previous
from vgx.gpqa.artifacts import FrozenCandidate
from vgx.gpqa.fresh import inputs
from vgx.gpqa.prompt import SYSTEM, build_verifier_prompt
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_collect import Pacer
from vgx.gpqa.signal_study import check_billing, runner_for

SHORT_INSTRUCTION = (
    'Solve the problem before choosing your final answer. Give a concise explanation '
    'of at most 200 words, focusing on the decisive calculation or evidence. '
    'Do not repeatedly reconsider the same alternatives. '
    'Then report your probability that your selected answer is correct.\n'
    'Always finish with exactly one final block, even if uncertain, and write nothing after it:\n'
    '<final>{"answer": "A"|"B"|"C"|"D", "p_correct": 0.0}</final>\n'
    'Replace the answer alternatives and example probability with your actual answer and probability. '
    'Put all explanation before the final block. Do not place any other JSON answer object in the explanation.'
)
ARMS = {'short_4096': (SHORT_INSTRUCTION, 4096),
        'original_8192': (previous.INSTRUCTION, 8192)}


def model_for(config, arm):
    model = deepcopy(config['model'])
    model['generation']['max_tokens'] = ARMS[arm][1]
    return model


def request_for(item, arm, operation=None):
    if item.partition != 'calibration':
        raise ValueError('completion study is calibration only')
    req = previous.generator_request(item, operation)
    return replace(req, prompt=req.prompt[:-len(previous.INSTRUCTION)] + ARMS[arm][0],
                   meta={**req.meta, 'prompt_version': arm})


def source(config):
    old, pilot, _ = previous.load(Path(config['previous_results']))
    if config['project'] != 'llm-applications-490420' or config['project'] != old['config']['project']:
        raise ValueError('unauthorized project')
    if config['dataset_root'] != old['config']['dataset_root']:
        raise ValueError('dataset changed')
    expected = next(m for m in old['config']['generators'] if m['id'] == 'qwen')
    if config['model'] != expected:
        raise ValueError('base model/settings changed')
    if not config['arms'] or len(set(config['arms'])) != len(config['arms']) or any(a not in ARMS for a in config['arms']):
        raise ValueError('invalid arms')
    manifest, all_items = inputs(Path(config['dataset_root']))
    if config['scope'] == 'pilot':
        items = pilot
    elif config['scope'] == 'all_calibration':
        items = [i for i in all_items if i.partition == 'calibration']
        selection = json.loads(Path(config['selection_file']).read_text())
        unseal(selection, 'selection_id')
        selected_plan, selected_items = load(Path(selection['source_root']))
        if selected_plan['study_id'] != selection['study_id'] or selected_plan['config']['scope'] != 'pilot':
            raise ValueError('selection source changed')
        expected_selection = gate(Path(selection['source_root']), write=False)
        if selection != expected_selection or config['arms'] != [selection['selected_arm']]:
            raise ValueError('expansion must use the completion-selected arm')
    else:
        raise ValueError('invalid scope')
    if any(i.partition != 'calibration' for i in items):
        raise ValueError('evaluation input entered study')
    return old, manifest, items


def implementation():
    return {**previous.implementation(), 'gpqa/completion_study.py': file_digest(Path(__file__))}


def prepare(config, root):
    old, manifest, items = source(config)
    cache = direct.budget(config, root)
    arms = []
    for arm in config['arms']:
        runner = runner_for(model_for(config, arm), config['project'])
        requests = [request_for(i, arm) for i in items]
        arms.append({'id': arm, 'identity': runner.identity,
                     'keys': [runner.cache_key(r) for r in requests],
                     'reservation_usd': sum(cache.reservation(runner, r) for r in requests)})
    plan = seal({'config': config, 'previous_study_id': old['study_id'],
                 'dataset_id': manifest['dataset_id'], 'input_hashes': [digest(asdict(i)) for i in items],
                 'implementation': implementation(), 'arms': arms, 'evaluation_requests': 0,
                 'selection_rule': 'Require 100% valid final answers/confidences; choose lowest mean output tokens, then arm id. No accuracy-based selection.'}, 'study_id')
    path = Path(root)/'plan.json'
    with file_lock(str(path)+'.lock'):
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('changed study; use another directory')
        atomic_json(path, plan)
    return plan


def load(root):
    plan = json.loads((Path(root)/'plan.json').read_text()); unseal(plan, 'study_id')
    old, manifest, items = source(plan['config'])
    if (plan['implementation'] != implementation() or plan['previous_study_id'] != old['study_id']
        or plan['dataset_id'] != manifest['dataset_id']
        or plan['input_hashes'] != [digest(asdict(i)) for i in items]):
        raise ValueError('source or implementation changed')
    return plan, items


def candidate(item, call, runner, req):
    parsed = previous.parse_final(call.response)
    return FrozenCandidate(item.item_id, item.partition, item.subject, item.question, item.choices,
        parsed.answer, parsed.p_correct, parsed.ok, parsed.failure, runner.model, call.key,
        direct.digest_text(req.prompt), direct.digest_text(call.response), SYSTEM,
        build_verifier_prompt(item, parsed.answer) if parsed.answer else None)


def collect(root, *, limit=None, allow_api=False):
    root = Path(root); plan, items = load(root); cfg = plan['config']
    if limit is not None:
        if type(limit) is not int or not 1 <= limit <= len(items): raise ValueError('invalid limit')
        items = items[:limit]
    cache = direct.budget(cfg, root); pacer = Pacer(cfg['model']['interval_seconds'])
    results = []; lock = threading.Lock()
    jobs = [(arm, item) for item in items for arm in cfg['arms']]
    def snapshot(error=None):
        with lock:
            value = {'study_id': plan['study_id'], 'completed': len(results), 'target': len(jobs),
                     'reused': sum(r['reused'] for r in results), 'results': list(results),
                     'error': error, 'budget': cache.summary()}
            atomic_json(root/f'collection_{len(items)}.json', value)
            return value
    def one(job):
        arm, item = job; runner = runner_for(model_for(cfg, arm), cfg['project'])
        req = request_for(item, arm, plan['study_id'])
        reused = direct.import_exact(cache, runner, req, cfg['reuse_caches'])
        if allow_api and not reused: pacer.acquire()
        call, cached = cache.get(runner, req, allow_api=allow_api)
        parsed = previous.parse_final(call.response)
        with lock:
            results.append({'arm': arm, 'item_id': item.item_id, 'key': call.key,
                            'reused': cached, 'failure': parsed.failure})
        snapshot()
    with file_lock(root/'collection.lock'):
        if allow_api: atomic_json(root/'billing_check.json', check_billing(cfg['project']))
        try: parallel_map(one, jobs, cfg['workers'])
        except Exception as exc:
            snapshot({'type': type(exc).__name__, 'http_status': getattr(exc, 'status', None)}); raise
        value = snapshot()
        if limit is None or limit == len(load(root)[1]):
            groups = {}
            for arm in cfg['arms']:
                runner = runner_for(model_for(cfg, arm), cfg['project'])
                groups[arm] = []
                for item in items:
                    req = request_for(item, arm, plan['study_id']); call, _ = cache.get(runner, req)
                    groups[arm].append(asdict(candidate(item, call, runner, req)))
            bundle = seal({'study_id': plan['study_id'], 'groups': groups}, 'bundle_id')
            path = root/'frozen_candidates.json'
            if path.exists() and digest(json.loads(path.read_text())) != digest(bundle):
                raise ValueError('frozen candidates changed')
            atomic_json(path, bundle)
        return value


def diagnostics(root):
    root = Path(root); plan, items = load(root); cfg = plan['config']; cache = direct.budget(cfg, root)
    summaries = {}; records = {}; projects = set()
    for arm in cfg['arms']:
        runner = runner_for(model_for(cfg, arm), cfg['project']); rows = []
        for item in items:
            req = request_for(item, arm, plan['study_id']); call, _ = cache.get(runner, req)
            parsed = previous.parse_final(call.response); estimate = estimate_call(vars(call), cfg['pricing'])
            provider = call.meta.get('provider_response', {})
            projects.add((call.meta.get('resource_project'), call.meta.get('quota_project')))
            rows.append({'item_id': item.item_id, 'answer': parsed.answer, 'p_correct': parsed.p_correct,
                         'valid': parsed.ok, 'failure': parsed.failure, 'key': call.key,
                         'output_tokens': estimate['usage']['output_tokens'], 'estimated_usd': estimate['estimated_usd'],
                         'length_stop': any(c.get('finish_reason') == 'length' for c in provider.get('choices', [])),
                         'provider_error': bool(provider.get('error'))})
        tokens = [r['output_tokens'] for r in rows if r['output_tokens'] is not None]
        summaries[arm] = {'n': len(rows), 'valid_answers': sum(r['answer'] is not None for r in rows),
                         'valid_answer_and_confidence': sum(r['valid'] for r in rows),
                         'length_stops': sum(r['length_stop'] for r in rows),
                         'provider_errors': sum(r['provider_error'] for r in rows),
                         'mean_output_tokens': sum(tokens)/len(tokens) if tokens else None,
                         'missing_usage': len(rows)-len(tokens),
                         'failures': dict(Counter(r['failure'] for r in rows if r['failure']))}
        records[arm] = rows
    return {'study_id': plan['study_id'], 'arms': summaries, 'records': records,
            'all_projects_match': projects == {(cfg['project'], cfg['project'])},
            'evaluation_requests': 0, 'new_spend': cache.summary(), 'confirmed_billing_usd': None}


def choose_arm(summaries):
    eligible = [(s['mean_output_tokens'], arm) for arm, s in summaries.items()
                if s['valid_answer_and_confidence'] == s['n'] and s['missing_usage'] == 0]
    return min(eligible)[1] if eligible else None


def gate(root, *, write=True):
    data = diagnostics(root)
    selected = choose_arm(data['arms'])
    result = seal({'study_id': data['study_id'], 'source_root': str(Path(root)),
                   'selected_arm': selected, 'arms': data['arms'],
                   'rule': '100% complete answer/confidence; minimize mean output tokens; no answer keys read.'}, 'selection_id')
    if write:
        atomic_json(Path(root)/'completion_gate.json', result)
        atomic_json(Path(root)/'completion_diagnostics.json', data)
    return result


def analyze(root):
    root = Path(root); plan, _ = load(root); cfg = plan['config']; data = diagnostics(root)
    manifest, _ = inputs(Path(cfg['dataset_root'])); path = Path(cfg['dataset_root'])/'labels.jsonl'
    if file_digest(path) != manifest['files']['labels.jsonl']: raise ValueError('labels changed')
    # Offline analysis only; evaluation labels never enter metrics or selection.
    labels = {r['item_id']: r['correct_index'] for r in read_jsonl(path) if r['partition'] == 'calibration'}
    for arm, rows in data['records'].items():
        valid = [r for r in rows if r['valid']]
        correct = lambda r: r['answer'] is not None and 'ABCD'.index(r['answer']) == labels[r['item_id']]
        wrong = [r for r in valid if not correct(r)]
        data['arms'][arm].update(correct=sum(correct(r) for r in rows),
            mean_confidence=sum(r['p_correct'] for r in valid)/len(valid) if valid else None,
            mean_wrong_confidence=sum(r['p_correct'] for r in wrong)/len(wrong) if wrong else None,
            raw_brier=sum((r['p_correct']-correct(r))**2 for r in valid)/len(valid) if valid else None)
    data['analysis_implementation'] = implementation()
    atomic_json(root/'report.json', data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'collect', 'gate', 'analyze'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path); parser.add_argument('--limit', type=int)
    parser.add_argument('--allow-api', action='store_true'); args = parser.parse_args()
    if args.command == 'prepare':
        value = prepare(json.loads(args.config.read_text()), args.root)
        print(json.dumps({'study_id': value['study_id'], 'arms': [{k:v for k,v in a.items() if k != 'keys'} for a in value['arms']]}))
    elif args.command == 'collect':
        value = collect(args.root, limit=args.limit, allow_api=args.allow_api)
        print(json.dumps({k:v for k,v in value.items() if k != 'results'}))
    else:
        value = gate(args.root) if args.command == 'gate' else analyze(args.root)
        print(json.dumps({k:v for k,v in value.items() if k != 'records'}))
