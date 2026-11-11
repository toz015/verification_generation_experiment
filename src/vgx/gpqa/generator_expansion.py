"""Collect Qwen/Llama on exactly the frozen Gemini question set.

No labels, verifier calls, or stopping-policy fitting. Evaluation responses are
frozen alongside calibration responses without scoring the evaluation partition.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import threading

from vgx.common.billing import estimate_call
from vgx.common.concurrency import parallel_map
from vgx.common.storage import atomic_json, digest, file_digest, file_lock
from vgx.gpqa import completion_study as completion, reasoning_study as previous, generator_study as direct
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.fresh import inputs
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_collect import Pacer
from vgx.gpqa.signal_study import check_billing, runner_for


def source(config):
    pilot, _ = completion.load(Path(config['completion_root']))
    selection = completion.gate(Path(config['completion_root']), write=False)
    if selection['selected_arm'] is None:
        raise ValueError('Qwen completion gate has not passed')
    old, _, _ = previous.load(Path(pilot['config']['previous_results']))
    if config['project'] != old['config']['project'] or config['project'] != 'llm-applications-490420':
        raise ValueError('unauthorized project')
    manifest, items = inputs(Path(old['config']['dataset_root']))
    bundle, baseline = load_candidates(Path(old['config']['baseline_bundle']))
    if bundle['dataset_id'] != manifest['dataset_id']:
        raise ValueError('baseline dataset differs')
    by_id = {i.item_id: i for i in items}
    if set(by_id) != {c.item_id for c in baseline}:
        raise ValueError('Gemini membership differs from prepared inputs')
    for c in baseline:
        i = by_id[c.item_id]
        if (c.question, tuple(c.choices), c.partition, c.subject) != (i.question, i.choices, i.partition, i.subject):
            raise ValueError('Gemini input content differs')
    models = {'qwen': completion.model_for(pilot['config'], selection['selected_arm']),
              'llama': next(m for m in old['config']['generators'] if m['id'] == 'llama')}
    return pilot, selection, manifest, bundle, list(items), models


def request_for(item, name, arm, operation=None):
    req = previous.generator_request(item, operation)
    if name == 'llama':
        return req
    if name != 'qwen': raise ValueError('unknown generator')
    return replace(req, prompt=req.prompt[:-len(previous.INSTRUCTION)] + completion.ARMS[arm][0],
                   meta={**req.meta, 'prompt_version': arm})


def implementation():
    return {**completion.implementation(), 'gpqa/generator_expansion.py': file_digest(Path(__file__))}


def prepare(config, root):
    pilot, selection, manifest, baseline, items, models = source(config)
    cache = direct.budget(config, root)
    arms = []
    for name, model in models.items():
        runner = runner_for(model, config['project'])
        requests = [request_for(i, name, selection['selected_arm']) for i in items]
        arms.append({'id': name, 'model': model, 'identity': runner.identity,
                     'keys': [runner.cache_key(r) for r in requests],
                     'reservation_usd': sum(cache.reservation(runner, r) for r in requests)})
    plan = seal({'config': config, 'completion_study_id': pilot['study_id'],
                 'selection': selection, 'dataset_id': manifest['dataset_id'],
                 'baseline_bundle_id': baseline['bundle_id'],
                 'input_hashes': [digest(asdict(i)) for i in items],
                 'partition_counts': dict(Counter(i.partition for i in items)),
                 'arms': arms, 'implementation': implementation(),
                 'evaluation_policy': 'Collect and freeze only; no evaluation-label access or tuning.'}, 'study_id')
    path = Path(root)/'plan.json'
    with file_lock(str(path)+'.lock'):
        if path.exists() and json.loads(path.read_text()) != plan: raise ValueError('study changed')
        atomic_json(path, plan)
    return plan


def load(root):
    plan = json.loads((Path(root)/'plan.json').read_text()); unseal(plan, 'study_id')
    pilot, selection, manifest, baseline, items, _ = source(plan['config'])
    if (plan['implementation'] != implementation() or plan['completion_study_id'] != pilot['study_id']
        or plan['selection'] != selection or plan['dataset_id'] != manifest['dataset_id']
        or plan['baseline_bundle_id'] != baseline['bundle_id']
        or plan['input_hashes'] != [digest(asdict(i)) for i in items]):
        raise ValueError('source or implementation changed')
    return plan, items


def collect(root, *, allow_api=False):
    root = Path(root); plan, items = load(root); cfg = plan['config']; cache = direct.budget(cfg, root)
    pacers = {a['id']: Pacer(a['model']['interval_seconds']) for a in plan['arms']}
    # At most one outstanding request per model; Qwen HTTP-200 concurrency
    # errors from the previous experiment are never regenerated.
    results = []; lock = threading.Lock()
    def snapshot(error=None):
        with lock:
            result = {'study_id': plan['study_id'], 'completed': len(results),
                      'target': len(items)*len(plan['arms']), 'results': list(results),
                      'reused': sum(r['reused'] for r in results), 'error': error, 'budget': cache.summary()}
            atomic_json(root/'collection.json', result); return result
    def arm_job(arm):
        name = arm['id']; runner = runner_for(arm['model'], cfg['project'])
        for item in items:
            req = request_for(item, name, plan['selection']['selected_arm'], plan['study_id'])
            reused = direct.import_exact(cache, runner, req, cfg['reuse_caches'])
            if allow_api and not reused: pacers[name].acquire()
            call, cached = cache.get(runner, req, allow_api=allow_api)
            parsed = previous.parse_final(call.response)
            with lock:
                results.append({'model': name, 'item_id': item.item_id, 'partition': item.partition,
                                'key': call.key, 'reused': cached, 'failure': parsed.failure})
            snapshot()
    with file_lock(root/'collection.lock'):
        if allow_api: atomic_json(root/'billing_check.json', check_billing(cfg['project']))
        try: parallel_map(arm_job, plan['arms'], 2)
        except Exception as exc:
            snapshot({'type': type(exc).__name__, 'http_status': getattr(exc, 'status', None)}); raise
        groups = {}; summary = {}; projects = set()
        for arm in plan['arms']:
            name = arm['id']; runner = runner_for(arm['model'], cfg['project']); groups[name] = []
            for partition in ('calibration', 'evaluation'):
                counts = Counter(); tokens = []
                for item in (i for i in items if i.partition == partition):
                    req = request_for(item, name, plan['selection']['selected_arm'], plan['study_id'])
                    call, _ = cache.get(runner, req); parsed = previous.parse_final(call.response)
                    groups[name].append(asdict(completion.candidate(item, call, runner, req)))
                    projects.add((call.meta.get('resource_project'), call.meta.get('quota_project')))
                    provider = call.meta.get('provider_response', {})
                    counts['requested'] += 1; counts['valid_answers'] += int(parsed.answer is not None)
                    counts['valid_answer_and_confidence'] += int(parsed.ok)
                    counts['length_stops'] += int(any(c.get('finish_reason') == 'length' for c in provider.get('choices', [])))
                    if parsed.failure: counts[parsed.failure] += 1
                    usage = estimate_call(vars(call), cfg['pricing'])['usage']['output_tokens']
                    if usage is not None: tokens.append(usage)
                summary[name+':'+partition] = {**counts, 'mean_output_tokens': sum(tokens)/len(tokens) if tokens else None}
        frozen = seal({'study_id': plan['study_id'], 'groups': groups}, 'bundle_id')
        path = root/'frozen_candidates.json'
        if path.exists() and digest(json.loads(path.read_text())) != digest(frozen): raise ValueError('frozen answers changed')
        atomic_json(path, frozen)
        atomic_json(root/'completion_report.json', {'study_id': plan['study_id'], 'groups': summary,
                    'all_projects_match': projects == {(cfg['project'], cfg['project'])},
                    'evaluation_labels_read': False, 'budget': cache.summary(), 'confirmed_billing_usd': None})
        return snapshot()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'collect']); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path); p.add_argument('--allow-api', action='store_true'); args = p.parse_args()
    if args.command == 'prepare':
        v = prepare(json.loads(args.config.read_text()), args.root)
        print(json.dumps({'study_id': v['study_id'], 'partition_counts': v['partition_counts'],
                          'generator_reservation_usd': sum(a['reservation_usd'] for a in v['arms'])}))
    else:
        v = collect(args.root, allow_api=args.allow_api)
        print(json.dumps({k:x for k,x in v.items() if k != 'results'}))
