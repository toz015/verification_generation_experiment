"""Independent calibration-only generator comparison; historical runs stay frozen.

Selection uses the original, pre-answer pilot manifest. Collection never reads
answer keys. Exact verifier requests can be reused across generator arms when
the chosen answer is identical; execution costs are counted once.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import threading

from vgx.common.api import OfflineCacheMiss, RequestCache
from vgx.common.budget import BudgetedRequestCache
from vgx.common.concurrency import parallel_map
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.fresh import candidate_for, generator_request, inputs
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_collect import Pacer
from vgx.gpqa.signal_study import check_billing, parse_signal, request_for, runner_for


def source(config):
    manifest, items = inputs(Path(config['dataset_root']))
    ids = manifest['pilot_calibration_ids']
    by_id = {i.item_id: i for i in items}
    if len(ids) != config['sample_size'] or len(set(ids)) != len(ids):
        raise ValueError('original pilot size/membership mismatch')
    selected = [by_id[i] for i in ids]
    if any(i.partition != 'calibration' for i in selected):
        raise ValueError('only calibration items may enter this study')
    bundle, candidates = load_candidates(Path(config['baseline_bundle']))
    if bundle.get('dataset_id') != manifest['dataset_id']:
        raise ValueError('baseline and public inputs have different dataset identities')
    baseline = {c.item_id: c for c in candidates}
    for item in selected:
        c = baseline[item.item_id]
        if (c.question, tuple(c.choices), c.subject, c.partition) != (
                item.question, tuple(item.choices), item.subject, item.partition):
            raise ValueError('baseline question/choice order mismatch')
        if c.generator_prompt_sha256 != digest_text(generator_request(item, None).prompt):
            raise ValueError('baseline generator prompt differs')
    return manifest, bundle, selected, [baseline[i] for i in ids]


def digest_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def implementation():
    root = Path(__file__).resolve().parents[1]
    return {name: file_digest(root/name) for name in (
        'gpqa/generator_study.py', 'gpqa/fresh.py', 'gpqa/prompt.py',
        'gpqa/signal_study.py', 'gpqa/signal_collect.py', 'gpqa/artifacts.py',
        'common/api.py', 'common/vertex.py', 'common/vertex_partner.py',
        'common/budget.py', 'common/billing.py', 'common/concurrency.py')}


def prepare(config, output):
    if config['project'] != 'llm-applications-490420':
        raise ValueError('unauthorized project')
    models = config['generators'] + config['verifiers']
    if len({m['id'] for m in models}) != len(models) or any(m['id'] == 'gemini_baseline' for m in models):
        raise ValueError('unique model identifiers required')
    manifest, bundle, items, baseline = source(config)
    cache = budget(config, output)
    arms = []
    for model in config['generators']:
        runner = runner_for(model, config['project'])
        requests = [generator_request(i, None) for i in items]
        arms.append({'id': model['id'], 'identity': runner.identity,
                     'request_keys': [runner.cache_key(r) for r in requests],
                     'total_reservation_usd': sum(cache.reservation(runner, r) for r in requests)})
    value = seal({'config': config, 'dataset_id': manifest['dataset_id'],
                  'baseline_bundle_id': bundle['bundle_id'],
                  'candidate_ids': [i.item_id for i in items],
                  'baseline_hashes': [digest(asdict(c)) for c in baseline],
                  'input_hashes': [digest(asdict(i)) for i in items],
                  'generator_arms': arms, 'implementation': implementation(),
                  'selection': 'Original subject-stratified pilot selected before generator outcomes.',
                  'evaluation_requests': 0}, 'study_id')
    path = Path(output)/'plan.json'
    with file_lock(str(path)+'.lock'):
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('changed study; use a new output directory')
        atomic_json(path, value)
    return value


def load(output):
    plan = json.loads((Path(output)/'plan.json').read_text())
    unseal(plan, 'study_id')
    manifest, bundle, items, baseline = source(plan['config'])
    if (plan['implementation'] != implementation()
        or plan['dataset_id'] != manifest['dataset_id']
        or plan['baseline_bundle_id'] != bundle['bundle_id']
        or plan['input_hashes'] != [digest(asdict(i)) for i in items]
        or plan['baseline_hashes'] != [digest(asdict(c)) for c in baseline]):
        raise ValueError('frozen source or implementation changed')
    return plan, items, baseline


def budget(config, output):
    return BudgetedRequestCache(Path(output)/'cache', project=config['project'],
        pricing=config['pricing'], limit_usd=config['stage_budget_usd'])


def import_exact(cache, runner, request, roots):
    try:
        cache.get(runner, request)
        return True
    except OfflineCacheMiss:
        pass
    for root in roots:
        try:
            call, _ = RequestCache(root).get(runner, request)
        except OfflineCacheMiss:
            continue
        if (call.model != runner.model or call.prompt != request.prompt
            or call.meta.get('system') != request.system
            or call.meta.get('runner_identity') != runner.identity
            or call.params != runner.params):
            raise ValueError('imported response provenance differs')
        cache.import_call(call)
        return True
    return False


def collect(output, stage, *, limit=None, allow_api=False):
    output = Path(output)
    plan, items, baseline = load(output)
    cfg = plan['config']; cache = budget(cfg, output)
    limit = len(items) if limit is None else limit
    if type(limit) is not int or not 1 <= limit <= len(items):
        raise ValueError('invalid candidate limit')
    items, baseline = items[:limit], baseline[:limit]
    models = cfg['generators'] if stage == 'generators' else cfg['verifiers']
    if stage not in ('generators', 'verifiers'):
        raise ValueError('invalid collection stage')
    pacers = {m['id']: Pacer(m['interval_seconds']) for m in models}
    results = []; lock = threading.Lock()
    jobs = []
    if stage == 'generators':
        jobs = [(m['id'], i, m) for i in items for m in models]
    else:
        groups = {'gemini_baseline': baseline}
        # All candidate generation must already be cached; no paid generator fallback.
        for m in cfg['generators']:
            runner = runner_for(m, cfg['project'])
            groups[m['id']] = [candidate_for(i, runner, cache, plan['study_id']) for i in items]
        for group, candidates in groups.items():
            for c in candidates:
                if c.answer is not None:  # confidence failure does not invalidate the answer
                    jobs.extend((group, c, m) for m in models)
    def snapshot(error=None):
        with lock:
            value = {'study_id': plan['study_id'], 'stage': stage, 'limit': limit,
                     'target_logical_requests': len(jobs), 'completed': len(results),
                     'cache_reused': sum(r['reused'] for r in results),
                     'failures': dict(Counter(r['failure'] for r in results if r['failure'])),
                     'error': error, 'budget': cache.summary(), 'results': list(results)}
            atomic_json(output/f'{stage}_{limit}.json', value)
            return value
    def one(job):
        group, item, model = job
        runner = runner_for(model, cfg['project'])
        req = (generator_request(item, plan['study_id']) if stage == 'generators'
               else request_for(item, model, 'probability', plan['study_id']))
        reused = import_exact(cache, runner, req, cfg['reuse_caches'])
        if allow_api and not reused:
            pacers[model['id']].acquire()
        call, cached = cache.get(runner, req, allow_api=allow_api)
        if stage == 'generators':
            candidate = candidate_for(item, runner, cache, plan['study_id'])
            failure = candidate.generator_failure
        else:
            _, failure = parse_signal(call.response, 'probability')
        with lock:
            results.append({'generator': group, 'model': model['id'], 'item_id': item.item_id,
                            'request_key': call.key, 'reused': cached, 'failure': failure})
        snapshot()
    with file_lock(output/'collection.lock'):
        if allow_api:
            atomic_json(output/'billing_check.json', check_billing(cfg['project']))
        try:
            parallel_map(one, jobs, cfg['workers'])
        except Exception as exc:
            snapshot({'type': type(exc).__name__, 'http_status': getattr(exc, 'status', None)})
            raise
        return snapshot()


def report(output):
    import numpy as np
    from vgx.gpqa.signal_analysis import cross_validate, dependence_summary, tail_diagnostics
    from vgx.gpqa.report import rate_interval
    from vgx.gpqa.score import binary_forecast_metrics, correctness_outcome
    output = Path(output); plan, items, baseline = load(output)
    cfg = plan['config']; cache = budget(cfg, output)
    # Scoring only: the collector never opens this file.
    path = Path(cfg['baseline_bundle'])/'calibration_records.jsonl'
    manifest = json.loads((Path(cfg['baseline_bundle'])/'bundle_manifest.json').read_text())
    if file_digest(path) != manifest['files'][path.name]:
        raise ValueError('calibration labels changed')
    labels = {r['item_id']: r['correct_index'] for r in read_jsonl(path)}
    groups = {'gemini_baseline': baseline}
    for m in cfg['generators']:
        groups[m['id']] = [candidate_for(i, runner_for(m, cfg['project']), cache, plan['study_id']) for i in items]
    summaries = {}; tags = [m['id'] for m in cfg['verifiers']]; all_rows = {}
    for group, candidates in groups.items():
        outcomes = [correctness_outcome(c.answer, labels[c.item_id]) for c in candidates]
        rows = []; failures = Counter()
        for c, y in zip(candidates, outcomes):
            if c.p_correct is None or c.answer is None:
                failures[c.generator_failure or 'invalid_candidate'] += 1
                continue
            row = {'item_id': c.item_id, 'partition': 'calibration', 'generator_answer': c.answer,
                   'generator_p_correct': c.p_correct, 'correct_index': labels[c.item_id],
                   'outcome': y, 'verifiers': {}}
            for m in cfg['verifiers']:
                req = request_for(c, m, 'probability', plan['study_id'])
                try:
                    call, _ = cache.get(runner_for(m, cfg['project']), req)
                    p, failure = parse_signal(call.response, 'probability')
                except OfflineCacheMiss:
                    p, failure = None, 'not_collected'
                if failure:
                    failures[m['id']+':'+failure] += 1
                else:
                    row['verifiers'][m['id']] = {'p_correct': p}
            rows.append(row)
        all_rows[group] = rows
        y = [r['outcome'] for r in rows]; p = [r['generator_p_correct'] for r in rows]
        paired = [r for r in rows if all(t in r['verifiers'] for t in tags)]
        summaries[group] = {'n': len(candidates), 'correct': int(sum(outcomes)),
            'accuracy_all': float(np.mean(outcomes)), 'accuracy_wilson95': rate_interval(outcomes),
            'valid_answer_n': sum(c.answer is not None for c in candidates),
            'confidence_n': len(rows), 'failures': dict(failures),
            'mean_confidence': float(np.mean(p)) if p else None,
            'confidence_values': dict(Counter(str(v) for v in p)),
            'mean_confidence_if_correct': float(np.mean([v for v,t in zip(p,y) if t])) if any(y) else None,
            'mean_confidence_if_incorrect': float(np.mean([v for v,t in zip(p,y) if not t])) if any(t==0 for t in y) else None,
            'raw_metrics': binary_forecast_metrics(y, p).to_dict(),
            'raw_tails': tail_diagnostics(y, {'raw': p}),
            'initial_calibration_oof': cross_validate(rows, [], {}, seed=cfg['seed']),
            'verifier_oof': cross_validate(paired, tags, {t:'probability' for t in tags}, seed=cfg['seed']),
            'dependence': dependence_summary(paired, tags)}
    value = {'study_id': plan['study_id'], 'generators': summaries,
             'new_collection_budget': cache.summary(), 'limitations': cfg['limitations']}
    atomic_json(output/'comparison.json', value)
    atomic_json(output/'analysis_records.json', all_rows)
    atomic_json(output/'frozen_candidates.json', seal(
        {'study_id': plan['study_id'], 'groups': {g:[asdict(c) for c in cs] for g,cs in groups.items()}}, 'bundle_id'))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','generators','verifiers','report'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        value = prepare(json.loads(args.config.read_text()), args.output)
        print(json.dumps({'study_id': value['study_id'], 'n': len(value['candidate_ids']),
                          'generator_reservations_usd': sum(a['total_reservation_usd'] for a in value['generator_arms'])}))
    elif args.command == 'report':
        value = report(args.output)
        print(json.dumps({g:{k:s[k] for k in ('n','correct','mean_confidence','failures')}
                          for g,s in value['generators'].items()}))
    else:
        value = collect(args.output, args.command, limit=args.limit, allow_api=args.allow_api)
        print(json.dumps({k:v for k,v in value.items() if k != 'results'}))


if __name__ == '__main__':
    main()
