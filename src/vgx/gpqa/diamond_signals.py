"""Collect probability-verifier scores for the frozen Diamond (evaluation) answers.

The five models, prompt, generation settings and parser are those of the
development signal study (the screen config, probability format only), so each
request is the development request applied to an evaluation candidate.

Inputs are the frozen Gemini and Qwen candidates and the public question
inputs. No labels file or correctness record is opened and no generator request
is sent. Without --allow-api nothing is sent (a dry run reports what would be);
paid requests stay within a local usage-estimate stage cap that is itself inside
the authorized total. Returned malformed replies are kept, never retried.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import threading
import time

from vgx.common.api import OfflineCacheMiss
from vgx.common.billing import estimate_call
from vgx.common.budget import BudgetedRequestCache
from vgx.common.concurrency import parallel_map
from vgx.common.llm import Call, Request
from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa.artifacts import FrozenCandidate, load_candidates
from vgx.gpqa.fresh import inputs
from vgx.gpqa.generator_calibration import index_cache
from vgx.gpqa.qwen_signal_completion import budget_inventory
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_collect import Pacer
from vgx.gpqa.signal_study import (PROMPT_VERSION, SYSTEM, check_billing, implementation_hashes, parse_signal,
                                   prompt, runner_for)

SIGNAL = 'probability'


def sources(cfg):
    """Frozen evaluation candidates of both generators, checked against the public inputs."""
    dataset, public = inputs(Path(cfg['dataset_root']))
    gemini_manifest, gemini = load_candidates(Path(cfg['gemini_bundle']))
    root = Path(cfg['generator_results'])
    plan = json.loads((root/'plan.json').read_text())
    unseal(plan, 'study_id')
    frozen = json.loads((root/'frozen_candidates.json').read_text())
    unseal(frozen, 'bundle_id')
    if frozen['study_id'] != plan['study_id']:
        raise ValueError('frozen Qwen candidates belong to another study')
    if not (dataset['dataset_id'] == gemini_manifest['dataset_id'] == plan['dataset_id']) \
            or plan['baseline_bundle_id'] != gemini_manifest['bundle_id']:
        raise ValueError('dataset or Gemini bundle differs from the development study')
    by_id = {i.item_id: i for i in public}
    expected = {i.item_id for i in public if i.partition == 'evaluation'}
    groups = {}
    for name, cs in [('gemini', gemini), ('qwen', [FrozenCandidate(**c) for c in frozen['groups']['qwen']])]:
        cs = sorted((c for c in cs if c.partition == 'evaluation'), key=lambda c: c.item_id)
        if len(cs) != len(expected) or {c.item_id for c in cs} != expected:
            raise ValueError(f'{name}: evaluation membership differs from the public inputs')
        for c in cs:
            i = by_id[c.item_id]
            if (c.question, tuple(c.choices), c.subject) != (i.question, tuple(i.choices), i.subject):
                raise ValueError(f'{name}: candidate input differs from the public inputs')
        groups[name] = cs
    provenance = {'dataset_id': dataset['dataset_id'], 'gemini_bundle_id': gemini_manifest['bundle_id'],
                  'generator_study_id': plan['study_id'], 'qwen_bundle_id': frozen['bundle_id']}
    return provenance, groups


def request_for(candidate, model, operation_id=None):
    """signal_study.request_for with evaluation metadata; prompt, system and runner (the cache key) are unchanged."""
    return Request(f"{model['id']}:{SIGNAL}|{candidate.item_id}", prompt(candidate, SIGNAL), SYSTEM,
                   {'role': 'verifier', 'partition': 'evaluation', 'item_id': candidate.item_id,
                    'candidate': candidate.answer, 'prompt_version': PROMPT_VERSION, 'signal': SIGNAL,
                    'verifier_id': model['id'], 'operation_id': operation_id})


def eligible(c):
    return c.answer is not None and c.p_correct is not None


def jobs_for(groups, models, project, seed):
    # Batches (--limit) follow a label-free hash order of the question IDs.
    rank = {q: n for n, q in enumerate(sorted({c.item_id for cs in groups.values() for c in cs},
                                              key=lambda q: digest({'seed': seed, 'item_id': q})))}
    jobs = []
    for name, cs in groups.items():
        for c in cs:
            if eligible(c):
                for m in models:
                    jobs.append({'generator': name, 'item_id': c.item_id, 'rank': rank[c.item_id], 'verifier': m['id'],
                                 'request_key': runner_for(m, project).cache_key(request_for(c, m))})
    return sorted(jobs, key=lambda j: (j['rank'], j['generator'], j['verifier']))


def load_config(config_path):
    cfg = json.loads(Path(config_path).read_text())
    screen = json.loads(Path(cfg['screen_config']).read_text())
    if cfg['signal'] != SIGNAL or cfg['generator_calls'] != 0 or cfg['labels_read'] is not False:
        raise ValueError('probability-only, label-free verifier collection required')
    if screen['project'] != cfg['project'] or not 0 < cfg['stage_budget_usd'] <= cfg['total_authorized_usd']:
        raise ValueError('unauthorized project or stage budget')
    if [m['id'] for m in screen['models']] != cfg['verifiers']:
        raise ValueError('verifier list differs from the development screen')
    return cfg, screen


def implementation():
    hashes = implementation_hashes()
    hashes['gpqa/diamond_signals.py'] = file_digest(Path(__file__))
    return hashes


def prepare(config_path, output):
    cfg, screen = load_config(config_path)
    provenance, groups = sources(cfg)
    if any(len(cs) != cfg['expected_evaluation_questions'] for cs in groups.values()):
        raise ValueError('unexpected number of evaluation questions')
    jobs = jobs_for(groups, screen['models'], cfg['project'], cfg['seed'])
    models = {m['id']: m for m in screen['models']}
    planning, seen = 0., set()
    by_key = {(g, c.item_id): c for g, cs in groups.items() for c in cs}
    for j in jobs:
        if j['request_key'] in seen:
            continue
        seen.add(j['request_key'])
        req = request_for(by_key[(j['generator'], j['item_id'])], models[j['verifier']])
        rate = screen['pricing']['usd_per_million_tokens'][models[j['verifier']]['model']]
        planning += (len((req.prompt+req.system).encode())+1024)*rate['input']/1e6 + 1024*rate['output']/1e6
    plan = seal({'schema': 1, 'config': cfg, 'config_sha256': file_digest(Path(config_path)),
                 'screen_sha256': file_digest(Path(cfg['screen_config'])), 'provenance': provenance,
                 'candidate_hashes': {g: digest([asdict(c) for c in cs]) for g, cs in groups.items()},
                 'questions': {g: len(cs) for g, cs in groups.items()},
                 'invalid_generator_outputs': {g: [c.item_id for c in cs if not eligible(c)] for g, cs in groups.items()},
                 'jobs': jobs, 'unique_requests': len(seen), 'implementation_hashes': implementation(),
                 'planning_usd_at_1024_output_tokens': planning, 'generator_calls': 0, 'labels_read': False,
                 'note': 'Planning estimate uses 1024 output tokens per call, as the development screen did; '
                         'it is not a bound on reasoning tokens. The stage ledger gates actual spending.'},
                'collection_id')
    path = Path(output)/'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('collection plan changed; use a new output directory')
    atomic_json(path, plan)
    return plan


def run(config_path, output, *, allow_api=False, limit=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output/'plan.json'
        plan = json.loads(path.read_text()) if path.exists() else prepare(config_path, output)
        unseal(plan, 'collection_id')
        cfg, screen = load_config(config_path)
        if cfg != plan['config'] or file_digest(Path(cfg['screen_config'])) != plan['screen_sha256']:
            raise ValueError('config changed')
        if implementation() != plan['implementation_hashes']:
            raise ValueError('implementation changed')
        provenance, groups = sources(cfg)
        if provenance != plan['provenance'] or any(digest([asdict(c) for c in groups[g]]) != h
                                                   for g, h in plan['candidate_hashes'].items()):
            raise ValueError('frozen candidates changed')
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError('limit must be a positive number of questions')
        selected = [j for j in plan['jobs'] if limit is None or j['rank'] < limit]
        candidates = {(g, c.item_id): c for g, cs in groups.items() for c in cs}
        cache = BudgetedRequestCache(output/'cache', project=cfg['project'], pricing=screen['pricing'],
                                     limit_usd=cfg['stage_budget_usd'])
        index = index_cache([r for r in cfg['reuse_caches'] if Path(r).exists()])
        results, mutex, stop = {}, threading.Lock(), threading.Event()

        def snapshot(stage, **extra):
            with mutex:
                done = list(results.values())
                value = {'collection_id': plan['collection_id'], 'stage': stage, 'allow_api': allow_api,
                         'questions_in_batch': limit or plan['config']['expected_evaluation_questions'],
                         'target_jobs': len(selected), 'completed_jobs': sum(r['score'] is not None or r['failure'] is not None
                                                                             for r in done),
                         'not_sent_offline': sum(r.get('not_sent', False) for r in done),
                         'new_calls_this_invocation': sum(r.get('new', False) for r in done),
                         'invalid_by_verifier': dict(Counter(r['verifier']+':'+r['failure'] for r in done if r['failure'])),
                         'valid_by_verifier': dict(Counter(r['verifier'] for r in done if r['score'] is not None)),
                         'budget': cache.summary(), 'labels_read': False, 'generator_calls': 0,
                         'updated_at': time.time(), **extra}
                atomic_json(output/'status.json', value)
                return value

        def model_job(model):
            runner = runner_for(model, cfg['project'])
            pacer = Pacer(screen['collection']['minimum_request_interval_seconds'][model['id']])
            try:
                for job in selected:
                    if job['verifier'] != model['id']:
                        continue
                    if stop.is_set():
                        return
                    req = request_for(candidates[(job['generator'], job['item_id'])], model, plan['collection_id'])
                    if runner.cache_key(req) != job['request_key']:
                        raise ValueError('unplanned request')
                    if job['request_key'] in index:  # an identical earlier response is reused, even if malformed
                        cache.import_call(Call(**index[job['request_key']]))
                    record = {'generator': job['generator'], 'item_id': job['item_id'], 'verifier': model['id'],
                              'request_key': job['request_key'], 'score': None, 'failure': None, 'estimated_usd': None}
                    try:
                        call, reused = cache.get(runner, req)
                    except OfflineCacheMiss:
                        if not allow_api:
                            with mutex:
                                results[(job['generator'], job['item_id'], model['id'])] = {**record, 'not_sent': True}
                            continue
                        pacer.acquire()
                        if stop.is_set():
                            return
                        call, reused = cache.get(runner, req, allow_api=True)
                    score, failure = parse_signal(call.response, SIGNAL)
                    estimate = estimate_call(asdict(call), screen['pricing'])
                    if estimate['estimated_usd'] is None:
                        raise ValueError('unpriced verifier response')
                    with mutex:
                        results[(job['generator'], job['item_id'], model['id'])] = {
                            **record, 'score': score, 'failure': failure, 'estimated_usd': estimate['estimated_usd'],
                            'new': not reused}
                    snapshot('collecting')
            except Exception:
                stop.set()
                raise

        try:
            snapshot('preflight')
            if allow_api:
                atomic_json(output/'billing_check.json', check_billing(cfg['project']))
                # The stage cap is a sublimit of the authorized total, rechecked right before sending.
                inventory = budget_inventory()
                remaining = cfg['stage_budget_usd'] - cache.summary()['estimated_usd']
                if inventory['committed_estimate_and_reservations_usd'] + remaining > cfg['total_authorized_usd']:
                    raise ValueError('authorized total budget would be exceeded')
            parallel_map(model_job, screen['models'], min(cfg['workers'], len(screen['models'])))
            if any(r.get('not_sent') for r in results.values()):
                return snapshot('dry_run', planning_usd_at_1024_output_tokens=plan['planning_usd_at_1024_output_tokens'],
                                unique_requests=plan['unique_requests'], questions=plan['questions'],
                                invalid_generator_outputs={g: len(v) for g, v in plan['invalid_generator_outputs'].items()})
            if len(results) != len(selected):
                raise ValueError('incomplete collection')
            if limit is not None:
                return snapshot('batch_complete')
            write_signals(output, plan, groups, results)
            return snapshot('complete', signals_sha256=file_digest(output/'signals.json'))
        except Exception as exc:
            snapshot('blocked', error_type=type(exc).__name__, error=str(exc)[:300], http_status=getattr(exc, 'status', None))
            raise


def write_signals(output, plan, groups, results):
    """Development-row shape without outcomes: answer letter, raw prior and one entry per verifier."""
    rows = {}
    for g, cs in groups.items():
        rows[g] = [{'item_id': c.item_id, 'partition': 'evaluation', 'answer': c.answer, 'prior': c.p_correct,
                    'signals': {v+':'+SIGNAL: {k: results[(g, c.item_id, v)][k]
                                               for k in ('score', 'failure', 'estimated_usd', 'request_key')}
                                for v in plan['config']['verifiers']} if eligible(c) else {}}
                   for c in cs]
    atomic_json(output/'signals.json', seal({'collection_id': plan['collection_id'], 'provenance': plan['provenance'],
                                             'signal': SIGNAL, 'labels_read': False, 'rows': rows}, 'signals_id'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/gpqa_diamond_signals.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=None, help='only the first N questions (label-free hash order)')
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, allow_api=args.allow_api, limit=args.limit), indent=1))
