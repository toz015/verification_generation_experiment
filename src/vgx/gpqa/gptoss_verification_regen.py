"""GPT-OSS-120B "complete" candidates (the 7 unusable replies re-requested once): same verification procedure.

Same procedure as sonnet_verification (copied, not imported, so both stay sealed);
only the candidate source, budget cap and generator label differ.

Uses unchanged request/parser/transport and exact nested-planner implementations.
The command boundaries enforce collection -> development fit -> public decisions
-> scoring. No generator requests, prior calibration, or candidate repairs.
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

import numpy as np

from vgx.common.api import OfflineCacheMiss, RequestCache
from vgx.common.billing import estimate_call
from vgx.common.concurrency import parallel_map
from vgx.common.llm import Call
from vgx.common.storage import atomic_json, digest, file_digest, read_jsonl
from vgx.gpqa import gptoss_regen7 as regen
from vgx.gpqa.artifacts import FrozenCandidate
from vgx.gpqa.dependence_triples import INF, ExactPlanner, JointLikelihood, all_orders, execute
from vgx.gpqa.diamond_score import boot
from vgx.gpqa.diamond_signals import request_for as evaluation_request
from vgx.gpqa.generator_calibration import index_cache
from vgx.gpqa.gptoss_retry179_ledgerfix import budget_inventory  # exempts only the pinned zero-cost QuantumKatas mock ledger
from vgx.gpqa.raw_prior_analysis import baseline_decision, candidate_valid, summarize_policy
from vgx.gpqa.replacement_generators import ScreeningCache
from vgx.gpqa.report import rate_interval
from vgx.gpqa.score import fit_verifier_likelihood
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_collect import Pacer
from vgx.gpqa.signal_study import implementation_hashes, parse_signal, request_for, runner_for, check_billing


def immutable(path, value):
    """Compare canonical JSON, so tuple/list serialization is immaterial."""
    if path.exists():
        if digest(json.loads(path.read_text())) != digest(value):
            raise ValueError('immutable artifact differs: '+str(path))
    else:
        atomic_json(path, value)


def implementation():
    base = Path(__file__).resolve().parents[1]
    files = ['gpqa/gptoss_verification_regen.py', 'gpqa/gptoss_regen7.py', 'gpqa/gptoss_finish445.py', 'gpqa/dependence_triples.py', 'gpqa/diamond_score.py',
        'gpqa/diamond_signals.py', 'gpqa/generator_calibration.py', 'gpqa/score.py',
        'gpqa/raw_prior_analysis.py', 'gpqa/report.py', 'gpqa/replacement_generators.py',
        'gpqa/qwen_signal_completion.py', 'gpqa/sequential.py', 'gpqa/signal_collect.py',
        'common/storage.py', 'common/concurrency.py']
    return {**implementation_hashes(), **{f: file_digest(base/f) for f in files}}


def sources(cfg):
    root = Path(cfg['generator_root'])
    path = root/regen.DIRECTORY/'frozen_candidates.json'
    frozen = json.loads(path.read_text()); unseal(frozen, 'bundle_id')
    completion = json.loads((root/regen.DIRECTORY/'completion_report.json').read_text())
    if completion['bundle_id'] != frozen['bundle_id']:
        raise ValueError('GPT-OSS complete bundle identity differs')
    candidates = [FrozenCandidate(**c) for c in frozen['groups']['gpt_oss_120b']]
    if len(candidates) != 445 or len({c.item_id for c in candidates}) != 445:
        raise ValueError('candidate membership differs')
    sonnet = json.loads(Path('results/gpqa_sonnet55_verification_20261016/protocol.json').read_text())
    return candidates, {'dataset_id': frozen['dataset_id'], 'bundle_id': frozen['bundle_id'],
        'candidate_file_sha256': file_digest(path), 'generator_expansion_id': frozen['expansion_id'],
        'regenerated_positions': frozen['replaced_positions'], 'source_bundle_id': frozen['source_bundle_id'],
        'dataset_root': sonnet['sources']['dataset_root'], 'baseline_bundle': sonnet['sources']['baseline_bundle']}


def req_for(c, m, op=None):
    return (request_for(c, m, 'probability', op) if c.partition == 'calibration'
            else evaluation_request(c, m, op))


def eligible(c):
    return candidate_valid({'answer': c.answer, 'prior': c.p_correct})


def cache_for(out, cfg, screen):
    # No automatic HTTP retries in this new extension. Historical malformed
    # responses and explicit rejection/unknown outcomes remain preserved.
    return ScreeningCache(out/'cache', project=cfg['project'], pricing=screen['pricing'],
                          limit_usd=cfg['stage_budget_usd'])


def check_call(call, runner, req, cfg, screen):
    if (call.key != runner.cache_key(req) or call.model != runner.model or call.prompt != req.prompt
        or call.params != runner.params or call.meta.get('system') != req.system
        or call.meta.get('runner_identity') != runner.identity
        or call.meta.get('role') != 'verifier'
        or any(call.meta.get(k) != cfg['project'] for k in ('resource_project', 'quota_project'))):
        raise ValueError('verifier cache provenance mismatch')
    e = estimate_call(asdict(call), screen['pricing'])
    if e['estimated_usd'] is None or e['issues']:
        raise ValueError('unpriced or unreconciled verifier usage')
    return e


def prepare(config_path, out):
    cfg = json.loads(config_path.read_text()); screen = json.loads(Path(cfg['screen_config']).read_text())
    cs, identity = sources(cfg)
    old_collection = json.loads(Path('results/gpqa_diamond_signals_20261012/plan.json').read_text())
    unseal(old_collection, 'collection_id')
    if file_digest(cfg['screen_config']) != old_collection['screen_sha256']:
        raise ValueError('five-verifier screen differs from historical Diamond collection')
    if (cfg['schema'] != 1 or cfg['project'] != 'llm-applications-490420'
        or cfg['project'] != screen['project'] or [m['id'] for m in screen['models']] != cfg['verifiers']
        or cfg['stage_budget_usd'] != 10 or cfg['total_authorized_usd'] != 200
        or cfg['settings']['prior'] != 'raw_unmodified_p_correct'
        or cfg['settings']['likelihood_model'] != 'conditional_independence'):
        raise ValueError('unexpected protocol or budget')
    inventory = budget_inventory()
    if inventory['committed_estimate_and_reservations_usd']+cfg['stage_budget_usd'] > cfg['total_authorized_usd']:
        raise ValueError('insufficient authorized overall budget')
    value = seal({'config': cfg, 'screen': screen, 'config_sha256': file_digest(config_path),
        'screen_sha256': file_digest(cfg['screen_config']), 'sources': identity,
        'candidate_hash': digest([asdict(c) for c in cs]), 'implementation': implementation(),
        'budget_before': inventory, 'candidate_counts': dict(Counter(c.partition for c in cs)),
        'eligible_counts': dict(Counter(c.partition for c in cs if eligible(c))),
        'confidence_calibrated': False, 'generator_calls': 0, 'evaluation_signals_read': False,
        'evaluation_generator_only_results_previously_seen': True}, 'protocol_id')
    immutable(out/'protocol.json', value)
    return value


def load(out):
    protocol = json.loads((out/'protocol.json').read_text()); unseal(protocol, 'protocol_id')
    cfg, screen = protocol['config'], protocol['screen']
    cs, identity = sources(cfg)
    if (protocol['implementation'] != implementation() or identity != protocol['sources']
        or digest([asdict(c) for c in cs]) != protocol['candidate_hash']
        or file_digest(cfg['screen_config']) != protocol['screen_sha256']):
        raise ValueError('frozen protocol, candidates, or implementation changed')
    return protocol, cfg, screen, cs


def verify_policy(out, protocol):
    path = out/'policy.json'; policy = json.loads(path.read_text()); unseal(policy, 'policy_id')
    sig = out/'calibration/signals.json'
    if (policy['protocol_id'] != protocol['protocol_id'] or policy['development_signals_sha256'] != file_digest(sig)
        or policy['implementation'] != implementation()):
        raise ValueError('frozen development policy differs')
    return policy


def collect(out, partition, allow_api=False, limit=None):
    protocol, cfg, screen, candidates = load(out)
    phase = out/partition; phase.mkdir(parents=True, exist_ok=True)
    with (out/'collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (phase/'status.json').exists() and json.loads((phase/'status.json').read_text())['stage'] == 'blocked':
            raise ValueError('saved blocked state requires review; no automatic retry')
        policy = verify_policy(out, protocol) if partition == 'evaluation' else None
        cs = sorted([c for c in candidates if c.partition == partition],
                    key=lambda c: digest({'seed': cfg['seed'], 'item_id': c.item_id}))
        roots = cfg['development_cache_roots'] if partition == 'calibration' else cfg['evaluation_cache_roots']
        index = index_cache(roots)
        cache = cache_for(out, cfg, screen)
        jobs, cached_hashes, planning, missing = [], {}, 0., Counter()
        for rank, c in enumerate([c for c in cs if eligible(c)]):
            for m in screen['models']:
                r = runner_for(m, cfg['project']); req = req_for(c, m); key = r.cache_key(req)
                jobs.append({'rank': rank, 'item_id': c.item_id, 'verifier': m['id'], 'request_key': key})
                if key in index:
                    check_call(Call(**index[key]), r, req, cfg, screen)
                    cached_hashes[key] = digest(index[key])
                else:
                    # Refuse to recreate uncertain historical provider outcomes.
                    for old_root in roots:
                        path = str(RequestCache(old_root).log(key).path)+'.attempts.jsonl'
                        attempts = read_jsonl(path, tolerate_tail=True)
                        started = {a['execution_id'] for a in attempts if a.get('event') == 'started'}
                        rejected = {a['execution_id'] for a in attempts if a.get('event') == 'rejected'}
                        if started-rejected:
                            raise ValueError('unresolved historical verifier execution')
                    missing[m['id']] += 1
                    rate = screen['pricing']['usd_per_million_tokens'][r.model]
                    planning += ((len((req.prompt+req.system).encode())+1024)*rate['input']+1024*rate['output'])/1e6
        plan = seal({'protocol_id': protocol['protocol_id'], 'partition': partition,
            'policy_id_before_evaluation': policy['policy_id'] if policy else None,
            'jobs': jobs, 'historical_response_hashes': cached_hashes,
            'missing_by_verifier': dict(missing), 'planning_new_usd_at_1024_output_tokens': planning,
            'questions': len(cs), 'eligible': sum(eligible(c) for c in cs)}, 'collection_id')
        immutable(phase/'plan.json', plan)
        selected = [j for j in jobs if limit is None or j['rank'] < limit]
        by_id = {c.item_id: c for c in cs}; results = {}; stop = threading.Event(); mutex = threading.Lock()
        def snapshot(stage, **extra):
            with mutex:
                rows = list(results.values())
                value = {'protocol_id': protocol['protocol_id'], 'collection_id': plan['collection_id'],
                    'stage': stage, 'partition': partition, 'target_jobs': len(selected),
                    'completed_jobs': sum(not x.get('not_sent', False) for x in rows),
                    'valid_by_verifier': dict(Counter(x['verifier'] for x in rows if x['score'] is not None)),
                    'invalid_by_verifier': dict(Counter(x['verifier'] for x in rows if x['failure'])),
                    'failures': dict(Counter(x['failure'] for x in rows if x['failure'])),
                    'new_calls_this_invocation': sum(x.get('new', False) for x in rows),
                    'reused_this_invocation': sum(not x.get('new', False) and not x.get('not_sent', False) for x in rows),
                    'budget': cache.summary(), 'updated_at_unix': time.time(),
                    'generator_calls': 0, 'labels_read': False, **extra}
                atomic_json(phase/'status.json', value)
                return value
        def one_model(m):
            r = runner_for(m, cfg['project'])
            pacer = Pacer(screen['collection']['minimum_request_interval_seconds'][m['id']])
            try:
                for j in selected:
                    if j['verifier'] != m['id']:
                        continue
                    if stop.is_set():
                        return
                    req = req_for(by_id[j['item_id']], m, plan['collection_id'])
                    if r.cache_key(req) != j['request_key']:
                        raise ValueError('unplanned request')
                    if j['request_key'] in index:
                        cache.import_call(Call(**index[j['request_key']]))
                    try:
                        call, reused = cache.get(r, req)
                    except OfflineCacheMiss:
                        if not allow_api:
                            with mutex:
                                results[j['request_key']] = {**j, 'score': None, 'failure': None, 'not_sent': True}
                            continue
                        pacer.acquire()
                        if stop.is_set():
                            return
                        call, reused = cache.get(r, req, allow_api=True)
                    e = check_call(call, r, req, cfg, screen)
                    score, failure = parse_signal(call.response, 'probability')
                    with mutex:
                        results[j['request_key']] = {**j, 'score': score, 'failure': failure,
                            'estimated_usd': e['estimated_usd'], 'new': not reused,
                            'response_record_hash': digest(asdict(call))}
                    snapshot('collecting')
            except Exception:
                stop.set()
                raise
        try:
            if allow_api:
                inventory = budget_inventory()
                remaining = cfg['stage_budget_usd']-cache.summary()['estimated_usd']
                if inventory['committed_estimate_and_reservations_usd']+remaining > cfg['total_authorized_usd']:
                    raise ValueError('authorized overall budget exceeded')
                atomic_json(phase/'billing_check.json', check_billing(cfg['project']))
            snapshot('preflight')
            parallel_map(one_model, screen['models'], cfg['workers'])
            if any(r.get('not_sent') for r in results.values()):
                return snapshot('dry_run', planning_new_usd_at_1024_output_tokens=planning,
                                missing_by_verifier=dict(missing))
            if len(results) != len(selected):
                raise ValueError('incomplete verifier collection')
            if limit is not None:
                state = snapshot('batch_complete')
                if max(state['invalid_by_verifier'].values(), default=0) > cfg['format_gate_max_invalid_per_verifier']:
                    raise ValueError('format gate failed')
                immutable(phase/'format_gate.json', state)
                return state
            rows = []
            for c in cs:
                signals = {}
                if eligible(c):
                    for m in screen['models']:
                        key = runner_for(m, cfg['project']).cache_key(req_for(c, m))
                        signals[m['id']+':probability'] = {k: results[key][k] for k in (
                            'score', 'failure', 'estimated_usd', 'request_key', 'response_record_hash')}
                rows.append({'item_id': c.item_id, 'partition': partition, 'answer': c.answer,
                             'prior': c.p_correct, 'signals': signals})
            value = seal({'protocol_id': protocol['protocol_id'], 'collection_id': plan['collection_id'],
                'partition': partition, 'rows': rows, 'labels_read': False}, 'signals_id')
            immutable(phase/'signals.json', value)
            return snapshot('complete', signals_id=value['signals_id'])
        except Exception as exc:
            snapshot('blocked', error_type=type(exc).__name__, error=str(exc)[:500],
                     http_status=getattr(exc, 'status', None))
            raise


def read_signals(out, part, protocol):
    path = out/part/'signals.json'
    value = json.loads(path.read_text()); unseal(value, 'signals_id')
    if value['protocol_id'] != protocol['protocol_id'] or value['partition'] != part:
        raise ValueError('signal dataset/partition differs')
    if any(r['partition'] != part for r in value['rows']):
        raise ValueError('signal row partition differs')
    return value


def planner_for(fit, order, loss, settings):
    m = [{1: fit['likelihoods'][t]['p_bin_if_correct'], 0: fit['likelihoods'][t]['p_bin_if_incorrect']} for t in order]
    costs = [fit['expected_cost_usd'][t] for t in order]
    return ExactPlanner(JointLikelihood(m, None, INF),
        [settings['utility_per_usd']*c for c in costs], settings['correct_reward'], loss), costs


def fit(out):
    protocol, cfg, screen, _ = load(out)
    if (out/'policy.json').exists():
        return verify_policy(out, protocol)
    if (out/'evaluation/signals.json').exists() or (out/'evaluation/plan.json').exists():
        raise ValueError('policy must be frozen before evaluation signal access')
    signals = read_signals(out, 'calibration', protocol); rows = signals['rows']
    bundle = Path(protocol['sources']['baseline_bundle'])
    manifest = json.loads((bundle/'bundle_manifest.json').read_text()); unseal(manifest, 'bundle_id')
    label_path = bundle/'calibration_records.jsonl'
    if file_digest(label_path) != manifest['files'][label_path.name]:
        raise ValueError('development labels changed')
    labels = read_jsonl(label_path)
    if any(r['partition'] != 'calibration' for r in labels) or {r['item_id'] for r in labels} != {r['item_id'] for r in rows}:
        raise ValueError('development label membership differs')
    lookup = {r['item_id']: r['correct_index'] for r in labels}
    valid = [{**r, 'outcome': int('ABCD'.index(r['answer']) == lookup[r['item_id']])}
             for r in rows if candidate_valid(r)]
    settings = cfg['settings']; tags = [v+':probability' for v in cfg['verifiers']]
    fitted = {'likelihoods': {}, 'expected_cost_usd': {}}
    for t in tags:
        usable = [r for r in valid if r['signals'][t]['score'] is not None]
        table = fit_verifier_likelihood([r['outcome'] for r in usable],
            [r['signals'][t]['score'] for r in usable], bins=settings['probability_bins'], laplace=settings['laplace'])
        fitted['likelihoods'][t] = {**asdict(table), 'fit_n': len(usable),
            'fit_correct_n': sum(r['outcome'] for r in usable), 'fit_wrong_n': sum(1-r['outcome'] for r in usable),
            'likelihood_ratio_by_bin': [a/b for a,b in zip(table.p_bin_if_correct,table.p_bin_if_incorrect)]}
        fitted['expected_cost_usd'][t] = float(np.mean([r['signals'][t]['estimated_usd'] for r in valid]))
    ranked = {}; selections = {}
    priors = np.array([r['prior'] for r in valid])
    for loss in settings['losses']:
        rankings = []
        for order in all_orders(tags):
            planner, costs = planner_for(fitted, order, loss, settings)
            rankings.append({'order': list(order), 'training_mean_J0': float(planner.j(0, priors).mean()),
                             'expected_costs_usd': costs})
        rankings.sort(key=lambda r: (-r['training_mean_J0'], r['order']))
        ranked[str(int(loss))] = rankings; selections[str(int(loss))] = rankings[0]
    value = seal({'protocol_id': protocol['protocol_id'], 'settings': settings, **fitted,
        'selected_triples': selections, 'ranked_60_orders': ranked,
        'development_questions': len(rows), 'development_valid': len(valid),
        'development_wrong_valid': sum(1-r['outcome'] for r in valid),
        'development_signals_sha256': file_digest(out/'calibration/signals.json'),
        'development_labels_sha256': file_digest(label_path), 'development_item_ids_sha256': digest(sorted(lookup)),
        'implementation': implementation(), 'frozen_at_unix': time.time(),
        'confidence_calibrated': False, 'evaluation_labels_read_for_fitting': False,
        'evaluation_signals_read_for_fitting': False}, 'policy_id')
    immutable(out/'policy.json', value)
    return value


class PrefixSignals:
    """Log every read; a scorer cannot silently inspect a future verifier."""
    def __init__(self, signals, order):
        self.signals, self.order, self.reads = signals, list(order), []

    def __getitem__(self, tag):
        if tag not in self.reads:
            if len(self.reads) >= len(self.order) or tag != self.order[len(self.reads)]:
                raise AssertionError('future verifier access')
            self.reads.append(tag)
        return self.signals[tag]


def decisions(out):
    protocol, cfg, _, _ = load(out); policy = verify_policy(out, protocol)
    signals = read_signals(out, 'evaluation', protocol); rows = signals['rows']
    if any('outcome' in r or 'correct_index' in r for r in rows):
        raise ValueError('labels reached public decisions')
    edges = tuple(np.linspace(0., 1., cfg['settings']['probability_bins']+1).tolist())
    by_loss, always_brier = {}, {}
    for loss in cfg['settings']['losses']:
        key = str(int(loss)); triple = policy['selected_triples'][key]['order']; arms = {}
        arms['prior_only'] = [baseline_decision(r['prior'], candidate_valid(r), loss,
            cfg['settings']['correct_reward']) for r in rows]
        for name, order in [('flash', ['gemini_flash:probability']), ('triple', triple)]:
            planner, costs = planner_for(policy, order, loss, cfg['settings']); ds=[]; forced=[]
            for r in rows:
                public = PrefixSignals(r['signals'], order)
                d = execute(r['prior'], candidate_valid(r), planner, order, costs, public, edges)
                if public.reads != list(order[:d['verifiers_used']]):
                    raise AssertionError('query prefix/count mismatch')
                # Future values removed: repeat the public run on only the prefix
                # actually consumed. The decision must be identical.
                replay = execute(r['prior'], candidate_valid(r), planner, order, costs,
                    PrefixSignals({t: r['signals'][t] for t in public.reads}, order), edges)
                if digest(d) != digest(replay):
                    raise AssertionError('prefix-only replay disagrees')
                ds.append(d)
                forced.append(execute(r['prior'], candidate_valid(r), planner, order, costs,
                    PrefixSignals(r['signals'], order), edges, force_all=True))
            arms[name] = ds
            always_brier.setdefault(key, {})[name] = forced
        by_loss[key] = {'order': triple, 'arms': arms}
    value = seal({'protocol_id': protocol['protocol_id'], 'policy_id': policy['policy_id'],
        'signals_id': signals['signals_id'], 'policy_sha256': file_digest(out/'policy.json'),
        'signals_sha256': file_digest(out/'evaluation/signals.json'), 'item_ids': [r['item_id'] for r in rows],
        'by_loss': by_loss, 'always_query_diagnostics': always_brier,
        'labels_read': False, 'confidence_calibrated': False,
        'future_signal_access': False, 'prefix_only_replay_checks': len(rows)*2*len(cfg['settings']['losses'])}, 'decisions_id')
    immutable(out/'evaluation/decisions.json', value)
    return value


def score(out):
    protocol, cfg, _, _ = load(out); policy = verify_policy(out, protocol)
    signals = read_signals(out, 'evaluation', protocol)
    d = json.loads((out/'evaluation/decisions.json').read_text()); unseal(d, 'decisions_id')
    if (d['policy_id'] != policy['policy_id'] or d['signals_id'] != signals['signals_id']
        or d['policy_sha256'] != file_digest(out/'policy.json')
        or d['signals_sha256'] != file_digest(out/'evaluation/signals.json')
        or d['item_ids'] != [r['item_id'] for r in signals['rows']]):
        raise ValueError('decisions, policy or signals changed')
    bundle = Path(protocol['sources']['baseline_bundle'])
    manifest = json.loads((bundle/'bundle_manifest.json').read_text()); unseal(manifest, 'bundle_id')
    path = bundle/'evaluation_labels.jsonl'
    if file_digest(path) != manifest['files'][path.name]:
        raise ValueError('evaluation key changed')
    records = read_jsonl(path); labels = {r['item_id']: r['correct_index'] for r in records}
    if len(records) != 196 or len(labels) != 196 or set(labels) != set(d['item_ids']):
        raise ValueError('evaluation key membership differs')
    rows = [{**r, 'outcome': int(r['answer'] is not None and r['answer'] == 'ABCD'[labels[r['item_id']]])}
            for r in signals['rows']]
    seed = int(policy['policy_id'][:8], 16); repeats = cfg['settings']['bootstrap_repeats']
    result = {}
    for k, by in d['by_loss'].items():
        summaries, utils = {}, {}
        for arm, ds in by['arms'].items():
            s = summarize_policy(rows, ds, cfg['settings'], float(k)); utils[arm] = np.array(s.pop('utility_per_item'))
            s['output_error_rate'] = s['wrong_released']/s['released'] if s['released'] else None
            s['output_error_wilson95'] = rate_interval([1-r['outcome'] for r,x in zip(rows,ds) if x['action']=='assert'])
            s['acceptance_rate'] = s['released']/len(rows)
            s['calls_per_question'] = s['queries']/len(rows)
            s['expected_usd_per_question'] = s['expected_cost_usd']/len(rows)
            s['usage_estimated_usd_per_question'] = s['usage_estimated_cost_usd']/len(rows)
            s['query_histogram'] = dict(Counter(str(x['verifiers_used']) for x in ds))
            s['mean_utility_ci95'] = boot(utils[arm], repeats, seed)['ci95']
            summaries[arm] = s
        # Always-query Brier is a separate diagnostic on matched valid signals;
        # the sequential policy's final beliefs include early-stop cases.
        forced = d['always_query_diagnostics'][k]
        common = [i for i,r in enumerate(rows) if candidate_valid(r) and all(not forced[a][i]['failure'] for a in ('flash','triple'))]
        brier, sq = {}, {}
        for arm in ('prior_only','flash','triple'):
            ps = [rows[i]['prior'] if arm=='prior_only' else forced[arm][i]['posterior'] for i in common]
            sq[arm] = np.array([(p-rows[i]['outcome'])**2 for p,i in zip(ps,common)])
            brier[arm] = float(sq[arm].mean()) if common else None
        result[k] = {'triple_order': by['order'], 'arms': summaries,
            'paired_utility_flash_minus_prior': boot(utils['flash']-utils['prior_only'],repeats,seed),
            'paired_utility_triple_minus_flash': boot(utils['triple']-utils['flash'],repeats,seed),
            'brier_always_query_common': {'n':len(common),'brier':brier,
                'flash_minus_prior':boot(sq['flash']-sq['prior_only'],repeats,seed) if common else None,
                'triple_minus_flash':boot(sq['triple']-sq['flash'],repeats,seed) if common else None}}
    value = seal({'protocol_id': protocol['protocol_id'], 'policy_id':policy['policy_id'],
        'decisions_id':d['decisions_id'], 'evaluation_signals_id':signals['signals_id'],
        'settings':cfg['settings'],'generator':'gpt_oss_120b','evaluation_n':len(rows),
        'results':result,'prefix_only_replay_checks':d['prefix_only_replay_checks'],
        'confidence_calibrated':False,'refit_on_evaluation':False,'evaluation_key_sha256':file_digest(path),
        'collection_budget':cache_for(out,cfg,protocol['screen']).summary(),
        'notes':cfg['notes']},'result_id')
    immutable(out/'result.json',value)
    return value


def pipeline(out):
    protocol, cfg, _, _ = load(out)
    def state(stage, **extra):
        atomic_json(out/'pipeline_status.json', {'protocol_id': protocol['protocol_id'],
            'stage': stage, 'updated_at_unix': time.time(), **extra})
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            state('development_format_check')
            if not (out/'calibration/format_gate.json').exists():
                collect(out,'calibration',True,cfg['format_gate_questions'])
            state('development_collection'); collect(out,'calibration',True)
            state('development_fit'); p=fit(out)
            state('evaluation_format_check',policy_id=p['policy_id'])
            if not (out/'evaluation/format_gate.json').exists():
                collect(out,'evaluation',True,cfg['format_gate_questions'])
            state('evaluation_collection',policy_id=p['policy_id']);collect(out,'evaluation',True)
            state('label_free_decisions',policy_id=p['policy_id']);decisions(out)
            state('offline_scoring',policy_id=p['policy_id']);r=score(out)
            state('complete',policy_id=p['policy_id'],result_id=r['result_id'])
            return {'stage':'complete','policy_id':p['policy_id'],'result_id':r['result_id']}
        except Exception as exc:
            state('blocked',error_type=type(exc).__name__,error=str(exc)[:500],http_status=getattr(exc,'status',None))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','collect','fit','decisions','score','pipeline'])
    p.add_argument('--config',type=Path,default=Path('configs/gpqa_gptoss_verification_regen_20261020.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partition',choices=['calibration','evaluation'],default='calibration')
    p.add_argument('--limit',type=int)
    p.add_argument('--allow-api',action='store_true')
    a=p.parse_args()
    if a.command=='prepare':
        v=prepare(a.config,a.output);v={k:v[k] for k in ('protocol_id','candidate_counts','eligible_counts')}
    elif a.command=='collect':v=collect(a.output,a.partition,a.allow_api,a.limit)
    elif a.command=='fit':
        v=fit(a.output);v={k:v[k] for k in ('policy_id','development_valid','development_wrong_valid','selected_triples')}
    elif a.command=='decisions':
        v=decisions(a.output);v={k:v[k] for k in ('decisions_id','prefix_only_replay_checks')}
    elif a.command=='score':
        v=score(a.output);v={k:v[k] for k in ('result_id','evaluation_n','results')}
    else:
        if not a.allow_api:p.error('pipeline requires --allow-api')
        v=pipeline(a.output)
    print(json.dumps(v,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
