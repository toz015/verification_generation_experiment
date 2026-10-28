"""Offline selection and execution of fixed nested singleton verifier layers.

All selection uses training-fold likelihoods, costs and raw priors. The executor
accepts only a public prior and a lazy observation callback. No API collection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from itertools import combinations, permutations, product
import json
from pathlib import Path

import numpy as np

from vgx.common.billing import estimate_call
from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa.generator_calibration import source, index_cache, audit_coverage
from vgx.gpqa.matched_coverage import release_weights
from vgx.gpqa.planner import NestedPlanner, ImpossibleObservation
from vgx.gpqa.raw_prior_analysis import candidate_valid, baseline_decision, summarize_policy
from vgx.gpqa.report import mean_interval
from vgx.gpqa.score import VerifierLikelihood
from vgx.gpqa.sequential import seal
from vgx.gpqa.signal_study import runner_for, request_for, parse_signal


def execute_public(prior, valid, planner, order, costs_usd, query, *, force_all=False):
    """Only query(stage, tag) can reveal an observation; outcome is not an input.

    force_all is an explicitly different baseline: all requests are charged even
    after a parse failure. It abstains if any returned signal cannot be used.
    """
    if not valid:
        return baseline_decision(prior, False, planner.loss, planner.reward)
    belief, observations, trace, failure = prior, [], [], None
    while True:
        stage = len(observations)
        decision = planner.decide(stage, belief)
        trace.append(decision)
        if stage == len(order) or (not force_all and decision['action'] != 'query'):
            break
        observation = dict(query(stage, order[stage]))
        observations.append(observation)
        if observation['score'] is None:
            failure = failure or 'missing_verifier_score'
        elif failure is None:
            try:
                belief = planner.update(stage, belief, observation['score'])
            except ImpossibleObservation:
                failure = 'impossible_verifier_observation'
        if failure and not force_all:
            break
    used = len(observations)
    # Independent product-form Bayes audit on the actually observed valid prefix.
    if not failure:
        numerator, alternative = prior, 1-prior
        for stage, observation in enumerate(observations):
            channel = planner.likelihoods[stage]
            index = channel.bin_index(observation['score'])
            numerator *= channel.p_bin_if_correct[index]
            alternative *= channel.p_bin_if_incorrect[index]
        if not np.isclose(belief, numerator/(numerator+alternative), rtol=1e-12, atol=1e-12):
            raise AssertionError('sequential and product-form Bayes disagree')
    return {'action': 'abstain' if failure else decision['action'], 'posterior': belief,
            'verifiers_used': used, 'failure': failure, 'signal_reads': used,
            'expected_cost_usd': sum(costs_usd[:used]),
            'usage_estimated_cost_usd': sum(o['estimated_usd'] for o in observations),
            'queried_tags': list(order[:used]), 'observations': observations, 'trace': trace}


def planner_for(order, arms, cfg, loss):
    channels = [VerifierLikelihood(**arms[t]['likelihood']) for t in order]
    costs = [arms[t]['expected_cost_usd'] for t in order]
    return NestedPlanner(channels, [cfg['utility_per_usd']*c for c in costs],
                         cfg['correct_reward'], loss, grid_size=cfg['grid_size']), costs


def select_plans(fold, training_priors, cfg, loss):
    """No held-out labels, priors, scores or costs enter selection."""
    arms = fold['arms']
    models = sorted({t.split(':')[0] for t in arms})
    ranked, best_by_combo = [], {}
    for combo in combinations(models, 3):
        key = '+'.join(combo)
        for ordered_models in permutations(combo):
            for formats in product(('binary', 'probability'), repeat=3):
                order = tuple(m+':'+s for m,s in zip(ordered_models, formats))
                planner, _ = planner_for(order, arms, cfg, loss)
                value = float(np.interp(training_priors, planner.grid, planner.j[0]).mean())
                entry = {'combination': key, 'order': list(order), 'training_model_value': value}
                ranked.append(entry)
                previous = best_by_combo.get(key)
                if previous is None or (-value, order) < (-previous['training_model_value'], tuple(previous['order'])):
                    best_by_combo[key] = entry
    ranked.sort(key=lambda r: (-r['training_model_value'], r['order']))
    singles = []
    for tag in sorted(arms):
        planner, _ = planner_for([tag], arms, cfg, loss)
        singles.append({'order': [tag], 'training_model_value': float(
            np.interp(training_priors, planner.grid, planner.j[0]).mean())})
    singles.sort(key=lambda r: (-r['training_model_value'], r['order']))
    return ranked, best_by_combo, singles


def evaluate(rows, fit, cfg, outdir):
    ids = [r['item_id'] for r in rows]
    if ids != fit['item_ids'] or [r['prior'] for r in rows] != fit['raw_priors']:
        raise ValueError('frozen candidate order/prior mismatch')
    if [r['outcome'] for r in rows] != fit['outcomes']:
        raise ValueError('calibration outcome mismatch')
    valid = [candidate_valid(r) for r in rows]
    lookup = {v:i for i,v in enumerate(ids)}
    result = {'item_ids': ids, 'fold_assignment': fit['fold_assignment'], 'seed': fit['seed'],
              'policies': {}, 'selections': {}, 'replay_agreements': 0}
    for loss in [cfg['primary_incorrect_loss'], *cfg['sensitivity_incorrect_losses']]:
        loss_key = str(loss)
        predictions = {'raw': [baseline_decision(r['prior'], v, loss, cfg['correct_reward'])
                               for r,v in zip(rows, valid)]}
        selections = []
        for fold_id, fold in enumerate(fit['folds']):
            tr, va = [lookup[x] for x in fold['train_ids']], [lookup[x] for x in fold['validation_ids']]
            if set(tr) & set(va): raise ValueError('training/validation overlap')
            priors = [rows[i]['prior'] for i in tr if valid[i]]
            ranked, combos, singles = select_plans(fold, priors, cfg, loss)
            choices = {'selected_three': ranked[0], 'selected_single': singles[0],
                       'always_selected_three': ranked[0],
                       **{'combo:'+k:v for k,v in combos.items()}}
            selections.append({'fold': fold_id, 'train_ids': fold['train_ids'],
                               'validation_ids': fold['validation_ids'], 'ranked_candidates': ranked,
                               'ranked_singles': singles})
            for name, selected in choices.items():
                order = selected['order']
                planner, costs = planner_for(order, fold['arms'], cfg, loss)
                predictions.setdefault(name, [None]*len(rows))
                if name == 'selected_three':
                    # Compressed actual J/Q tables and reconstruction metadata.
                    stem = outdir/f'loss_{loss_key}_fold_{fold_id}'
                    stem.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(str(stem)+'.npz', grid=planner.grid, J=np.array(planner.j), Q=np.array(planner.q))
                    atomic_json(str(stem)+'.json', {'order': order, 'expected_cost_usd': costs,
                        'likelihoods': [asdict(x) for x in planner.likelihoods], 'grid_size': len(planner.grid),
                        'reward': planner.reward, 'loss': planner.loss, 'utility_per_usd': cfg['utility_per_usd'],
                        'tables_sha256': file_digest(str(stem)+'.npz')})
                for i in va:
                    reads = []
                    def query(stage, tag):
                        if stage != len(reads) or tag != order[stage]:
                            raise AssertionError('non-prefix observation access')
                        reads.append(tag)
                        return rows[i]['signals'][tag]
                    forced = name == 'always_selected_three'
                    decision = execute_public(rows[i]['prior'], valid[i], planner, order, costs, query, force_all=forced)
                    if reads != order[:decision['verifiers_used']]: raise AssertionError('query accounting mismatch')
                    if not forced and valid[i]:
                        class LazyScores:
                            def __len__(self): return len(order)
                            def __getitem__(self, stage): return query(stage, order[stage])['score']
                        reads.clear()
                        replay = planner.replay(rows[i]['prior'], LazyScores())
                        if any(replay[k] != decision[k] for k in ('action','posterior','verifiers_used','failure')):
                            raise AssertionError('public executor / replay mismatch')
                        result['replay_agreements'] += 1
                    predictions[name][i] = decision
        summaries = {}
        for name, decisions in predictions.items():
            if any(x is None for x in decisions): raise ValueError('missing held-out prediction')
            s = summarize_policy(rows, decisions, cfg, loss)
            utility = np.array(s.pop('utility_per_item'))
            s['query_count_histogram'] = dict(sorted(Counter(d['verifiers_used'] for d in decisions).items()))
            s['queried_tags'] = dict(Counter(t for d in decisions for t in d.get('queried_tags', [])))
            s['mean_utility_interval_fixed_oof'] = mean_interval(utility, cfg['bootstrap_repeats'], fit['seed'])
            raw = summarize_policy(rows, predictions['raw'], cfg, loss)['utility_per_item']
            single = summarize_policy(rows, predictions['selected_single'], cfg, loss)['utility_per_item']
            s['paired_utility_vs_raw'] = mean_interval(utility-np.array(raw), cfg['bootstrap_repeats'], fit['seed'])
            s['paired_utility_vs_selected_single'] = mean_interval(utility-np.array(single), cfg['bootstrap_repeats'], fit['seed'])
            weights = np.zeros(len(rows))
            for fold in fit['folds']:
                ix = [lookup[x] for x in fold['validation_ids']]
                k = sum(decisions[i]['action']=='assert' for i in ix)
                weights[ix] = release_weights([rows[i]['prior'] for i in ix], [valid[i] for i in ix], k)
            gross = np.array([cfg['correct_reward'] if r['outcome'] else -loss for r in rows])
            s['paired_utility_vs_matched_coverage'] = mean_interval(utility-weights*gross, cfg['bootstrap_repeats'], fit['seed'])
            summaries[name] = s
        result['policies'][loss_key] = summaries
        result['selections'][loss_key] = selections
        atomic_json(outdir/f'decisions_loss_{loss_key}.json', predictions)
    return result


def analyze(config_path, output):
    config_path, output = Path(config_path), Path(output)
    protocol_cfg = json.loads(config_path.read_text())
    if protocol_cfg['new_api_requests'] != 0 or protocol_cfg['evaluation_labels_read']:
        raise ValueError('offline calibration-partition analysis only')
    cfg = json.loads(Path(protocol_cfg['source_config']).read_text())
    source_root = Path(protocol_cfg['source_results'])
    previous = json.loads((source_root/'protocol.json').read_text())
    if previous['config'] != cfg: raise ValueError('source config mismatch')
    plan, groups, labels, pilot, label_hash, bundle_id = source(Path(cfg['generator_results']))
    screen = json.loads(Path(cfg['screen_config']).read_text())
    index = index_cache(cfg['cache_roots'])
    coverage = audit_coverage(groups, labels, screen['models'], screen['project'], index)
    if coverage['missing_requests']: raise ValueError('incomplete cache coverage')
    source_paths = [source_root/g/c/f'seed_{s}.json' for g in groups for c in cfg['cohorts'] for s in cfg['seeds']]
    impl = ('nested_combinations.py','planner.py','score.py','signal_study.py','generator_calibration.py',
            'raw_prior_analysis.py','matched_coverage.py','report.py')
    protocol = seal({'config': protocol_cfg, 'config_sha256': file_digest(config_path),
        'source_config_sha256': file_digest(protocol_cfg['source_config']),
        'source_analysis_id': previous['analysis_id'], 'bundle_id': bundle_id,
        'calibration_label_sha256': label_hash,
        'source_candidates': {g:digest([asdict(c) for c in cs]) for g,cs in groups.items()},
        'source_files': {str(p):file_digest(p) for p in source_paths},
        'used_cache_record_hashes': coverage['used_cache_record_hashes'],
        'implementation_sha256': {n:file_digest(Path(__file__).with_name(n)) for n in impl},
        'new_api_requests': 0, 'evaluation_labels_read': False,
        'limitations': ['Development-set selection after earlier screening, not independent confirmation.',
            'Training J0 ranking assumes the fitted probabilities and conditional independence are correct.',
            'Initial raw priors are not calibrated; verifier likelihoods still use training labels.',
            'Valid-bin tables omit parse-failure risk; a queried invalid signal forces abstention.',
            'Each fold fixes one order before seeing validation answers; no within-question adaptive verifier choice.',
            'Fixed OOF intervals do not include refitting/model-selection uncertainty; no multiple-comparison correction.',
            'Feedback is a controller belief update, not an observed LLM private-belief update; no reward training or terminal audit.',
            'Historical Diamond use precludes describing it as globally untouched; this run reads no Diamond answer keys.',
            'Policy cost is counterfactual cached-call token pricing, not new spend or confirmed billing.']}, 'analysis_id')
    path = output/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('protocol changed; new output required')
    atomic_json(path, protocol)
    summaries = {}
    for g, candidates in groups.items():
        rows = []
        for c in candidates:
            if c.partition != 'calibration': raise ValueError('evaluation candidate encountered')
            row = {'item_id':c.item_id, 'partition':c.partition, 'answer':c.answer, 'prior':c.p_correct,
                   'outcome': int(c.answer is not None and 'ABCD'.index(c.answer)==labels[c.item_id]), 'signals':{}}
            if candidate_valid(row):
                for m in screen['models']:
                    runner = runner_for(m, screen['project'])
                    for signal in screen['signals']:
                        key = runner.cache_key(request_for(c,m,signal)); call = index[key]
                        est = estimate_call(call,screen['pricing'])['estimated_usd']
                        if est is None: raise ValueError('unpriced call')
                        score, failure = parse_signal(call['response'],signal)
                        row['signals'][m['id']+':'+signal] = {'score':score,'failure':failure,
                            'estimated_usd':est,'request_key':key}
            rows.append(row)
        for cohort in cfg['cohorts']:
            subset = rows if cohort=='all_calibration' else [r for r in rows if r['item_id'] not in pilot]
            for seed in cfg['seeds']:
                rel = Path(g)/cohort/f'seed_{seed}'
                fit = json.loads((source_root/(str(rel)+'.json')).read_text())
                value = evaluate(subset, fit, cfg, output/rel)
                atomic_json(output/rel/'results.json', value)
                summaries[str(rel)] = {k:v for k,v in value.items() if k not in ('item_ids','fold_assignment','selections')}
                print(json.dumps({'finished':str(rel),'replay_agreements':value['replay_agreements']}),flush=True)
    _, end_groups, _, _, end_label, end_bundle = source(Path(cfg['generator_results']))
    end_index = index_cache(cfg['cache_roots'])
    if end_label!=label_hash or end_bundle!=bundle_id or any(digest([asdict(c) for c in end_groups[g]])!=protocol['source_candidates'][g] for g in groups):
        raise ValueError('frozen source changed')
    if any(digest(end_index[k])!=v for k,v in protocol['used_cache_record_hashes'].items()): raise ValueError('cached response changed')
    if any(file_digest(p)!=h for p,h in protocol['source_files'].items()): raise ValueError('source fit changed')
    result = {'analysis_id':protocol['analysis_id'],'results':summaries,'new_api_spend_usd':0,
              'evaluation_labels_read':False,'source_integrity_unchanged':True,'confirmed_billing_usd':None,
              'project':screen['project'],'limitations':protocol['limitations']}
    atomic_json(output/'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/gpqa_nested_combinations.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.config,args.output)
