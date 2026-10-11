"""Phase II: risk certification with separate fitting, risk-calibration and evaluation roles.

Workflow (each stage reads only the labels of its own role):
  1. fit       - pooled Flash likelihood, expected cost and the frozen sequential planner,
                 fitted on fit-role questions only.
  2. calibrate - score calibration-role questions without labels, then count accepted (m)
                 and incorrect accepted (k) for every rule of the predeclared family; exact
                 one-sided binomial bounds with Bonferroni over the whole family; select the
                 certified rule with the highest acceptance rate (cost, then declared order,
                 break ties). The selection is sealed before any evaluation label is read.
  3. evaluate  - apply the sealed rule to evaluation-role questions and report observed
                 conditional error, acceptance rate, queries and usage-based cost.

Partitions are by question ID and shared by all generators. A certification claim needs
calibration and evaluation data that were not used to design the family; the existing
development data were inspected repeatedly, so a split of them is a software dry run only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa.risk_control import (
    METHODS, certify_bonferroni, certify_fixed_sequence, clopper_pearson, fit_pooled, generator_rows,
    min_accepted, accepted_for_power, partition_questions, rule_counts, score_method, sequential_planner, table_dict,
)
from vgx.gpqa.sequential import seal, unseal

IMPLEMENTATION = ('risk_control.py', 'risk_certification.py', 'planner.py', 'raw_prior_analysis.py', 'score.py')


def declared_family(cfg):
    """Every (generator, method, threshold) rule eligible for joint selection, in declared order."""
    family = cfg['certification']['family']
    rules = []
    for generator in family['generators']:
        for method in family['methods']:
            for threshold in sorted(family['thresholds'][method]):
                rules.append({'rule_id': f'{generator}:{method}:{threshold:g}', 'generator': generator,
                              'method': method, 'threshold': float(threshold)})
    return rules


def unlabeled(row):
    return {k: v for k, v in row.items() if k != 'correct'}


def fit_models(export, roles, cfg):
    """Fit-role rows only (labels included); nothing from calibration or evaluation roles."""
    settings, models = cfg['frozen_settings'], {}
    for generator in cfg['certification']['family']['generators']:
        rows = [r for q, r in generator_rows(export, generator).items() if roles.get(q) == 'fit']
        likelihood, cost = fit_pooled(rows, settings['probability_bins'], settings['laplace'])
        models[generator] = {'likelihood': likelihood, 'expected_cost_usd': cost,
                             'planner': sequential_planner(likelihood, cost, settings, cfg['sequential_base_loss']),
                             'fit_questions': len(rows), 'fit_correct': sum(r['correct'] for r in rows if r['valid']),
                             'fit_incorrect': sum(1 - r['correct'] for r in rows if r['valid'])}
    return models


def score_role(export, models, roles, role, cfg):
    """Label-free scores for every question of one role, per generator and method."""
    out = {}
    for generator, model in models.items():
        rows = [unlabeled(r) for q, r in sorted(generator_rows(export, generator).items()) if roles.get(q) == role]
        out[generator] = {m: [score_method(m, r, likelihood=model['likelihood'], expected_cost_usd=model['expected_cost_usd'],
                                           planner=model['planner']) for r in rows] for m in METHODS}
    return out


def role_labels(export, generator, roles, role):
    return {q: r['correct'] for q, r in generator_rows(export, generator).items() if roles.get(q) == role}


def rule_statistics(scored, labels, rules):
    stats = []
    for rule in rules:
        c = rule_counts(scored[rule['generator']][rule['method']], labels[rule['generator']], rule['threshold'])
        stats.append({**rule, **c})
    return stats


def calibrate(export, roles, cfg, alpha):
    """Fit on the fit role, count on the calibration role, certify, and return a sealable selection."""
    models = fit_models(export, roles, cfg)
    scored = score_role(export, models, roles, 'calibration', cfg)
    labels = {g: role_labels(export, g, roles, 'calibration') for g in models}
    rules = declared_family(cfg)
    stats = rule_statistics(scored, labels, rules)
    table, selected = certify_bonferroni(stats, alpha, cfg['delta'])
    chains = [[s for s in stats if (s['generator'], s['method']) == (g, m)][::-1]
              for g in cfg['certification']['family']['generators'] for m in cfg['certification']['family']['methods']]
    alt_table, alt_selected = certify_fixed_sequence(chains, alpha, cfg['delta'])
    best_per_family = {}
    for g in models:
        for m in METHODS:
            sub = [r for r in table if r['generator'] == g and r['method'] == m and r['certified']]
            best_per_family[f'{g}:{m}'] = (min(sub, key=lambda r: (-r['m']/r['N'], r['expected_cost_usd']/r['N'],
                                                                   r['family_order']))['rule_id'] if sub else None)
    selection = seal({'alpha': alpha, 'delta': cfg['delta'], 'family_size': len(rules), 'family': rules,
                      'procedure': cfg['certification']['procedure'],
                      'selected_rule': selected['rule_id'] if selected else None,
                      'status': 'certified_on_calibration' if selected else cfg['certification']['no_rule_message'],
                      'best_certified_per_generator_method': best_per_family,
                      'models': {g: {'likelihood': table_dict(m['likelihood']), 'expected_cost_usd': m['expected_cost_usd'],
                                     'fit_questions': m['fit_questions'], 'fit_correct': m['fit_correct'],
                                     'fit_incorrect': m['fit_incorrect']} for g, m in models.items()},
                      'roles_sha256': digest(roles)}, 'selection_id')
    return {'selection': selection, 'certification_table': table, 'fixed_sequence_table': alt_table,
            'fixed_sequence_selected': alt_selected['rule_id'] if alt_selected else None,
            'calibration_items': {g: {m: v for m, v in d.items()} for g, d in scored.items()}}


def evaluate(export, roles, cfg, selection):
    """Apply sealed rules to the evaluation role. Reads evaluation labels only."""
    body = unseal(selection, 'selection_id')
    if body['roles_sha256'] != digest(roles):
        raise ValueError('partition differs from the one used for selection')
    models = fit_models(export, roles, cfg)
    for g, m in models.items():
        if (digest(table_dict(m['likelihood'])) != digest(body['models'][g]['likelihood'])
                or m['expected_cost_usd'] != body['models'][g]['expected_cost_usd']):
            raise ValueError('refitted model differs from the sealed model')
    scored = score_role(export, models, roles, 'evaluation', cfg)
    labels = {g: role_labels(export, g, roles, 'evaluation') for g in models}
    wanted = {r for r in [body['selected_rule'], *body['best_certified_per_generator_method'].values()] if r}
    rows = {rule['rule_id']: evaluation_row(rule, scored, labels, body['selected_rule']) for rule in body['family']}
    out = {'selection_id': selection['selection_id'], 'status': body['status'], 'selected_rule': body['selected_rule'],
           'evaluation_items': scored,
           'rules': {r: v for r, v in rows.items() if r in wanted},
           'evaluation_questions': sum(1 for v in roles.values() if v == 'evaluation'),
           'descriptive_family_on_evaluation': list(rows.values()),
           'descriptive_note': 'Evaluation counts for every family rule, including rules that were not certified or '
                               'selected. Descriptive only: the sealed selection was fixed before these labels were read.'}
    if body['selected_rule'] is None:
        out['abstain_all'] = {g: {'N': len(scored[g]['A']), 'valid_candidates': sum(1 for s in scored[g]['A'] if s['eligible']),
                                  'm': 0, 'queries': 0, 'usage_cost_usd': 0.,
                                  'note': 'no rule was certified; abstaining on everything is not a certified 0% coverage rule'}
                              for g in scored}
    return out


def evaluation_row(rule, scored, labels, selected_rule):
    items = scored[rule['generator']][rule['method']]
    c = rule_counts(items, labels[rule['generator']], rule['threshold'])
    return {**rule, **c, 'valid_candidates': sum(1 for s in items if s['failure'] != 'invalid_generator'),
            'conditional_error_cp95': clopper_pearson(c['k'], c['m']),
            'acceptance_rate_cp95': clopper_pearson(c['m'], c['N']),
            'usage_cost_per_question': c['usage_cost_usd']/c['N'] if c['N'] else None,
            'usage_cost_per_accepted': c['usage_cost_usd']/c['m'] if c['m'] else None,
            'selected_overall': rule['rule_id'] == selected_rule}


def simulate_procedures(chains, n, alpha, delta, repeats, seed):
    """Monte Carlo check of both certification procedures on synthetic nested rules.

    Each chain lists score levels from the top: (fraction of questions, error rate). Rule i
    of a chain accepts its top i levels, so rules are nested and their counts dependent,
    as in the real family. Returns how often any rule whose true conditional error exceeds
    alpha (or equals it, the boundary case) is certified, and how often a rule is selected.
    """
    rng = np.random.default_rng(seed)
    truth, rules = [], []
    for c, levels in enumerate(chains):
        mass = np.cumsum([f for f, _ in levels])
        wrong = np.cumsum([f*e for f, e in levels])
        truth.append(wrong/mass)
        rules.append([{'rule_id': f'{c}:{i}', 'chain': c} for i in range(len(levels))])
    family = [r for chain in rules for r in chain]
    risky = {r['rule_id'] for c, chain in enumerate(rules) for i, r in enumerate(chain) if truth[c][i] >= alpha}
    counts = {'bonferroni': 0, 'fixed_sequence': 0}
    selected = {'bonferroni': 0, 'fixed_sequence': 0}
    for _ in range(repeats):
        stats = {}
        for c, levels in enumerate(chains):
            fractions = [f for f, _ in levels]
            drawn = rng.multinomial(n, [*fractions, 1 - sum(fractions)])[:-1]
            errors = rng.binomial(drawn, [e for _, e in levels])
            for i, rule in enumerate(rules[c]):
                stats[rule['rule_id']] = {**rule, 'N': n, 'm': int(drawn[:i + 1].sum()), 'k': int(errors[:i + 1].sum()),
                                          'expected_cost_usd': 0.}
        table, best = certify_bonferroni([stats[r['rule_id']] for r in family], alpha, delta)
        counts['bonferroni'] += any(r['certified'] and r['rule_id'] in risky for r in table)
        selected['bonferroni'] += best is not None
        ordered = [[stats[r['rule_id']] for r in chain] for chain in rules]
        table, best = certify_fixed_sequence(ordered, alpha, delta)
        counts['fixed_sequence'] += any(r['certified'] and r['rule_id'] in risky for r in table)
        selected['fixed_sequence'] += best is not None
    return {'n_per_chain': n, 'alpha': alpha, 'delta': delta, 'repeats': repeats, 'seed': seed,
            'true_conditional_error': [t.tolist() for t in truth], 'rules_at_or_above_alpha': len(risky),
            'family_size': len(family),
            'familywise_error': {k: v/repeats for k, v in counts.items()},
            'familywise_error_mc_se': {k: float(np.sqrt(max(v/repeats*(1 - v/repeats), 1e-12)/repeats)) for k, v in counts.items()},
            'any_rule_selected': {k: v/repeats for k, v in selected.items()}}


def validation_scenarios(alpha):
    """Synthetic scenarios (planning and software checks only, no development data)."""
    flat = [(.1, alpha)]*5
    top_heavy = [(.05, 3*alpha), (.1, alpha/4), (.15, alpha/4), (.1, alpha), (.1, 2*alpha)]
    low = [(.1, .4*alpha)]*5
    return {'all_rules_at_alpha': [flat]*6, 'non_monotone_top_error': [top_heavy]*6, 'all_rules_below_alpha': [low]*6}


def sample_size_plan(cfg, regions=None):
    """Accepted-example requirements and implied question counts (planning inputs, not results)."""
    plan, delta = cfg['sample_size'], cfg['delta']
    minimum, power = [], []
    for alpha in plan['alphas']:
        for family in plan['family_sizes']:
            for k in plan['observed_errors']:
                minimum.append({'alpha': alpha, 'delta': delta, 'family_size': family, 'observed_errors': k,
                                'min_accepted': min_accepted(k, alpha, delta/family)})
            for risk in plan['true_risks'][f'{alpha:.2f}']:
                power.append({'alpha': alpha, 'delta': delta, 'family_size': family, 'true_conditional_error': risk,
                              'power': plan['power'],
                              'accepted_needed': accepted_for_power(risk, alpha, delta/family, plan['power'])})
    reference = [{'check': f'zero errors, alpha=delta=0.05, M={family}', 'expected': expected,
                  'computed': min_accepted(0, .05, .05/family)} for family, expected in ((1, 59), (20, 117))]
    implied = []
    for rule in regions or []:
        rate, observed = rule['acceptance_rate'], rule['observed_conditional_error']
        for alpha in plan['alphas']:
            for family in plan['family_sizes']:
                row = {**rule, 'alpha': alpha, 'family_size': family}
                if not rate:
                    implied.append({**row, 'status': 'no accepted development examples; cannot be sized'})
                    continue
                for risk in [*plan['true_risks'][f'{alpha:.2f}'], observed]:
                    needed = accepted_for_power(risk, alpha, delta/family, plan['power']) if risk < alpha else None
                    implied.append({**row, 'assumed_true_conditional_error': risk,
                                    'assumption': 'observed development k/m' if risk == observed else 'planning grid',
                                    'accepted_needed': needed,
                                    'calibration_questions_needed': None if needed is None else math.ceil(needed/rate),
                                    'status': 'ok' if needed is not None else 'assumed risk not below alpha; not certifiable'})
    return {'reference_checks': reference, 'min_accepted_for_k_errors': minimum, 'accepted_for_power': power,
            'implied_calibration_questions': implied,
            'notes': ['accepted_needed is the number of accepted calibration examples for the stated power, per rule.',
                      'calibration_questions_needed = accepted_needed / exploratory acceptance rate; that rate is an '
                      'in-sample, optimistic estimate from development data.',
                      'Fix the calibration size before collection; do not add data after seeing calibration results.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('calibrate', 'evaluate', 'plan', 'validate'):
        p = sub.add_parser(name)
        p.add_argument('--config', default='configs/gpqa_risk_acceptance.json')
        p.add_argument('--output', required=True)
        if name == 'validate':
            p.add_argument('--repeats', type=int, default=20000)
            p.add_argument('--seed', type=int, default=20261011)
            p.add_argument('--sizes', type=int, nargs='+', default=[500, 2000])
        elif name != 'plan':
            p.add_argument('--observations', required=True)
            p.add_argument('--dry-run-on-development-data', action='store_true',
                           help='required acknowledgement: the only available data are inspected development data')
        else:
            p.add_argument('--exploration', help='exploration.json from Phase I (for implied question counts)')
    sub.choices['evaluate'].add_argument('--selection', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    if args.command == 'validate':
        results = []
        for alpha in cfg['alpha'].values():
            for name, chains in validation_scenarios(alpha).items():
                for n in args.sizes:
                    results.append({'scenario': name, **simulate_procedures(chains, n, alpha, cfg['delta'], args.repeats,
                                                                            args.seed)})
        atomic_json(out/'procedure_validation.json', {'status': 'synthetic Monte Carlo; validates the procedures, '
                                                      'not the GPQA data', 'results': results})
        return
    if args.command == 'plan':
        regions = None
        if args.exploration:
            regions = _plan_regions(json.loads(Path(args.exploration).read_text()), cfg)
        atomic_json(out/'sample_size_plan.json', sample_size_plan(cfg, regions))
        return
    if not args.dry_run_on_development_data:
        raise SystemExit('Only development data are available; pass --dry-run-on-development-data to acknowledge '
                         'that results are a software dry run, not a certification.')
    export = json.loads(Path(args.observations).read_text())
    part = cfg['certification']['dry_run_partition']
    ids = export['generators'][cfg['primary_generator']]['cohorts']['all_calibration']['question_ids']
    for g in cfg['certification']['family']['generators']:
        if sorted(export['generators'][g]['cohorts']['all_calibration']['question_ids']) != sorted(ids):
            raise ValueError('generators must share the same question set for a shared partition')
    roles = partition_questions(ids, part['fractions'], part['seed'])
    here = Path(__file__)
    provenance = {'observations_sha256': file_digest(args.observations), 'config_sha256': file_digest(args.config),
                  'implementation_sha256': {n: file_digest(here.with_name(n)) for n in IMPLEMENTATION},
                  'status': part['status'], 'new_api_requests': 0}
    if args.command == 'calibrate':
        atomic_json(out/'partition.json', {'roles': roles, 'roles_sha256': digest(roles), **provenance})
        for label, alpha in cfg['alpha'].items():
            result = calibrate(export, roles, cfg, alpha)
            atomic_json(out/f'selection_alpha_{alpha:g}.json', result['selection'])
            atomic_json(out/f'calibration_alpha_{alpha:g}.json', {k: v for k, v in result.items() if k != 'selection'}
                        | {'provenance': provenance, 'alpha_role': label})
            print(json.dumps({'alpha': alpha, 'status': result['selection']['status'],
                              'selected': result['selection']['selected_rule']}))
    else:
        selection = json.loads(Path(args.selection).read_text())
        result = evaluate(export, roles, cfg, selection)
        atomic_json(out/f"evaluation_alpha_{unseal(selection, 'selection_id')['alpha']:g}.json", result | {'provenance': provenance})
        print(json.dumps({'status': result['status'], 'rules': list(result['rules'])}))


def _plan_regions(exploration, cfg):
    """Out-of-fold development rates of each declared family rule (249-question cohort, median over seeds)."""
    out = []
    for rule in declared_family(cfg):
        found = [r for r in exploration['family_rules'] if r['cohort'] == 'all_calibration'
                 and (r['generator'], r['method'], r['threshold']) == (rule['generator'], rule['method'], rule['threshold'])]
        errors = [r['conditional_error'] for r in found if r['m']]
        out.append({**rule, 'acceptance_rate': float(np.median([r['acceptance_rate'] for r in found])),
                    'observed_conditional_error': float(np.median(errors)) if errors else None,
                    'source': 'out-of-fold development scores, 249-question cohort, median over 5 seeds (optimistic: '
                              'the family was declared after these data were inspected)'})
    return out


if __name__ == '__main__':
    main()
