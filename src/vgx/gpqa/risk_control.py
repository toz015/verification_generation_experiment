"""Risk-controlled acceptance: exported observations, scoring, exact bounds and certification.

Target: maximize the acceptance rate P(A) subject to the conditional error among
accepted answers P(W | A) <= alpha, certified with confidence 1 - delta. This is
conditional error among released answers, not P(W and A), FDR or expected utility.

Everything here is offline. Observations come from frozen artifacts; no provider
client is constructed or called. Scores (raw confidence, Bayes beliefs) are treated as
selection scores, not as calibrated probabilities.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

from vgx.common.storage import digest, file_digest
from vgx.gpqa.planner import NestedPlanner
from vgx.gpqa.raw_prior_analysis import decide_public
from vgx.gpqa.score import VerifierLikelihood, fit_verifier_likelihood

METHODS = ('A', 'B', 'C')
METHOD_NAMES = {'A': 'generator confidence only', 'B': 'always-query Flash, pooled Bayes belief',
                'C': 'frozen sequential Flash policy plus acceptance gate on terminal belief'}


# ---------------------------------------------------------------- observations

def _collect_signals(trace_root, generator, tag):
    """Cached verifier records as recorded in frozen nested-combination traces, keyed by question."""
    records, files = {}, []
    for path in sorted((trace_root/generator).glob('*/seed_*/decisions_loss_*.json')):
        ids = json.loads((path.parent/'results.json').read_text())['item_ids']
        files += [path, path.parent/'results.json']
        for decisions in json.loads(path.read_text()).values():
            for item, d in zip(ids, decisions):
                for queried, obs in zip(d.get('queried_tags', []), d.get('observations', [])):
                    if queried != tag:
                        continue
                    value = {'score': obs['score'], 'failure': obs['failure'],
                             'usage_usd': obs['estimated_usd'], 'request_key': obs['request_key']}
                    if item in records and records[item] != value:
                        raise ValueError('inconsistent cached verifier record for one question')
                    records[item] = value
    return records, sorted(set(files))


def _minimal(decision):
    return {k: decision[k] for k in ('action', 'posterior', 'verifiers_used', 'expected_cost_usd',
                                     'usage_estimated_cost_usd', 'failure')}


def export_observations(root, cfg):
    """Build the compact observation export from a frozen project or review package.

    Verifies input files against FILE_MANIFEST.json when present, cross-checks every
    question's confidence/outcome/validity across cohorts and seeds, and keeps the
    frozen pooled tables, costs and decisions needed to re-verify the source policies.
    """
    root = Path(root)
    source_root, trace_root, tag = root/cfg['source_results'], root/cfg['signal_trace_results'], cfg['verifier']
    inputs, out = [root/cfg['source_config']], {'schema': 1, 'verifier': tag, 'generators': {}}
    source_cfg = json.loads((root/cfg['source_config']).read_text())
    for key, value in cfg['frozen_settings'].items():
        if source_cfg[key] != value:
            raise ValueError(f'frozen setting {key} differs from the source analysis')
    for generator in cfg['generators']:
        signals, files = _collect_signals(trace_root, generator, tag)
        inputs += files
        items, cohorts = {}, {}
        for cohort in cfg['cohorts']:
            cohorts[cohort] = {'question_ids': None, 'seeds': {}}
            for seed in cfg['seeds']:
                path = source_root/generator/cohort/f'seed_{seed}.json'
                inputs.append(path)
                fit = json.loads(path.read_text())
                raw = fit['decisions'][str(cfg['sequential_base_loss'])]['raw']
                for qid, prior, outcome, d in zip(fit['item_ids'], fit['raw_priors'], fit['outcomes'], raw):
                    valid = d.get('failure') != 'invalid_generator'
                    entry = {'question_id': qid, 'valid': valid, 'raw_confidence': prior, 'correct': int(outcome),
                             'signal': signals.get(qid) if valid else None}
                    if valid and qid not in signals:
                        raise ValueError(f'missing cached verifier record for valid candidate {qid}')
                    if items.setdefault(qid, entry) != entry:
                        raise ValueError('question data differ across cohorts or seeds')
                if cohorts[cohort]['question_ids'] is None:
                    cohorts[cohort]['question_ids'] = fit['item_ids']
                elif cohorts[cohort]['question_ids'] != fit['item_ids']:
                    raise ValueError('cohort question order differs across seeds')
                folds = []
                for fold in fit['folds']:
                    arm = fold['arms'][tag]
                    folds.append({'pooled_likelihood': arm['likelihood'], 'expected_cost_usd': arm['expected_cost_usd'],
                                  'fit_correct': arm['fit_correct'], 'fit_wrong': arm['fit_wrong']})
                frozen = {}
                for loss, policies in fit['decisions'].items():
                    frozen[loss] = {'always_query': [_minimal(d) for d in policies['always_query:'+tag]],
                                    'sequential': [_minimal(d) for d in policies['sequential:'+tag]]}
                cohorts[cohort]['seeds'][str(seed)] = {'fold_assignment': fit['fold_assignment'], 'folds': folds,
                                                       'frozen_decisions': frozen}
        out['generators'][generator] = {'items': [items[q] for q in sorted(items)], 'cohorts': cohorts}
    hashes = {str(p.relative_to(root)): file_digest(p) for p in inputs}
    manifest = root/'FILE_MANIFEST.json'
    check = None
    if manifest.exists():
        listed = json.loads(manifest.read_text())['files']
        bad = [p for p, h in hashes.items() if listed.get(p) != h]
        if bad:
            raise ValueError(f'input files differ from the package manifest: {bad[:3]}')
        check = {'manifest_sha256': file_digest(manifest), 'files_checked': len(hashes)}
    out['provenance'] = {'input_sha256': hashes, 'package_manifest_check': check,
                         'original_request_caches_reparsed': False,
                         'signal_table_sha256': {g: digest([i['signal'] for i in v['items']])
                                                 for g, v in out['generators'].items()}}
    return out


def generator_rows(export, generator):
    return {item['question_id']: item for item in export['generators'][generator]['items']}


# ---------------------------------------------------------------- fitting and scoring

def fit_pooled(rows, bins, laplace):
    """Pooled P(bin | Y) and expected cost from the given (fitting) rows only."""
    fitting = [r for r in rows if r['valid'] and r['signal']['score'] is not None]
    likelihood = fit_verifier_likelihood([r['correct'] for r in fitting], [r['signal']['score'] for r in fitting],
                                         bins=bins, laplace=laplace)
    cost_rows = [r for r in rows if r['valid']]  # returned malformed responses were paid for
    return likelihood, float(np.mean([r['signal']['usage_usd'] for r in cost_rows]))


def likelihood_from(table):
    return VerifierLikelihood(tuple(table['edges']), tuple(table['p_bin_if_correct']), tuple(table['p_bin_if_incorrect']))


def score_generator_only(row):
    return {'question_id': row['question_id'], 'eligible': bool(row['valid']),
            'score': row['raw_confidence'] if row['valid'] else None, 'queries': 0,
            'expected_cost_usd': 0., 'usage_cost_usd': 0., 'failure': None if row['valid'] else 'invalid_generator'}


def score_always_query(row, likelihood, expected_cost_usd, query):
    """Every valid candidate is verified; a malformed response is paid for and cannot be accepted."""
    base = {'question_id': row['question_id'], 'eligible': False, 'score': None, 'queries': 0,
            'expected_cost_usd': 0., 'usage_cost_usd': 0., 'failure': 'invalid_generator'}
    if not row['valid']:
        return base
    signal = query()
    base.update(queries=1, expected_cost_usd=expected_cost_usd, usage_cost_usd=signal['usage_usd'])
    if signal['score'] is None:
        base['failure'] = signal['failure'] or 'invalid_signal'
        return base
    base.update(eligible=True, failure=None, score=likelihood.posterior(row['raw_confidence'], signal['score']))
    return base


def score_sequential(row, planner, expected_cost_usd, query):
    """Frozen query/stopping policy; only candidates it would release can pass the added gate."""
    def observe():
        signal = query()
        return {'score': signal['score'], 'failure': signal['failure'], 'estimated_usd': signal['usage_usd']}
    decision = decide_public(row['raw_confidence'], row['valid'], planner, observe, expected_cost_usd=expected_cost_usd)
    released = decision['action'] == 'assert'
    return {'question_id': row['question_id'], 'eligible': released,
            'score': decision['posterior'] if released else None, 'queries': decision['verifiers_used'],
            'expected_cost_usd': decision['expected_cost_usd'], 'usage_cost_usd': decision['usage_estimated_cost_usd'],
            'failure': decision['failure'], 'base_action': decision['action'],
            'terminal_belief': decision['posterior'],
            'stage0': {k: decision['decision'][k] for k in ('action', 'assert_value', 'continuation_value', 'value')}
            if 'decision' in decision else None}


def lazy_signal(row, reads):
    def query():
        reads.append(row['question_id'])
        return row['signal']
    return query


def score_method(method, row, likelihood=None, expected_cost_usd=None, planner=None):
    """Score one held-out row. The signal is reachable only through a lazy query that records reads."""
    reads = []
    query = lazy_signal(row, reads)
    if method == 'A':
        result = score_generator_only(row)
    elif method == 'B':
        result = score_always_query(row, likelihood, expected_cost_usd, query)
    elif method == 'C':
        result = score_sequential(row, planner, expected_cost_usd, query)
    else:
        raise ValueError('unknown method')
    if len(reads) != result['queries']:
        raise AssertionError('signal read outside a selected query')
    return result


def sequential_planner(likelihood, expected_cost_usd, settings, loss):
    return NestedPlanner([likelihood], [settings['utility_per_usd']*expected_cost_usd], settings['correct_reward'],
                         loss, grid_size=settings['grid_size'])


def replay_check(row, planner, scored):
    """Independent replay of the sequential policy through NestedPlanner.replay."""
    if not row['valid']:
        return True
    reads = []

    class Lazy:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            if index != 0:
                raise AssertionError('future signal')
            reads.append(index)
            return row['signal']['score']
    replay = planner.replay(row['raw_confidence'], Lazy())
    return (replay['action'] == scored['base_action'] and replay['verifiers_used'] == scored['queries']
            and replay['posterior'] == scored['terminal_belief'] and len(reads) == scored['queries'])


# ---------------------------------------------------------------- rules

def accepts(scored, threshold):
    """Rule semantics: accept iff the method makes the candidate eligible and score >= threshold."""
    return bool(scored['eligible']) and scored['score'] >= threshold


def rule_counts(scored_items, correct, threshold):
    """Acceptance and error counts. The denominator N includes invalid candidates and failures."""
    accepted = [s for s in scored_items if accepts(s, threshold)]
    m = len(accepted)
    k = sum(1 - correct[s['question_id']] for s in accepted)
    return {'N': len(scored_items), 'm': m, 'k': int(k), 'acceptance_rate': m/len(scored_items) if scored_items else None,
            'conditional_error': k/m if m else None,
            'queries': sum(s['queries'] for s in scored_items),
            'expected_cost_usd': float(sum(s['expected_cost_usd'] for s in scored_items)),
            'usage_cost_usd': float(sum(s['usage_cost_usd'] for s in scored_items))}


# ---------------------------------------------------------------- exact binomial bounds

def binomial_upper_bound(k, m, error_probability):
    """One-sided exact (Clopper-Pearson) upper bound for a binomial rate.

    Returns None when m == 0 (conditional risk undefined), 1.0 when k == m.
    """
    if not (isinstance(k, (int, np.integer)) and isinstance(m, (int, np.integer))) or k < 0 or m < 0 or k > m:
        raise ValueError('need integers 0 <= k <= m')
    if not 0 < error_probability < 1:
        raise ValueError('error probability must be in (0, 1)')
    if m == 0:
        return None
    if k == m:
        return 1.0
    return float(beta.ppf(1 - error_probability, k + 1, m - k))


def clopper_pearson(k, m, confidence=.95):
    """Two-sided exact interval for reporting (not used for certification)."""
    if m == 0:
        return (None, None)
    tail = (1 - confidence)/2
    low = 0. if k == 0 else float(beta.ppf(tail, k, m - k + 1))
    high = 1. if k == m else float(beta.ppf(1 - tail, k + 1, m - k))
    return (low, high)


def certify_bonferroni(rules, alpha, delta):
    """Finite-family certification with Bonferroni over every rule eligible for selection.

    `rules` is the full declared family, in declared order, each with m, k, N and
    expected_cost_usd computed on risk-calibration data. Selection: highest m/N among
    rules with m > 0 and U <= alpha; ties by lower expected verifier cost per question,
    then declared family order. Labels never enter the tie-break.
    """
    family = len(rules)
    if family == 0:
        raise ValueError('empty rule family')
    per_rule = delta/family
    table = []
    for order, rule in enumerate(rules):
        upper = binomial_upper_bound(rule['k'], rule['m'], per_rule)
        table.append({**rule, 'family_order': order, 'family_size': family, 'per_rule_error': per_rule,
                      'upper_bound': upper, 'certified': rule['m'] > 0 and upper is not None and upper <= alpha})
    return table, select_certified(table)


def select_certified(table):
    passing = [r for r in table if r['certified']]
    if not passing:
        return None
    return min(passing, key=lambda r: (-r['m']/r['N'], r['expected_cost_usd']/r['N'], r['family_order']))


def certify_fixed_sequence(chains, alpha, delta):
    """Alternative (secondary): fixed-sequence testing inside each predeclared chain, Bonferroni across chains.

    Each chain is an ordered list declared before calibration (e.g. thresholds from strict
    to lenient). Within a chain every test uses delta/len(chains); testing stops at the
    first rule that fails, so later rules in that chain are not certified even if their
    own bound would pass. Familywise error <= delta under the same i.i.d. assumptions,
    without assuming monotone risk; non-monotone risk only costs power.
    """
    if not chains:
        raise ValueError('empty chain family')
    level = delta/len(chains)
    table, order = [], 0
    for chain_id, chain in enumerate(chains):
        open_chain = True
        for position, rule in enumerate(chain):
            upper = binomial_upper_bound(rule['k'], rule['m'], level)
            passed = rule['m'] > 0 and upper is not None and upper <= alpha
            certified = open_chain and passed
            table.append({**rule, 'chain': chain_id, 'chain_position': position, 'family_order': order,
                          'per_test_error': level, 'upper_bound': upper, 'tested': open_chain,
                          'certified': certified})
            open_chain = certified
            order += 1
    return table, select_certified(table)


# ---------------------------------------------------------------- sample size

def max_errors_allowed(m, alpha, error_probability):
    """Largest k whose upper bound is <= alpha with m accepted examples (-1 if none). U is increasing in k."""
    if m <= 0:
        return -1
    low, high = -1, m - 1  # U(k) <= alpha for k <= low; U(k) > alpha for k > high
    while low < high:
        middle = (low + high + 1)//2
        if binomial_upper_bound(middle, m, error_probability) <= alpha:
            low = middle
        else:
            high = middle - 1
    return low


def max_errors_allowed_many(ms, alpha, error_probability):
    """Vectorized max_errors_allowed via U(k, m) <= alpha  <=>  P(Bin(m, alpha) <= k) <= error_probability."""
    ms = np.asarray(ms)
    k = binom.ppf(error_probability, ms, alpha).astype(int)  # smallest k with cdf >= error_probability
    k = np.where(binom.cdf(k, ms, alpha) <= error_probability, k, k - 1)
    return np.minimum(k, ms - 1)


def min_accepted(k, alpha, error_probability, limit=1_000_000):
    """Smallest m with U(k, m) <= alpha for a fixed observed error count k."""
    if k == 0:
        m = max(1, math.ceil(math.log(error_probability)/math.log(1 - alpha)) - 1)
    else:
        m = k + 1
    while m <= limit:
        if binomial_upper_bound(k, m, error_probability) <= alpha:
            return m
        m += 1
    raise ValueError('no feasible m below limit')


def pass_probability(m, true_risk, alpha, error_probability):
    """P(certify) for one rule with m accepted examples whose true conditional error is true_risk."""
    kmax = max_errors_allowed(m, alpha, error_probability)
    return float(binom.cdf(kmax, m, true_risk)) if kmax >= 0 else 0.


def accepted_for_power(true_risk, alpha, error_probability, power=.8, limit=200_000, window=25):
    """Smallest m whose pass probability is >= power and stays there for the next `window` values.

    Pass probability is a sawtooth in m, so a first crossing alone can be followed by dips.
    """
    if true_risk >= alpha:
        return None
    start, stop = min_accepted(0, alpha, error_probability), 0
    while stop < limit:  # scan growing blocks so small answers stay cheap
        stop = min(limit, max(2*stop, start + 2048))
        ms = np.arange(start, stop + window + 1)
        kmax = max_errors_allowed_many(ms, alpha, error_probability)
        ok = np.where(kmax >= 0, binom.cdf(kmax, ms, true_risk), 0.) >= power
        run = np.convolve(ok.astype(int), np.ones(window + 1, dtype=int), mode='valid') == window + 1
        hits = np.flatnonzero(run)
        if hits.size:
            return int(ms[hits[0]])
    return None


# ---------------------------------------------------------------- partitions

def partition_questions(question_ids, fractions, seed):
    """Deterministic, label-free question-level partition (shared by all generators).

    fractions: ordered mapping role -> fraction. Questions are ordered by
    sha256(f'{seed}|{question_id}') and cut into consecutive blocks.
    """
    ids = sorted(set(question_ids))
    if not math.isclose(sum(fractions.values()), 1., abs_tol=1e-9):
        raise ValueError('fractions must sum to 1')
    ranked = sorted(ids, key=lambda q: hashlib.sha256(f'{seed}|{q}'.encode()).hexdigest())
    roles, start, n = {}, 0, len(ranked)
    names = list(fractions)
    for index, role in enumerate(names):
        stop = n if index == len(names) - 1 else start + round(fractions[role]*n)
        for q in ranked[start:stop]:
            roles[q] = role
        start = stop
    return roles


def table_dict(likelihood):
    return asdict(likelihood)
