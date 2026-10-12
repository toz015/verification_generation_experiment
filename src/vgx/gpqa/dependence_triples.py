"""Offline dependence-aware combination of three cached probability verifiers.

The probability-triples experiment multiplies per-verifier binned likelihoods,
i.e. assumes verifiers are independent given correctness. Here the three bins
are modelled jointly per correctness class: the empirical joint table shrunk
toward that independence product with Dirichlet weight alpha (alpha = inf is
exactly the independence model). The Green-Laffont recursion is unchanged
except that the next verifier's likelihood depends on the bins already seen,
so the state is (stage, earlier bins, belief). Values use exact recursion
(three bins, three stages), not a belief grid.

Inputs are the committed probability-triples review bundle: derived
development rows and frozen source fits. No GPQA text, no API calls, no
evaluation labels. The raw generator confidence is the prior, unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from itertools import combinations, permutations
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa.probability_triples_review import unpack_decisions
from vgx.gpqa.raw_prior_analysis import candidate_valid, baseline_decision
from vgx.gpqa.report import mean_interval
from vgx.gpqa.score import fit_verifier_likelihood

INF = math.inf


def bin_of(score, edges):
    """Same binning as VerifierLikelihood.bin_index."""
    index = int(np.searchsorted(edges, score, side='right') - 1)
    return min(max(index, 0), len(edges)-2)


def alpha_label(alpha):
    return 'inf' if math.isinf(alpha) else f'{alpha:g}'


class JointLikelihood:
    """P(bins of an ordered verifier triple | correctness), shrunk to independence."""

    def __init__(self, marginals, counts, alpha):
        # marginals[k][y]: P(bin of verifier k | Y=y); counts[y]: training cell counts.
        if not (alpha > 0):
            raise ValueError('alpha must be positive (inf for independence)')
        self.alpha, self.table = alpha, {}
        for y in (0, 1):
            base = reduce(np.multiply.outer, [np.asarray(m[y], dtype=float) for m in marginals])
            if math.isinf(alpha):
                self.table[y] = base
            else:
                c = np.asarray(counts[y], dtype=float)
                self.table[y] = (c + alpha*base)/(c.sum() + alpha)
            if not np.isclose(self.table[y].sum(), 1.) or (self.table[y] <= 0).any():
                raise ValueError('joint table must be a strictly positive distribution')
        self._cache = {}

    def conditional(self, history, y):
        """P(next verifier bin | earlier bins, Y=y); only earlier bins are inputs."""
        key = (tuple(history), y)
        if key not in self._cache:
            t, k = self.table[y], len(history)
            m = t.sum(axis=tuple(range(k+1, t.ndim))) if k+1 < t.ndim else t
            v = m[tuple(history)]
            self._cache[key] = v/v.sum()
        return self._cache[key]

    def likelihood_ratio(self, cell):
        return self.table[1][tuple(cell)]/self.table[0][tuple(cell)]


def cell_counts(rows, order, edges):
    counts = {y: np.zeros((len(edges)-1,)*len(order)) for y in (0, 1)}
    for r in rows:
        counts[r['outcome']][tuple(bin_of(r['signals'][t]['score'], edges) for t in order)] += 1
    return counts


def complete(rows, order):
    return [r for r in rows if candidate_valid(r) and all(r['signals'][t]['score'] is not None for t in order)]


def fit_joint(train_rows, order, marginals, alpha, edges):
    return JointLikelihood(marginals, cell_counts(complete(train_rows, order), order, edges), alpha)


def choose_alpha(train_rows, order, alphas, edges, cfg, seed):
    """Inner cross-validation on training rows only: class-conditional log-likelihood
    of held-out bin cells. The prior is not used. Ties go to the larger alpha."""
    rows = complete(train_rows, order)
    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(rows)) % cfg['inner_folds']
    scores = {}
    for alpha in alphas:
        total = 0.
        for f in range(cfg['inner_folds']):
            fit = [r for r, k in zip(rows, folds) if k != f]
            held = [r for r, k in zip(rows, folds) if k == f]
            marginals = []
            for t in order:
                m = fit_verifier_likelihood([r['outcome'] for r in fit], [r['signals'][t]['score'] for r in fit],
                                            bins=cfg['probability_bins'], laplace=cfg['laplace'])
                marginals.append({1: m.p_bin_if_correct, 0: m.p_bin_if_incorrect})
            joint = fit_joint(fit, order, marginals, alpha, edges)
            total += sum(math.log(joint.table[r['outcome']][tuple(bin_of(r['signals'][t]['score'], edges)
                                                                 for t in order)]) for r in held)
        scores[alpha_label(alpha)] = total
    best = max(alphas, key=lambda a: (scores[alpha_label(a)], a))
    return best, scores


class ExactPlanner:
    """Algorithm 1 by exact recursion; likelihoods may depend on earlier bins."""

    def __init__(self, joint, costs, reward, loss):
        self.joint, self.costs, self.reward, self.loss = joint, tuple(costs), reward, loss
        self.threshold = loss/(reward+loss)

    def stop(self, b):
        return np.maximum(0., (self.reward+self.loss)*b - self.loss)

    def q(self, k, b, history=()):
        b = np.asarray(b, dtype=float)
        value = np.full(b.shape, -float(self.costs[k]))
        p1, p0 = self.joint.conditional(history, 1), self.joint.conditional(history, 0)
        for x in range(len(p1)):
            mass = b*p1[x] + (1-b)*p0[x]
            updated = np.divide(b*p1[x], mass, out=np.zeros_like(mass), where=mass > 0)
            value = value + mass*self.j(k+1, updated, (*history, x))
        return value

    def j(self, k, b, history=()):
        stop = self.stop(np.asarray(b, dtype=float))
        return stop if k == len(self.costs) else np.maximum(stop, self.q(k, b, history))

    def decide(self, k, b, history):
        stop = float(self.stop(b))
        q = float(self.q(k, b, history)) if k < len(self.costs) else None
        query = q is not None and q > stop  # ties stop
        return {'stage': k, 'belief': b, 'history': list(history), 'stop_value': stop,
                'continuation_value': q,
                'action': 'query' if query else ('assert' if b >= self.threshold else 'abstain')}

    def update(self, b, history, x):
        p1, p0 = self.joint.conditional(history, 1)[x], self.joint.conditional(history, 0)[x]
        return b*p1/(b*p1 + (1-b)*p0)


def execute(prior, valid, planner, order, costs_usd, signals, edges, *, force_all=False):
    """Reads a verifier's cached score only after the policy has queried it."""
    if not valid:
        return {**baseline_decision(prior, False, planner.loss, planner.reward), 'trace': []}
    reads = []

    def read(stage):
        if stage != len(reads):
            raise AssertionError('future or out-of-order signal access')
        reads.append(order[stage])
        return signals[order[stage]]

    belief, history, failure, trace = prior, (), None, []
    while True:
        k = len(reads)
        decision = planner.decide(k, belief, history)
        trace.append(decision)
        if k == len(order) or (not force_all and decision['action'] != 'query'):
            break
        observation = read(k)
        if observation['score'] is None:
            failure = failure or 'missing_verifier_score'
            if not force_all:
                break
        elif failure is None:
            x = bin_of(observation['score'], edges)
            belief = planner.update(belief, history, x)
            history = (*history, x)
    if not failure and history:
        joint = planner.joint
        lr = joint.likelihood_ratio(history) if len(history) == len(order) else None
        if lr is not None and not np.isclose(belief, prior*lr/(prior*lr + 1-prior), rtol=1e-10, atol=1e-12):
            raise AssertionError('sequential and joint-table Bayes disagree')
    used = len(reads)
    return {'action': 'abstain' if failure else decision['action'], 'posterior': belief,
            'verifiers_used': used, 'failure': failure, 'queried_tags': list(order[:used]),
            'bins': list(history), 'expected_cost_usd': float(sum(costs_usd[:used])),
            'usage_estimated_cost_usd': float(sum(signals[t]['estimated_usd'] for t in order[:used])),
            'trace': trace}


def all_orders(tags):
    return [tuple(o) for c in combinations(sorted(tags), 3) for o in permutations(c)]


def order_id(order):
    return '>'.join(t.split(':')[0] for t in order)


def combo_id(order):
    return '+'.join(sorted(t.split(':')[0] for t in order))


def utility(rows, decisions, loss, cfg):
    y = np.array([r['outcome'] for r in rows])
    released = np.array([d['action'] == 'assert' for d in decisions])
    cost = np.array([d['expected_cost_usd'] for d in decisions])
    return released*np.where(y == 1, cfg['correct_reward'], -loss) - cfg['utility_per_usd']*cost


def summarize(rows, decisions, loss, cfg):
    y = np.array([r['outcome'] for r in rows])
    released = np.array([d['action'] == 'assert' for d in decisions])
    u = utility(rows, decisions, loss, cfg)
    posts = [d['posterior'] for d in decisions if d['action'] == 'assert']
    return {'n': len(rows), 'released': int(released.sum()), 'wrong_released': int((released & (y == 0)).sum()),
            'queries': int(sum(d['verifiers_used'] for d in decisions)),
            'expected_cost_usd': float(sum(d['expected_cost_usd'] for d in decisions)),
            'usage_estimated_cost_usd': float(sum(d['usage_estimated_cost_usd'] for d in decisions)),
            'mean_utility': float(u.mean()),
            'released_accuracy': float(y[released].mean()) if released.any() else None,
            'mean_posterior_on_released': float(np.mean(posts)) if posts else None,
            'query_histogram': {str(k): v for k, v in sorted(Counter(d['verifiers_used'] for d in decisions).items())},
            'signal_failures': int(sum(d['failure'] == 'missing_verifier_score' for d in decisions))}


def fold_marginals(fold, order):
    return [{1: fold['arms'][t]['likelihood']['p_bin_if_correct'], 0: fold['arms'][t]['likelihood']['p_bin_if_incorrect']}
            for t in order]


def check_fold(rows, fold, tags, cfg):
    """The frozen per-verifier fits must equal a refit on the training IDs."""
    lookup = {r['item_id']: r for r in rows}
    if set(fold['train_ids']) & set(fold['validation_ids']):
        raise ValueError('training/held-out overlap')
    for t in tags:
        usable = [lookup[i] for i in fold['train_ids'] if candidate_valid(lookup[i])
                  and lookup[i]['signals'][t]['score'] is not None]
        m = fit_verifier_likelihood([r['outcome'] for r in usable], [r['signals'][t]['score'] for r in usable],
                                    bins=cfg['probability_bins'], laplace=cfg['laplace'])
        arm = fold['arms'][t]['likelihood']
        if not (np.allclose(m.p_bin_if_correct, arm['p_bin_if_correct'], rtol=0, atol=1e-15)
                and np.allclose(m.p_bin_if_incorrect, arm['p_bin_if_incorrect'], rtol=0, atol=1e-15)):
            raise ValueError('frozen likelihood differs from training refit')
        cost = np.mean([lookup[i]['signals'][t]['estimated_usd'] for i in fold['train_ids'] if candidate_valid(lookup[i])])
        if not np.isclose(cost, fold['arms'][t]['expected_cost_usd'], rtol=1e-12, atol=1e-15):
            raise ValueError('frozen training cost differs')


def run_split(rows, fit, cfg, tags, committed):
    """One generator/cohort/seed. Fitting and selection see training rows only."""
    ids = [r['item_id'] for r in rows]
    if ids != fit['item_ids'] or [r['outcome'] for r in rows] != fit['outcomes']:
        raise ValueError('frozen split mismatch')
    lookup = {q: i for i, q in enumerate(ids)}
    edges = tuple(np.linspace(0., 1., cfg['probability_bins']+1).tolist())
    valid = [candidate_valid(r) for r in rows]
    orders = all_orders(tags)
    combos = sorted({tuple(sorted(o)) for o in orders})
    flash = (cfg['reference_verifier'],)
    fixed_alphas = [float(a) for a in cfg['fixed_alphas']]
    out = {'seed': fit['seed'], 'folds': [], 'policies': {}, 'checks': Counter(), 'calibration': {}}
    for loss in cfg['losses']:
        lk = str(float(loss))
        decisions = {}

        def put(name, i, d):
            decisions.setdefault(name, [None]*len(rows))[i] = d

        calibration = {'independence': [], 'dependence': []}
        for fold_id, fold in enumerate(fit['folds']):
            check_fold(rows, fold, tags, cfg)
            train = [rows[lookup[q]] for q in fold['train_ids']]
            held = [lookup[q] for q in fold['validation_ids']]
            priors = np.array([r['prior'] for r in train if candidate_valid(r)])
            seed = cfg['inner_seed'] + 1000*fit['seed'] % 997 + fold_id
            alphas = {c: choose_alpha(train, c, [float(a) for a in cfg['alpha_grid']] + [INF], edges, cfg, seed)
                      for c in combos}
            costs = {t: fold['arms'][t]['expected_cost_usd'] for t in tags}

            def planner(order, alpha):
                joint = fit_joint(train, order, fold_marginals(fold, order), alpha, edges)
                return ExactPlanner(joint, [cfg['utility_per_usd']*costs[t] for t in order],
                                    cfg['correct_reward'], float(loss))

            ranking = {}
            for model, alpha_of in [('independence', lambda c: INF), ('dependence', lambda c: alphas[c][0]),
                                    *[(f'dependence_alpha_{a:g}', (lambda a: lambda c: a)(a)) for a in fixed_alphas]]:
                rows_rank = []
                for o in orders:
                    p = planner(o, alpha_of(tuple(sorted(o))))
                    rows_rank.append({'order': list(o), 'order_id': order_id(o), 'alpha': alpha_label(alpha_of(tuple(sorted(o)))),
                                      'training_model_value': float(p.j(0, priors).mean())})
                rows_rank.sort(key=lambda r: (-r['training_model_value'], r['order']))
                ranking[model] = rows_rank
            fp = planner(flash, INF)
            fold_record = {'fold': fold_id, 'alpha_by_combination': {combo_id(c): {'chosen': alpha_label(a), 'inner_cv_loglik': s}
                                                                    for c, (a, s) in alphas.items()},
                           'selected': {m: r[0] for m, r in ranking.items()},
                           'flash_training_model_value': float(fp.j(0, priors).mean())}
            out['folds'].append({'loss': lk, **fold_record})
            for i in held:
                r = rows[i]
                put('raw', i, {**baseline_decision(r['prior'], valid[i], float(loss), cfg['correct_reward'])})
                put('raw_ties_abstain', i, {**baseline_decision(r['prior'], valid[i] and r['prior'] > fp.threshold,
                                                                float(loss), cfg['correct_reward'])})
                put('abstain_all', i, {**baseline_decision(r['prior'], False, float(loss), cfg['correct_reward'])})
                put('single_flash', i, execute(r['prior'], valid[i], fp, flash, [costs[t] for t in flash], r['signals'], edges))
            for model, rows_rank in ranking.items():
                best = tuple(rows_rank[0]['order'])
                alpha = INF if model == 'independence' else (alphas[tuple(sorted(best))][0] if model == 'dependence'
                                                             else float(model.rsplit('_', 1)[1]))
                p = planner(best, alpha)
                for i in held:
                    r = rows[i]
                    put(f'{model}:selected_triple', i, execute(r['prior'], valid[i], p, best, [costs[t] for t in best],
                                                                r['signals'], edges))
                    if model in ('independence', 'dependence'):
                        put(f'{model}:always_selected_triple', i, execute(
                            r['prior'], valid[i], p, best, [costs[t] for t in best], r['signals'], edges, force_all=True))
            for o in orders:
                for model, alpha in [('independence', INF), ('dependence', alphas[tuple(sorted(o))][0])]:
                    p = planner(o, alpha)
                    for i in held:
                        r = rows[i]
                        put(f'{model}:triple:{order_id(o)}', i, execute(r['prior'], valid[i], p, o, [costs[t] for t in o],
                                                                         r['signals'], edges))
            # Held-out fit of the verifier model alone (no prior): class-conditional
            # log-likelihood of the observed cell, and posterior after all three.
            base_rate = float(np.mean([r['outcome'] for r in train if candidate_valid(r)]))
            flash_like = fold['arms'][cfg['reference_verifier']]['likelihood']
            for c in combos:
                a = alphas[c][0]
                for model, alpha in [('independence', INF), ('dependence', a)]:
                    joint = fit_joint(train, c, fold_marginals(fold, c), alpha, edges)
                    for i in held:
                        r = rows[i]
                        if not candidate_valid(r) or any(r['signals'][t]['score'] is None for t in c):
                            continue
                        cell = tuple(bin_of(r['signals'][t]['score'], edges) for t in c)
                        lr = joint.likelihood_ratio(cell)
                        fs = r['signals'][cfg['reference_verifier']]['score']
                        fb = bin_of(fs, edges) if fs is not None else None
                        calibration[model].append({'combination': combo_id(c), 'fold': fold_id, 'item_id': r['item_id'],
                            'outcome': r['outcome'], 'prior': r['prior'], 'training_base_rate': base_rate,
                            'cell_loglik': math.log(joint.table[r['outcome']][cell]), 'likelihood_ratio': float(lr),
                            'flash_likelihood_ratio': (flash_like['p_bin_if_correct'][fb]/flash_like['p_bin_if_incorrect'][fb]
                                                       if fb is not None else None),
                            'posterior_all_three': r['prior']*lr/(r['prior']*lr + 1-r['prior'])})
        # The independence model with exact recursion must reproduce the committed
        # grid-based fixed-order decisions wherever the grid and exact values agree.
        if committed is not None:
            stored = committed[lk]
            for o in orders:
                for a, b in zip(decisions[f'independence:triple:{order_id(o)}'], stored['triple:'+order_id(o)]):
                    same = a['action'] == b['action'] and a['verifiers_used'] == b['verifiers_used'] \
                        and (a['posterior'] == b['posterior'] or (None not in (a['posterior'], b['posterior'])
                             and math.isclose(a['posterior'], b['posterior'], rel_tol=1e-9, abs_tol=1e-12)))
                    out['checks'][f'committed_fixed_order_agree_{lk}'] += int(bool(same))
                    out['checks'][f'committed_fixed_order_differ_{lk}'] += int(not same)
            for a, b in zip(decisions['single_flash'], stored['single:gemini_flash']):
                out['checks'][f'committed_flash_agree_{lk}'] += int(a['action'] == b['action'] and a['verifiers_used'] == b['verifiers_used'])
        if any(d is None for v in decisions.values() for d in v):
            raise ValueError('incomplete held-out decisions')
        u_flash = utility(rows, decisions['single_flash'], float(loss), cfg)
        policies = {}
        for name, ds in decisions.items():
            s = summarize(rows, ds, float(loss), cfg)
            delta = utility(rows, ds, float(loss), cfg) - u_flash
            s['utility_difference_vs_flash'] = float(delta.mean())
            if ':triple:' not in name:
                s['paired_utility_vs_flash'] = mean_interval(delta, cfg['bootstrap_repeats'], fit['seed'])
            policies[name] = s
        out['policies'][lk] = policies
        if lk == str(float(cfg['losses'][0])):
            for model, recs in calibration.items():
                out['calibration'][model] = recs
            out['decisions_primary'] = {k: [{f: d[f] for f in ('action', 'posterior', 'verifiers_used', 'failure', 'bins')
                                              if f in d} for d in v]
                                        for k, v in decisions.items() if ':triple:' not in k}
    out['checks'] = {k: int(v) for k, v in out['checks'].items()}
    return out


def load_bundle(path):
    with zipfile.ZipFile(path) as z:
        protocol = json.loads(z.read('experiment/protocol.json'))
        inputs = json.loads(z.read('inputs/development_rows.json'))
        fits = {n[len('source_fits/'):-len('.json')]: json.loads(z.read(n))
                for n in z.namelist() if n.startswith('source_fits/')}
        committed = {}
        for n in z.namelist():
            if n.endswith('.packed.json'):
                run, name = n[len('experiment/'):].rsplit('/', 1)
                loss = name[len('decisions_loss_'):-len('.packed.json')]
                committed.setdefault(run, {})[loss] = unpack_decisions(json.loads(z.read(n)))['decisions']
    return protocol, inputs, fits, committed


_WORKER = {}


def _init_worker(bundle):
    _WORKER['bundle'] = load_bundle(bundle)


def _run_one(task):
    run, cfg, tags = task
    _, inputs, fits, committed = _WORKER['bundle']
    g, cohort, _ = run.split('/')
    rows = inputs['rows'][g]
    if cohort == 'remaining_calibration':
        rows = [r for r in rows if r['item_id'] not in set(inputs['pilot_ids'])]
    result = run_split(rows, fits[run], cfg, tags, committed[run])
    result['item_ids'] = [r['item_id'] for r in rows]
    return run, result


def analyze(config_path, output, jobs=1):
    config_path, output = Path(config_path), Path(output)
    cfg = json.loads(config_path.read_text())
    if cfg['new_api_requests'] != 0 or cfg['evaluation_labels_read'] or cfg['confidence_calibration']:
        raise ValueError('offline, development-only, uncalibrated protocol required')
    bundle = Path(cfg['bundle'])
    if file_digest(bundle) != cfg['bundle_sha256']:
        raise ValueError('review bundle changed')
    protocol, _, fits, _ = load_bundle(bundle)
    if protocol['analysis_id'] != cfg['source_analysis_id']:
        raise ValueError('unexpected source analysis')
    tags = [v+':probability' for v in protocol['config']['verifiers']]
    source = protocol['source_config']
    for k in ('correct_reward', 'laplace', 'probability_bins'):
        if cfg[k] != source[k]:
            raise ValueError('setting differs from the source experiment: '+k)
    cfg['utility_per_usd'] = protocol['config']['primary_utility_per_usd']
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'config': cfg, 'config_sha256': file_digest(config_path),
                'implementation_sha256': file_digest(Path(__file__)), 'runs': {}}
    tasks = [(run, cfg, tags) for run in sorted(fits)]
    with ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(bundle,)) as pool:
        for run, result in pool.map(_run_one, tasks):
            target = output/(run+'.json')
            atomic_json(target, result)
            manifest['runs'][run] = file_digest(target)
            print(json.dumps({'run': run, 'checks': result['checks']}), flush=True)
    manifest['analysis_id'] = digest(manifest)
    atomic_json(output/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/gpqa_dependence_triples.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    analyze(args.config, args.output, args.jobs)
