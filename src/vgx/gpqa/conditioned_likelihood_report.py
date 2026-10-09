"""Tables, fitted parameters and per-item traces for the confidence-conditioned likelihood run.

Reads only the analysis output directory; no API calls, no source artifacts.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from vgx.common.storage import atomic_json, file_digest

PRIMARY = {'cohort': 'all_calibration', 'seed': '20261005', 'loss': '19.0'}
POLICY_ORDER = ['raw_confidence', 'pooled', 'conditioned', 'conditioned_tau_5', 'conditioned_tau_20', 'conditioned_tau_100']
LABEL = {'raw_confidence': 'A. raw confidence only', 'pooled': 'B. Flash, pooled likelihood',
         'conditioned': 'C. Flash, conditioned (inner-CV tau)', 'conditioned_tau_5': 'C. conditioned, tau=5',
         'conditioned_tau_20': 'C. conditioned, tau=20', 'conditioned_tau_100': 'C. conditioned, tau=100'}
METRICS = ['utility', 'released', 'wrong_released', 'queries']


def _runs(root):
    for path in sorted(root.glob('*/*/seed_*/results.json')):
        generator, cohort, seed = path.parts[-4], path.parts[-3], path.parts[-2].removeprefix('seed_')
        yield generator, cohort, seed, path.parent


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ci(entry):
    return entry['estimate'], entry['ci95']['low'], entry['ci95']['high']


def _fmt(x, digits=3, sign=False):
    if x is None:
        return '—'
    return f'{x:+.{digits}f}' if sign else f'{x:.{digits}f}'


def build(results, output):
    results, output = Path(results), Path(output)
    policy, groups, paired, transitions, brier, tables, taus, sizes = [], [], [], [], [], [], [], []
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(output/'item_decisions.jsonl.gz', 'wt') as traces:
        for generator, cohort, seed, run in _runs(results):
            key = {'generator': generator, 'cohort': cohort, 'seed': seed}
            result = json.loads((run/'results.json').read_text())
            items = json.loads((run/'decisions.json').read_text())
            for fold in result['folds']:
                f = {**key, 'fold': fold['fold']}
                sel = fold['tau_selection']
                taus.append({**f, 'selected_tau': sel['tau'], 'reason': sel['reason'], 'scheme': sel['scheme'],
                             'inner_folds': sel['inner_folds'], 'pooled_mean_log_likelihood': sel['pooled_mean_log_likelihood'],
                             **{f'mean_log_likelihood_tau_{t}': v for t, v in (sel['mean_log_likelihood'] or {}).items()}})
                for g, n in fold['group_sizes'].items():
                    sizes.append({**f, 'group': g, 'train_correct': n['correct'], 'train_incorrect': n['incorrect'],
                                  'bin_counts_correct': ' '.join(map(str, fold['bin_counts'][g]['correct'])),
                                  'bin_counts_incorrect': ' '.join(map(str, fold['bin_counts'][g]['incorrect'])),
                                  'expected_cost_usd': fold['expected_cost_usd']})
                named = {'pooled': {'all': fold['pooled_table']}, **fold['tables']}
                for name, by_group in named.items():
                    for g, t in by_group.items():
                        edges = t['edges']
                        for k, (p1, p0) in enumerate(zip(t['p_bin_if_correct'], t['p_bin_if_incorrect'])):
                            tables.append({**f, 'table': name, 'tau': fold['variants'].get(name), 'group': g, 'bin': k,
                                           'lower': edges[k], 'upper': edges[k+1], 'p_bin_given_correct': p1,
                                           'p_bin_given_incorrect': p0, 'likelihood_ratio': p1/p0,
                                           'fallback_cells': ' '.join(fold['fallbacks'].get(name, []))})
            for loss, policies in result['policies'].items():
                k = {**key, 'loss': loss}
                for name in POLICY_ORDER:
                    s = policies[name]
                    base = {kk: s[kk] for kk in ('n', 'released', 'correct_released', 'wrong_released', 'coverage',
                            'released_accuracy', 'queries', 'expected_cost_usd', 'usage_estimated_cost_usd', 'mean_utility',
                            'mean_utility_at_observed_usage', 'mean_predicted_value', 'mean_posterior_released',
                            'expected_wrong_released', 'signal_failures_queried')}
                    est, lo, hi = _ci(s['mean_utility_interval'])
                    policy.append({**k, 'policy': name, **base, 'mean_utility_ci_low': lo, 'mean_utility_ci_high': hi,
                                   'decision_posterior_brier_matched': s['decision_posterior_brier_matched'],
                                   **{f'stop_{c}': n for c, n in sorted(s['stopping'].items())}})
                    for g, sub in s['by_group'].items():
                        groups.append({**k, 'policy': name, 'group': g, **{kk: sub[kk] for kk in (
                            'n', 'released', 'wrong_released', 'released_accuracy', 'coverage', 'queries',
                            'expected_cost_usd', 'mean_utility', 'mean_predicted_value', 'mean_posterior_released',
                            'expected_wrong_released')}, **{f'stop_{c}': n for c, n in sorted(sub['stopping'].items())}})
                    if 'paired_vs_pooled' not in s:
                        continue
                    pv = s['paired_vs_pooled']
                    for scope, entries in [('all', pv), *pv['by_group'].items()]:
                        for metric in METRICS:
                            est, lo, hi = _ci(entries[metric])
                            paired.append({**k, 'policy': name, 'scope': scope, 'metric': metric + '_per_item',
                                           'estimate': est, 'ci95_low': lo, 'ci95_high': hi, 'total': entries[metric]['sum']})
                    est, lo, hi = _ci(pv['decision_posterior_brier_matched'])
                    paired.append({**k, 'policy': name, 'scope': 'matched_eligible', 'metric': 'decision_posterior_brier',
                                   'estimate': est, 'ci95_low': lo, 'ci95_high': hi, 'total': None})
                    for move, n in pv['stopping_transitions_from_pooled'].items():
                        transitions.append({**k, 'policy': name, 'pooled_to_policy': move, 'items': n})
            forecasts = result['forecasts']
            for name, entry in forecasts['signal_posterior'].items():
                row = {**key, 'policy': name, 'scope': 'matched_eligible', 'n': forecasts['n_matched_eligible'],
                       'raw_prior_brier': forecasts['raw_prior_brier'], 'brier': entry['brier'],
                       'mean_posterior': entry['mean_posterior']}
                if 'brier_difference_vs_pooled' in entry:
                    row.update(zip(('brier_diff_vs_pooled', 'ci95_low', 'ci95_high'), _ci(entry['brier_difference_vs_pooled'])))
                brier.append(row)
                for g, sub in entry['by_group'].items():
                    row = {**key, 'policy': name, 'scope': g, 'n': sub['n'], 'accuracy': sub['accuracy'],
                           'mean_raw_prior': sub['mean_raw_prior'], 'mean_posterior': sub['mean_posterior'], 'brier': sub['brier']}
                    if 'brier_difference_vs_pooled' in sub:
                        row.update(zip(('brier_diff_vs_pooled', 'ci95_low', 'ci95_high'), _ci(sub['brier_difference_vs_pooled'])))
                    brier.append(row)
            for loss, by_policy in items['decisions'].items():
                for name, decisions in by_policy.items():
                    for i, d in enumerate(decisions):
                        traces.write(json.dumps({**key, 'loss': loss, 'policy': name, 'item_id': items['item_ids'][i],
                            'fold': items['fold_assignment'][i], 'confidence_group': items['groups'][i],
                            'raw_confidence': items['priors'][i], 'outcome': items['outcomes'][i],
                            'decision': d}, sort_keys=True) + '\n')
    files = {'policy_results.csv': policy, 'group_results.csv': groups, 'paired_vs_pooled.csv': paired,
             'stopping_transitions.csv': transitions, 'signal_posterior_brier.csv': brier,
             'fitted_likelihoods.csv': tables, 'tau_selection.csv': taus, 'training_group_sizes.csv': sizes}
    for name, rows in files.items():
        _write_csv(output/name, rows)
    (output/'TABLES.md').write_text(tables_markdown(policy, groups, paired, transitions, brier, taus, sizes))
    protocol = json.loads((results/'protocol.json').read_text())
    report = json.loads((results/'report.json').read_text())
    atomic_json(output/'manifest.json', {'analysis_id': protocol['analysis_id'], 'protocol': protocol,
        'report': report, 'files_sha256': {p.name: file_digest(p) for p in sorted(output.iterdir())
                                           if p.name != 'manifest.json' and p.is_file()}})


def _pick(rows, **match):
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in match.items())]


def tables_markdown(policy, groups, paired, transitions, brier, taus, sizes):
    out = ['# Confidence-conditioned Flash likelihoods: result tables', '',
           'Generated by `vgx.gpqa.conditioned_likelihood_report`. Primary setting: 249-question cohort, seed 20261005, '
           'R=1, L=19, 10 utility per USD, Flash probability verifier. Utility and coverage use all items as denominator; '
           'released accuracy uses released items. Costs are counterfactual cached-call estimates, not billing.', '']
    for generator in ('gemini', 'qwen'):
        out += [f'## {generator.capitalize()} generator', '', '### Policies (primary setting)', '',
                '| Policy | Released | Wrong | Accuracy | Coverage | Queries | Cost USD | Mean utility [95% CI] | Predicted value | Mean posterior (released) | Σ(1−posterior) vs wrong | Decision Brier (matched) |',
                '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|']
        for r in _pick(policy, generator=generator, **PRIMARY):
            out.append(f"| {LABEL[r['policy']]} | {r['released']} | {r['wrong_released']} | {_fmt(r['released_accuracy'])} | "
                       f"{_fmt(r['coverage'])} | {r['queries']} | {r['expected_cost_usd']:.4f} | {_fmt(r['mean_utility'], sign=True)} "
                       f"[{_fmt(r['mean_utility_ci_low'], sign=True)}, {_fmt(r['mean_utility_ci_high'], sign=True)}] | "
                       f"{_fmt(r['mean_predicted_value'], sign=True)} | {_fmt(r['mean_posterior_released'], 4)} | "
                       f"{r['expected_wrong_released']:.2f} vs {r['wrong_released']} | {_fmt(r['decision_posterior_brier_matched'], 4)} |")
        out += ['', '### Paired differences against pooled (primary setting, per item, bootstrap 95% CI; total in parentheses)', '',
                '| Policy | Scope | Utility | Released | Wrong releases | Queries |', '|---|---|---|---|---|---|']
        for name in POLICY_ORDER:
            if name == 'pooled':
                continue
            for scope in ('all', 'low', 'high'):
                cells = []
                for metric in METRICS:
                    m = _pick(paired, generator=generator, policy=name, scope=scope, metric=metric+'_per_item', **PRIMARY)
                    if m:
                        m = m[0]
                        cells.append(f"{_fmt(m['estimate'], sign=True)} [{_fmt(m['ci95_low'], sign=True)}, {_fmt(m['ci95_high'], sign=True)}] ({m['total']:+g})")
                if cells:
                    out.append(f"| {LABEL[name]} | {scope} | " + ' | '.join(cells) + ' |')
        out += ['', '### By original-confidence group (primary setting)', '',
                '| Policy | Group | n | Released | Wrong | Accuracy | Queries | Mean utility | Predicted value | Mean posterior (released) | Σ(1−posterior) |',
                '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in _pick(groups, generator=generator, **PRIMARY):
            out.append(f"| {LABEL[r['policy']]} | {r['group']} | {r['n']} | {r['released']} | {r['wrong_released']} | "
                       f"{_fmt(r['released_accuracy'])} | {r['queries']} | {_fmt(r['mean_utility'], sign=True)} | "
                       f"{_fmt(r['mean_predicted_value'], sign=True)} | {_fmt(r['mean_posterior_released'], 4)} | {r['expected_wrong_released']:.2f} |")
        out += ['', '### Stopping changes relative to pooled (primary setting; unchanged rows omitted)', '',
                '| Policy | Pooled → policy | Items |', '|---|---|---:|']
        for r in _pick(transitions, generator=generator, **PRIMARY):
            a, b = r['pooled_to_policy'].split('->')
            if a != b:
                out.append(f"| {LABEL[r['policy']]} | {a} → {b} | {r['items']} |")
        out += ['', '### Posterior Brier after observing Flash (matched eligible items, independent of stopping)', '',
                '| Policy | Scope | n | Accuracy | Mean raw prior | Mean posterior | Brier | Δ vs pooled [95% CI] |', '|---|---|---:|---:|---:|---:|---:|---|']
        for r in _pick(brier, generator=generator, cohort=PRIMARY['cohort'], seed=PRIMARY['seed']):
            diff = (f"{_fmt(r['brier_diff_vs_pooled'], 4, True)} [{_fmt(r['ci95_low'], 4, True)}, {_fmt(r['ci95_high'], 4, True)}]"
                    if r.get('brier_diff_vs_pooled') is not None else '—')
            out.append(f"| {LABEL[r['policy']]} | {r['scope']} | {r['n']} | {_fmt(r.get('accuracy'))} | {_fmt(r.get('mean_raw_prior'))} | "
                       f"{_fmt(r['mean_posterior'], 4)} | {_fmt(r['brier'], 4)} | {diff} |")
        out += ['', '### Stability: conditioned (inner-CV tau) minus pooled, all seeds, cohorts and losses', '',
                '| Cohort | Loss | Seed | Δ utility/item [95% CI] | Δ released | Δ wrong releases | Δ queries | Δ signal-posterior Brier |',
                '|---|---|---|---|---:|---:|---:|---:|']
        for r in policy:
            if r['generator'] != generator or r['policy'] != 'conditioned':
                continue
            match = dict(generator=generator, cohort=r['cohort'], seed=r['seed'], loss=r['loss'], policy='conditioned', scope='all')
            u = _pick(paired, metric='utility_per_item', **match)[0]
            rel = _pick(paired, metric='released_per_item', **match)[0]
            wr = _pick(paired, metric='wrong_released_per_item', **match)[0]
            q = _pick(paired, metric='queries_per_item', **match)[0]
            b = _pick(brier, generator=generator, cohort=r['cohort'], seed=r['seed'], policy='conditioned', scope='matched_eligible')[0]
            out.append(f"| {r['cohort']} | {r['loss']} | {r['seed']} | {_fmt(u['estimate'], sign=True)} [{_fmt(u['ci95_low'], sign=True)}, "
                       f"{_fmt(u['ci95_high'], sign=True)}] | {rel['total']:+g} | {wr['total']:+g} | {q['total']:+g} | "
                       f"{_fmt(b['brier_diff_vs_pooled'], 4, True)} |")
        out += ['', '### Training cells per fold (primary cohort, all seeds): correct / incorrect', '',
                '| Seed | Fold | Low group | High group | Selected tau |', '|---|---:|---|---|---:|']
        for t in _pick(taus, generator=generator, cohort=PRIMARY['cohort']):
            cell = {s['group']: f"{s['train_correct']} / {s['train_incorrect']}" for s in
                    _pick(sizes, generator=generator, cohort=t['cohort'], seed=t['seed'], fold=t['fold'])}
            out.append(f"| {t['seed']} | {t['fold']} | {cell['low']} | {cell['high']} | {t['selected_tau']:g} |")
        out.append('')
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='conditioned_likelihood output directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.input, args.output)
