"""Tables and figures for the risk-controlled acceptance study (reads saved outputs only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vgx.gpqa.risk_control import accepted_for_power, binomial_upper_bound

COLORS = {'A': '#2a78d6', 'B': '#eb6834', 'C': '#1baf7a'}  # reference categorical slots 1-3
LABELS = {'A': 'A: generator confidence', 'B': 'B: always-query Flash', 'C': 'C: sequential Flash + gate'}
NAMES = {'qwen': 'Qwen', 'gemini': 'Gemini'}
COHORTS = {'all_calibration': '249-question cohort', 'remaining_calibration': '199-question cohort (pilot excluded)'}


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '-'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f'{x:.{digits}f}'


def span(values, digits=3):
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not values:
        return '-'
    low, high = fmt(min(values), digits), fmt(max(values), digits)
    return low if low == high else f'{low} to {high}'


def table(rows, headers):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def risk_coverage_figure(curves, output, alphas, min_accepted=10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), sharex=True, sharey=True)
    for r, cohort in enumerate(COHORTS):
        for c, generator in enumerate(NAMES):
            ax = axes[r][c]
            for j, alpha in enumerate(alphas):
                ax.axhline(alpha, color='#8a8986', lw=1, ls='--', zorder=1,
                           label=f"alpha = {' and '.join(f'{a:g}' for a in alphas)}" if j == 0 else None)
            # C is drawn first and wider so B stays visible where the two coincide
            for method, width in (('C', 3.2), ('B', 1.3), ('A', 1.6)):
                sub = curves[(curves.generator == generator) & (curves.cohort == cohort) & (curves.method == method)
                             & (curves.m >= min_accepted)]
                for i, (seed, g) in enumerate(sub.groupby('seed')):
                    g = g.sort_values('acceptance_rate')
                    ax.step(g.acceptance_rate, g.conditional_error, where='post', color=COLORS[method], lw=width,
                            alpha=.55 if method == 'C' else .85, label=LABELS[method] if i == 0 else None, zorder=2)
            ax.set_title(f'{NAMES[generator]}, {COHORTS[cohort]}', fontsize=9.5, loc='left')
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, .42)
            ax.grid(color='#e6e5e1', lw=.6)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if r == 1:
                ax.set_xlabel('acceptance rate m/N (invalid and failed items stay in N)')
            if c == 0:
                ax.set_ylabel('conditional error among accepted k/m')
    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [labels.index(LABELS[m]) for m in 'ABC'] + [i for i, l in enumerate(labels) if l.startswith('alpha')]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(.5, .995))
    fig.text(.5, .005, f'Out-of-fold development scores; one line per fold seed (5); points with m < {min_accepted} omitted; '
             'where B and C coincide the orange line runs inside the green band. Exploratory: not certified.',
             ha='center', fontsize=8, color='#52514e')
    fig.tight_layout(rect=(0, .02, 1, .96))
    fig.savefig(output, dpi=160, facecolor='#fcfcfb')
    plt.close(fig)


def sample_size_figure(cfg, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    delta, power = cfg['delta'], cfg['sample_size']['power']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    family_colors = {1: '#2a78d6', 13: '#eb6834', 26: '#1baf7a'}
    rows = []
    for ax, alpha in zip(axes, cfg['sample_size']['alphas']):
        risks = np.round(np.linspace(.005, alpha*.8, 12), 4)
        for family, color in family_colors.items():
            needed = [accepted_for_power(float(r), alpha, delta/family, power) for r in risks]
            rows += [{'alpha': alpha, 'family_size': family, 'true_conditional_error': float(r), 'accepted_needed': n}
                     for r, n in zip(risks, needed)]
            ax.plot(risks, needed, marker='o', ms=4, lw=2, color=color, label=f'M = {family}')
        ax.set_yscale('log')
        ax.set_title(f'alpha = {alpha:g}, delta = {delta:g}, power {power:g}', fontsize=9.5, loc='left')
        ax.set_xlabel('assumed true conditional error of the rule')
        ax.set_ylabel('accepted calibration examples needed')
        ax.grid(color='#e6e5e1', lw=.6, which='both')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160, facecolor='#fcfcfb')
    plt.close(fig)
    return rows


def build(cfg, phase1, phase2, plan, validation, output):
    output = Path(output)
    for sub in ('figures', 'tables'):
        (output/sub).mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(phase1/'risk_coverage_curves.csv')
    exploration = json.loads((phase1/'exploration.json').read_text())
    alphas = [cfg['alpha']['primary'], cfg['alpha']['secondary']]
    risk_coverage_figure(curves, output/'figures'/'risk_coverage.png', alphas)
    power_rows = sample_size_figure(cfg, output/'figures'/'sample_size.png')
    pd.DataFrame(power_rows).to_csv(output/'tables'/'sample_size_curve.csv', index=False)
    md = ['# Risk-controlled acceptance: tables', '',
          'Generated by `python -m vgx.gpqa.risk_report` from saved outputs. Phase I numbers are exploratory '
          '(development data, out-of-fold scores, thresholds chosen on the same data). Phase II numbers are a software '
          'dry run on a split of the same development data and are not a certification.', '']

    # 1. method summary
    summary = pd.DataFrame(exploration['summary'])
    rows = []
    for (g, cohort, m), s in summary.groupby(['generator', 'cohort', 'method'], sort=False):
        rows.append([NAMES[g], cohort, m, fmt(s.N.iloc[0]), fmt(s.valid_candidates.iloc[0]), span(s.eligible.tolist(), 0),
                     span(s.invalid_generator.tolist(), 0), span(s.signal_failures.tolist(), 0), span(s.queries.tolist(), 0),
                     span((s.usage_cost_usd/s.N).tolist(), 5), span((s.expected_cost_usd/s.N).tolist(), 5)])
    md += ['## 1. Denominators, failures, queries and costs (Phase I, range over 5 seeds)', '',
           'Eligible = candidates a method can accept at some threshold (A: valid; B: valid with a parsed Flash score; '
           'C: released by the frozen sequential policy). Usage cost is the cached-call usage estimate (not billing).', '',
           table(rows, ['generator', 'cohort', 'method', 'N', 'valid', 'eligible', 'invalid candidate', 'signal failures',
                        'queries', 'usage USD / question', 'expected USD / question']), '']

    # 2. ranking
    ranking = pd.DataFrame(exploration['ranking'])
    rows = []
    for (g, cohort), s in ranking.groupby(['generator', 'cohort'], sort=False):
        rows.append([NAMES[g], cohort, fmt(s.n.iloc[0]), fmt(s.n_incorrect.iloc[0]), span(s.auroc_A.tolist()),
                     span(s.auroc_B.tolist()), span(s.auroc_B_minus_A.tolist()),
                     span([c[0] for c in s.ci95]), span([c[1] for c in s.ci95])])
    md += ['## 2. Ranking of correct over incorrect answers (AUROC, range over seeds)', '',
           'Items with a valid candidate and a parsed Flash score. CI: paired item bootstrap of B minus A within one seed.', '',
           table(rows, ['generator', 'cohort', 'n', 'incorrect', 'AUROC A', 'AUROC B', 'B - A', 'CI low', 'CI high']), '']

    # 3. matched coverage
    matched = pd.DataFrame([{**{k: v for k, v in r.items() if k != 'point'}, **(r['point'] or {})}
                            for r in exploration['matched']])
    rows = []
    for (g, cohort, target), s in matched.groupby(['generator', 'cohort', 'coverage_target'], sort=False):
        cells = []
        for method in 'ABC':
            x = s[s.method == method]
            if x.acceptance_rate.isna().all():
                cells.append('not reached')
            else:
                cells.append(f"{span(x.acceptance_rate.tolist(), 2)} / {span(x.conditional_error.tolist())}")
        rows.append([NAMES[g], cohort, fmt(target, 1), *cells])
    md += ['## 3. Matched coverage (exploratory)', '',
           'For each target, the strictest threshold whose acceptance rate reaches the target. Cells: actual acceptance '
           'rate / conditional error k/m (range over seeds). Tied scores make exact matching impossible, '
           'especially for A, whose confidence values are coarse.', '',
           table(rows, ['generator', 'cohort', 'target', 'A', 'B', 'C']), '']

    # 4. exploratory low-risk regions
    rows = []
    for r_alpha in alphas:
        for g in NAMES:
            for cohort in COHORTS:
                for method in 'ABC':
                    found = [r['region'] for r in exploration['regions'] if (r['generator'], r['cohort'], r['method'],
                             r['alpha']) == (g, cohort, method, r_alpha)]
                    have = [x for x in found if x]
                    rows.append([fmt(r_alpha, 2), NAMES[g], cohort, method, f'{len(have)}/{len(found)}',
                                 span([x['acceptance_rate'] for x in have]), span([x['m'] for x in have], 0),
                                 span([x['k'] for x in have], 0), span([x['conditional_error'] for x in have]),
                                 span([x['upper_bound_single_rule_95'] for x in have])])
    md += ['## 4. Exploratory low-risk regions (in-sample, optimistic)', '',
           'Largest acceptance rate whose development k/m <= alpha, chosen on the same out-of-fold scores. '
           'U95 is a single-rule exact bound with no multiplicity correction; it is shown only to indicate distance '
           'from a certificate.', '',
           table(rows, ['alpha', 'generator', 'cohort', 'method', 'seeds with region', 'acceptance rate', 'm', 'k', 'k/m',
                        'single-rule U95']), '']

    # 5. family rules on development data
    family = pd.DataFrame(exploration['family_rules'])
    rows = []
    for (g, method, t), s in family[family.cohort == 'all_calibration'].groupby(['generator', 'method', 'threshold'], sort=False):
        bounds = [binomial_upper_bound(int(k), int(m), cfg['delta']/26) for k, m in zip(s.k, s.m)]
        rows.append([NAMES[g], method, fmt(t, 3), span(s.acceptance_rate.tolist()), span(s.m.tolist(), 0),
                     span(s.k.tolist(), 0), span([x for x in s.conditional_error.tolist() if x == x]), span(bounds)])
    md += ['## 5. Declared family rules on development data (249-question cohort, out of fold)', '',
           'The last column applies the Phase II bound (delta/26) to each seed\'s development counts. It is illustrative '
           'only: these data were used to design the family, so this is not a certificate.', '',
           table(rows, ['generator', 'method', 'threshold', 'acceptance rate', 'm', 'k', 'k/m', 'U at delta/26 (illustrative)']),
           '']

    # 6. B vs C acceptance overlap
    overlap = pd.DataFrame(exploration['b_c_overlap'])
    rows = []
    for (g, cohort), s in overlap.groupby(['generator', 'cohort'], sort=False):
        rows.append([NAMES[g], cohort, f"{int(s.identical_sets.sum())}/{len(s)}",
                     span((s.accepted_both/s[['accepted_B', 'accepted_C']].max(axis=1).replace(0, np.nan)).tolist())])
    c_vs_b = summary.pivot_table(index=['generator', 'cohort', 'seed'], columns='method', values='queries').reset_index()
    c_vs_b['order'] = c_vs_b.generator.map(list(NAMES).index)
    c_vs_b = c_vs_b.sort_values(['order', 'cohort', 'seed'])
    md += ['## 6. B and C at the declared thresholds (>= 0.95)', '',
           'C only releases candidates whose terminal belief clears the L = 19 bar (0.95), so at thresholds >= 0.95 its '
           'accepted set is compared with B. Identical = same question IDs accepted (count over thresholds x seeds).', '',
           table(rows, ['generator', 'cohort', 'identical accepted sets', 'shared / larger set']), '',
           table([[NAMES[g], cohort, span(s.C.tolist(), 0), span(s.B.tolist(), 0),
                   span((1 - s.C/s.B).tolist(), 3)] for (g, cohort), s in c_vs_b.groupby(['generator', 'cohort'], sort=False)],
                 ['generator', 'cohort', 'C queries', 'B queries', 'fraction of B queries C skips']), '']

    # 7. secondary posterior diagnostics and utility
    items = pd.read_csv(phase1/'item_scores.csv', usecols=['generator', 'cohort', 'seed', 'method', 'eligible', 'score', 'correct'])
    rows = []
    for (g, method), s in items[items.cohort == 'all_calibration'].groupby(['generator', 'method'], sort=False):
        for t in cfg['certification']['family']['thresholds'][method]:
            acc = s[(s.eligible) & (s.score >= t)]
            if len(acc) == 0:
                rows.append([NAMES[g], method, fmt(t, 3), 0, '-', '-'])
                continue
            per_seed = acc.groupby('seed').agg(pred=('score', lambda x: 1 - x.mean()), obs=('correct', lambda x: 1 - x.mean()))
            rows.append([NAMES[g], method, fmt(t, 3), int(len(acc)/acc.seed.nunique()), span(per_seed.pred.tolist()),
                         span(per_seed.obs.tolist())])
    util = curves[(curves.cohort == 'all_calibration') & (curves.threshold >= .95)]
    md += ['## 7. Secondary: posterior diagnostics and utility', '',
           'Mean of (1 - score) over accepted items, read as a probability, against observed k/m. Scores are selection '
           'scores; a gap means they are not calibrated probabilities of correctness.', '',
           table(rows, ['generator', 'method', 'threshold', 'accepted (mean per seed)', 'mean 1 - score', 'observed k/m']), '',
           'Mean utility at L = 19 (R = 1, cost x 10 per USD) when accepting at threshold 0.95 (the lowest declared '
           'threshold, equal to the L = 19 release bar), range over seeds:', '',
           table([[NAMES[g], method, span(u.secondary_mean_utility_L19.tolist())]
                  for (g, method), u in util.groupby(['generator', 'method'], sort=False)
                  for u in [u.sort_values('threshold').groupby('seed').head(1)]],
                 ['generator', 'method', 'mean utility (secondary)']), '']

    # 8. Phase II dry run
    partition = json.loads((phase2/'partition.json').read_text())
    counts = pd.Series(partition['roles']).value_counts()
    md += ['## 8. Phase II software dry run (development data; not a certification)', '',
           f"Question-level partition shared by both generators (seed {cfg['certification']['dry_run_partition']['seed']}): "
           + ', '.join(f'{r} {counts.get(r, 0)}' for r in ('fit', 'calibration', 'evaluation')) + '.', '']
    for alpha in alphas:
        calibration = json.loads((phase2/f'calibration_alpha_{alpha:g}.json').read_text())
        selection = json.loads((phase2/f'selection_alpha_{alpha:g}.json').read_text())
        rows = [[r['rule_id'], r['N'], r['m'], r['k'], fmt(r['conditional_error']), fmt(r['upper_bound']),
                 'yes' if r['certified'] else 'no'] for r in calibration['certification_table']]
        alt = sum(r['certified'] for r in calibration['fixed_sequence_table'])
        md += [f'### alpha = {alpha:g}', '',
               f"Status: **{selection['status']}** Selected rule: {selection['selected_rule'] or 'none'}. "
               f"Fixed-sequence alternative: {alt} rules certified, selected {calibration['fixed_sequence_selected'] or 'none'}.", '',
               table(rows, ['rule', 'N', 'm', 'k', 'k/m', f"U (delta/{selection['family_size']})", 'certified']), '']
    evaluation = json.loads((phase2/f'evaluation_alpha_{alphas[0]:g}.json').read_text())
    rows = [[r['rule_id'], r['N'], r['valid_candidates'], r['m'], r['k'], fmt(r['conditional_error']),
             f"{fmt(r['acceptance_rate'])} [{fmt(r['acceptance_rate_cp95'][0])}, {fmt(r['acceptance_rate_cp95'][1])}]",
             r['queries'], fmt(r['usage_cost_usd'], 4)] for r in evaluation['descriptive_family_on_evaluation']]
    md += ['### Evaluation partition, descriptive only', '',
           'No rule was selected, so there is no certified rule to evaluate. These counts are for transparency and '
           'were read after the selection files were sealed.', '',
           table(rows, ['rule', 'N', 'valid', 'm', 'k', 'k/m', 'm/N [95% CI]', 'queries', 'usage USD']), '']

    # 9. sample size
    plan_value = json.loads((plan/'sample_size_plan.json').read_text())
    minimum = pd.DataFrame(plan_value['min_accepted_for_k_errors'])
    power = pd.DataFrame(plan_value['accepted_for_power'])
    md += ['## 9. Sample-size planning', '',
           'Reference checks: ' + '; '.join(f"{r['check']}: {r['computed']} (expected {r['expected']})"
                                            for r in plan_value['reference_checks']) + '.', '',
           'Minimum accepted calibration examples m for which k observed errors still certify:', '']
    ks = sorted(minimum.observed_errors.unique())
    rows = [[fmt(a, 2), fmt(f), *[fmt(int(minimum[(minimum.alpha == a) & (minimum.family_size == f)
                                                   & (minimum.observed_errors == k)].min_accepted.iloc[0])) for k in ks]]
            for (a, f), _ in minimum.groupby(['alpha', 'family_size'])]
    md += [table(rows, ['alpha', 'M', *[f'k = {k}' for k in ks]]), '',
           'Accepted calibration examples for 80% probability that one rule with the given true conditional error '
           'certifies:', '']
    risks = sorted(power.true_conditional_error.unique())
    rows = [[fmt(a, 2), fmt(f), *[fmt(_cell(power, a, f, r)) for r in risks]]
            for (a, f), _ in power.groupby(['alpha', 'family_size'])]
    md += [table(rows, ['alpha', 'M', *[f'risk {r:g}' for r in risks]]), '']
    implied = pd.DataFrame(plan_value['implied_calibration_questions'])
    if len(implied):
        sub = implied[(implied.family_size == 26) & (implied.status == 'ok')]
        rows = [[r.rule_id, fmt(r.alpha, 2), fmt(r.acceptance_rate), fmt(r.observed_conditional_error),
                 fmt(r.assumed_true_conditional_error), fmt(r.accepted_needed), fmt(r.calibration_questions_needed)]
                for r in sub.itertuples() if r.assumed_true_conditional_error <= r.alpha/2 + 1e-12
                or r.assumption == 'observed development k/m']
        md += ['Implied calibration questions for the M = 26 family (questions = accepted needed / development '
               'acceptance rate). Development rates are optimistic; rows assuming the observed development k/m are '
               'shown for scale only.', '',
               table(rows, ['rule', 'alpha', 'dev m/N', 'dev k/m', 'assumed risk', 'accepted needed', 'questions needed']), '']

    # 10. procedure validation
    if validation and Path(validation).exists():
        value = json.loads(Path(validation).read_text())
        rows = [[r['scenario'], fmt(r['alpha'], 2), r['n_per_chain'], r['family_size'], r['rules_at_or_above_alpha'],
                 fmt(r['familywise_error']['bonferroni'], 4), fmt(r['familywise_error']['fixed_sequence'], 4),
                 fmt(r['any_rule_selected']['bonferroni'], 3), fmt(r['any_rule_selected']['fixed_sequence'], 3)]
                for r in value['results']]
        md += ['## 10. Procedure validation (synthetic Monte Carlo)', '',
               f"{value['results'][0]['repeats']} repetitions per row; 6 chains of 5 nested rules (M = 30). FWER = share "
               'of repetitions in which any rule whose true conditional error is >= alpha was certified; it should '
               'not exceed delta = 0.05.', '',
               table(rows, ['scenario', 'alpha', 'N per chain', 'M', 'rules with risk >= alpha', 'FWER Bonferroni',
                            'FWER fixed-sequence', 'any rule selected (Bonferroni)', 'any rule selected (fixed-seq.)']), '']
    (output/'TABLES.md').write_text('\n'.join(md) + '\n')


def _cell(power, a, f, r):
    x = power[(power.alpha == a) & (power.family_size == f) & (power.true_conditional_error == r)].accepted_needed
    return None if x.empty or pd.isna(x.iloc[0]) else int(x.iloc[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/gpqa_risk_acceptance.json')
    parser.add_argument('--phase1', required=True)
    parser.add_argument('--phase2', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--validation')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(json.loads(Path(args.config).read_text()), Path(args.phase1), Path(args.phase2), Path(args.plan),
          args.validation, args.output)


if __name__ == '__main__':
    main()
