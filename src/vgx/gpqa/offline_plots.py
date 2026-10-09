"""Render aggregate figures from an existing offline validation report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = ['#226baf', '#00846b', '#cc7722', '#9958a6', '#67788a']


def render(report_path, output):
    report = json.loads(Path(report_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 130,
                         'svg.hashsalt': 'gpqa-offline-validation'})

    def save(fig, name):
        fig.savefig(output / f'{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(output / f'{name}.svg', bbox_inches='tight', metadata={'Date': None})
        plt.close(fig)

    forecasts = report['forecasts']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    names = [('raw_generator', 'Generator confidence'),
             ('joint_jev_choice_v1', 'Generator + Jev Choice'),
             ('joint_verifier_2', 'Generator + Gemini'),
             ('joint_three_verifiers', 'Generator + all three')]
    for (name, label), color in zip(names, COLORS):
        points = forecasts['matched_coverage'][name]
        axes[0].plot([p['coverage'] for p in points],
                     [1-p['expected_accuracy'] for p in points],
                     label=label, color=color, linewidth=1.8)
    axes[0].set(xlim=(.3, 1.), ylim=(0., .12), xlabel='Coverage (denominator: all 196 items)',
                ylabel='Error rate among releases', title='Ranking at matched coverage')
    axes[0].axhline(.05, color='#bbbbbb', linestyle=':', linewidth=1.)
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(alpha=.15)

    comparisons = [('joint_jev_choice_v1', 'Jev Choice'), ('joint_verifier_1', 'Llama'),
                   ('joint_verifier_2', 'Gemini'), ('joint_three_verifiers', 'All three')]
    for i, (name, label) in enumerate(comparisons):
        value = forecasts['paired_evaluation'][name+'_minus_calibrated_generator']['brier_delta']
        ci = value['ci95']
        axes[1].plot([ci['low'], ci['high']], [i, i], color=COLORS[i], linewidth=2)
        axes[1].scatter(value['estimate'], i, color=COLORS[i], s=35, zorder=3)
    axes[1].set(yticks=range(4), yticklabels=[v[1] for v in comparisons],
                xlabel='Brier difference vs calibrated generator (lower is better)',
                title='Additional forecast information')
    axes[1].invert_yaxis()
    axes[1].axvline(0., linestyle='--', color='#888888', linewidth=1.)
    axes[1].grid(axis='x', alpha=.15)
    fig.suptitle('Exploratory GPQA forecasts — frozen answers, 195 valid candidates', fontsize=13)
    fig.supxlabel('Joint forecasts use all specified signals. Ties use expected random selection; intervals condition on fitted models.', fontsize=9)
    save(fig, 'forecasts')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    kinds = ['none', 'gate', 'myopic', 'nested', 'always']
    for ax, prior in zip(axes, ('raw', 'calibrated')):
        labels = []
        for i, kind in enumerate(kinds):
            metric = report['policy_study']['policies'][f'{prior}_{kind}']
            value = metric['mean_utility']
            ci = value['ci95']
            ax.plot([ci['low'], ci['high']], [i, i], color=COLORS[i], linewidth=2)
            ax.scatter(value['estimate'], i, color=COLORS[i], s=40, zorder=3)
            labels.append(f"{kind.title()}: {metric['released']} released, "
                          f"{metric['wrong_released']} wrong; {metric['queries']} queries")
        ax.set(yticks=range(len(kinds)), yticklabels=labels, xlim=(-.1, .8),
               xlabel='Mean utility per evaluation item',
               title=f'{prior.title()} generator prior')
        ax.invert_yaxis()
        ax.axvline(0., color='#999999', linestyle=':', linewidth=1.)
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('Stopping rules — same fixed order: Jev Choice → Llama → Gemini', fontsize=13)
    fig.supxlabel('Correct +1; wrong −19; expected query USD × 10. Intervals condition on fitted models. Zero new API requests.', fontsize=9)
    save(fig, 'policies')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    cases = ['independent_correct_prior', 'correlated_correct_prior',
             'independent_overconfident_prior', 'correlated_overconfident_prior']
    labels = ['Correct\nassumptions', 'Dependent\nsignals', 'Biased\nprior', 'Dependence\n+ biased prior']
    x = np.arange(4)
    for offset, key, label, color in [(-.25, 'predicted', 'Model-predicted nested value', COLORS[0]),
                                     (0., 'nested', 'Actual nested utility', COLORS[2]),
                                     (.25, 'oracle', 'Exact oracle utility', COLORS[1])]:
        values = []
        for name in cases:
            means = report['simulations']['cases'][name]['mean_over_prior_grid']
            values.append(means['nested']['predicted_nested_value'] if key == 'predicted'
                          else means[key]['utility'])
        axes[0].bar(x+offset, values, .24, color=color, label=label)
    axes[0].set(xticks=x, xticklabels=labels, ylabel='Mean utility across the fixed prior grid',
                ylim=(0., .86), title='Correct recursion can still use a wrong model')
    axes[0].legend(fontsize=8, loc='upper left')
    learning = report['simulations']['finite_calibration_learning']
    ns = [v['calibration_n'] for v in learning]
    means = [v['mean_regret'] for v in learning]
    low = [v['regret_interval_across_simulated_calibrations'][0] for v in learning]
    high = [v['regret_interval_across_simulated_calibrations'][1] for v in learning]
    axes[1].plot(ns, means, 'o-', color=COLORS[0], label='Mean regret')
    axes[1].fill_between(ns, low, high, color=COLORS[0], alpha=.15, label='2.5–97.5% across calibration replicates')
    axes[1].set(xscale='log', xlabel='Number of simulated calibration examples',
                ylabel='Utility lost relative to the exact oracle', title='Finite likelihood estimation matters')
    axes[1].set_xticks(ns, [str(n) for n in ns])
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.15)
    fig.suptitle('Controlled simulations — exact evaluation under known distributions', fontsize=13)
    fig.supxlabel('Oracle obeys the same fixed order and sees no future signals. These constructed scenarios do not establish GPQA gains.', fontsize=9)
    save(fig, 'simulations')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.), layout='constrained')
    for ax, prior in zip(axes, ('raw', 'calibrated')):
        for kind, color in zip(('none', 'myopic', 'nested'), COLORS):
            points = [p for p in report['policy_study']['cost_loss_sensitivity']
                      if p['prior'] == prior and p['kind'] == kind and p['loss'] == 19.]
            ax.plot([p['cost_multiplier'] for p in points],
                    [p['mean_utility']['estimate'] for p in points], 'o-', color=color, label=kind.title())
        ax.set(xscale='log', xlabel='Multiplier on estimated verification cost',
               ylabel='Mean utility', title=f'{prior.title()} generator prior')
        ax.axvline(1., color='#bbbbbb', linestyle=':')
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle('Cost sensitivity at wrong-release loss 19 — exploratory, no setting chosen from evaluation', fontsize=11)
    save(fig, 'cost_sensitivity')

    aggregate = {k: report[k] for k in ('schema', 'protocol', 'calibration_n', 'evaluation_n',
                 'calibration_valid_n', 'evaluation_valid_n', 'calibration_incorrect_valid_n',
                 'evaluation_incorrect_valid_n', 'costs_utility', 'selection', 'policy_study',
                 'refit_bootstrap', 'simulations', 'api_calls', 'source_artifacts_unchanged')}
    aggregate['forecasts'] = {k: v for k, v in forecasts.items() if k != 'matched_coverage'}
    aggregate['matched_coverage_131_raw_generator'] = forecasts['matched_coverage']['raw_generator'][130]
    (output/'aggregate.json').write_text(json.dumps(aggregate, indent=2, sort_keys=True)+'\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('results/gpqa_offline_validation_20261004/report.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/gpqa_offline_validation_20261004'))
    args = parser.parse_args()
    print(render(args.report, args.output))
