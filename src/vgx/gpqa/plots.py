"""Export aggregate scientific figures; never read question text or call APIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_policy(policy, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    table=policy['planner']
    belief=np.asarray(table['grid'])
    stop=np.maximum(0.,(table['correct_reward']+table['incorrect_loss'])*belief-table['incorrect_loss'])
    fig,axes=plt.subplots(1,len(table['Q']),figsize=(11,4),squeeze=False,layout='constrained')
    threshold=table['incorrect_loss']/(table['correct_reward']+table['incorrect_loss'])
    for stage,ax in enumerate(axes[0]):
        q=np.asarray(table['Q'][stage]);j=np.asarray(table['J'][stage])
        ax.fill_between(belief,0,1,where=q>stop,transform=ax.get_xaxis_transform(),color='#e7a33e',alpha=.16,label='Query region')
        ax.plot(belief,stop,color='#333333',linewidth=1.8,label='Stop: max(0, A(b))')
        ax.plot(belief,q,color='#c77713',linewidth=1.7,label='Q(b): query value')
        ax.plot(belief,j,color='#216a9b',linestyle='--',linewidth=1.5,label='J(b): optimal value')
        query_points=belief[q>stop]
        left=max(0.,min(threshold,float(query_points.min()) if len(query_points) else threshold)-.03)
        ax.axvline(threshold,color='#888888',linestyle=':',linewidth=1,label='Terminal assertion threshold')
        ax.set(xlim=(left,1),ylim=(-.06,1.04),xlabel='Current correctness belief b (boundary detail)',ylabel='Expected utility',
               title=f"Before verifier {stage+1} • cost {table['costs'][stage]:.4f}")
        ax.grid(alpha=.18)
        ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Nested verification value tables — fitted on calibration only',fontsize=13)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.png'),dpi=180)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


def plot_evaluation(report, output, scenario='primary_95'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    evaluation=report['evaluation_metrics']
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    forecasts=[('Generator',evaluation['generator_confidence'],'#216a9b'),
               ('Both-verifier posterior',evaluation['sequential_independence_posterior'],'#c77713')]
    ax=axes[0]
    ax.plot([0,1],[0,1],color='#888888',linestyle=':',label='Perfect calibration')
    for name,forecast,color in forecasts:
        rows=forecast['reliability']
        x=[r['mean_confidence'] for r in rows];y=[r['empirical_accuracy'] for r in rows]
        errors=[[max(0.,r['empirical_accuracy']-r['empirical_accuracy_ci95']['low']) for r in rows],
                [max(0.,r['empirical_accuracy_ci95']['high']-r['empirical_accuracy']) for r in rows]]
        ax.errorbar(x,y,yerr=errors,fmt='o',capsize=3,color=color,label=f'{name} (n={forecast["n"]})')
        for xx,yy,row in zip(x,y,rows):
            ax.annotate(f"n={row['count']}",(xx,yy),xytext=(5,5 if name=='Generator' else -12),
                        textcoords='offset points',fontsize=7,color=color)
    ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='Mean reported probability',ylabel='Observed correctness',title='Evaluation calibration • Wilson 95% intervals')
    ax.legend(fontsize=8);ax.grid(alpha=.18)
    ax=axes[1]
    policies=evaluation['routing_scenarios'][scenario]['policies']
    labels={'always_answer':'Always answer','confidence_only':'Confidence only','query_one_verifier':'Llama only',
            'query_second_verifier':'Gemini only','query_all_verifiers':'Both verifiers','sequential_stopping':'Sequential'}
    import numpy as np
    rows=[policies[name] for name in labels]
    positions=np.arange(len(rows))
    ax.barh(positions+.18,[r['release_coverage'] for r in rows],height=.34,color='#216a9b',label='Coverage')
    values=[r['accuracy_among_released'] or 0. for r in rows]
    errors=[[0. if r['accuracy_among_released'] is None else max(0.,r['accuracy_among_released']-r['accuracy_among_released_ci95']['low']) for r in rows],
            [0. if r['accuracy_among_released'] is None else max(0.,r['accuracy_among_released_ci95']['high']-r['accuracy_among_released']) for r in rows]]
    ax.barh(positions-.18,values,height=.34,xerr=errors,capsize=2,color='#c77713',label='Accuracy among released')
    for index,(name,label) in enumerate(labels.items()):
        row=policies[name]
        if row['accuracy_among_released'] is None:
            ax.text(.02,index-.18,'N/A (no releases)',fontsize=8)
    ax.axvline(.95,color='#888888',linestyle=':',label='Nominal 95% target')
    ax.set(xlim=(0,1.025),yticks=positions,yticklabels=list(labels.values()),xlabel='Proportion',title='Primary policy • accuracy and coverage')
    ax.invert_yaxis()
    ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2)
    ax.grid(axis='x',alpha=.18)
    fig.suptitle(f"Filtered GPQA Diamond evaluation (n={evaluation['n']})",fontsize=13)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.png'),dpi=180)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    plot_policy(json.loads(args.policy.read_text()),args.output_dir/'value_tables')
    if args.report:
        plot_evaluation(json.loads(args.report.read_text()),args.output_dir/'evaluation')


if __name__=='__main__':
    main()
