"""Aggregate-only plots for the direct/rationale paired pilot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root,output):
    data=json.loads((Path(root)/'comparison.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    colors=['#2166ac','#e08214'];x=np.arange(2);width=.35
    for offset,condition in enumerate(('direct','reasoned')):
        summaries=[data['generators'][g+':'+condition] for g in ('qwen','llama')]
        values=[s['correct']/s['n'] for s in summaries]
        position=x+(offset-.5)*width
        axes[0].bar(position,values,width,label=condition.title(),color=colors[offset])
        for pos,s,v in zip(position,summaries,values):
            axes[0].text(pos,v+.025,f"{s['correct']}/{s['n']}",ha='center',fontsize=9)
        axes[1].bar(position,[s['mean_confidence_if_wrong'] for s in summaries],width,color=colors[offset])
        axes[2].bar(position,[data['paired_conditions'][g]['matched_'+condition+'_brier'] for g in ('qwen','llama')],width,color=colors[offset])
    axes[0].set_title('Correct answers / 50 requested');axes[0].set_ylim(0,1.12);axes[0].legend(fontsize=9)
    axes[1].set_title('Mean confidence on wrong answers');axes[1].set_ylim(0,1.05)
    axes[2].set_title('Paired raw Brier (lower is better)')
    for ax in axes:
        ax.set_xticks(x,['Qwen3 235B','Llama 3.3 70B'])
        ax.spines[['top','right']].set_visible(False)
    axes[2].set_xticks(x,[label+f"\n(paired n={data['paired_conditions'][g]['raw_brier_on_matched_confidence']['n']})"
                          for g,label in [('qwen','Qwen3 235B'),('llama','Llama 3.3 70B')]])
    fig.suptitle('Same models, settings and 50 questions; new rationale-before-answer prompt',fontsize=14)
    fig.text(.5,.015,'Exploratory follow-up, not a held-out test. Invalid answers remain in accuracy; confidence metrics exclude invalid confidence.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.95));output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'prompt_comparison.png';fig.savefig(path,dpi=180);plt.close(fig)
    return str(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(plot(args.root,args.output))
