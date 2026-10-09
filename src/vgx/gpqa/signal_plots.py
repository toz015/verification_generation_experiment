"""Aggregate phase-one screening figures. No question text or candidate IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def create_figures(root, output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'signal_report.json').read_text())
    analysis=json.loads((root/'calibration_comparison.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    records=json.loads((root/'signal_records.json').read_text())
    if report['study_id']!=analysis['study_id'] or report['study_id']!=plan['study_id']:
        raise ValueError('mismatched figure inputs')
    models=[m['id'] for m in plan['config']['models']]
    names={'gemini_flash':'Gemini Flash','gemini_lite':'Gemini Flash-Lite',
           'mistral_small':'Mistral Small','mistral_medium':'Mistral Medium','claude_haiku':'Claude Haiku'}
    fig,axes=plt.subplots(2,len(models),figsize=(16,6),sharey=True)
    colors=['#ca562c','#247a96']
    for col,model in enumerate(models):
        for row,signal in enumerate(('binary','probability')):
            ax=axes[row,col]
            subset=[r for r in records if r['model']==model and r['signal']==signal]
            if signal=='binary':
                labels=['0','1','Invalid'];bins=2
            else:
                labels=['0-.2','.2-.4','.4-.6','.6-.8','.8-1','Invalid'];bins=5
            for y in (0,1):
                items=[r for r in subset if r['correct']==y]
                scores=[r['score'] for r in items if r['failure'] is None]
                counts=np.histogram(scores,bins=np.linspace(0,1,bins+1))[0].tolist()
                counts.append(sum(r['failure'] is not None for r in items))
                heights=np.array(counts)/max(1,len(items))
                ax.bar(np.arange(len(labels))+(y-.5)*.38,heights,width=.38,color=colors[y],
                       label=('Candidate wrong' if y==0 else 'Candidate correct'))
            ax.set_xticks(np.arange(len(labels)),labels,rotation=35 if signal=='probability' else 0,fontsize=8)
            ax.set_ylim(0,1.08);ax.grid(axis='y',alpha=.15)
            ax.set_title(names.get(model,model)+' / '+signal,fontsize=10)
            if col==0:ax.set_ylabel('Fraction within correctness class')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.95),frameon=False)
    fig.suptitle('Verifier signals on frozen calibration candidates',fontsize=15,y=1.)
    fig.text(.5,.01,'Each class is normalized separately. Invalid replies remain visible; no evaluation items are included.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.89));fig.savefig(output/'signal_distributions.png',dpi=180,bbox_inches='tight');plt.close(fig)

    seeds=analysis['fold_seed_sensitivity'];positions=np.arange(len(models))
    fig,ax=plt.subplots(figsize=(10,5))
    for offset,signal,color in [(-.18,'binary','#247a96'),(.18,'probability','#c27725')]:
        values=[]
        for model in models:
            key='joint:'+model+':'+signal
            values.append([s['differences_vs_calibrated_generator'][key]['brier'] for s in seeds
                           if s['differences_vs_calibrated_generator'] is not None])
        if any(not v for v in values):raise ValueError('insufficient fold results for plot')
        means=np.array([np.mean(v) for v in values]);low=np.array([min(v) for v in values]);high=np.array([max(v) for v in values])
        ax.bar(positions+offset,means,width=.32,color=color,label=signal,
               yerr=np.maximum(0.,np.array([means-low,high-means])),capsize=4,error_kw={'linewidth':1})
    ax.axhline(0,color='black',linewidth=.8)
    ax.set_xticks(positions,[names.get(m,m) for m in models])
    ax.set_ylabel('Brier change vs calibrated generator (negative is better)')
    common=analysis['all_models_common_cohort']
    ax.set_title(f'Added forecast information: common cohort n={common["n"]}, errors={common["wrong"]}')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    fig.text(.5,.01,'Bars: mean over five fixed fold seeds. Whiskers: seed range, NOT a confidence interval. Calibration only.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(output/'forecast_increment.png',dpi=180,bbox_inches='tight');plt.close(fig)
    return {'figures':['signal_distributions.png','forecast_increment.png']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(create_figures(args.root,args.output)))


if __name__=='__main__':main()
