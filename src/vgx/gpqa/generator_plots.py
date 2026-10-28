"""Aggregate-only figures for the frozen, 50-question generator comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root, output):
    root, output = Path(root), Path(output)
    data = json.loads((root/'comparison.json').read_text())
    groups = data['generators']
    names = list(groups)
    labels = {'gemini_baseline':'Gemini 3.8 Flash', 'qwen':'Qwen3 235B Instruct', 'llama':'Llama 3.3 70B'}
    x = np.arange(len(names)); width = .35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    accuracy = [groups[n]['accuracy_all'] for n in names]
    confidence = [groups[n]['mean_confidence'] for n in names]
    axes[0].bar(x-width/2, accuracy, width, label='Observed answer accuracy', color='#2166ac')
    axes[0].bar(x+width/2, confidence, width, label='Mean reported confidence', color='#e08214')
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title('Accuracy and self-reported confidence')
    axes[0].set_ylabel('Fraction')
    for i,n in enumerate(names):
        axes[0].text(i-width/2, accuracy[i]+.025, f"{groups[n]['correct']}/{groups[n]['n']}", ha='center', fontsize=9)
        axes[0].text(i+width/2, confidence[i]+.025, f'{confidence[i]:.2f}', ha='center', fontsize=9)
    raw = [groups[n]['raw_metrics']['brier'] for n in names]
    calibrated = [groups[n]['initial_calibration_oof'].get('metrics',{}).get('calibrated_generator',{}).get('brier',float('nan')) for n in names]
    axes[1].bar(x-width/2, raw, width, label='Raw confidence', color='#e08214')
    axes[1].bar(x+width/2, calibrated, width, label='Calibrated (3-fold OOF)', color='#2166ac')
    axes[1].set_title('Correctness forecasts: lower Brier is better')
    axes[1].set_ylabel('Brier score')
    for ax in axes:
        ax.set_xticks(x, [labels.get(n,n) for n in names], fontsize=9)
        ax.legend(fontsize=8, loc='upper left' if ax is axes[1] else 'lower left')
        ax.spines[['top','right']].set_visible(False)
    axes[1].set_xticks(x,[labels.get(n,n)+f"\n(confidence n={groups[n]['confidence_n']})" for n in names],fontsize=9)
    fig.suptitle('Same 50 calibration questions; independent generator arms', fontsize=14)
    fig.text(.5,.015,'Exploratory pilot; model/settings differ. Accuracy: n=50 each; confidence metrics exclude invalid confidence.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95))
    output.mkdir(parents=True, exist_ok=True)
    path=output/'generator_comparison.png';fig.savefig(path,dpi=180);plt.close(fig)
    return str(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(plot(args.root,args.output))
