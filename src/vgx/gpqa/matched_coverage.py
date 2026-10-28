"""Offline, descriptive confidence-only controls matched to policy coverage.

No labels are used to rank or break ties. Boundary ties are averaged analytically
under uniform selection. Matching uses held-out policy release counts, so this
is a retrospective comparator, not a deployable threshold-selection rule.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa.report import mean_interval
from vgx.gpqa.sequential import seal


def release_weights(priors, eligible, k):
    indices = [i for i, valid in enumerate(eligible) if valid]
    if not 0 <= k <= len(indices):
        raise ValueError('matched coverage exceeds eligible candidates')
    w = np.zeros(len(priors))
    if k:
        boundary = sorted((priors[i] for i in indices), reverse=True)[k-1]
        above = [i for i in indices if priors[i] > boundary]
        tied = [i for i in indices if priors[i] == boundary]
        w[above] = 1.
        w[tied] = (k-len(above))/len(tied)
    if not np.isclose(w.sum(), k):
        raise AssertionError('coverage not matched')
    return w


def analyze(root, output):
    root, output = Path(root), Path(output)
    protocol = json.loads((root/'protocol.json').read_text())
    cfg = protocol['config']
    paths = sorted(root.glob('*/*/seed_*.json'))
    if not paths:
        raise ValueError('no completed OOF analyses')
    binding = seal({'source_analysis_id': protocol['analysis_id'],
        'implementation_sha256': file_digest(Path(__file__)),
        'sources': {str(p): file_digest(p) for p in paths},
        'ranking': 'descending unchanged generator confidence',
        'matching': 'each validation fold release count of the verifier policy',
        'ties': 'expected performance under uniform boundary-tie selection',
        'labels_used_for_selection': False, 'evaluation_labels_read': False,
        'new_api_requests': 0, 'deployable_policy': False}, 'comparison_id')
    dest = output/'protocol.json'
    if dest.exists() and json.loads(dest.read_text()) != binding:
        raise ValueError('inputs changed; use a new output directory')
    atomic_json(dest, binding)
    summaries = {}
    for path in paths:
        f = json.loads(path.read_text())
        n = f['n']; y = np.array(f['outcomes']); fold = np.array(f['fold_assignment'])
        priors = f['raw_priors']; data = {}
        for loss_key, decisions in f['decisions'].items():
            loss = float(loss_key)
            eligible = [d['failure'] != 'invalid_generator' for d in decisions['raw']]
            gross = np.where(y == 1, cfg['correct_reward'], -loss)
            for arm, values in decisions.items():
                if not arm.startswith('sequential:'):
                    continue
                chosen = np.array([d['action'] == 'assert' for d in values])
                weights = np.zeros(n)
                for fold_id in sorted(set(fold)):
                    ix = np.flatnonzero(fold == fold_id)
                    weights[ix] = release_weights([priors[i] for i in ix],
                        [eligible[i] for i in ix], int(chosen[ix].sum()))
                k = int(chosen.sum())
                expected_cost = np.array([d['expected_cost_usd'] for d in values])
                policy_u = chosen*gross-cfg['utility_per_usd']*expected_cost
                control_u = weights*gross
                delta = policy_u-control_u
                value = {'n': n, 'released': k, 'coverage': k/n,
                    'control_expected_correct_released': float((weights*y).sum()),
                    'control_expected_wrong_released': float((weights*(1-y)).sum()),
                    'control_expected_accuracy': float((weights*y).sum()/k) if k else None,
                    'control_mean_utility': float(control_u.mean()),
                    'policy_correct_released': int((chosen*y).sum()),
                    'policy_wrong_released': int((chosen*(1-y)).sum()),
                    'policy_accuracy': float((chosen*y).sum()/k) if k else None,
                    'policy_mean_utility': float(policy_u.mean()),
                    'policy_expected_cost_usd': float(expected_cost.sum()),
                    'paired_utility_difference': mean_interval(delta, cfg['bootstrap_repeats'], f['seed']),
                    'fractional_tie_candidates': int(((weights>0)&(weights<1)).sum()),
                    'confidence_only_weights': weights.tolist()}
                data.setdefault(loss_key,{})[arm] = value
        rel = path.relative_to(root)
        atomic_json(output/rel, {'source': str(path), 'item_ids': f['item_ids'], 'comparisons': data})
        summaries[str(rel)] = {loss:{arm:{k:v for k,v in d.items() if k!='confidence_only_weights'}
            for arm,d in arms.items()} for loss,arms in data.items()}
    if any(file_digest(Path(p)) != h for p,h in binding['sources'].items()):
        raise ValueError('source changed during analysis')
    result = {'comparison_id': binding['comparison_id'], 'results': summaries,
        'limitations': ['Retrospective matched coverage, not an independently fitted operational policy.',
            'Expected boundary tie selection; fractional counts are expectations, not extra observed answers.',
            'Intervals resample fixed OOF paired outcomes, excluding refitting, tie-draw and selection uncertainty.',
            'Single-verifier exploratory development results; no final verifier selection.']}
    atomic_json(output/'report.json',result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args();r=analyze(a.root,a.output)
    print(json.dumps({'comparison_id':r['comparison_id'],'completed_analyses':len(r['results'])}))
