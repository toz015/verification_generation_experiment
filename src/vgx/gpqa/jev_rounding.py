"""Post hoc, offline sensitivity to rounded Choice probabilities.

Never changes sealed live policies, original parser behavior, or cached responses.
"""
import argparse
import copy
import json
import math
from pathlib import Path

from vgx.common.api import RequestCache
from vgx.common.storage import atomic_json
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.report import build_report
from vgx.gpqa.verifiers import _prob, load_specs


def rounded_choice(payload, candidate, model, *, normalize=False):
    """Allow only mass errors consistent with four two-decimal marginals.

    Keep the reported candidate marginal or renormalize, as two separate analyses.
    This does not assert which operation reproduces the provider's full precision.
    """
    try:
        if payload['model'] != model:
            raise ValueError('model mismatch')
        answer = payload['answers']['verification']
        p = answer['probabilities']
        if answer['type']!='choice' or set(p)!=set('ABCD') or candidate not in p or any(not _prob(v) for v in p.values()):
            raise ValueError('invalid distribution')
        mass = sum(p.values())
        exact = math.isclose(mass,1.,abs_tol=1e-6)
        rounded = all(math.isclose(v*100,round(v*100),abs_tol=1e-8) for v in p.values())
        if not exact and (not rounded or abs(mass-1.)>.020000001):
            raise ValueError('mass error exceeds two-decimal rounding allowance')
        return p[candidate]/mass if normalize else p[candidate], not exact
    except (KeyError,TypeError,ValueError,AttributeError):
        return None,False


def run(root):
    root = Path(root)
    config = json.loads((root/'config.json').read_text())
    source = Path(config['source_root'])
    manifest,candidates = load_candidates(source/'frozen')
    records = json.loads((root/'analysis_records.json').read_text())
    cache = RequestCache(root/'cache')
    spec = next(s for s in load_specs(config) if s.mode=='choice')
    policy = json.loads((root/'policies'/'choice_only.json').read_text())
    outputs = {'status':'post_hoc_offline_sensitivity_only',
               'note':'Original live policies and strict parsing results remain unchanged. Both rounding conventions are reported, without selecting one using evaluation performance.',
               'variants':{}}
    for normalize in (False,True):
        name = 'accept_reported_marginal' if not normalize else 'renormalize_probability_mass'
        mapping = {}
        recovered = {'calibration':0,'evaluation':0}
        for c in candidates:
            if c.answer is None:
                mapping[c.item_id]=None
                continue
            call,_ = cache.get(spec.runner(),spec.request(c),allow_api=False)
            score, adjusted = rounded_choice(json.loads(call.response),c.answer,spec.model,normalize=normalize)
            mapping[c.item_id]=score
            recovered[c.partition]+=adjusted
        updated = []
        for row in records:
            p = mapping[row['item_id']]
            updated.append({**copy.deepcopy(row),'verifiers':{'verifier_1':{'p_correct':p,'ok':p is not None,
                                                                          'failure':None if p is not None else 'invalid_signal_or_candidate'}}})
        analysis = {'models':{'verifiers':[spec.model]},'sample_size':len(candidates),
                    'calibration_size':len(manifest['calibration_ids']),
                    'routing_scenarios':[{'name':'primary_95','correct_reward':config['correct_reward'],
                                         'incorrect_loss':config['incorrect_loss'],'verifier_costs':policy['planner']['costs']}]}
        report = build_report(updated,analysis,bootstrap_repeats=config['bootstrap_repeats'],seed=config['seed'])
        outputs['variants'][name] = {'recovered_signals':recovered,'report':report}
    atomic_json(root/'rounding_sensitivity.json',outputs)
    print(json.dumps({'output':str(root/'rounding_sensitivity.json'),'api_calls':0}))
    return outputs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/gpqa_jev_20261004'))
    run(parser.parse_args().root)
