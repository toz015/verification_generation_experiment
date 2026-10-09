"""Matched-cohort and usage audit supplement for generator_study; offline only."""
from itertools import combinations
import argparse
import json
from pathlib import Path

import numpy as np

from vgx.common.api import RequestCache
from vgx.common.billing import summarize_vertex_usage
from vgx.common.storage import atomic_json, file_digest, read_jsonl
from vgx.gpqa.generator_study import load
from vgx.gpqa.score import binary_forecast_metrics, correctness_outcome
from vgx.gpqa.signal_analysis import likelihood_stability
from vgx.gpqa.signal_study import parse_signal, request_for, runner_for


def analyze(root):
    root=Path(root);plan,items,_=load(root);cfg=plan['config']
    from vgx.gpqa.sequential import unseal
    from vgx.gpqa.artifacts import FrozenCandidate
    bundle=json.loads((root/'frozen_candidates.json').read_text());unseal(bundle,'bundle_id')
    if bundle['study_id']!=plan['study_id']:raise ValueError('different study candidates')
    records=json.loads((root/'analysis_records.json').read_text())
    # Label provenance was checked by report; check again before this analysis.
    source=Path(cfg['baseline_bundle'])
    manifest=json.loads((source/'bundle_manifest.json').read_text())
    if file_digest(source/'calibration_records.jsonl')!=manifest['files']['calibration_records.jsonl']:
        raise ValueError('calibration labels changed')
    labels={r['item_id']:r['correct_index'] for r in read_jsonl(source/'calibration_records.jsonl')}
    common=set.intersection(*[{r['item_id'] for r in rows} for rows in records.values()])
    ids=[i.item_id for i in items if i.item_id in common]
    matched={g:[{r['item_id']:r for r in rows}[i] for i in ids] for g,rows in records.items()}
    metrics={g:binary_forecast_metrics([r['outcome'] for r in rows],[r['generator_p_correct'] for r in rows]).to_dict()
             for g,rows in matched.items()}
    rng=np.random.default_rng(cfg['seed']);indices=rng.integers(0,len(ids),(2000,len(ids))) if ids else None
    pairs={}
    for a,b in combinations(matched,2):
        # Different generators have different correctness outcomes on the same question.
        # Resample question IDs jointly; do not feed both models one generator's labels.
        ra,rb=matched[a],matched[b]
        delta=np.array([(y['generator_p_correct']-y['outcome'])**2-(x['generator_p_correct']-x['outcome'])**2
                        for x,y in zip(ra,rb)])
        pairs[f'{b}_minus_{a}']={'n':len(ids),'raw_brier_difference':float(delta.mean()) if ids else None,
            'paired_question_bootstrap95':np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist() if ids else None}
    cache=RequestCache(root/'cache');signals={};answers={}
    for group,raw in bundle['groups'].items():
        candidates=[FrozenCandidate(**r) for r in raw]
        answers[group]={c.item_id:c.answer for c in candidates}
        rows=[]
        for c in candidates:
            if c.answer is None:continue
            row={'item_id':c.item_id,'outcome':correctness_outcome(c.answer,labels[c.item_id]),'verifiers':{}}
            for model in cfg['verifiers']:
                call,_=cache.get(runner_for(model,cfg['project']),request_for(c,model,'probability',plan['study_id']))
                p,failure=parse_signal(call.response,'probability')
                if failure is None:row['verifiers'][model['id']]={'p_correct':p}
            rows.append(row)
        signals[group]={}
        for m in cfg['verifiers']:
            tag=m['id'];usable=[r for r in rows if tag in r['verifiers']]
            signals[group][tag]={'valid':len(usable),'invalid':len(rows)-len(usable),
                'wrong_n':sum(r['outcome']==0 for r in usable),
                'wrong_rejected_at_0_5':sum(r['outcome']==0 and r['verifiers'][tag]['p_correct']<.5 for r in usable),
                'correct_rejected_at_0_5':sum(r['outcome']==1 and r['verifiers'][tag]['p_correct']<.5 for r in usable),
                'likelihood':likelihood_stability(rows,tag,'probability',seed=cfg['seed'])}
    all_calls=[r for log in cache.logs().values() for r in log.records() if r.get('error') is None]
    usage=summarize_vertex_usage(cache.logs(),cfg['pricing'])
    audit={'study_id':plan['study_id'],'collection_calls_in_cache':len(all_calls),
           'unique_executions':len({r['meta']['execution_id'] for r in all_calls}),
           'all_resource_and_quota_projects_match':all(r['meta'].get('resource_project')==cfg['project']
                 and r['meta'].get('quota_project')==cfg['project'] for r in all_calls),
           'evaluation_calls':sum(r['meta'].get('partition')=='evaluation' for r in all_calls),
           'generator_calls_by_model':{model:sum(r['model']==model and r['meta'].get('role')=='generator' for r in all_calls)
                                       for model in sorted({r['model'] for r in all_calls if r['meta'].get('role')=='generator'})},
           'provider_model_labels':{model:sorted({r['meta'].get('provider_response_model') or 'unreported'
                                                 for r in all_calls if r['model']==model}) for model in sorted({r['model'] for r in all_calls})},
           'cache_usage_including_historical_imports':usage,
           'note':'Cache usage includes imported historical verifier executions; do not add it to historical experiment totals.'}
    value={'study_id':plan['study_id'],'implementation_sha256':file_digest(Path(__file__)),
           'common_confidence_n':len(ids),'common_confidence_raw_metrics':metrics,
           'paired_raw_forecasts':pairs,'verifier_diagnostics_all_valid_answers':signals,
           'answer_agreement':{f'{a}:{b}':sum(answers[a][i.item_id]==answers[b][i.item_id] for i in items)
                               for a,b in combinations(answers,2)},
           'note':'Exploratory screening; 0.5 rejection is descriptive, not a release threshold. Different generators require their own outcomes/likelihoods.'}
    atomic_json(root/'matched_comparison.json',value);atomic_json(root/'audit.json',audit)
    return {k:v for k,v in value.items() if k not in ('verifier_diagnostics_all_valid_answers','implementation_sha256')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(analyze(parser.parse_args().root)))
