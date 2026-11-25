"""Descriptive development-only dependence checks; no fitted prior or API calls."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa.generator_calibration import source, index_cache, audit_coverage
from vgx.gpqa.sequential import seal
from vgx.gpqa.signal_study import runner_for, request_for, parse_signal


def association(a,b):
    pairs=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
    n=len(pairs)
    if n<3:return {'n':n,'spearman':None,'reason':'fewer_than_three'}
    x,y=np.array(pairs).T
    if len(set(x))<2 or len(set(y))<2:return {'n':n,'spearman':None,'reason':'constant_signal'}
    return {'n':n,'spearman':float(spearmanr(x,y).statistic)}


def analyze(config_path, output):
    cfg=json.loads(Path(config_path).read_text());root=Path(output)
    plan,groups,labels,pilot,label_hash,bundle=source(Path(cfg['generator_results']))
    screen=json.loads(Path(cfg['screen_config']).read_text());index=index_cache(cfg['cache_roots'])
    audit=audit_coverage(groups,labels,screen['models'],screen['project'],index)
    complete=[g for g in groups if not any(v['missing'] for v in audit['coverage'][g].values())]
    spec=seal({'config_sha256':file_digest(Path(config_path)), 'implementation_sha256':file_digest(Path(__file__)),
        'source_bundle_id':bundle,'calibration_label_sha256':label_hash,
        'candidate_hashes':{g:digest([asdict(c) for c in groups[g]]) for g in complete},
        'cache_hashes':audit['used_cache_record_hashes'],
        'prior_cutpoint':0.95,'bin_edges':[0.,1/3,2/3,1.],
        'evaluation_labels_read':False,'api_calls':0,
        'purpose':'Descriptive conditional dependence diagnostics, not proof of independence or a fitted policy'},'diagnostic_id')
    dest=root/'protocol.json'
    if dest.exists() and json.loads(dest.read_text())!=spec:raise ValueError('new inputs require new output')
    atomic_json(dest,spec);result={'diagnostic_id':spec['diagnostic_id'],'generators':{}}
    for g in groups:
        if g not in complete:
            result['generators'][g]={'status':'blocked_incomplete_collection'};continue
        rows=[]
        for c in groups[g]:
            if c.answer is None or c.p_correct is None:continue
            signals={}
            for model in screen['models']:
                runner=runner_for(model,screen['project'])
                for signal in screen['signals']:
                    key=runner.cache_key(request_for(c,model,signal))
                    signals[model['id']+':'+signal]=parse_signal(index[key]['response'],signal)[0]
            rows.append({'id':c.item_id,'prior':c.p_correct,'y':int('ABCD'.index(c.answer)==labels[c.item_id]),'signals':signals})
        cohorts={}
        for cohort in ('all_calibration','remaining_calibration'):
            subset=rows if cohort=='all_calibration' else [r for r in rows if r['id'] not in pilot]
            strata={}
            for y in (0,1):
                rr=[r for r in subset if r['y']==y];arms={};pairs={}
                tags=[m['id']+':'+s for m in screen['models'] for s in screen['signals']]
                for tag in tags:
                    parts={}
                    for label,condition in [('raw_lt_095',lambda p:p<.95),('raw_ge_095',lambda p:p>=.95)]:
                        eligible=[r for r in rr if condition(r['prior'])]
                        scores=[r['signals'][tag] for r in eligible if r['signals'][tag] is not None]
                        bins=2 if tag.endswith(':binary') else 3
                        counts=np.bincount([min(int(s*bins),bins-1) for s in scores],minlength=bins)
                        parts[label]={'n':len(eligible),'valid_signals':len(scores),'counts':counts.tolist(),
                            'frequencies':(counts/len(scores)).tolist() if scores else None}
                    arms[tag]={'prior_signal_association':association([r['prior'] for r in rr],[r['signals'][tag] for r in rr]),
                        'likelihood_by_prior_stratum':parts}
                for signal in screen['signals']:
                    ts=[m['id']+':'+signal for m in screen['models']]
                    pairs[signal]={}
                    for i,a in enumerate(ts):
                        for b in ts[i+1:]:
                            pairs[signal][a+'|'+b]=association([r['signals'][a] for r in rr],[r['signals'][b] for r in rr])
                strata[str(y)]={'n':len(rr),'arms':arms,'between_verifier_associations':pairs}
            cohorts[cohort]=strata
        result['generators'][g]={'status':'complete','cohorts':cohorts}
    end=index_cache(cfg['cache_roots'])
    if any(digest(end[k])!=v for k,v in spec['cache_hashes'].items()):raise ValueError('responses changed')
    result['limitations']=['Nonzero association can challenge independence; zero correlation cannot establish it.',
        'Small error groups and constant signals limit these diagnostics; no significance-based model selection.',
        'Uses development labels for diagnostic stratification only; initial confidence is unchanged.',
        'Likelihood strata are descriptive, not substituted into the frozen likelihood fits.']
    atomic_json(root/'report.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=analyze(a.config,a.output)
    print(json.dumps({'diagnostic_id':r['diagnostic_id'],'status':{g:v['status'] for g,v in r['generators'].items()}}))
