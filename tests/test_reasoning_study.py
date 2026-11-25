from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_generator_study import prepared
from test_fresh_gpqa import fresh_data, fake_vertex
from vgx.common.api import execute_request, OfflineCacheMiss
from vgx.common.vertex import VertexBatchRunner
from vgx.gpqa import generator_study as direct, reasoning_study as study


@pytest.mark.parametrize('text,answer,failure',[
    ('Reasoning. <final>{"answer":"C","p_correct":0.8}</final>','C',None),
    ('Draft {"answer":"A","p_correct":0.9}\n<final>{"answer":"B","p_correct":0.6}</final>','B',None),
    ('{"answer":"A","p_correct":0.8}',None,'missing_or_multiple_final_blocks'),
    ('<final>{"answer":"A","p_correct":0.8}',None,'missing_or_multiple_final_blocks'),
    ('<final>{}</final><final>{}</final>',None,'missing_or_multiple_final_blocks'),
    ('<final>{"answer":"A","p_correct":0.8}</final> changed',None,'invalid_final_block'),
    ('<final>prose {"answer":"A","p_correct":0.8}</final>',None,'invalid_final_json'),
    ('<final>{"answer":"B","p_correct":"0.8"}</final>','B','invalid_confidence'),
])
def test_final_block_parser(text,answer,failure):
    result=study.parse_final(text)
    assert result.answer==answer and result.failure==failure


@pytest.fixture
def reasoning_prepared(prepared, fake_vertex, monkeypatch):
    cfg,old=prepared
    monkeypatch.setattr(direct,'check_billing',lambda project:{'project':project})
    direct.collect(old,'generators',allow_api=True)
    direct.collect(old,'verifiers',allow_api=True)
    direct.report(old)
    cfg=deepcopy(cfg);cfg['direct_results']=str(old);cfg['reuse_caches']=[str(old/'cache')]
    cfg['protocol']={'fold_seeds':[20261005],'bootstrap_repeats':20}
    out=old.parent/'reasoning'
    study.prepare(cfg,out);fake_vertex.clear()
    return cfg,out,old


def test_prompt_only_change_and_model_settings_frozen(reasoning_prepared):
    cfg,out,old=reasoning_prepared
    a=direct.load(old)[0];b,items,_=study.load(out)
    for left,right in zip(a['generator_arms'],b['arms']):
        assert left['identity']==right['identity']
        assert set(left['request_keys']).isdisjoint(right['keys'])
    bad=deepcopy(cfg);bad['generators'][0]['generation']['temperature']=0.5
    with pytest.raises(ValueError,match='unchanged generators'):
        study.prepare(bad,out.parent/'bad')


def test_no_answer_keys_resume_and_answer_only_verification(reasoning_prepared,monkeypatch):
    cfg,out,old=reasoning_prepared
    calls=[]
    def fake(self,requests,log):
        req=requests[0];calls.append(req)
        assert req.meta['role']=='generator' and req.meta['partition']=='calibration'
        text='Reasoning marker SECRET_RATIONALE. <final>{"answer":"B","p_correct":0.8}</final>'
        execute_request(self,req,log,lambda:(200,{'choices':[{'message':{'content':text}}],
             'usage':{'prompt_tokens':100,'completion_tokens':30,'total_tokens':130}}))
    monkeypatch.setattr(VertexBatchRunner,'run',fake)
    monkeypatch.setattr(study,'check_billing',lambda project:{'project':project})
    original=Path.open
    def guard(path,*args,**kwargs):
        assert path.name not in ('labels.jsonl','calibration_records.jsonl','evaluation_labels.jsonl')
        return original(path,*args,**kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(Path,'open',guard)
        with pytest.raises(OfflineCacheMiss):study.collect(out,'verifiers',allow_api=True)
        assert calls==[]
        assert study.collect(out,'generators',allow_api=True)['completed']==24
        assert study.collect(out,'generators')['reused']==24
        v=study.collect(out,'verifiers',allow_api=True)
        assert v['completed']==120 and v['reused']==120
        assert len(calls)==24  # all answer-only verifier calls imported from direct condition
    bundle=json.loads((out/'frozen_candidates.json').read_text())
    for name,cs in bundle['groups'].items():
        for c in cs:assert 'SECRET_RATIONALE' not in (c['verifier_prompt'] or '')
    from vgx.gpqa.reasoning_analysis import analyze
    result=analyze(out)
    assert result['paired_conditions']['qwen']['accuracy']['difference']==0
    assert result['generators']['qwen:reasoned']['valid_confidence']==12


def test_paired_scores_use_each_conditions_own_outcomes():
    from vgx.gpqa.reasoning_analysis import paired_summary
    old=[{'item_id':'x','outcome':0,'generator_answer':'A','generator_p_correct':.9}]
    new=[{'item_id':'x','outcome':1,'generator_answer':'B','generator_p_correct':.9}]
    result=paired_summary(old,new,repeats=10)
    assert result['accuracy']['difference']==1
    assert result['wrong_to_correct']==1
    assert result['raw_brier_on_matched_confidence']['difference']==pytest.approx(-.8)
    assert result['matched_direct_brier']==pytest.approx(.81)
    assert result['matched_reasoned_brier']==pytest.approx(.01)


def test_paired_confidence_failure_does_not_remove_accuracy():
    from vgx.gpqa.reasoning_analysis import paired_summary
    old=[{'item_id':'x','outcome':0,'generator_answer':'A','generator_p_correct':None}]
    new=[{'item_id':'x','outcome':1,'generator_answer':'B','generator_p_correct':.9}]
    result=paired_summary(old,new,repeats=10)
    assert result['accuracy']['n']==1
    assert result['raw_brier_on_matched_confidence']['n']==0


def test_paired_incomplete_is_separate_from_wrong_answer():
    from vgx.gpqa.reasoning_analysis import paired_summary
    old=[{'item_id':'x','outcome':1,'generator_answer':'A','generator_p_correct':.9}]
    new=[{'item_id':'x','outcome':0,'generator_answer':None,'generator_p_correct':None}]
    result=paired_summary(old,new,repeats=10)
    assert result['accuracy']['difference']==-1
    assert result['correct_to_wrong']==0 and result['correct_to_incomplete']==1
