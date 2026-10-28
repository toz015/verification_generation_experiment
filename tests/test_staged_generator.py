from pathlib import Path

import pytest

from test_completion_study import completion_prepared
from test_reasoning_study import reasoning_prepared
from test_generator_study import prepared
from test_fresh_gpqa import fresh_data, fake_vertex
from vgx.common.api import execute_request, OfflineCacheMiss
from vgx.common.vertex import VertexBatchRunner
from vgx.gpqa import completion_study as controls, staged_generator as study


def test_stages_reuse_drafts_and_gate_expansion_without_labels(completion_prepared,monkeypatch):
    cfg,control_root=completion_prepared;calls=[]
    def fake(self,requests,log):
        req=requests[0];calls.append(req)
        text='{"answer":"B","p_correct":0.7}' if req.meta['role']=='generator_final' else 'Unfinished scratch work'
        execute_request(self,req,log,lambda:(200,{'choices':[{'message':{'content':text},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':100,'completion_tokens':30,'total_tokens':130}}))
    monkeypatch.setattr(VertexBatchRunner,'run',fake)
    monkeypatch.setattr(controls,'check_billing',lambda p:{'project':p})
    monkeypatch.setattr(study,'check_billing',lambda p:{'project':p})
    controls.prepare(cfg,control_root);controls.collect(control_root,allow_api=True)
    pilot_cfg={k:cfg[k] for k in ('project','pricing','previous_results')}
    pilot_cfg.update(scope='pilot',stage_budget_usd=2.,reuse_caches=[str(control_root/'cache')])
    root=control_root.parent/'staged';study.prepare(pilot_cfg,root)
    original=Path.open
    def guard(path,*args,**kwargs):
        assert path.name not in ('labels.jsonl','calibration_records.jsonl','evaluation_labels.jsonl')
        return original(path,*args,**kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(Path,'open',guard)
        before=len(calls);r=study.collect(root,allow_api=True)
        assert len(calls)-before==12
        assert all(x['draft_reused'] for x in r['results'])
        assert all(r.meta['role']=='generator_final' for r in calls[before:])
        assert study.gate(root)['passed']
        assert study.collect(root)['completed']==12
        assert len(calls)-before==12
        expanded=dict(pilot_cfg,scope='all_questions',pilot_root=str(root),reuse_caches=[str(root/'cache'),str(control_root/'cache')])
        target=root.parent/'staged_full';p=study.prepare(expanded,target)
        assert p['partition_counts']['evaluation']>0
        study.collect(target,allow_api=True)
        assert study.gate(target)['passed']
    from vgx.gpqa.expansion_analysis import analyze
    assert analyze(target,staged=True)['summaries']['qwen:all_calibration']['n']==12


def test_missing_draft_does_not_trigger_paid_regeneration(completion_prepared,monkeypatch):
    cfg,root=completion_prepared
    config={k:cfg[k] for k in ('project','pricing','previous_results')}
    config.update(scope='pilot',stage_budget_usd=2.,reuse_caches=[])
    study.prepare(config,root)
    monkeypatch.setattr(study,'check_billing',lambda p:{'project':p})
    def reject(*args,**kwargs):raise AssertionError('unexpected API call')
    monkeypatch.setattr(VertexBatchRunner,'run',reject)
    with pytest.raises(OfflineCacheMiss):study.collect(root,allow_api=True)


@pytest.mark.parametrize('text,valid',[
    ('{"answer":"A","p_correct":0.8}',True),
    ('Analysis {"answer":"A","p_correct":0.8}',False),
    ('{"answer":"A","p_correct":0.8} extra',False),
    ('{"answer":"A","p_correct":"bad"}',False),
])
def test_strict_final_json(text,valid):
    assert study.parse_final_json(text).ok==valid
