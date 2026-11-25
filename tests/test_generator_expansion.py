from pathlib import Path

import pytest

from test_completion_study import completion_prepared
from test_reasoning_study import reasoning_prepared
from test_generator_study import prepared
from test_fresh_gpqa import fresh_data, fake_vertex
from vgx.common.api import execute_request
from vgx.common.vertex import VertexBatchRunner
from vgx.gpqa import completion_study as pilot, generator_expansion as study


def test_full_membership_frozen_prompts_resume_without_labels(completion_prepared, monkeypatch):
    cfg, root = completion_prepared; calls = []
    def fake(self, requests, log):
        req = requests[0]; calls.append((self.model, req))
        execute_request(self, req, log, lambda: (200, {'choices': [{'message': {'content':
            'Analysis. <final>{"answer":"B","p_correct":0.7}</final>'}, 'finish_reason':'stop'}],
            'usage': {'prompt_tokens':100, 'completion_tokens':30 if self.max_tokens == 4096 else 60,
                      'total_tokens':130 if self.max_tokens == 4096 else 160}}))
    monkeypatch.setattr(VertexBatchRunner, 'run', fake)
    monkeypatch.setattr(pilot, 'check_billing', lambda p: {'project':p})
    monkeypatch.setattr(study, 'check_billing', lambda p: {'project':p})
    pilot.prepare(cfg, root); pilot.collect(root, allow_api=True)
    assert pilot.gate(root)['selected_arm'] == 'short_4096'
    config = {k:cfg[k] for k in ('project','pricing')}
    config.update(completion_root=str(root), stage_budget_usd=5., reuse_caches=[str(root/'cache')])
    target = root.parent/'expanded'
    original = Path.open
    def guard(path, *args, **kwargs):
        assert path.name not in ('labels.jsonl','calibration_records.jsonl','evaluation_labels.jsonl')
        return original(path,*args,**kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(Path,'open',guard)
        plan = study.prepare(config,target); _, items = study.load(target)
        assert plan['partition_counts']['calibration'] == 12
        assert plan['partition_counts']['evaluation'] > 0
        item = next(i for i in items if i.partition == 'calibration')
        assert study.request_for(item,'qwen','short_4096').prompt == pilot.request_for(item,'short_4096').prompt
        assert study.request_for(item,'llama','short_4096').prompt == study.previous.generator_request(item).prompt
        before = len(calls); result = study.collect(target,allow_api=True)
        assert result['completed'] == 2*len(items)
        assert result['reused'] == 12  # Qwen pilot reused; Llama not imported in this test
        assert len(calls)-before == result['completed']-12
        before = len(calls)
        assert study.collect(target)['reused'] == 2*len(items)
        assert len(calls) == before
    def analysis_guard(path, *args, **kwargs):
        assert path.name not in ('labels.jsonl', 'evaluation_labels.jsonl')
        return original(path,*args,**kwargs)
    from vgx.gpqa.expansion_analysis import analyze
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, 'open', analysis_guard)
        summary = analyze(target)
        assert summary['evaluation_labels_read'] is False
        assert summary['summaries']['qwen:all_calibration']['n'] == 12


def test_expansion_blocked_if_no_arm_completed(completion_prepared, monkeypatch):
    cfg, root = completion_prepared
    def fake(self, requests, log):
        execute_request(self, requests[0], log, lambda: (200, {'choices': [{'message': {'content':'unfinished'}, 'finish_reason':'length'}],
            'usage': {'prompt_tokens':100,'completion_tokens':30,'total_tokens':130}}))
    monkeypatch.setattr(VertexBatchRunner,'run',fake)
    monkeypatch.setattr(pilot,'check_billing',lambda p:{'project':p})
    pilot.prepare(cfg,root); pilot.collect(root,allow_api=True)
    config = {k:cfg[k] for k in ('project','pricing')}
    config.update(completion_root=str(root),stage_budget_usd=5.,reuse_caches=[])
    with pytest.raises(ValueError,match='gate has not passed'):
        study.prepare(config,root.parent/'expanded')
