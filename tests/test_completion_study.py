from copy import deepcopy
from pathlib import Path

import pytest

from test_reasoning_study import reasoning_prepared
from test_generator_study import prepared
from test_fresh_gpqa import fresh_data, fake_vertex
from vgx.common.api import execute_request
from vgx.common.vertex import VertexBatchRunner
from vgx.gpqa import reasoning_study as old, completion_study as study


@pytest.fixture
def completion_prepared(reasoning_prepared, monkeypatch):
    cfg, root, _ = reasoning_prepared
    monkeypatch.setattr(old, 'check_billing', lambda project: {'project': project})
    old.collect(root, 'generators', allow_api=True)
    config = {k: cfg[k] for k in ('project', 'dataset_root', 'pricing')}
    config.update(previous_results=str(root), model=cfg['generators'][0], scope='pilot',
                  arms=['short_4096', 'original_8192'], workers=2, stage_budget_usd=2., reuse_caches=[])
    return config, root.parent/'completion'


def test_only_intended_factor_changes(completion_prepared):
    cfg, root = completion_prepared
    plan = study.prepare(cfg, root); _, items = study.load(root)
    long = study.request_for(items[0], 'original_8192')
    assert long.prompt == old.generator_request(items[0]).prompt
    assert study.model_for(cfg, 'short_4096') == cfg['model']
    high = study.model_for(cfg, 'original_8192')
    high['generation']['max_tokens'] = 4096
    assert high == cfg['model']
    assert set(plan['arms'][0]['keys']).isdisjoint(plan['arms'][1]['keys'])
    changed = deepcopy(cfg); changed['model']['generation']['temperature'] = .5
    with pytest.raises(ValueError, match='settings changed'): study.prepare(changed, root.parent/'bad')


def test_no_labels_cache_resume_and_gate(completion_prepared, monkeypatch):
    cfg, root = completion_prepared; calls = []
    def fake(self, requests, log):
        req = requests[0]; calls.append(req)
        execute_request(self, req, log, lambda: (200, {'choices': [{'message': {'content':
            'Brief analysis. <final>{"answer":"B","p_correct":0.7}</final>'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 30, 'total_tokens': 130}}))
    monkeypatch.setattr(VertexBatchRunner, 'run', fake)
    monkeypatch.setattr(study, 'check_billing', lambda project: {'project': project})
    original = Path.open
    def guard(path, *args, **kwargs):
        assert path.name not in ('labels.jsonl', 'calibration_records.jsonl', 'evaluation_labels.jsonl')
        return original(path, *args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, 'open', guard)
        study.prepare(cfg, root)
        assert study.collect(root, allow_api=True)['completed'] == 24
        assert study.collect(root)['reused'] == 24
        selected = study.gate(root)
        assert selected['selected_arm'] == 'original_8192'  # deterministic tie break
        assert len(calls) == 24


def test_selection_rejects_incomplete_and_uses_no_accuracy():
    complete = {'n':50, 'valid_answer_and_confidence':50, 'missing_usage':0, 'mean_output_tokens':200}
    incomplete = {**complete, 'valid_answer_and_confidence':49, 'mean_output_tokens':100}
    assert study.choose_arm({'a':incomplete, 'b':complete}) == 'b'
    assert study.choose_arm({'a':incomplete}) is None
    assert study.choose_arm({'a':complete, 'b':{**complete, 'mean_output_tokens':150}}) == 'b'


def test_evaluation_request_rejected():
    from vgx.gpqa.fresh import PublicItem
    item = PublicItem('x', 'evaluation', 'Physics', 'q', ('a','b','c','d'))
    with pytest.raises(ValueError, match='calibration only'): study.request_for(item, 'short_4096')
