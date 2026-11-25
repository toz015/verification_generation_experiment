import builtins
from dataclasses import asdict, replace
import io
import json
from pathlib import Path

import pytest

from vgx.common.api import execute_request
from vgx.common.storage import digest
from vgx.common.vertex import VertexBatchRunner
from vgx.gpqa import diamond_signals as ds
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.sequential import seal, unseal

ROOT = Path(__file__).resolve().parents[1]


def setup(frozen_run, monkeypatch, tmp_path):
    """Two Vertex chat models from the development screen, synthetic frozen candidates."""
    screen = json.loads((ROOT/'configs/gpqa_signal_study_4096.json').read_text())
    screen['project'] = 'synthetic-project'
    screen['models'] = [m for m in screen['models'] if m['id'] in ('gemini_flash', 'gemini_lite')]
    (tmp_path/'screen.json').write_text(json.dumps(screen))
    _, candidates = load_candidates(frozen_run.bundle)
    gemini = sorted((c for c in candidates if c.partition == 'evaluation'), key=lambda c: c.item_id)
    # Qwen: one invalid output, one different answer, the rest identical prompts to Gemini.
    qwen = [replace(gemini[0], answer=None, p_correct=None, verifier_prompt=None),
            replace(gemini[1], answer='ABCD'[('ABCD'.index(gemini[1].answer)+1) % 4]), *gemini[2:]]
    groups = {'gemini': gemini, 'qwen': qwen}
    provenance = {'dataset_id': 'd', 'gemini_bundle_id': 'g', 'generator_study_id': 's', 'qwen_bundle_id': 'q'}
    monkeypatch.setattr(ds, 'sources', lambda cfg: (provenance, groups))
    cfg = json.loads((ROOT/'configs/gpqa_diamond_signals.json').read_text())
    cfg.update(project='synthetic-project', screen_config=str(tmp_path/'screen.json'),
               verifiers=['gemini_flash', 'gemini_lite'], expected_evaluation_questions=len(gemini),
               reuse_caches=[str(tmp_path/'none')])
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ds, 'check_billing', lambda project: {'project': project})
    return path, groups


def guard_labels(monkeypatch):
    for target in (builtins, io):
        original = target.open

        def guarded(file, *args, _original=original, **kwargs):
            name = str(file)
            if 'label' in name or 'calibration_records' in name:
                raise AssertionError('labels opened: '+name)
            return _original(file, *args, **kwargs)
        monkeypatch.setattr(target, 'open', guarded)


def fake_vertex(monkeypatch, calls, text=lambda req: '{"p_correct": 0.8}'):
    def fake(self, requests, log):
        req = requests[0]
        calls.append(req)
        assert req.meta['role'] == 'verifier' and req.meta['partition'] == 'evaluation'
        execute_request(self, req, log, lambda: (200, {'choices': [{'message': {'content': text(req)}}],
                                                      'usage': {'prompt_tokens': 100, 'completion_tokens': 20,
                                                                'total_tokens': 120}}))
    monkeypatch.setattr(VertexBatchRunner, 'run', fake)


def test_dry_run_sends_nothing_and_plans_unique_requests(frozen_run, monkeypatch, tmp_path):
    config, groups = setup(frozen_run, monkeypatch, tmp_path)
    guard_labels(monkeypatch)
    calls = []
    fake_vertex(monkeypatch, calls)
    status = ds.run(config, tmp_path/'out')
    n = len(groups['gemini'])
    plan = json.loads((tmp_path/'out/plan.json').read_text())
    unseal(plan, 'collection_id')
    assert calls == [] and status['stage'] == 'dry_run'
    assert len(plan['jobs']) == 2*(2*n - 1)  # Qwen's invalid output gets no verifier request
    assert plan['unique_requests'] == 2*(n + 1)  # identical prompts are one provider request
    assert plan['invalid_generator_outputs']['qwen'] == [groups['qwen'][0].item_id]
    assert status['not_sent_offline'] == len(plan['jobs']) and status['labels_read'] is False


def test_collection_is_batched_reuses_identical_requests_and_keeps_failures(frozen_run, monkeypatch, tmp_path):
    config, groups = setup(frozen_run, monkeypatch, tmp_path)
    guard_labels(monkeypatch)
    calls = []
    bad = groups['gemini'][2].item_id
    fake_vertex(monkeypatch, calls, lambda req: 'not json' if req.meta['item_id'] == bad else '{"p_correct": 0.8}')
    first = ds.run(config, tmp_path/'out', allow_api=True, limit=1)
    assert first['stage'] == 'batch_complete' and not (tmp_path/'out/signals.json').exists()
    sent = len(calls)
    final = ds.run(config, tmp_path/'out', allow_api=True)
    plan = json.loads((tmp_path/'out/plan.json').read_text())
    assert final['stage'] == 'complete' and len(calls) == plan['unique_requests'] > sent
    again = ds.run(config, tmp_path/'out', allow_api=True)
    assert len(calls) == plan['unique_requests'] and again['new_calls_this_invocation'] == 0
    signals = json.loads((tmp_path/'out/signals.json').read_text())
    unseal(signals, 'signals_id')
    rows = {(g, r['item_id']): r for g, rs in signals['rows'].items() for r in rs}
    assert rows[('qwen', groups['qwen'][0].item_id)]['signals'] == {}
    flash = rows[('gemini', bad)]['signals']['gemini_flash:probability']
    assert flash['score'] is None and flash['failure'] == 'invalid_json' and flash['estimated_usd'] > 0
    assert rows[('gemini', groups['gemini'][3].item_id)]['signals']['gemini_lite:probability']['score'] == 0.8
    assert 'outcome' not in json.dumps(signals) and 'question' not in json.dumps(signals)
    assert final['invalid_by_verifier'] == {'gemini_flash:invalid_json': 2, 'gemini_lite:invalid_json': 2}


def test_prompt_and_cache_key_match_the_development_request(frozen_run, monkeypatch, tmp_path):
    from vgx.gpqa.signal_study import request_for as development_request, runner_for
    setup(frozen_run, monkeypatch, tmp_path)
    _, candidates = load_candidates(frozen_run.bundle)
    c = next(c for c in candidates if c.partition == 'evaluation')
    model = json.loads((tmp_path/'screen.json').read_text())['models'][0]
    runner = runner_for(model, 'synthetic-project')
    ours, theirs = ds.request_for(c, model), development_request(c, model, 'probability')
    assert (ours.prompt, ours.system) == (theirs.prompt, theirs.system)
    assert runner.cache_key(ours) == runner.cache_key(theirs) and ours.meta['partition'] == 'evaluation'


def test_changed_candidates_or_config_block_collection(frozen_run, monkeypatch, tmp_path):
    config, groups = setup(frozen_run, monkeypatch, tmp_path)
    ds.run(config, tmp_path/'out')
    groups['gemini'][0] = replace(groups['gemini'][0], p_correct=0.5)
    with pytest.raises(ValueError, match='frozen candidates changed'):
        ds.run(config, tmp_path/'out')
    cfg = json.loads(config.read_text())
    cfg['labels_read'] = True
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        ds.run(config, tmp_path/'out2')


def test_sources_checks_membership_content_and_provenance(frozen_run, monkeypatch, tmp_path):
    from vgx.gpqa.fresh import PublicItem
    manifest, candidates = load_candidates(frozen_run.bundle)
    public = [PublicItem(c.item_id, c.partition, c.subject, c.question, tuple(c.choices)) for c in candidates]
    monkeypatch.setattr(ds, 'inputs', lambda root: ({'dataset_id': 'D'}, public))
    monkeypatch.setattr(ds, 'load_candidates', lambda root: ({'dataset_id': 'D', 'bundle_id': 'B'}, candidates))
    root = tmp_path/'staged'
    root.mkdir()
    plan = seal({'dataset_id': 'D', 'baseline_bundle_id': 'B'}, 'study_id')
    (root/'plan.json').write_text(json.dumps(plan))

    def write(qwen):
        (root/'frozen_candidates.json').write_text(json.dumps(seal(
            {'study_id': plan['study_id'], 'groups': {'qwen': [asdict(c) for c in qwen]}}, 'bundle_id')))
    cfg = {'dataset_root': 'x', 'gemini_bundle': 'y', 'generator_results': str(root)}
    write(candidates)
    provenance, groups = ds.sources(cfg)
    evaluation = {c.item_id for c in candidates if c.partition == 'evaluation'}
    assert {c.item_id for c in groups['qwen']} == {c.item_id for c in groups['gemini']} == evaluation
    assert provenance['gemini_bundle_id'] == 'B'
    write([c for c in candidates if c.item_id != sorted(evaluation)[0]])
    with pytest.raises(ValueError, match='membership'):
        ds.sources(cfg)
    write([replace(c, question=c.question+' edited') if c.partition == 'evaluation' else c for c in candidates])
    with pytest.raises(ValueError, match='input differs'):
        ds.sources(cfg)
    write(candidates)
    monkeypatch.setattr(ds, 'load_candidates', lambda root: ({'dataset_id': 'D', 'bundle_id': 'other'}, candidates))
    with pytest.raises(ValueError, match='Gemini bundle'):
        ds.sources(cfg)
