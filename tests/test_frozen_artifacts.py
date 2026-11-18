from dataclasses import asdict, replace
import json

import pytest

from vgx.common.api import OfflineCacheMiss, RequestCache
from vgx.common.storage import file_digest
from vgx.gpqa.artifacts import load_candidates, recover, validate_bundle, validate_original
from vgx.gpqa.workflow import collect_frozen, prepare_comparison, prepare_policy


def test_recovery_is_idempotent_and_keeps_candidates_and_prompts(frozen_run, monkeypatch):
    run = frozen_run
    before = file_digest(run.bundle/'candidates.jsonl')
    recover(run.source, run.sample, run.run_dir, run.bundle, run.cache, run.run_id)
    assert file_digest(run.bundle/'candidates.jsonl') == before
    manifest, candidates = load_candidates(run.bundle)
    assert len(candidates) == 10
    assert all('correct_index' not in asdict(c) and 'verifiers' not in asdict(c) for c in candidates)
    assert {c.item_id:c.answer for c in candidates} == {r['item_id']:r['generator_answer'] for r in run.records}
    # Imported exact provider requests are usable without authentication or inference.
    monkeypatch.setattr('vgx.common.vertex.VertexBatchRunner.run', lambda *a, **k: pytest.fail('must not collect'))
    result = collect_frozen(run.bundle, run.specs, run.cache, run.root/'collected.json')
    assert all(row['used_cached_response'] for row in result['observations'])
    assert manifest == validate_bundle(run.bundle)


def test_missing_artifacts_never_generate(original_run, monkeypatch):
    run = original_run
    (run.run_dir/'generator.jsonl').unlink()
    monkeypatch.setattr('vgx.common.vertex.VertexBatchRunner.run', lambda *a, **k: pytest.fail('generator called'))
    with pytest.raises(FileNotFoundError, match='generator.jsonl'):
        recover(run.source, run.sample, run.run_dir, run.root/'frozen', RequestCache(run.root/'cache'), run.run_id)
    assert not (run.root/'frozen').exists()


@pytest.mark.parametrize('target', ['source', 'candidate', 'prompt', 'settings', 'split'])
def test_recovery_rejects_changed_original(original_run, target):
    run = original_run
    if target == 'source':
        run.source.write_text(run.source.read_text()+'\n')
    elif target == 'candidate':
        path = run.run_dir/'pilot_records.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows[0]['generator_answer'] = 'D' if rows[0]['generator_answer'] != 'D' else 'A'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif target in ('prompt', 'settings'):
        path = run.run_dir/'generator.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if target == 'prompt':
            rows[0]['prompt'] += ' changed'
        else:
            rows[0]['params']['max_tokens'] = 99
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    else:
        payload = json.loads(run.sample.read_text())
        payload['evaluation_ids'] = payload['calibration_ids']
        run.sample.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_original(run.source, run.sample, run.run_dir, run.run_id)


def test_frozen_bundle_tampering_rejected(frozen_run):
    run = frozen_run
    path = run.bundle/'candidates.jsonl'
    path.write_text(path.read_text().replace('"p_correct": 0.6', '"p_correct": 0.9', 1))
    with pytest.raises(ValueError, match='checksum'):
        load_candidates(run.bundle)


def test_verifier_addition_and_repricing_do_not_repeat_generator(frozen_run, monkeypatch):
    run = frozen_run
    generator_calls_before = sum(r.get('meta', {}).get('role') == 'generator'
                                 for log in run.cache.logs().values() for r in log.records())
    monkeypatch.setattr('vgx.common.vertex.VertexBatchRunner.run', lambda *a, **k: pytest.fail('no provider call expected'))
    extra = replace(run.specs[0], id='same_request_another_label')
    result = collect_frozen(run.bundle, (*run.specs, extra), run.cache, run.root/'expanded.json')
    assert all(row['used_cached_response'] for row in result['observations'])
    from vgx.common.billing import summarize_vertex_usage
    a = summarize_vertex_usage(run.cache.logs(), {'usd_per_million_tokens': {}})
    b = summarize_vertex_usage(run.cache.logs(), {'checked_on': 'tomorrow', 'usd_per_million_tokens': {}})
    assert a['successful_calls'] == b['successful_calls']
    assert generator_calls_before == sum(r.get('meta', {}).get('role') == 'generator'
                                        for log in run.cache.logs().values() for r in log.records())


def test_changed_prompt_or_generation_misses_cache(frozen_run):
    run = frozen_run
    _, candidates = load_candidates(run.bundle)
    c = candidates[0]
    spec = run.specs[0]
    with pytest.raises(OfflineCacheMiss):
        run.cache.get(spec.runner(), spec.request(replace(c, verifier_prompt=c.verifier_prompt+' changed')))
    changed = replace(spec, generation={**spec.generation, 'max_tokens': 99})
    with pytest.raises(OfflineCacheMiss):
        run.cache.get(changed.runner(), changed.request(c))


def test_fit_never_reads_evaluation_labels_or_outputs(frozen_run, monkeypatch):
    run = frozen_run
    # Fit succeeds even if evaluation files and responses are inaccessible.
    (run.bundle/'evaluation_labels.jsonl').unlink()
    manifest, candidates = load_candidates(run.bundle)
    for c in candidates:
        if c.partition == 'evaluation':
            for spec in run.specs:
                run.cache.log(spec.runner().cache_key(spec.request(c))).path.unlink()
    policy = prepare_policy(run.bundle, run.specs, run.cache, correct_reward=1, incorrect_loss=4,
                            normalized_costs=[.01,.01], grid_size=101)
    assert all(f['correct_n'] and f['incorrect_n'] for f in policy['fit_diagnostics'])
    assert policy['cost_mapping']['kind'] == 'normalized_sensitivity'


def test_money_policy_uses_only_calibration_usage_and_frozen_mapping(frozen_run):
    run = frozen_run
    pricing = {'usd_per_million_tokens': {s.model: {'input': 1., 'output': 2.} for s in run.specs}}
    policy = prepare_policy(run.bundle, run.specs, run.cache, correct_reward=1, incorrect_loss=4,
                            pricing=pricing, utility_per_usd=100., grid_size=101)
    assert policy['planner']['costs'] == pytest.approx([.014,.014])
    assert all(f['mean_calibration_usd'] == pytest.approx(.00014) for f in policy['fit_diagnostics'])
    assert policy['cost_mapping']['pricing_snapshot'] == pricing
    with pytest.raises(ValueError, match='unresolved calibration cost'):
        prepare_policy(run.bundle, run.specs, run.cache, correct_reward=1, incorrect_loss=4,
                       pricing={}, utility_per_usd=100.)


def test_comparison_is_reviewable_and_has_zero_generator_calls(frozen_run):
    run = frozen_run
    config = {'schema': 1, 'verifiers': [asdict(s) for s in run.specs],
              'arms': [{'name': 'first', 'order': [run.specs[0].id]}]}
    result = prepare_comparison(run.bundle, config, run.cache)
    assert result['generator_calls_planned'] == 0
    assert result['collection_status'] == 'prepared_only_no_calls'
    assert all(row['cache']['evaluation']['missing_n'] == 0 for row in result['verifiers'])


def test_recovery_identity_is_independent_of_archive_location(frozen_run):
    import shutil
    run = frozen_run
    copies = run.root/'archive-copy'
    copies.mkdir()
    shutil.copy(run.source,copies/'renamed-source.csv')
    shutil.copy(run.sample,copies/'renamed-manifest.json')
    shutil.copytree(run.run_dir,copies/'run')
    repeated = recover(copies/'renamed-source.csv',copies/'renamed-manifest.json',copies/'run',
                       run.bundle,run.cache,run.run_id)
    assert repeated == run.manifest
