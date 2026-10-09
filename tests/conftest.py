"""Synthetic, fully local historical artifacts for frozen-workflow regression tests."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vgx.common.api import RequestCache
from vgx.common.llm import Call, CallLog, Request
from vgx.common.storage import file_digest
from vgx.gpqa.artifacts import recover, vertex_from_config
from vgx.gpqa.load import load_items, prepare_pilot, write_manifest
from vgx.gpqa.prompt import SYSTEM, build_generator_prompt, build_verifier_prompt, parse_generator_response, parse_verifier_response
from vgx.gpqa.run_pilot import run_identity
from vgx.gpqa.verifiers import VerifierSpec


@pytest.fixture
def original_run(tmp_path, monkeypatch):
    source = tmp_path/'source.csv'
    with source.open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3', 'High-level domain'])
        for i in range(20):
            writer.writerow([f'Synthetic fixture question {i}', f'correct {i}', f'wrong a {i}', f'wrong b {i}', f'wrong c {i}', 'physics'])
    split = prepare_pilot(load_items(source), 10, 123)
    sample = tmp_path/'sample.json'
    write_manifest(split, sample, file_digest(source), [])
    generator, models = 'google/gemini-3.8-flash', ['meta/llama-3.3-70b-instruct-maas', 'google/gemini-3.7-flash']
    config = {'dataset': 'gpqa_main', 'sample_size': 10, 'calibration_size': len(split.calibration),
              'source_file': str(source), 'manifest': str(sample), 'seed': 123,
              'models': {'generator': generator, 'verifiers': models},
              'inference': {'provider': 'vertex_ai', 'project_id': 'synthetic-project',
                            'model_locations': {generator: 'global', models[0]: 'us-central1', models[1]: 'global'}},
              'generation': {'max_tokens': 32, 'temperature': 0., 'top_p': 1., 'reasoning_effort': 'low'},
              'routing_scenarios': []}
    monkeypatch.setattr('vgx.gpqa.run_pilot.subprocess.run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='synthetic-project'))
    run_id, manifest = run_identity(config, split)
    run_dir = tmp_path/'original'/run_id
    run_dir.mkdir(parents=True)
    (run_dir/'run_manifest.json').write_text(json.dumps({'run_id': run_id, **manifest}))
    partitions = {i.item_id: partition for partition in ('calibration','evaluation') for i in getattr(split, partition)}
    positions = {i.item_id: n for partition in ('calibration','evaluation') for n,i in enumerate(getattr(split, partition))}
    rows = []
    for item in split.sample:
        partition = partitions[item.item_id]
        right = positions[item.item_id] % 2 == 0
        candidate = item.correct_letter if right else 'ABCD'[(item.correct_index+1)%4]
        text = json.dumps({'answer': candidate, 'p_correct': .6})
        request = Request(f'generator|{item.item_id}', build_generator_prompt(item), SYSTEM,
                          {'item_id': item.item_id, 'partition': partition, 'role': 'generator'})
        runner = vertex_from_config(config, generator)
        def log_call(runner, request, response, tag):
            call = Call(runner.legacy_cache_key(request), runner.model, request.prompt, response,
                        runner.params, 'managed_api', .1, 1.,
                        meta={**request.meta, 'logical_key': request.key, 'system': request.system,
                              'runner_identity': runner.identity, 'provider_response_model': runner.model,
                              'usage': {'prompt_tokens': 100, 'completion_tokens': 5,
                                        'completion_tokens_details': {'reasoning_tokens': 15}, 'total_tokens': 120},
                              'response_id': tag+'-'+item.item_id})
            CallLog(run_dir/f'{tag}.jsonl').append(call)
        log_call(runner, request, text, 'generator')
        answer = parse_generator_response(text)
        verifiers = {}
        for index, model in enumerate(models, 1):
            tag = f'verifier_{index}'
            request = Request(f'{tag}|{item.item_id}', build_verifier_prompt(item, candidate), SYSTEM,
                {'item_id': item.item_id, 'partition': partition, 'role': 'verifier', 'verifier_index': index, 'candidate': candidate})
            response = json.dumps({'p_correct': .9 if right else .1})
            log_call(vertex_from_config(config, model), request, response, tag)
            verifiers[tag] = asdict(parse_verifier_response(response))
        rows.append({'item_id': item.item_id, 'partition': partition, 'subject': item.subject,
                     'correct_index': item.correct_index, 'generator_answer': candidate, 'generator_p_correct': .6,
                     'generator_ok': True, 'generator_failure': None, 'verifiers': verifiers})
    (run_dir/'pilot_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    from vgx.gpqa.report import build_report
    metrics = build_report(rows, config, bootstrap_repeats=12, synthetic=True)
    metrics['run_id'] = run_id
    (run_dir/'pilot_metrics.json').write_text(json.dumps(metrics))
    return SimpleNamespace(source=source, sample=sample, split=split, config=config, run_dir=run_dir,
                           run_id=run_id, records=rows, root=tmp_path)


@pytest.fixture
def frozen_run(original_run):
    run = original_run
    run.bundle = run.root/'frozen'
    run.cache = RequestCache(run.root/'cache')
    run.manifest = recover(run.source, run.sample, run.run_dir, run.bundle, run.cache, run.run_id)
    run.specs = tuple(VerifierSpec(id=f'verifier_{i}', provider='vertex_ai', model=model,
        location=run.config['inference']['model_locations'][model], project='synthetic-project',
        generation=run.config['generation']) for i,model in enumerate(run.config['models']['verifiers'],1))
    return run
