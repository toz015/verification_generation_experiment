"""Recover a complete historical run into immutable, label-separated artifacts.

This module never generates an answer, creates a new sample, or contacts an API.
The live runner reads candidates.jsonl only; labels and historical verifier
outputs remain in separate files for offline fitting/scoring.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import os
import shutil

from vgx.common.api import RequestCache
from vgx.common.llm import Call, Request
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.common.vertex import VertexBatchRunner
from vgx.gpqa.load import load_items_excluding_duplicate_choices, restore_pilot
from vgx.gpqa.prompt import SYSTEM, build_generator_prompt, build_verifier_prompt, parse_generator_response, parse_verifier_response

ORIGINAL_RUN = 'a5d92086584910dbe156297b5c1083bb8bf47765670b46a0dfc1b7ddc49cbce7'


@dataclass(frozen=True)
class FrozenCandidate:
    item_id: str
    partition: str
    subject: str
    question: str
    choices: tuple[str, str, str, str]
    answer: str | None
    p_correct: float | None
    generator_ok: bool
    generator_failure: str | None
    generator_model: str
    generator_call_key: str
    generator_prompt_sha256: str
    generator_response_sha256: str
    system: str
    verifier_prompt: str | None

    def __post_init__(self):
        if self.partition not in ('calibration', 'evaluation') or len(self.choices) != 4:
            raise ValueError('invalid frozen candidate schema')
        if self.answer is not None and self.answer not in ('A', 'B', 'C', 'D'):
            raise ValueError('invalid frozen answer')
        if self.p_correct is not None and (isinstance(self.p_correct, bool) or not 0 <= self.p_correct <= 1):
            raise ValueError('invalid frozen confidence')
        if self.answer is not None and not self.verifier_prompt:
            raise ValueError('missing frozen verifier prompt')


def vertex_from_config(config: dict, model: str) -> VertexBatchRunner:
    generation = config['generation']
    return VertexBatchRunner(model, config['inference']['model_locations'][model],
                             config['inference']['project_id'],
                             max_tokens=generation['max_tokens'],
                             temperature=generation.get('temperature', 0.0),
                             top_p=generation.get('top_p', 1.0),
                             reasoning_effort=generation.get('reasoning_effort', 'low'))


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _legacy_digest(payload) -> str:
    return _sha_text(json.dumps(payload, sort_keys=True, allow_nan=False))


def _checked_call(rows: list[dict], runner, request: Request) -> Call:
    wanted = {runner.legacy_cache_key(request), runner.cache_key(request)}
    matches = [r for r in rows if r.get('key') in wanted and r.get('error') is None]
    if not matches:
        raise ValueError(f'missing matching frozen response: {request.key}')
    if len({r['response'] for r in matches}) != 1:
        raise ValueError(f'conflicting frozen responses: {request.key}')
    for row in matches:
        meta = row.get('meta') or {}
        if (row.get('model') != runner.model or row.get('prompt') != request.prompt
                or meta.get('system') != request.system or meta.get('runner_identity') != runner.identity
                or meta.get('logical_key') != request.key or row.get('params') != runner.params):
            raise ValueError(f'provider request/provenance mismatch: {request.key}')
        for key, value in request.meta.items():
            if meta.get(key) != value:
                raise ValueError(f'candidate or partition metadata mismatch: {request.key}')
    return Call(**matches[0])


def validate_original(source: Path, sample_manifest: Path, run_dir: Path,
                      expected_run_id: str = ORIGINAL_RUN) -> dict:
    required = [source, sample_manifest, run_dir/'run_manifest.json',
                run_dir/'pilot_records.jsonl', run_dir/'pilot_metrics.json', run_dir/'generator.jsonl']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('original artifacts unavailable: ' + ', '.join(missing))
    manifest = json.loads((run_dir/'run_manifest.json').read_text())
    payload = {k: v for k, v in manifest.items() if k != 'run_id'}
    if manifest.get('run_id') != expected_run_id or _legacy_digest(payload) != expected_run_id:
        raise ValueError('original run manifest identity does not validate')
    config = manifest['config']
    source_hash = file_digest(source)
    if source_hash != manifest.get('source_sha256'):
        raise ValueError('source checksum differs from original run')
    items, exclusions = load_items_excluding_duplicate_choices(source)
    split = restore_pilot(items, sample_manifest, source_hash, exclusions)
    sample = json.loads(sample_manifest.read_text())
    by_id = {item.item_id: item for item in split.sample}
    ordered = [by_id[item_id] for item_id in sample['sample_ids']]
    if _legacy_digest([asdict(item) for item in ordered]) != manifest.get('sample_content_sha256'):
        raise ValueError('original choice order or sample content differs')
    if (len(split.sample) != config['sample_size'] or len(split.calibration) != config['calibration_size']
            or split.seed != config['seed'] or split.seed != manifest['split_seed']):
        raise ValueError('original split sizes or seed differ')
    for partition in ('calibration', 'evaluation'):
        actual = [i.item_id for i in getattr(split, partition)]
        if len(set(manifest[partition+'_ids'])) != len(actual) or set(actual) != set(manifest[partition+'_ids']):
            raise ValueError('run/sample split membership differs')
    raw_records = read_jsonl(run_dir/'pilot_records.jsonl')
    records = {r['item_id']: r for r in raw_records}
    if len(records) != len(raw_records) or set(records) != set(by_id):
        raise ValueError('original records are incomplete or duplicated')
    models = {'generator': config['models']['generator'],
              **{f'verifier_{i}': m for i, m in enumerate(config['models']['verifiers'], 1)}}
    logs = {}
    for tag in models:
        path = run_dir/f'{tag}.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'original artifacts unavailable: {path}')
        logs[tag] = read_jsonl(path)  # complete historical recovery is strict
    candidates, checked = [], []
    for item in ordered:
        partition = 'calibration' if item.item_id in manifest['calibration_ids'] else 'evaluation'
        row = records[item.item_id]
        if row['partition'] != partition or row['correct_index'] != item.correct_index or row['subject'] != item.subject:
            raise ValueError('record label, subject or split differs from pinned source')
        request = Request(f'generator|{item.item_id}', build_generator_prompt(item), SYSTEM,
                          {'item_id': item.item_id, 'partition': partition, 'role': 'generator'})
        runner = vertex_from_config(config, models['generator'])
        call = _checked_call(logs['generator'], runner, request)
        answer = parse_generator_response(call.response)
        if (row['generator_answer'], row['generator_p_correct'], row['generator_ok'], row['generator_failure']) != (
                answer.answer, answer.p_correct, answer.ok, answer.failure):
            raise ValueError('candidate record disagrees with original generator response')
        checked.append((runner, request, call))
        candidates.append(FrozenCandidate(item.item_id, partition, item.subject, item.question, item.choices,
            answer.answer, answer.p_correct, answer.ok, answer.failure, runner.model, call.key,
            _sha_text(request.prompt), _sha_text(call.response), SYSTEM,
            build_verifier_prompt(item, answer.answer) if answer.answer else None))
        for index, model in enumerate(config['models']['verifiers'], 1):
            tag = f'verifier_{index}'
            if answer.answer is None:
                if row['verifiers'][tag].get('p_correct') is not None:
                    raise ValueError('verifier signal exists without a valid candidate')
                continue
            request = Request(f'{tag}|{item.item_id}', build_verifier_prompt(item, answer.answer), SYSTEM,
                {'item_id': item.item_id, 'partition': partition, 'role': 'verifier',
                 'verifier_index': index, 'candidate': answer.answer})
            runner = vertex_from_config(config, model)
            verifier_call = _checked_call(logs[tag], runner, request)
            if asdict(parse_verifier_response(verifier_call.response)) != row['verifiers'][tag]:
                raise ValueError('verifier record disagrees with original response')
            checked.append((runner, request, verifier_call))
    known_keys = {call.key for _, _, call in checked}
    if any(row.get('error') is None and row.get('key') not in known_keys
           for rows in logs.values() for row in rows):
        raise ValueError('unexpected successful requests in original run logs')
    metrics = json.loads((run_dir/'pilot_metrics.json').read_text())
    if metrics.get('run_id') != expected_run_id or metrics.get('planned_sample_size') != len(ordered):
        raise ValueError('reported metrics belong to a different run or sample')
    from vgx.gpqa.score import binary_forecast_metrics, correctness_outcome
    for partition in ('all', 'calibration', 'evaluation'):
        rows = [r for r in raw_records if partition == 'all' or r['partition'] == partition]
        outcomes = [correctness_outcome(r['generator_answer'], r['correct_index']) for r in rows]
        section = metrics[partition+'_metrics']
        if section['n'] != len(rows) or section['correct_n'] != sum(outcomes):
            raise ValueError('reported outcome counts disagree with original responses')
        valid = [r for r in rows if r['generator_answer'] is not None and r['generator_p_correct'] is not None]
        forecast = binary_forecast_metrics(
            [correctness_outcome(r['generator_answer'], r['correct_index']) for r in valid],
            [r['generator_p_correct'] for r in valid]).to_dict()
        for key in ('n', 'brier', 'log_loss'):
            observed = section['generator_confidence'][key]
            expected = forecast[key]
            if (observed is None) != (expected is None) or (expected is not None and
                    not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError('reported generator forecasts disagree with original responses')
    # Artifact identity depends on bytes, not where an archive was unpacked.
    files = {'dataset_source': file_digest(source), 'sample_manifest': file_digest(sample_manifest),
             **{p.name: file_digest(p) for p in required[2:]},
             **{f'{tag}.jsonl': file_digest(run_dir/f'{tag}.jsonl') for tag in models}}
    return {'manifest': manifest, 'config': config, 'candidates': candidates, 'records': raw_records,
            'checked_calls': checked, 'file_hashes': files}


def recover(source: Path, sample_manifest: Path, run_dir: Path, output: Path,
            cache: RequestCache, expected_run_id: str = ORIGINAL_RUN) -> dict:
    validated = validate_original(source, sample_manifest, run_dir, expected_run_id)
    config = validated['config']
    public = [asdict(c) for c in validated['candidates']]
    calibration = [r for r in validated['records'] if r['partition'] == 'calibration']
    labels = [{'item_id': r['item_id'], 'correct_index': r['correct_index']}
              for r in validated['records'] if r['partition'] == 'evaluation']
    contents = {'candidates.jsonl': public, 'calibration_records.jsonl': calibration,
                'evaluation_labels.jsonl': labels}
    bodies = {name: ''.join(json.dumps(row, sort_keys=True, allow_nan=False)+'\n' for row in rows)
              for name, rows in contents.items()}
    manifest = {'schema': 1, 'original_run_id': expected_run_id,
                'generator_model': config['models']['generator'],
                'collection_config': config, 'candidate_count': len(public),
                'calibration_ids': [c['item_id'] for c in public if c['partition'] == 'calibration'],
                'evaluation_ids': [c['item_id'] for c in public if c['partition'] == 'evaluation'],
                'files': {name: _sha_text(body) for name, body in bodies.items()},
                'source_artifact_hashes': validated['file_hashes']}
    manifest['bundle_id'] = digest(manifest)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(str(output)+'.lock'):
        if output.exists():
            existing = validate_bundle(output)
            if existing != manifest:
                raise ValueError('refusing to overwrite a different frozen bundle')
        else:
            temporary = Path(tempfile.mkdtemp(prefix='.frozen-', dir=output.parent))
            try:
                for name, body in bodies.items():
                    path = temporary/name
                    with path.open('w') as stream:
                        stream.write(body)
                        stream.flush()
                        os.fsync(stream.fileno())
                atomic_json(temporary/'bundle_manifest.json', manifest)
                temporary.rename(output)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    # Import can be resumed independently after an interrupted recovery.
    for runner, request, call in validated['checked_calls']:
        cache.import_call(replace(call, key=runner.cache_key(request), meta={**call.meta,
            'original_cache_key': call.key, 'original_run_id': expected_run_id,
            'legacy_row_fingerprint': digest(asdict(call))}))
    return manifest


def _manifest(bundle: Path) -> dict:
    manifest = json.loads((Path(bundle)/'bundle_manifest.json').read_text())
    if manifest.get('schema') != 1 or manifest.get('bundle_id') != digest({k:v for k,v in manifest.items() if k != 'bundle_id'}):
        raise ValueError('frozen bundle manifest checksum invalid')
    if set(manifest['files']) != {'candidates.jsonl', 'calibration_records.jsonl', 'evaluation_labels.jsonl'}:
        raise ValueError('unexpected bundle files')
    return manifest


def validate_bundle(bundle: Path) -> dict:
    manifest = _manifest(bundle)
    for name, expected in manifest['files'].items():
        if file_digest(Path(bundle)/name) != expected:
            raise ValueError(f'frozen artifact changed: {name}')
    load_candidates(bundle)
    return manifest


def load_candidates(bundle: Path) -> tuple[dict, tuple[FrozenCandidate, ...]]:
    """Live input boundary: never opens labels or historical verifier outputs."""
    manifest = _manifest(bundle)
    path = Path(bundle)/'candidates.jsonl'
    if file_digest(path) != manifest['files']['candidates.jsonl']:
        raise ValueError('frozen candidates checksum invalid')
    candidates = tuple(FrozenCandidate(**{**row, 'choices': tuple(row['choices'])}) for row in read_jsonl(path))
    ids = [c.item_id for c in candidates]
    if len(ids) != manifest['candidate_count'] or len(set(ids)) != len(ids):
        raise ValueError('incomplete or duplicated frozen candidates')
    for partition in ('calibration', 'evaluation'):
        expected = manifest[partition+'_ids']
        if len(set(expected)) != len(expected) or set(expected) != {c.item_id for c in candidates if c.partition == partition}:
            raise ValueError('frozen split membership invalid')
    return manifest, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sample-manifest', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('results/request_cache'))
    parser.add_argument('--expected-run-id', default=ORIGINAL_RUN)
    args = parser.parse_args()
    manifest = recover(args.source, args.sample_manifest, args.run_dir, args.output,
                       RequestCache(args.cache), args.expected_run_id)
    print(json.dumps({'bundle_id': manifest['bundle_id'], 'candidates': manifest['candidate_count'],
                      'status': 'validated_and_frozen'}))


if __name__ == '__main__':
    main()
