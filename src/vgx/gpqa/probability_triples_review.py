"""Portable offline review bundle without GPQA text, API replies, or credentials.

The bundle carries derived development rows, original source fits, all result
tables and losslessly pooled per-item decisions. Pooling removes repeated trace
and observation dictionaries; it does not remove policies or decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

from vgx.common.storage import digest, file_digest
from vgx.gpqa.probability_triples import evaluate, load_rows
from vgx.gpqa.sequential import unseal


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def pack_decisions(value):
    pools = {k: [] for k in ('observations', 'trace', 'feedback_events', 'records')}
    indexes = {k: {} for k in pools}

    def intern(kind, record):
        key = encoded(record)
        if key not in indexes[kind]:
            indexes[kind][key] = len(pools[kind])
            pools[kind].append(record)
        return indexes[kind][key]

    policies = {}
    for policy, decisions in value['decisions'].items():
        refs = []
        for decision in decisions:
            record = dict(decision)
            for field in ('observations', 'trace', 'feedback_events'):
                if field in record:
                    record[field] = [intern(field, entry) for entry in record[field]]
            refs.append(intern('records', record))
        policies[policy] = refs
    return {'schema': 1, 'logical_sha256': digest(value), 'item_ids': value['item_ids'],
            'pools': pools, 'policies': policies}


def unpack_decisions(packed):
    if packed['schema'] != 1:
        raise ValueError('unknown decision pooling schema')
    pools = packed['pools']
    records = []
    for record in pools['records']:
        decision = dict(record)
        for field in ('observations', 'trace', 'feedback_events'):
            if field in decision:
                decision[field] = [pools[field][i] for i in decision[field]]
        records.append(decision)
    value = {'item_ids': packed['item_ids'],
             'decisions': {p: [records[i] for i in refs] for p, refs in packed['policies'].items()}}
    if digest(value) != packed['logical_sha256']:
        raise ValueError('decision pooling checksum mismatch')
    return value


def check_export_fields(value):
    prohibited = {'question', 'choices', 'prompt', 'system', 'response', 'correct_index',
                  'api_key', 'access_token', 'refresh_token', 'private_key', 'authorization'}
    if isinstance(value, dict):
        if prohibited.intersection(k.lower() for k in value):
            raise ValueError('unexpected source text or secret field in export')
        for item in value.values():
            check_export_fields(item)
    elif isinstance(value, list):
        for item in value:
            check_export_fields(item)


def export_bundle(root, output):
    root, output = Path(root), Path(output)
    protocol = json.loads((root/'protocol.json').read_text())
    unseal(protocol, 'analysis_id')
    status = json.loads((root/'status.json').read_text())
    if status['status'] != 'complete' or status['analysis_id'] != protocol['analysis_id']:
        raise ValueError('completed experiment required')
    rows, pilot, identity = load_rows(protocol['source_config'], protocol['config'])
    if identity != protocol['frozen_sources']:
        raise ValueError('frozen inputs changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 1, 'analysis_id': protocol['analysis_id'], 'files': {},
        'original_result_files': {}, 'decision_logical_hashes': {},
        'content': 'Development scores/outcomes only. No GPQA text, answer keys, raw provider replies or credentials.',
        'limitations': 'Reparsing provider replies and independently checking billed tokens requires the original local caches.'}
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        def put(name, data):
            check_export_fields(json.loads(data))
            archive.writestr(name, data)
            manifest['files'][name] = hashlib.sha256(data).hexdigest()
        put('inputs/development_rows.json', encoded({'rows': rows, 'pilot_ids': sorted(pilot)}))
        for source_path, sha in protocol['source_files'].items():
            p = Path(source_path)
            if file_digest(p) != sha:
                raise ValueError('source fit changed')
            rel = p.relative_to(protocol['config']['source_results'])
            put('source_fits/'+rel.as_posix(), p.read_bytes())
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            manifest['original_result_files'][rel] = file_digest(path)
            if path.suffix == '.gz':
                with gzip.open(path, 'rt') as f:
                    value = json.load(f)
                packed = pack_decisions(value)
                if unpack_decisions(packed) != value:
                    raise ValueError('lossless pooling round trip failed')
                name = 'experiment/'+rel[:-8]+'.packed.json'
                put(name, encoded(packed))
                manifest['decision_logical_hashes'][rel] = packed['logical_sha256']
            elif path.suffix == '.json':
                put('experiment/'+rel, path.read_bytes())
            else:
                raise ValueError('unexpected result file type')
        archive.writestr('manifest.json', encoded(manifest))
    return {'path': str(output), 'bytes': output.stat().st_size, 'sha256': file_digest(output),
            'files': len(manifest['files']), 'analysis_id': protocol['analysis_id']}


def verify_bundle(bundle):
    counts = Counter()
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if set(archive.namelist()) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('archive membership mismatch')
        for name, expected in manifest['files'].items():
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('archive file hash mismatch: '+name)
            counts['files'] += 1
            if name.endswith('.packed.json'):
                value = unpack_decisions(json.loads(data))
                counts['decision_records'] += sum(len(v) for v in value['decisions'].values())
                counts['decision_files'] += 1
        protocol = json.loads(archive.read('experiment/protocol.json'))
        unseal(protocol, 'analysis_id')
        if manifest['analysis_id'] != protocol['analysis_id']:
            raise ValueError('analysis identity mismatch')
    return {'status': 'passed', 'counts': dict(counts), 'analysis_id': manifest['analysis_id'],
            'bundle_sha256': file_digest(bundle), 'api_requests': 0}


def replay_bundle(bundle, run, output):
    """Recompute a named run; compare all policies, selections and decision content."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('use an empty replay output directory')
    with zipfile.ZipFile(bundle) as archive:
        protocol = json.loads(archive.read('experiment/protocol.json'))
        inputs = json.loads(archive.read('inputs/development_rows.json'))
        g, cohort, _ = run.split('/')
        fit = json.loads(archive.read('source_fits/'+run+'.json'))
        frozen = json.loads(archive.read('experiment/'+run+'/results.json'))
        rows = inputs['rows'][g]
        if cohort == 'remaining_calibration':
            rows = [r for r in rows if r['item_id'] not in set(inputs['pilot_ids'])]
        elif cohort != 'all_calibration':
            raise ValueError('development cohorts only')
        cfg = dict(protocol['source_config'])
        cfg.update(utility_per_usd=protocol['config']['primary_utility_per_usd'],
                   bootstrap_repeats=protocol['config']['bootstrap_repeats'])
        tags = [v+':probability' for v in protocol['config']['verifiers']]
        result = evaluate(rows, fit, cfg, protocol['config'], tags, output)
        expected = {k: v for k, v in frozen.items() if k not in ('analysis_id', 'output_hashes')}
        if result != expected:
            raise ValueError('recomputed results differ; inspect runtime versions and code')
        for loss in protocol['config']['losses']:
            with gzip.open(output/f'decisions_loss_{loss}.json.gz', 'rt') as f:
                decisions = json.load(f)
            stored = unpack_decisions(json.loads(archive.read(
                f'experiment/{run}/decisions_loss_{loss}.packed.json')))
            if decisions != stored:
                raise ValueError('recomputed per-item decisions differ')
    return {'status': 'passed', 'run': run, 'api_requests': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('export'); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('verify'); p.add_argument('--bundle', required=True)
    p = sub.add_parser('replay'); p.add_argument('--bundle', required=True)
    p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'export':
        result = export_bundle(args.root, args.output)
    elif args.command == 'verify':
        result = verify_bundle(args.bundle)
    else:
        result = replay_bundle(args.bundle, args.run, args.output)
    print(json.dumps(result, indent=2))
