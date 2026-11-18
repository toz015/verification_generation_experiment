"""Fresh, pinned GPQA data and resumable candidate generation for API experiments.

    python -m vgx.gpqa.fresh prepare --root results/gpqa_fresh --config ...

Dataset text stays in ignored local files. Collection is offline by default.
The source answer key is separated before generator requests are constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from vgx.common.api import RequestCache
from vgx.common.concurrency import parallel_map
from vgx.common.llm import Request
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.gpqa.artifacts import FrozenCandidate, load_candidates, validate_bundle, vertex_from_config
from vgx.gpqa.load import DuplicateChoicesError, _read_rows, _row_to_item, shuffle_choices, stratified_sample
from vgx.gpqa.prompt import SYSTEM, build_generator_prompt, build_verifier_prompt, parse_generator_response
from vgx.gpqa.sequential import seal, unseal

REPO = 'Idavidrein/gpqa'


def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def _sha(body):
    return hashlib.sha256(body.encode()).hexdigest()


def _body(rows):
    return ''.join(json.dumps(row, sort_keys=True, allow_nan=False)+'\n' for row in rows)


def _write_immutable(path, body):
    path = Path(path)
    if path.exists() and path.read_text() != body:
        raise ValueError(f'refusing to change frozen artifact {path.name}')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via JSON helper for JSON objects; JSONL needs its own
        # temporary sibling, followed by atomic replacement.
        import os, tempfile
        fd, temporary = tempfile.mkstemp(prefix='.fresh-', dir=path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def download(destination: Path, revision=None):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    api.whoami()  # use the user's gated access, never an unauthenticated mirror
    pinned = api.dataset_info(REPO, revision=revision).sha
    destination = Path(destination)/pinned
    for name in ('main', 'diamond', 'extended'):
        hf_hub_download(REPO, f'gpqa_{name}.csv', repo_type='dataset', revision=pinned, local_dir=destination)
    return destination, pinned


def prepare(source_dir: Path, revision: str, config: dict, root: Path):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('pin a full Hugging Face commit SHA')
    raw, valid, exclusions, counts, hashes = {}, {}, {}, {}, {}
    for subset in ('main', 'diamond', 'extended'):
        path = Path(source_dir)/f'gpqa_{subset}.csv'
        hashes[subset] = file_digest(path)
        raw[subset], valid[subset], exclusions[subset] = {}, {}, []
        rows = _read_rows(path)
        counts[subset] = len(rows)
        for row in rows:
            if not isinstance(row.get('Question'), str) or not normalized(row['Question']):
                raise ValueError('missing source question')
            key = digest(normalized(row['Question']))
            if key in raw[subset]:
                raise ValueError('duplicate normalized question in source')
            fields = ['Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']
            if any(not isinstance(row.get(f), str) or not normalized(row[f]) for f in fields):
                raise ValueError('missing source answer choice')
            raw[subset][key] = {'options': sorted(normalized(row[f]) for f in fields),
                               'correct': normalized(row['Correct Answer'])}
            try:
                item = replace(_row_to_item(row), item_id=key)
                if len({normalized(c) for c in item.choices}) != 4:
                    raise DuplicateChoicesError('normalized duplicate choices')
                valid[subset][key] = item
            except DuplicateChoicesError:
                exclusions[subset].append({'item_id': key, 'reason': 'duplicate_answer_choices'})
    if not raw['diamond'].keys() <= raw['main'].keys() <= raw['extended'].keys():
        raise ValueError('published subset nesting does not hold for this snapshot')
    for subset in ('main', 'diamond'):
        if any(raw[subset][key] != raw['extended'][key] for key in raw[subset]):
            raise ValueError('overlapping questions have different options or answer keys')
    # Membership uses all original Diamond IDs, including invalid questions.
    cal = [item for key,item in valid['main'].items() if key not in raw['diamond']]
    evaluation = list(valid['diamond'].values())
    pilot = [i.item_id for i in stratified_sample(cal, min(50, len(cal)), config['seed'])]
    public, labels = [], []
    for partition, items in (('calibration', cal), ('evaluation', evaluation)):
        for item in sorted(items, key=lambda i:i.item_id):
            item = shuffle_choices(item, config['seed'])
            public.append({'item_id': item.item_id, 'partition': partition, 'subject': item.subject,
                           'question': item.question, 'choices': item.choices})
            labels.append({'item_id': item.item_id, 'partition': partition, 'correct_index': item.correct_index})
    manifest = seal({'schema': 1, 'dataset_repo': REPO, 'dataset_revision': revision,
        'source_sha256': hashes, 'raw_counts': counts, 'exclusions': exclusions,
        'split': 'calibration=Main-minus-Diamond; evaluation=Diamond', 'seed': config['seed'],
        'counts': {'calibration': len(cal), 'evaluation': len(evaluation)},
        'subject_counts': {p:dict(Counter(r['subject'] for r in public if r['partition']==p))
                           for p in ('calibration','evaluation')},
        'pilot_calibration_ids': pilot, 'overlap_n': 0,
        'files': {'inputs.jsonl': _sha(_body(public)), 'labels.jsonl': _sha(_body(labels))}}, 'dataset_id')
    root = Path(root)
    with file_lock(root/'prepare.lock'):
        for name,body in [('inputs.jsonl',_body(public)), ('labels.jsonl',_body(labels)),
                          ('dataset_manifest.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')]:
            _write_immutable(root/name, body)
    return manifest


@dataclass(frozen=True)
class PublicItem:
    item_id: str
    partition: str
    subject: str
    question: str
    choices: tuple


def inputs(root):
    root = Path(root)
    manifest = json.loads((root/'dataset_manifest.json').read_text())
    unseal(manifest, 'dataset_id')
    if file_digest(root/'inputs.jsonl') != manifest['files']['inputs.jsonl']:
        raise ValueError('prepared public inputs changed')
    rows = tuple(PublicItem(**{**r, 'choices': tuple(r['choices'])}) for r in read_jsonl(root/'inputs.jsonl'))
    if len({r.item_id for r in rows}) != len(rows):
        raise ValueError('duplicate prepared inputs')
    return manifest, rows


def generator_request(item, operation):
    return Request(f'generator|{item.item_id}', build_generator_prompt(item), SYSTEM,
                   {'role': 'generator', 'item_id': item.item_id, 'partition': item.partition,
                    'operation_id': operation})


def candidate_for(item, runner, cache, operation, allow_api=False):
    request = generator_request(item, operation)
    call, cached = cache.get(runner, request, allow_api=allow_api)
    parsed = parse_generator_response(call.response)
    return FrozenCandidate(item.item_id, item.partition, item.subject, item.question, item.choices,
        parsed.answer, parsed.p_correct, parsed.ok, parsed.failure, runner.model, call.key,
        _sha(request.prompt), _sha(call.response), SYSTEM,
        build_verifier_prompt(item, parsed.answer) if parsed.answer else None)


def collect_generators(root, config, cache, *, subset='all', allow_api=False, workers=4):
    manifest, items = inputs(root)
    runner = vertex_from_config(config, config['models']['generator'])
    identity = seal({'dataset_id': manifest['dataset_id'], 'runner': runner.identity,
                     'requests': [runner.cache_key(generator_request(i, None)) for i in items]}, 'generation_id')
    # Prices and verifier lists are excluded: neither changes a generator call.
    with file_lock(Path(root)/'generation.lock'):
        _write_immutable(Path(root)/'generation_manifest.json', json.dumps(identity,indent=2,sort_keys=True)+'\n')
    if subset == 'pilot':
        items = [i for i in items if i.item_id in manifest['pilot_calibration_ids']]
    elif subset in ('calibration', 'evaluation'):
        items = [i for i in items if i.partition == subset]
    elif subset != 'all':
        raise ValueError('unknown generator subset')
    operation = 'generator:'+identity['generation_id']
    def collect(item):
        return candidate_for(item, runner, cache, operation, allow_api)
    return tuple(parallel_map(collect, items, workers))


def freeze(root, config, cache):
    """Offline assembly only. Cached responses cannot trigger model calls."""
    root = Path(root)
    manifest, _ = inputs(root)
    candidates = collect_generators(root, config, cache, allow_api=False)
    if file_digest(root/'labels.jsonl') != manifest['files']['labels.jsonl']:
        raise ValueError('prepared answer keys changed')
    labels = {r['item_id']:r for r in read_jsonl(root/'labels.jsonl')}
    if len(labels) != len(candidates) or set(labels) != {c.item_id for c in candidates}:
        raise ValueError('labels and candidates differ')
    records = [{'item_id':c.item_id,'partition':c.partition,'subject':c.subject,
                'correct_index':labels[c.item_id]['correct_index'], 'generator_answer':c.answer,
                'generator_p_correct':c.p_correct,'generator_ok':c.generator_ok,
                'generator_failure':c.generator_failure,'verifiers':{}} for c in candidates]
    bodies = {'candidates.jsonl':_body([asdict(c) for c in candidates]),
              'calibration_records.jsonl':_body([r for r in records if r['partition']=='calibration']),
              'evaluation_labels.jsonl':_body([{'item_id':r['item_id'],'correct_index':r['correct_index']}
                                             for r in records if r['partition']=='evaluation'])}
    bundle = seal({'schema':1,'origin':'fresh_huggingface','dataset_id':manifest['dataset_id'],
                   'generator_model':config['models']['generator'],'collection_config':config,
                   'candidate_count':len(candidates),
                   'calibration_ids':[c.item_id for c in candidates if c.partition=='calibration'],
                   'evaluation_ids':[c.item_id for c in candidates if c.partition=='evaluation'],
                   'files':{name:_sha(body) for name,body in bodies.items()},
                   'source_artifact_hashes':manifest['source_sha256']},'bundle_id')
    destination = root/'frozen'
    with file_lock(root/'freeze.lock'):
        for name,body in bodies.items():
            _write_immutable(destination/name,body)
        _write_immutable(destination/'bundle_manifest.json',json.dumps(bundle,indent=2,sort_keys=True)+'\n')
    validate_bundle(destination)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','generate','freeze'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path)
    parser.add_argument('--revision')
    parser.add_argument('--allow-api',action='store_true')
    parser.add_argument('--subset',choices=['all','pilot','calibration','evaluation'],default='all')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if args.command=='prepare':
        wanted_revision = args.revision or config.get('dataset_source',{}).get('revision')
        source, revision = (args.source_dir,wanted_revision) if args.source_dir else download(Path('data/gpqa/hf'),wanted_revision)
        result=prepare(source,revision,config,args.root)
    else:
        from vgx.common.budget import BudgetedRequestCache
        cache=BudgetedRequestCache(args.root/'cache',project=config['inference']['project_id'],
                                   pricing=config['api_pricing'],limit_usd=config['budget_usd'])
        result=freeze(args.root,config,cache) if args.command=='freeze' else {
            'candidates':len(collect_generators(args.root,config,cache,subset=args.subset,allow_api=args.allow_api))}
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','collection_config')}))


if __name__=='__main__':
    main()
