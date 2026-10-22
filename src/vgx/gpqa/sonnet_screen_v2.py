"""User-approved final-JSON parsing amendment; reuse the frozen Sonnet requests."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re
import time

from vgx.common.api import OfflineCacheMiss
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.gpqa import sonnet_screen as original
from vgx.gpqa import replacement_generators as shared
from vgx.gpqa import generator_study as direct
from vgx.gpqa.artifacts import FrozenCandidate
from vgx.gpqa.fresh import generator_request
from vgx.gpqa.prompt import GeneratorResponse, SYSTEM, build_verifier_prompt, parse_generator_response
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_collect import Pacer
from vgx.gpqa.signal_study import check_billing


PARSER_VERSION = 'sonnet_unique_final_json_v2'


def _unique_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON key')
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError('non-finite JSON number')


def parse_call(call):
    """Only final text is inspected; require one unambiguous, terminal object."""
    if call.meta.get('provider_response', {}).get('stop_reason') != 'end_turn':
        return GeneratorResponse(None, None, False, 'non_end_turn')
    text = call.response.strip()
    decoder = json.JSONDecoder(object_pairs_hook=_unique_fields, parse_constant=_invalid_constant)
    objects = []
    for match in re.finditer(r'\{', text):
        try:
            payload, length = decoder.raw_decode(text[match.start():])
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict):
            objects.append((match.start(), match.start()+length, payload))
    if len(objects) != 1:
        return GeneratorResponse(None, None, False,
                                 'ambiguous_json_objects' if objects else 'missing_valid_json')
    start, end, payload = objects[0]
    prefix, suffix = text[:start], text[end:].strip()
    fenced = re.search(r'(?:^|\n)```(?:json)?\s*$', prefix) is not None
    if suffix not in ('', '```') or (suffix == '```') != fenced:
        return GeneratorResponse(None, None, False, 'nonterminal_json_or_unbalanced_fence')
    if set(payload) != {'answer', 'p_correct'}:
        return GeneratorResponse(None, None, False, 'invalid_final_fields')
    if type(payload['p_correct']) not in (int, float):
        return GeneratorResponse(None, None, False, 'nonnumeric_confidence')
    return parse_generator_response(json.dumps(payload))


def amendment_dir(root):
    return Path(root)/'parser_v2'


def prepare(root):
    root = Path(root); out = amendment_dir(root)
    plan, items, _ = original.load(root)
    cfg = plan['config']; runner = original.SonnetRunner(cfg); cache = shared.cache_for(cfg, root)
    if json.loads((root/'status.json').read_text())['stage'] != 'format_failed':
        raise ValueError('expected preserved original format-gate failure')
    gate = []; hashes = {}
    for n, item in enumerate(items[:5], 1):
        call, _ = cache.get(runner, generator_request(item, plan['study_id']))
        original.priced_usage(call, cfg)
        parsed = parse_call(call)
        gate.append({'pilot_position': n, 'strict_ok': original.parse_call(call).ok,
                     'v2_ok': parsed.ok, 'v2_failure': parsed.failure})
        hashes[str(cache.log(call.key).path)] = file_digest(cache.log(call.key).path)
    if not all(r['v2_ok'] for r in gate):
        raise ValueError('cached first-five gate failed under approved v2 parser')
    value = seal({'parent_study_id': plan['study_id'],
        'parent_plan_sha256': file_digest(root/'plan.json'),
        'parser_version': PARSER_VERSION,
        'implementation_sha256': file_digest(__file__),
        'authorization': 'User approved accepting explanation followed by a unique final answer JSON, reusing the five replies and continuing the other 45 development items.',
        'rule': 'Exactly one decodable JSON object in final text, terminal apart from a matched optional code fence; exact answer/p_correct keys, no duplicate keys, finite numeric probability in [0,1], valid answer, end_turn, expected provider model, priced usage.',
        'cached_gate': gate, 'first_five_cache_sha256': hashes,
        'request_keys': plan['request_keys'], 'runner_identity': runner.identity,
        'request_changes': False, 'confidence_calibrated': False,
        'shared_cache': str(root/'cache'), 'stage_budget_usd': cfg['stage_budget_usd'],
        'first_five_estimated_usd': cache.summary()['estimated_usd'],
        'additional_paid_requests_planned': 45, 'evaluation_requests': 0, 'verifier_requests': 0}, 'amendment_id')
    shared.immutable(out/'plan.json', value)
    return {'stage': 'cached_gate_passed', 'amendment_id': value['amendment_id'],
            'reused': 5, 'v2_valid': 5, 'strict_valid': sum(r['strict_ok'] for r in gate),
            'new_api_calls': 0, 'remaining_requests': 45, 'confidence_calibrated': False}


def load(root):
    root = Path(root)
    parent, items, baselines = original.load(root)
    amendment = json.loads((amendment_dir(root)/'plan.json').read_text())
    unseal(amendment, 'amendment_id')
    if (amendment['parent_study_id'] != parent['study_id']
        or amendment['parent_plan_sha256'] != file_digest(root/'plan.json')
        or amendment['implementation_sha256'] != file_digest(__file__)
        or amendment['request_keys'] != parent['request_keys']
        or amendment['runner_identity'] != parent['runner']):
        raise ValueError('frozen requests, protocol or implementation changed')
    if any(file_digest(p) != h for p, h in amendment['first_five_cache_sha256'].items()):
        raise ValueError('an original cached reply changed')
    return parent, amendment, items, baselines


def collect(root, allow_api=False):
    root = Path(root); out = amendment_dir(root)
    parent, amendment, items, _ = load(root); cfg = parent['config']
    runner = original.SonnetRunner(cfg); cache = shared.cache_for(cfg, root)
    rows, errors = [], []
    def snapshot(stage):
        value = {'parent_study_id': parent['study_id'], 'amendment_id': amendment['amendment_id'],
            'stage': stage, 'updated_at_unix': time.time(), 'target': 50,
            'responses': len(rows), 'valid': sum(x['v2_ok'] for x in rows),
            'strict_valid': sum(x['strict_ok'] for x in rows),
            'invalid': dict(Counter(x['failure'] for x in rows if x['failure'])),
            'reused': sum(x['reused'] for x in rows),
            'new_calls_this_invocation': sum(not x['reused'] for x in rows),
            'errors': errors, 'budget': cache.summary(), 'confidence_calibrated': False}
        atomic_json(out/'status.json', value)
        return value
    with file_lock(root/'collection.lock'):
        if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['stage'] == 'blocked':
            raise ValueError('blocked; review before retrying')
        current_key = None; started = time.time()
        try:
            # Recheck the approved gate from cache without any network fallback.
            for item in items[:5]:
                call, _ = cache.get(runner, generator_request(item, parent['study_id']))
                original.priced_usage(call, cfg)
                if not parse_call(call).ok:
                    raise ValueError('cached v2 gate no longer passes')
            if allow_api:
                original.budget_audit(cfg, root)
                atomic_json(out/'billing_check.json', check_billing(cfg['project']))
            snapshot('running')
            pacer = Pacer(cfg['interval_seconds'])
            for item in items:
                req = generator_request(item, amendment['amendment_id'])
                current_key = runner.cache_key(req)
                try:
                    call, reused = cache.get(runner, req)
                except OfflineCacheMiss:
                    if not allow_api:
                        raise
                    pacer.acquire()
                    call, reused = cache.get(runner, req, allow_api=True)
                original.priced_usage(call, cfg)
                parsed = parse_call(call)
                rows.append({'v2_ok': parsed.ok, 'strict_ok': original.parse_call(call).ok,
                             'failure': parsed.failure, 'reused': reused})
                snapshot('running')
        except Exception as exc:
            error = {'type': type(exc).__name__, 'http_status': getattr(exc, 'status', None)}
            if isinstance(exc, (ValueError, OfflineCacheMiss)):
                error['message'] = str(exc)
            if current_key:
                for record in cache.log(current_key).records():
                    if record.get('error') and record.get('created', 0) >= started:
                        error['provider_error'] = record.get('meta', {}).get('provider_error')
            errors.append(error)
        value = snapshot('blocked' if errors else 'complete')
        shared.immutable(out/f'collection_{digest(value)[:12]}.json', value)
        return value


def report(root):
    """All fifty calls must exist; only development labels are then opened."""
    import numpy as np
    from vgx.gpqa.report import forecast, rate_interval, risk_coverage
    from vgx.gpqa.score import correctness_outcome
    root = Path(root); out = amendment_dir(root)
    parent, amendment, items, groups = load(root); cfg = parent['config']
    runner = original.SonnetRunner(cfg); cache = shared.cache_for(cfg, root)
    candidates, costs, formats = [], [], []
    for n, item in enumerate(items, 1):
        req = generator_request(item, amendment['amendment_id']); call, _ = cache.get(runner, req)
        costs.append(original.priced_usage(call, cfg)); parsed = parse_call(call)
        body = call.meta['provider_response']; usage = body.get('usage', {})
        formats.append({'pilot_position': n, 'strict_ok': original.parse_call(call).ok,
            'v2_ok': parsed.ok, 'failure': parsed.failure, 'stop_reason': body.get('stop_reason'),
            'thinking_block_n': sum(b.get('type') == 'thinking' for b in body.get('content', [])),
            'reported_thinking_tokens': usage.get('output_tokens_details', {}).get('thinking_tokens')})
        candidates.append(FrozenCandidate(item.item_id, item.partition, item.subject, item.question,
            item.choices, parsed.answer, parsed.p_correct, parsed.ok, parsed.failure, runner.model,
            call.key, direct.digest_text(req.prompt), direct.digest_text(call.response), SYSTEM,
            build_verifier_prompt(item, parsed.answer) if parsed.answer else None))
    groups['sonnet_5_5'] = candidates
    shared.immutable(out/'frozen_candidates.json', seal({'parent_study_id': parent['study_id'],
        'amendment_id': amendment['amendment_id'], 'parser_version': PARSER_VERSION,
        'groups': {g: [asdict(c) for c in cs] for g, cs in groups.items()}}, 'bundle_id'))
    bundle = Path(cfg['baseline_bundle'])
    manifest = json.loads((bundle/'bundle_manifest.json').read_text())
    path = bundle/'calibration_records.jsonl'
    if file_digest(path) != manifest['files'][path.name]:
        raise ValueError('development labels changed')
    labels = {x['item_id']: x['correct_index'] for x in read_jsonl(path)}
    summaries = {}
    for group, cs in groups.items():
        y = [correctness_outcome(c.answer, labels[c.item_id]) for c in cs]
        pairs = [(c.p_correct, t) for c, t in zip(cs, y) if c.answer is not None and c.p_correct is not None]
        p = [p for p, t in pairs]; yp = [t for p, t in pairs]
        summaries[group] = {'n': len(cs), 'valid_answer_n': sum(c.answer is not None for c in cs),
            'confidence_n': len(pairs), 'correct': sum(y), 'accuracy_all': float(np.mean(y)),
            'accuracy_wilson95': rate_interval(y), 'mean_confidence': float(np.mean(p)) if p else None,
            'confidence_values': dict(Counter(str(v) for v in p)),
            'mean_confidence_if_correct': float(np.mean([p for p,t in pairs if t])) if any(yp) else None,
            'mean_confidence_if_incorrect': float(np.mean([p for p,t in pairs if not t])) if 0 in yp else None,
            'raw_metrics': forecast(yp, p, seed=cfg['seed']),
            'risk_coverage': risk_coverage(yp, p, len(cs)),
            'failures': dict(Counter(c.generator_failure for c in cs if c.generator_failure))}
    spend = sum(x['estimated_usd'] for x in costs)
    value = {'parent_study_id': parent['study_id'], 'amendment_id': amendment['amendment_id'],
        'parser_version': PARSER_VERSION, 'generators': summaries,
        'sonnet_usage': {'input_tokens': sum(x['usage']['input_tokens'] for x in costs),
            'output_tokens_including_thinking': sum(x['usage']['output_tokens'] for x in costs),
            'estimated_usd': spend, 'first_five_estimated_usd': amendment['first_five_estimated_usd'],
            'continuation_estimated_usd': spend-amendment['first_five_estimated_usd'],
            'confirmed_billing_usd': None},
        'formats': formats, 'strict_valid_n': sum(x['strict_ok'] for x in formats),
        'v2_valid_n': sum(x['v2_ok'] for x in formats),
        'budget': cache.summary(), 'confidence_calibrated': False,
        'limitations': [
            'Exploratory original 50 development items; no Diamond collection or scoring.',
            'Parser was amended after observing the first five formats, with user approval and no access to correctness labels. Strict-format results are retained.',
            'The five original replies were reused exactly; only the next 45 requests were new.',
            'No confidence calibration, likelihood fitting or verifier queries.',
            'Historical Gemini and two-step Qwen have different generation settings and serving dates; model effects cannot be isolated.',
            'Recorded provider output usage, including thinking, determines estimated cost; not a confirmed invoice.']}
    shared.immutable(out/'comparison.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'collect', 'report'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    value = (prepare(args.root) if args.command == 'prepare' else
             collect(args.root, args.allow_api) if args.command == 'collect' else report(args.root))
    print(json.dumps(value, indent=2, allow_nan=False))
    if value.get('stage') == 'blocked':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
