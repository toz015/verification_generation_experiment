"""Re-request the 7 unusable GPT-OSS-120B GPQA replies once (owner request, 2026-10-20).

The finish445 bundle had 5 replies with malformed final JSON ("p_correct": 0. nine)
and 2 provider aborts. Each of these inputs is sent once more with the identical
request (same runner, prompt, system, parser and prices) in a fresh cache with its
own ledger (USD 1 cap). A reply that parses replaces the unusable one in a new
"complete" bundle; one that still fails stays invalid. Originals are kept and the
replaced positions are listed, so results can be reported with and without this step.
No valid reply is ever re-requested.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from vgx.common.api import OfflineCacheMiss, ProviderRejectedError, UncertainRequestError
from vgx.common.budget import BudgetedRequestCache
from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa import gptoss_continuation as base
from vgx.gpqa import gptoss_finish445 as finish
from vgx.gpqa.fresh import generator_request
from vgx.gpqa.sequential import seal, unseal

ROOT = finish.ROOT
DIRECTORY = 'regen7_20261020'
REPORT = Path('reports/gpqa_gptoss_regen7_20261020')
SPACING = 30


def sources(root):
    plan, parent, items, runner, _ = finish.load(root)
    frozen = json.loads((root/finish.DIRECTORY/'frozen_candidates.json').read_text()); unseal(frozen, 'bundle_id')
    rows = frozen['groups']['gpt_oss_120b']
    targets = [p for p, c in enumerate(rows, 1) if c['answer'] is None or c['p_correct'] is None]
    return plan, parent, items, runner, frozen, rows, targets


def cache(root, parent):
    cfg = parent['config']
    return BudgetedRequestCache(root/DIRECTORY/'cache', project=cfg['project'], pricing=cfg['pricing'], limit_usd=1.0)


def prepare(root):
    out = root/DIRECTORY
    plan, parent, items, runner, frozen, rows, targets = sources(root)
    if len(targets) != 7:
        raise ValueError('expected exactly 7 unusable replies')
    value = seal({'finish_expansion_id': plan['expansion_id'], 'source_bundle_id': frozen['bundle_id'],
                  'implementation_sha256': file_digest(__file__), 'runner': runner.identity,
                  'target_positions': targets, 'original_failures': {str(p): rows[p-1]['generator_failure'] for p in targets},
                  'policy': 'one identical re-request per unusable reply; valid replies are never re-requested',
                  'authorization': '2026-10-20 owner asked to re-run the 5 invalid-JSON and 2 aborted replies for complete results',
                  'budget_cap_usd': 1.0, 'answer_keys_read': False}, 'regen_id')
    base.immutable(out/'plan.json', value)
    return value


def load(root):
    value = json.loads((root/DIRECTORY/'plan.json').read_text()); unseal(value, 'regen_id')
    plan, parent, items, runner, frozen, rows, targets = sources(root)
    if (value['implementation_sha256'] != file_digest(__file__) or value['source_bundle_id'] != frozen['bundle_id']
            or value['target_positions'] != targets or value['runner'] != runner.identity):
        raise ValueError('regeneration plan inputs changed')
    return value, parent, items, runner, frozen, rows


def collect(root, allow_api=False):
    value, parent, items, runner, frozen, rows = load(root)
    c = cache(root, parent); out = root/DIRECTORY; results = {}; last = 0.
    for pos in value['target_positions']:
        item = items[pos-1]; req = generator_request(item, value['regen_id'])
        try:
            call, reused = c.get(runner, req, allow_api=False)
        except OfflineCacheMiss:
            if not allow_api:
                results[str(pos)] = {'status': 'not_sent'}; continue
            time.sleep(max(0., last+SPACING-time.time()))
            try:
                call, reused = c.get(runner, req, allow_api=True)
            except (UncertainRequestError, ProviderRejectedError) as exc:
                last = time.time()
                results[str(pos)] = {'status': 'no_reply', 'error_type': type(exc).__name__,
                                     'http_status': getattr(exc, 'status', None)}
                continue
            last = time.time()
        except UncertainRequestError:
            results[str(pos)] = {'status': 'no_reply', 'error_type': 'UncertainRequestError'}; continue
        base.checked(call, runner, req, parent['config'])
        parsed = base.original.parse_call(call)
        results[str(pos)] = {'status': 'valid' if parsed.ok else 'still_invalid', 'failure': parsed.failure,
                             'reused_cache': reused}
    status = {'regen_id': value['regen_id'], 'results': results, 'budget': c.summary(),
              'stage': 'complete' if all(r['status'] != 'not_sent' for r in results.values()) else 'dry_run'}
    atomic_json(out/'status.json', status)
    if status['stage'] == 'complete':
        finalize(root)
    return status


def finalize(root):
    value, parent, items, runner, frozen, rows = load(root)
    c = cache(root, parent); complete = [dict(r) for r in rows]; replaced = []
    for pos in value['target_positions']:
        item = items[pos-1]; req = generator_request(item, value['regen_id'])
        try:
            call, _ = c.get(runner, req)
        except (OfflineCacheMiss, UncertainRequestError):
            continue
        cand = finish.candidate(item, req, call)
        if cand['answer'] is not None and cand['p_correct'] is not None:
            complete[pos-1] = cand; replaced.append(pos)
    bundle = seal({'expansion_id': frozen['expansion_id'], 'parent_expansion_id': frozen['parent_expansion_id'],
                   'dataset_id': frozen['dataset_id'], 'source_bundle_id': frozen['bundle_id'],
                   'regen_id': value['regen_id'], 'replaced_positions': replaced,
                   'groups': {'gpt_oss_120b': complete}}, 'bundle_id')
    base.immutable(root/DIRECTORY/'frozen_candidates.json', bundle)
    report = {'regen_id': value['regen_id'], 'bundle_id': bundle['bundle_id'], 'replaced_positions': replaced,
              'still_unusable': [p for p in value['target_positions'] if p not in replaced],
              'valid_total': sum(r['answer'] is not None and r['p_correct'] is not None for r in complete),
              'budget': c.summary(), 'confirmed_billing_usd': None}
    base.immutable(root/DIRECTORY/'completion_report.json', report)
    REPORT.mkdir(parents=True, exist_ok=True)
    for name in ('plan.json', 'status.json', 'completion_report.json'):
        (REPORT/name).write_text((root/DIRECTORY/name).read_text())
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'collect', 'finalize'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--allow-api', action='store_true')
    a = p.parse_args()
    v = prepare(a.root) if a.command == 'prepare' else collect(a.root, a.allow_api) if a.command == 'collect' else finalize(a.root)
    print(json.dumps(v, indent=2, allow_nan=False, default=str)[:3000])


if __name__ == '__main__':
    main()
