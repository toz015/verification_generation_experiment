"""Re-send, once, the MATH-500 generator requests whose outcome was unknown (owner request 2026-10-21).

Same runner, prompt, system and parser as the frozen run; a fresh cache with its own ledger (USD 1 cap)
is used because the original cache keeps those requests blocked as uncertain. Only
`uncertain_request` rows are re-sent (format failures and truncations at temperature 0 are kept).
Writes frozen_candidates_complete.json next to the original bundle and lists the replaced items, so
results can be reported with and without this step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from vgx.common.api import ProviderRejectedError, UncertainRequestError
from vgx.common.budget import BudgetedRequestCache
from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa.sequential import seal, unseal
from vgx.math_trial import collect, prompt
from vgx.quantum_katas import runners
from vgx.quantum_katas.load import sha256_text
from vgx.quantum_katas.run_pilot import AbandonedRequests


def run(out, allow_api=False, attempt=1):
    out = Path(out)
    plan = json.loads((out/'plan.json').read_text()); unseal(plan, 'plan_id')
    # attempt 2 (owner request 2026-10-21): the still-unknown ones again, 900 s read timeout, own cache.
    src = 'frozen_candidates.json' if attempt == 1 else 'frozen_candidates_complete.json'
    bundle = json.loads((out/src).read_text()); unseal(bundle, 'bundle_id')
    suffix = '' if attempt == 1 else str(attempt)
    if attempt > 1:
        import requests as _requests
        original = _requests.Session.post
        def post(session, *a, **kw):
            kw['timeout'] = 900
            return original(session, *a, **kw)
        _requests.Session.post = post
    cfg = plan['config']; runner = collect.spec(cfg).runner(); b = cfg['budget']
    cache = BudgetedRequestCache(out/f'regen{suffix}_cache', project=b['project'], pricing=b['pricing'], limit_usd=1.0)
    items = {it['item_id']: it for rows in plan['sample'].values() for it in rows}
    rows = [dict(r) for r in bundle['rows']]; report = {}
    abandoned = AbandonedRequests(SimpleNamespace(root=out/f'regen{suffix}'))
    targets = {runner.cache_key(collect.request(items[r['item_id']], cfg['generation_id'], plan['prompt_version'])):
               {'role': 'generator', 'item_id': r['item_id']} for r in rows if r.get('failure') == 'uncertain_request'}
    abandoned.settle_prior(cache, targets)
    for i, r in enumerate(rows):
        if r.get('failure') != 'uncertain_request':
            continue
        req = collect.request(items[r['item_id']], cfg['generation_id'], plan['prompt_version'])
        key = runner.cache_key(req)
        if key in abandoned.keys():
            report[r['item_id']] = {'status': 'still_failed', 'error': 'UncertainRequestError'}; continue
        try:
            call, _ = cache.get(runner, req, allow_api=allow_api)
        except UncertainRequestError:
            abandoned.abandon(cache, key, targets[key])
            report[r['item_id']] = {'status': 'still_failed', 'error': 'UncertainRequestError'}; continue
        except ProviderRejectedError as exc:
            report[r['item_id']] = {'status': 'still_failed', 'error': type(exc).__name__}; continue
        finish = runners.finish_status(call)
        p = prompt.parse_generator_response(call.response, truncated=finish == 'truncated')
        new = {'item_id': r['item_id'], 'status': 'frozen', 'answer': p.answer, 'p_correct': p.p_correct,
               'failure': p.failure, 'confidence_status': p.confidence_status, 'finish_status': finish,
               'request_key': call.key, 'response_sha256': sha256_text(call.response), 'reused_cache': False,
               'latency_s': call.latency_s, 'usage': None, 'regenerated': True}
        rows[i] = new; report[r['item_id']] = {'status': 'replaced', 'failure': p.failure}
    value = seal({'plan_id': plan['plan_id'], 'source_bundle_id': bundle['bundle_id'], 'regen_sha256': file_digest(__file__),
                  'replaced': sorted(k for k, v in report.items() if v['status'] == 'replaced'), 'report': report,
                  'budget': cache.summary(), 'rows': rows}, 'bundle_id')
    atomic_json(out/f'frozen_candidates_complete{suffix}.json', value)
    return {k: value[k] for k in ('replaced', 'report', 'budget')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('results/math500_full_qwen3_20261021'))
    p.add_argument('--allow-api', action='store_true')
    p.add_argument('--attempt', type=int, default=1)
    a = p.parse_args()
    print(json.dumps(run(a.out, a.allow_api, a.attempt), indent=2, default=str))
