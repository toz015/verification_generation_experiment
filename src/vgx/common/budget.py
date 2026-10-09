"""A local usage-estimate budget; this is not a provider billing limit."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from vgx.common.api import OfflineCacheMiss, ProviderRejectedError, RequestCache
from vgx.common.billing import estimate_call
from vgx.common.storage import atomic_json, digest, file_lock, read_jsonl


class BudgetLimitError(RuntimeError):
    pass


class BudgetedRequestCache(RequestCache):
    """Reserve costs before sending; reconcile only the selected request's log.

    Concurrent requests share the reservation ledger. Unknown outcomes retain
    their reservation. No other candidate/verifier response is opened here.
    """

    def __init__(self, root, *, project, pricing, limit_usd, gemini_output_reserve=131072):
        super().__init__(root)
        if isinstance(limit_usd,bool) or not isinstance(limit_usd,(int,float)) or not math.isfinite(limit_usd) or limit_usd <= 0:
            raise ValueError('positive finite budget required')
        if not isinstance(project,str) or not project:
            raise ValueError('explicit billing project required')
        if isinstance(gemini_output_reserve,bool) or not isinstance(gemini_output_reserve,int) or gemini_output_reserve<1:
            raise ValueError('positive integer output reservation required')
        self.project, self.pricing, self.limit = project, pricing, limit_usd
        self.output_reserve = gemini_output_reserve
        self.path = Path(root)/'budget.json'
        self.binding = {'project': project, 'limit_usd': limit_usd,
                        'pricing_sha256': digest(pricing),
                        'gemini_output_reserve': gemini_output_reserve}

    def _load(self):
        if self.path.exists():
            value = json.loads(self.path.read_text())
            if value['binding'] != self.binding:
                raise ValueError('budget/project/pricing changed; review the existing ledger')
            return value
        return {'binding': self.binding, 'requests': {},
                'basis': 'provider usage estimates, not confirmed billing',
                'confirmed_billing_usd': None}

    def reservation(self, runner, request):
        if getattr(runner, 'project', None) != self.project:
            raise ValueError('provider request does not target the authorized project')
        rate = self.pricing['usd_per_million_tokens'][runner.model]
        if any(isinstance(rate.get(k),bool) or not isinstance(rate.get(k),(int,float))
               or not math.isfinite(rate[k]) or rate[k]<0 for k in ('input','output')):
            raise ValueError('finite nonnegative input/output prices required before collection')
        # UTF-8 bytes plus framing is a conservative text input allowance.
        # Reserve a generous reasoning/output allowance for Gemini instead of
        # assuming the visible max_tokens also bounds internal reasoning.
        inputs = len((request.prompt + (request.system or '')).encode()) + 1024
        outputs = max(runner.max_tokens, self.output_reserve) if 'gemini' in runner.model else runner.max_tokens
        return (inputs*rate['input'] + outputs*rate['output']) / 1_000_000

    def _settle(self, key):
        rows = [r for r in self.log(key).records() if r.get('key') == key and r.get('error') is None]
        with file_lock(str(self.path)+'.lock'):
            state = self._load()
            entry = state['requests'].get(key)
            if entry is None:
                return  # response predates this budget's paid collection
            previous = dict(entry)
            if rows:
                estimate = estimate_call(rows[0], self.pricing)
                entry.update(estimated_usd=estimate['estimated_usd'], issues=estimate['issues'],
                             execution_id=rows[0].get('meta', {}).get('execution_id'),
                             status='settled' if estimate['estimated_usd'] is not None else 'unpriced')
            else:
                attempts=read_jsonl(str(self.log(key).path)+'.attempts.jsonl',tolerate_tail=True)
                started={e['execution_id'] for e in attempts if e.get('event')=='started'}
                rejected={e['execution_id'] for e in attempts if e.get('event')=='rejected'}
                if started and started <= rejected:
                    # Google explicitly states non-200 responses are not
                    # charged. This cache is restricted to the Vertex project.
                    entry.update(status='rejected',estimated_usd=0.)
                elif started:
                    entry.update(status='unresolved',estimated_usd=None)
            if entry != previous:
                atomic_json(self.path, state)
            if entry['status'] == 'unpriced':
                raise BudgetLimitError('provider usage is unpriced; pause collection for reconciliation')
            if rows and entry['estimated_usd'] > entry['reserved_usd']:
                raise BudgetLimitError('request exceeded its reservation; review the budget before continuing')

    def _get_once(self, runner, request, *, allow_api=False):
        # Even a cached response must belong to the authorized project.
        reserve = self.reservation(runner, request)
        key = runner.cache_key(request)
        try:
            result = super().get(runner, request, allow_api=False)
            self._settle(key)
            return result
        except OfflineCacheMiss:
            if not allow_api:
                raise
        # Serialize duplicates of this exact request while other requests run.
        with file_lock(self.root/'budget-request-locks'/f'{digest(key)}.lock'):
            try:
                result = super().get(runner, request, allow_api=False)
                self._settle(key)
                return result
            except OfflineCacheMiss:
                pass
            with file_lock(str(self.path)+'.lock'):
                state = self._load()
                if any(e['status'] in ('unpriced','unresolved') for e in state['requests'].values()):
                    raise BudgetLimitError('unpriced or unresolved execution requires reconciliation')
                if key not in state['requests'] or state['requests'][key]['status']=='rejected':
                    committed = sum(e.get('estimated_usd') if e['status'] in ('settled','rejected') else e['reserved_usd']
                                    for e in state['requests'].values())
                    if committed + reserve > self.limit:
                        raise BudgetLimitError('insufficient budget for the next request reservation')
                    state['requests'][key] = {'status': 'reserved', 'reserved_usd': reserve,
                                              'model': runner.model, 'role': request.meta.get('role')}
                    atomic_json(self.path, state)
            try:
                return super().get(runner, request, allow_api=True)
            finally:
                self._settle(key)

    def get(self, runner, request, *, allow_api=False):
        def attempt():
            # Only explicit HTTP 429 rejections are retried. Lost responses,
            # timeouts, unpriced usage, and other HTTP errors are never retried.
            for number in range(4):
                try:
                    return self._get_once(runner,request,allow_api=allow_api)
                except ProviderRejectedError as exc:
                    if exc.status!=429 or number==3:
                        raise
                    time.sleep(min(5*2**number,20))
        if allow_api and runner.model.startswith('meta/'):
            # The managed Llama endpoint rejected the initial concurrent burst.
            # Serialize this model, including backoff, across processes.
            with file_lock(self.root/'provider-locks'/f'{digest(runner.model)}.lock'):
                return attempt()
        return attempt()

    def summary(self):
        with file_lock(str(self.path)+'.lock'):
            state = self._load()
        entries = list(state['requests'].values())
        return {'project': self.project, 'limit_usd': self.limit,
                'successful_priced_calls': sum(e['status'] == 'settled' for e in entries),
                'estimated_usd': sum(e['estimated_usd'] for e in entries if e['status'] == 'settled'),
                'rejected_requests':sum(e['status']=='rejected' for e in entries),
                'outstanding_reserved_usd': sum(e['reserved_usd'] for e in entries if e['status'] not in ('settled','rejected')),
                'confirmed_billing_usd': None}
