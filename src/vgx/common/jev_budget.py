"""Separate, conservative Jev usage budget; no inference retries or Vertex calls."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

from vgx.common.api import OfflineCacheMiss, RequestCache
from vgx.common.billing import estimate_call
from vgx.common.budget import BudgetLimitError
from vgx.common.storage import atomic_json, digest, file_lock


def load_jev_key(path=Path('.env')):
    """Read one named credential without executing shell syntax or logging values."""
    if os.environ.get('TYPESAFE_API_KEY'):
        return
    path = Path(path)
    values = []
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip().removeprefix('export ').strip()
            name, sep, value = line.partition('=')
            if sep and name.strip() == 'TYPESAFE_API_KEY':
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                values.append(value)
    if len(values) != 1 or not values[0] or any(c.isspace() for c in values[0]):
        raise ValueError('One nonempty TYPESAFE_API_KEY is required; credential value omitted')
    os.environ['TYPESAFE_API_KEY'] = values[0]


class JevBudgetCache(RequestCache):
    """Reserve a full 64k input context per request at the saved input price.

    Calls are serialized. Rejections and unknown outcomes retain their full
    reservation and block further collection, since billing is not confirmed.
    The ledger is an estimate and does not change the provider's account limit.
    """
    def __init__(self, root, *, pricing, limit_usd, model, account_scope):
        super().__init__(root)
        if isinstance(limit_usd, bool) or not isinstance(limit_usd, (int, float)) or not math.isfinite(limit_usd) or limit_usd <= 0:
            raise ValueError('positive finite Jev budget required')
        rate = pricing.get('usd_per_million_tokens', {}).get(model, {})
        if (isinstance(rate.get('input'), bool) or not isinstance(rate.get('input'), (int, float))
                or not math.isfinite(rate['input']) or rate['input'] <= 0 or rate.get('output') != 0):
            raise ValueError('confirmed positive input rate and free output required')
        self.pricing, self.limit, self.model, self.account_scope = pricing, limit_usd, model, account_scope
        self.reserve = 64000 * rate['input'] / 1_000_000
        self.path = self.root/'jev_budget.json'
        self.binding = {'provider': 'typesafe_systemone', 'model': model, 'account_scope': account_scope,
                        'limit_usd': limit_usd, 'input_reserve_tokens': 64000, 'pricing_sha256': digest(pricing)}

    def _load(self):
        if not self.path.exists():
            return {'binding': self.binding, 'requests': {}, 'confirmed_billing_usd': None}
        state = json.loads(self.path.read_text())
        if state['binding'] != self.binding:
            raise ValueError('Jev budget, account or pricing binding changed')
        return state

    def _settle(self, state, key, call):
        if key not in state['requests']:
            return
        estimate = estimate_call(asdict(call), self.pricing)
        entry = state['requests'][key]
        usd = estimate['estimated_usd']
        status = 'unpriced' if usd is None else ('exceeded_reservation' if usd > entry['reserved_usd'] else 'settled')
        entry.update(estimated_usd=usd, execution_id=call.meta.get('execution_id'), status=status)
        atomic_json(self.path, state)
        if status != 'settled':
            raise BudgetLimitError('Jev usage unpriced or exceeded reservation; review before further calls')

    def get(self, runner, request, *, allow_api=False):
        identity = runner.identity
        if (identity.get('provider') != 'typesafe_systemone' or runner.model != self.model
                or identity.get('account_scope') != self.account_scope
                or identity.get('endpoint') != 'https://api.typesafe.ai/v1/systemone'):
            raise ValueError('request outside the authorized Jev account/model/endpoint')
        key = runner.cache_key(request)
        with file_lock(str(self.path)+'.lock'):
            state = self._load()
            try:
                call, _ = super().get(runner, request, allow_api=False)
                self._settle(state, key, call)
                return call, True
            except OfflineCacheMiss:
                if not allow_api:
                    raise
            if any(e['status'] != 'settled' for e in state['requests'].values()):
                raise BudgetLimitError('unresolved Jev request or unpriced usage; automatic retry disabled')
            committed = sum(e['estimated_usd'] for e in state['requests'].values())
            if committed + self.reserve > self.limit:
                raise BudgetLimitError('insufficient Jev budget for next request reservation')
            state['requests'][key] = {'status': 'reserved', 'reserved_usd': self.reserve,
                                      'estimated_usd': None, 'operation_id': request.meta.get('operation_id')}
            atomic_json(self.path, state)
            call, cached = super().get(runner, request, allow_api=True)
            self._settle(state, key, call)
            return call, cached

    def summary(self):
        with file_lock(str(self.path)+'.lock'):
            state = self._load()
        rows = list(state['requests'].values())
        return {'limit_usd': self.limit, 'successful_priced_calls': sum(e['status']=='settled' for e in rows),
                'estimated_usd': sum(e['estimated_usd'] for e in rows if e['status']=='settled'),
                'outstanding_reserved_usd': sum(e['reserved_usd'] for e in rows if e['status']!='settled'),
                'confirmed_billing_usd': None}
