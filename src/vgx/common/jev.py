"""TypeSafe System One transport; credentials are used only for explicit API calls."""
from __future__ import annotations

import json
import os
import re

from vgx.common.api import execute_request, request_key
from vgx.common.llm import CallLog, Request
from vgx.common.storage import canonical


class JevRunner:
    def __init__(self, model: str, account_scope: str = 'default'):
        if not re.fullmatch(r'jev-\d+\.\d+\.\d+', model):
            raise ValueError('pin Jev to an explicit version, not latest')
        self.model, self.account_scope = model, account_scope

    @property
    def identity(self):
        return {'schema': 1, 'provider': 'typesafe_systemone', 'model': self.model,
                'endpoint': 'https://api.typesafe.ai/v1/systemone', 'account_scope': self.account_scope}

    @property
    def params(self):
        return {}

    def cache_key(self, request):
        return request_key(self.identity, request)

    @staticmethod
    def response_text(result):
        return canonical(result)

    def run(self, requests_to_run: list[Request], log: CallLog):
        if len({r.key for r in requests_to_run}) != len(requests_to_run):
            raise ValueError('request keys must be unique')
        outputs = {}
        for request in requests_to_run:
            cached = log.responses().get(self.cache_key(request))
            if cached is not None:
                outputs[request.key] = cached
                continue
            if request.system is not None:
                raise ValueError('Jev requests use structured state/questions, no system message')
            body = json.loads(request.prompt)
            if set(body) != {'state', 'questions'} or len(body['questions']) != 1:
                raise ValueError('one verifier signal per Jev request; speculative fan-out disabled')
            token = os.environ.get('TYPESAFE_API_KEY')
            if not token:
                raise RuntimeError('TYPESAFE_API_KEY is not configured')
            import requests
            session = requests.Session()

            def send():
                response = session.post(self.identity['endpoint'],
                    headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
                    json={**body, 'model': self.model}, timeout=300)
                try:
                    payload = response.json()
                except ValueError:
                    payload = {'invalid_response': True}
                return response.status_code, payload

            try:
                outputs[request.key] = execute_request(self, request, log, send).response
            finally:
                if hasattr(session, 'close'):
                    session.close()
        return outputs
