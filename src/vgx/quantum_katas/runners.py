"""Model specifications for QuantumKatas roles, reusing the durable Vertex transports.

Every provider request goes through vgx.common.api.execute_request via the
existing runners, so request identity, uncertain-outcome blocking and usage
logging are unchanged. The mock provider exists for offline dry runs only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

from vgx.common.api import execute_request, request_key
from vgx.common.llm import CallLog, Request
from vgx.common.storage import digest
from vgx.common.vertex import VertexBatchRunner
from vgx.common.vertex_partner import VertexPartnerRunner

PROVIDERS = ('vertex_openai', 'vertex_anthropic', 'mock')
OPENAI_GENERATION = {'max_tokens', 'temperature', 'top_p', 'reasoning_effort'}
ANTHROPIC_GENERATION = {'max_tokens', 'temperature', 'thinking', 'output_config'}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    model: str
    location: str | None = None
    project: str | None = None
    generation: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id or self.provider not in PROVIDERS or not self.model:
            raise ValueError(f'invalid model specification: {self.id}')
        if not isinstance(self.generation.get('max_tokens'), int) or isinstance(self.generation.get('max_tokens'), bool) \
                or self.generation['max_tokens'] < 1:
            raise ValueError(f'explicit positive max_tokens required: {self.id}')
        if self.provider == 'mock':
            if set(self.generation) != {'max_tokens'}:
                raise ValueError('mock provider takes max_tokens only')
            return
        if not self.location or not self.project:
            raise ValueError(f'explicit Vertex location and project required: {self.id}')
        allowed = OPENAI_GENERATION if self.provider == 'vertex_openai' else ANTHROPIC_GENERATION
        if set(self.generation) - allowed:
            raise ValueError(f'unsupported generation settings for {self.provider}: {sorted(set(self.generation) - allowed)}')
        if self.provider == 'vertex_anthropic' and not self.model.startswith('anthropic/'):
            raise ValueError('vertex_anthropic requires an anthropic/ publisher model')

    @property
    def identity(self) -> str:
        return digest(asdict(self))

    def runner(self, mock_responses: dict | None = None):
        if self.provider == 'vertex_openai':
            return VertexBatchRunner(self.model, self.location, self.project, **self.generation)
        if self.provider == 'vertex_anthropic':
            return AnthropicVertexRunner(self.model, self.location, self.project, self.generation)
        return MockRunner(self.model, self.generation['max_tokens'], mock_responses or {})


class AnthropicVertexRunner(VertexPartnerRunner):
    """Claude on Vertex rawPredict with the reviewed generation body passed verbatim.

    VertexPartnerRunner only accepts max_tokens/temperature; thinking and
    output_config settings are carried here as part of identity and body.
    """

    def __init__(self, model, location, project, generation):
        super().__init__(model, location, project, max_tokens=generation['max_tokens'],
                         temperature=generation.get('temperature', 0.0))
        self.generation = json.loads(json.dumps(generation, allow_nan=False))

    @property
    def params(self):
        return json.loads(json.dumps(self.generation))

    @property
    def identity(self):
        return {**super().identity, 'quantum_katas_adapter': 1,
                'extended_thinking': 'thinking' in self.generation,
                'sampling_parameters': 'as_declared' if 'temperature' in self.generation else 'omitted_provider_defaults'}


class MockRunner:
    """Offline provider double keyed by exact prompt digest; never contacts a network."""

    location = 'offline'
    project = 'offline-mock'

    def __init__(self, model: str, max_tokens: int, responses: dict):
        self.model, self.max_tokens, self.responses = model, max_tokens, responses

    @property
    def params(self):
        return {'max_tokens': self.max_tokens}

    @property
    def identity(self):
        return {'schema': 1, 'provider': 'offline_mock', 'model': self.model, 'max_tokens': self.max_tokens}

    def cache_key(self, request: Request) -> str:
        return request_key(self.identity, request)

    response_text = staticmethod(VertexBatchRunner.response_text)

    def run(self, requests_to_run, log: CallLog):
        outputs = {}
        for request in requests_to_run:
            entry = self.responses.get(digest({'model': self.model, 'prompt': request.prompt, 'system': request.system}))

            def send():
                if entry is None:
                    return 404, {'error': {'code': 404, 'message': 'no mock response for this prompt'}}
                text, finish = entry['text'], entry.get('finish_reason', 'stop')
                prompt_tokens = len((request.prompt + (request.system or '')).encode()) // 4
                completion = max(1, len(text.encode()) // 4)
                return 200, {'id': 'mock-' + digest(request.prompt)[:16], 'model': self.model,
                             'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': finish}],
                             'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion + 7,
                                       'completion_tokens_details': {'reasoning_tokens': 7},
                                       'total_tokens': prompt_tokens + completion + 7}}
            outputs[request.key] = execute_request(self, request, log, send).response
        return outputs


def finish_status(call) -> str:
    """Provider-declared completion state of the final text."""
    body = (call.meta or {}).get('provider_response') or {}
    if 'stop_reason' in body:  # Anthropic Messages
        reason = body.get('stop_reason')
        return 'complete' if reason == 'end_turn' else ('truncated' if reason == 'max_tokens' else f'other:{reason}')
    try:
        reason = body['choices'][0].get('finish_reason')
    except (KeyError, IndexError, TypeError, AttributeError):
        return 'unknown'
    if reason == 'stop':
        return 'complete'
    return 'truncated' if reason in ('length', 'max_tokens') else f'other:{reason}'


def load_specs(config: dict) -> tuple[ModelSpec, tuple[ModelSpec, ...]]:
    generator = ModelSpec(**config['generator'])
    verifiers = tuple(ModelSpec(**v) for v in config['verifiers'])
    ids = [generator.id, *(v.id for v in verifiers)]
    if len(ids) != len(set(ids)) or not verifiers:
        raise ValueError('model spec IDs must be unique and at least one verifier is required')
    identities = [digest(v.runner().identity) for v in verifiers]
    if len(set(identities)) != len(identities):
        raise ValueError('two verifiers would send identical provider requests')
    return generator, verifiers


def call_record(call, reused: bool) -> dict:
    meta = call.meta or {}
    return {'request_key': call.key, 'provider_execution_id': meta.get('execution_id'),
            'response_id': meta.get('response_id'), 'provider_response_model': meta.get('provider_response_model'),
            'model_version': meta.get('model_version'), 'latency_s': call.latency_s,
            'usage': meta.get('usage'), 'used_cached_response': reused}
