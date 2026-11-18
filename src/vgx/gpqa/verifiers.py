"""Frozen verifier specifications and signal extraction; no generator adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math

from vgx.common.jev import JevRunner
from vgx.common.llm import Request
from vgx.common.storage import canonical, digest
from vgx.common.vertex import VertexBatchRunner
from vgx.gpqa.artifacts import FrozenCandidate
from vgx.gpqa.prompt import parse_verifier_response

JEV_CHOICE_INSTRUCTION = 'Which answer option correctly answers this question?'
JEV_NOUL_INSTRUCTION = 'Is the fixed candidate answer correct for this question?'


@dataclass(frozen=True)
class VerifierSpec:
    id: str
    provider: str
    model: str
    mode: str = 'fixed_candidate'
    location: str | None = None
    project: str | None = None
    generation: dict = field(default_factory=dict)
    prompt_version: str = 'frozen_original'
    account_scope: str = 'default'

    def __post_init__(self):
        if not self.id or self.provider not in ('vertex_ai', 'typesafe'):
            raise ValueError('invalid verifier specification')
        if self.provider == 'vertex_ai':
            if self.mode != 'fixed_candidate' or self.prompt_version != 'frozen_original' or not self.location or not self.project:
                raise ValueError('Vertex verifier must use frozen original prompts and an explicit endpoint')
            if 'max_tokens' not in self.generation or set(self.generation) - {'max_tokens', 'temperature', 'top_p', 'reasoning_effort'}:
                raise ValueError('explicit supported Vertex generation settings required')
        elif self.mode not in ('choice', 'noul') or self.prompt_version != 'gpqa_jev_v1':
            raise ValueError('unsupported Jev signal mode or prompt version')

    @property
    def identity(self):
        value = asdict(self)
        if self.provider == 'typesafe':
            value['instructions'] = JEV_CHOICE_INSTRUCTION if self.mode == 'choice' else JEV_NOUL_INSTRUCTION
        return digest(value)

    def runner(self):
        if self.provider == 'typesafe':
            return JevRunner(self.model, self.account_scope)
        return VertexBatchRunner(self.model, self.location, self.project, **self.generation)

    def request(self, candidate: FrozenCandidate, operation_id: str | None = None) -> Request:
        if candidate.answer is None:
            raise ValueError('cannot verify a missing frozen answer')
        meta = {'item_id': candidate.item_id, 'partition': candidate.partition,
                'role': 'verifier', 'candidate': candidate.answer, 'verifier_id': self.id,
                'operation_id': operation_id}
        if self.provider == 'vertex_ai':
            return Request(f'{self.id}|{candidate.item_id}', candidate.verifier_prompt, candidate.system, meta)
        options = dict(zip('ABCD', candidate.choices))
        state = {'subject': candidate.subject, 'question': candidate.question, 'options': options}
        if self.mode == 'choice':
            question = {'type': 'choice', 'instructions': JEV_CHOICE_INSTRUCTION, 'criteria': options}
        else:
            state['fixed_candidate'] = {'option': candidate.answer, 'text': options[candidate.answer]}
            question = {'type': 'noul', 'instructions': JEV_NOUL_INSTRUCTION}
        return Request(f'{self.id}|{candidate.item_id}', canonical({'state': state, 'questions': {'verification': question}}), None, meta)

    def signal(self, response: str, candidate: FrozenCandidate) -> tuple[float | None, str | None]:
        if self.provider == 'vertex_ai':
            result = parse_verifier_response(response)
            return result.p_correct, result.failure
        try:
            payload = json.loads(response)
            if payload.get('model') != self.model:
                return None, 'model_revision_mismatch'
            answer = payload['answers']['verification']
            if answer['type'] != self.mode:
                raise ValueError('wrong signal type')
            if self.mode == 'choice':
                probabilities = answer['probabilities']
                if set(probabilities) != set('ABCD') or any(not _prob(p) for p in probabilities.values()):
                    raise ValueError('invalid option probabilities')
                if not math.isclose(sum(probabilities.values()), 1., abs_tol=1e-6):
                    raise ValueError('option probabilities do not sum to one')
                score = probabilities[candidate.answer]  # never confidence or argmax
            else:
                score = answer['noul']
            if not _prob(score):
                raise ValueError('invalid probability')
            return float(score), None
        except (ValueError, KeyError, TypeError, AttributeError):
            return None, 'invalid_jev_signal'


def _prob(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def load_specs(value: dict) -> tuple[VerifierSpec, ...]:
    if value.get('schema') != 1:
        raise ValueError('unsupported verifier-spec schema')
    specs = tuple(VerifierSpec(**v) for v in value['verifiers'])
    if len({s.id for s in specs}) != len(specs) or not specs:
        raise ValueError('verifier IDs must be unique and nonempty')
    for spec in specs:
        spec.runner()  # validate provider identity without authentication or calls
    return specs
