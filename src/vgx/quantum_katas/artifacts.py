"""Frozen code candidates with provenance, and the whitelisted verifier view.

A candidate row is written once and never revised. Its identity binds the task,
generation, provider request, raw response and code; later verification cannot
change it. Raw r and any later calibrated prior are separate fields/files.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from vgx.common.storage import append_jsonl, digest, file_lock, read_jsonl
from vgx.quantum_katas.load import PARTITIONS, PublicTask, sha256_text

CONFIDENCE_STATUSES = ('valid', 'missing', 'non_numeric', 'boolean', 'non_finite', 'out_of_range', 'not_applicable')


@dataclass(frozen=True)
class FrozenCodeCandidate:
    task_id: str
    generation_id: str
    candidate_id: str
    partition: str
    raw_code: str | None
    code_sha256: str | None
    raw_p_correct: float | None
    generator_response_status: str      # 'code_extracted' | 'format_failure'
    generator_failure: str | None
    confidence_status: str
    finish_status: str
    model_identity: dict
    request_key: str
    provider_execution_id: str | None
    prompt_sha256: str
    system_sha256: str
    raw_response_sha256: str
    prompt_version: str
    dataset_revision: str
    task_prompt_sha256: str

    def __post_init__(self):
        if self.partition not in PARTITIONS:
            raise ValueError(f'unknown candidate partition: {self.partition}')
        if self.generator_response_status not in ('code_extracted', 'format_failure'):
            raise ValueError('invalid generator response status')
        if self.confidence_status not in CONFIDENCE_STATUSES:
            raise ValueError('invalid confidence status')
        has_code = self.generator_response_status == 'code_extracted'
        if has_code != (self.raw_code is not None) or has_code == (self.generator_failure is not None):
            raise ValueError('code presence must match the response status')
        if (self.raw_code is None) != (self.code_sha256 is None) or (
                self.raw_code is not None and self.code_sha256 != sha256_text(self.raw_code)):
            raise ValueError('code hash mismatch')
        if (self.raw_p_correct is not None) != (self.confidence_status == 'valid'):
            raise ValueError('raw confidence must be present exactly when valid')
        if self.raw_p_correct is not None and (isinstance(self.raw_p_correct, bool) or not 0 <= self.raw_p_correct <= 1):
            raise ValueError('invalid raw confidence')
        if self.candidate_id != candidate_identity(self.task_id, self.generation_id, self.request_key,
                                                   self.raw_response_sha256, self.code_sha256):
            raise ValueError('candidate identity does not match its provenance')

    @property
    def has_code(self) -> bool:
        return self.raw_code is not None

    @property
    def has_valid_confidence(self) -> bool:
        return self.raw_p_correct is not None


def candidate_identity(task_id, generation_id, request_key, raw_response_sha256, code_sha256) -> str:
    return digest({'schema': 1, 'task_id': task_id, 'generation_id': generation_id, 'request_key': request_key,
                   'raw_response_sha256': raw_response_sha256, 'code_sha256': code_sha256})


def verifier_view(candidate: FrozenCodeCandidate, task: PublicTask) -> dict:
    """The only fields a verifier prompt may depend on (no r, model, label or provenance)."""
    if not candidate.has_code or candidate.task_id != task.task_id:
        raise ValueError('verifier view requires extractable code for the same task')
    return {'task_id': task.task_id, 'prompt': task.prompt, 'entry_point': task.entry_point,
            'required_functions': list(task.required_functions), 'code': candidate.raw_code}


class CandidateStore:
    """Append-only frozen_candidates.jsonl; one row per (task_id, generation_id)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def all(self) -> list[FrozenCodeCandidate]:
        rows = [FrozenCodeCandidate(**r) for r in read_jsonl(self.path)]
        keys = [(c.task_id, c.generation_id) for c in rows]
        if len(keys) != len(set(keys)) or len({c.candidate_id for c in rows}) != len(rows):
            raise ValueError('duplicate frozen candidates')
        return rows

    def by_task(self) -> dict[str, FrozenCodeCandidate]:
        return {c.task_id: c for c in self.all()}

    def freeze(self, candidate: FrozenCodeCandidate) -> bool:
        """Return True if newly frozen; an identical existing row is reused, a different one is refused."""
        with file_lock(str(self.path) + '.lock'):
            for existing in self.all():
                if (existing.task_id, existing.generation_id) == (candidate.task_id, candidate.generation_id):
                    if asdict(existing) != asdict(candidate):
                        raise ValueError(f'frozen candidate for {candidate.task_id} cannot be revised')
                    return False
            append_jsonl(self.path, asdict(candidate))
            return True


def read_rows(path) -> list[dict]:
    return read_jsonl(path)


def write_jsonl_once(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
