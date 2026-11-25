"""Pinned QuantumKatas sources, the fixed 28-task pilot, and the public/private split.

Tasks are keyed by task_id: entry_point repeats across the 350 upstream tasks.
PublicTask never carries canonical_solution or test; OracleTask does and is
written only under the run's private/ directory.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from vgx.common.storage import file_digest

REVISION = '3790fb3e9892537b550c884ad54b65c73c13f1ad'
DATASET = 'qiskit-community/Qiskit-QuantumKatas'
SOURCE_SHA256 = {
    'dataset/qiskit_quantumkatas.jsonl': '03abb872a87bc62c36c9ea453659cb741d7a8f4892a4c1b52c39a08f882e7b8c',
    'benchmark/evaluator.py': 'f91f4f6da9efbdb8fb88dbbd964979a9272119fb517c844a6680cedceb1b1e2e',
    'uv.lock': 'd7608adaa5bbec1354bd1ffcc4a22ec5da3d779706e362b4fc63e8267c50a3c8',
    'LICENSE': 'e6f27c58c270f578b26ec807c965a6aa2d3e6e6d77cc8004e34b88d47fc62dd3',
}
FIELDS = ('task_id', 'prompt', 'canonical_solution', 'test', 'entry_point')
PILOT_TASK_COUNT = 28
PARTITIONS = ('development',) + tuple(f'fold_{k}' for k in range(5))
DEFAULT_SOURCE = Path('tmp/quantum_four_dataset_review_20261018/Qiskit-QuantumKatas')
DEFAULT_PILOT = Path('configs/quantum_katas_pilot_20261018.json')
DEFAULT_PROTOCOL = Path('configs/quantum_katas_official_protocol_20261018.json')
DEFAULT_REQUIREMENTS = Path('configs/quantum_katas_pilot_requirements.txt')


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    revision: str
    partition: str
    group_id: str
    entry_point: str
    required_functions: tuple[str, ...]
    prompt: str
    prompt_sha256: str

    def __post_init__(self):
        if self.partition not in PARTITIONS:
            raise ValueError(f'unknown partition: {self.partition}')
        if self.prompt_sha256 != sha256_text(self.prompt):
            raise ValueError(f'public prompt hash mismatch: {self.task_id}')
        if self.entry_point not in self.required_functions:
            raise ValueError(f'entry point is not a required function: {self.task_id}')

    def to_row(self) -> dict:
        return {**asdict(self), 'required_functions': list(self.required_functions)}

    @classmethod
    def from_row(cls, row: dict) -> 'PublicTask':
        if set(row) != {f for f in cls.__dataclass_fields__}:
            raise ValueError('public task row has unexpected or missing fields')
        return cls(**{**row, 'required_functions': tuple(row['required_functions'])})


@dataclass(frozen=True)
class OracleTask:
    task_id: str
    entry_point: str
    test: str
    canonical_solution: str
    test_sha256: str
    canonical_solution_sha256: str

    def __post_init__(self):
        if (self.test_sha256 != sha256_text(self.test)
                or self.canonical_solution_sha256 != sha256_text(self.canonical_solution)):
            raise ValueError(f'oracle hash mismatch: {self.task_id}')


def required_functions(prompt: str) -> tuple[str, ...]:
    """Top-level function stubs in the upstream prompt, e.g. both Alice and Bob in 2.6."""
    tree = ast.parse(prompt)
    names = tuple(n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    if len(set(names)) != len(names) or not names:
        raise ValueError('prompt must declare unique top-level function stubs')
    return names


def verify_sources(source: str | Path) -> dict[str, str]:
    source = Path(source)
    for rel, expected in SOURCE_SHA256.items():
        if file_digest(source/rel) != expected:
            raise ValueError(f'pinned source hash mismatch: {rel}')
    return dict(SOURCE_SHA256)


def load_pilot_ids(pilot_config: str | Path) -> tuple[list[str], dict]:
    config = json.loads(Path(pilot_config).read_text())
    ids = config['task_ids']
    if (config.get('revision') != REVISION or config.get('dataset') != DATASET
            or config.get('partition') != 'development'):
        raise ValueError('pilot config does not target the pinned development revision')
    if len(ids) != PILOT_TASK_COUNT or len(set(ids)) != len(ids):
        raise ValueError('pilot requires exactly 28 unique task IDs')
    return ids, config


def load_tasks(source: str | Path, pilot_config: str | Path) -> tuple[list[PublicTask], list[OracleTask]]:
    """Return the fixed pilot in config order; no task is dropped or edited."""
    ids, _ = load_pilot_ids(pilot_config)
    return _project(source, [(tid, 'development') for tid in ids])


def load_split_tasks(source: str | Path, split_path: str | Path) -> tuple[list[PublicTask], list[OracleTask]]:
    """All 322 non-development tasks, ordered by fold then task_id, partition fold_k."""
    from vgx.quantum_katas import split as split_rules
    value = json.loads(Path(split_path).read_text())
    if value != split_rules.build(source):
        raise ValueError('pre-registered split file does not reproduce from its rules')
    order = sorted(value['task_fold'].items(), key=lambda kv: (kv[1], kv[0]))
    return _project(source, [(tid, f'fold_{fold}') for tid, fold in order])


def _project(source, assignments) -> tuple[list[PublicTask], list[OracleTask]]:
    verify_sources(source)
    ids = [tid for tid, _ in assignments]
    rows = [json.loads(line) for line in (Path(source)/'dataset/qiskit_quantumkatas.jsonl').read_text().splitlines()
            if line.strip()]
    by_id = {}
    for row in rows:
        if set(row) != set(FIELDS) or not all(isinstance(row[f], str) and row[f] for f in FIELDS):
            raise ValueError(f'upstream row with missing or invalid fields: {row.get("task_id")}')
        if row['task_id'] in by_id:
            raise ValueError(f'duplicate upstream task_id: {row["task_id"]}')
        by_id[row['task_id']] = row
    missing = [t for t in ids if t not in by_id]
    if missing:
        raise ValueError(f'pilot task IDs missing upstream: {missing}')
    public, oracle = [], []
    for tid, partition in assignments:
        row = by_id[tid]
        public.append(PublicTask(tid, REVISION, partition, tid.split('/')[0], row['entry_point'],
                                 required_functions(row['prompt']), row['prompt'], sha256_text(row['prompt'])))
        oracle.append(OracleTask(tid, row['entry_point'], row['test'], row['canonical_solution'],
                                 sha256_text(row['test']), sha256_text(row['canonical_solution'])))
    return public, oracle


def read_public_tasks(path: str | Path) -> list[PublicTask]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    tasks = [PublicTask.from_row(r) for r in rows]
    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError('duplicate public task_id')
    return tasks


def read_oracle_tasks(path: str | Path) -> dict[str, OracleTask]:
    rows = [OracleTask(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({r.task_id for r in rows}) != len(rows):
        raise ValueError('duplicate oracle task_id')
    return {r.task_id: r for r in rows}
