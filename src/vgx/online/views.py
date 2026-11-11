"""Learner and evaluator views of the cached data.

The learner gets PublicItem records (task id, fold, group, raw r, whether code exists) and a
ScoreSource that hands out cached verifier outputs only through per-round access objects that log
every read. Labels live in a separate LabelStore that only the evaluator reads; the labeled
reference arm D gets a LabelGate that releases a label only for an audited round of the current
item, and the initializer gets the labels of D_init only.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path

import numpy as np

BINS = 3
EDGES = tuple(np.linspace(0., 1., BINS+1).tolist())
CODES = np.array([0., .5, 1.])  # audit score coding A in {0, 1/2, 1}; Z-hat is invariant to affine recoding


def bin_of(score: float) -> int:
    """Equal-width bins on [0,1], identical to vgx.gpqa.dependence_triples.bin_of."""
    index = int(np.searchsorted(EDGES, score, side='right') - 1)
    return min(max(index, 0), BINS-1)


@dataclass(frozen=True)
class PublicItem:
    task_id: str
    fold: int
    group: str
    raw_r: float | None
    has_code: bool

    @property
    def processable(self) -> bool:
        """Public rule: extractable code and a valid raw confidence. No label information."""
        r = self.raw_r
        return bool(self.has_code and isinstance(r, (int, float)) and math.isfinite(r) and 0 <= r <= 1)


class AccessError(RuntimeError):
    pass


class ScoreSource:
    """Cached verifier outputs, readable only through an open RoundAccess."""

    def __init__(self, signals: dict[str, dict], verifiers: tuple[str, ...]):
        self._signals = signals
        self.verifiers = tuple(verifiers)
        self._open = None

    def open(self, task_id: str, purpose: str):
        if self._open is not None:
            raise AccessError('another round is still open')
        self._open = RoundAccess(self, task_id, purpose)
        return self._open

    def _close(self, access):
        if self._open is not access:
            raise AccessError('closing a round that is not open')
        self._open = None

    def _read(self, task_id, verifier):
        s = self._signals[task_id][verifier]
        score = s['score']
        return {'bin': None if score is None else bin_of(score), 'failure': None if score is not None else
                (s.get('failure') or 'missing_verifier_score'), 'usd': float(s.get('estimated_usd') or 0.)}


class RoundAccess:
    def __init__(self, source, task_id, purpose):
        self.source, self.task_id, self.purpose = source, task_id, purpose
        self.reads, self.closed, self._cache = [], False, {}

    def read(self, verifier: str) -> dict:
        if self.closed:
            raise AccessError('round already closed')
        if verifier not in self._cache:
            self._cache[verifier] = self.source._read(self.task_id, verifier)
            self.reads.append(verifier)
        return self._cache[verifier]

    def already_read(self, verifier):
        return verifier in self._cache

    def close(self):
        self.source._close(self)
        self.closed = True


class LabelStore:
    """Evaluator-only labels. Missing official label for a processable candidate counts as incorrect."""

    def __init__(self, labels: dict[str, int | None]):
        self._labels = labels

    def outcome(self, task_id) -> int:
        y = self._labels[task_id]
        return 0 if y is None else int(y)

    def missing(self, task_ids):
        return [t for t in task_ids if self._labels[t] is None]


class InitLabels:
    """Labels of D_init only (labeled initialization)."""

    def __init__(self, store: LabelStore, init_ids):
        self._store, self._ids = store, frozenset(init_ids)

    def outcome(self, task_id):
        if task_id not in self._ids:
            raise AccessError('initialization may read labels of D_init only')
        return self._store.outcome(task_id)


class LabelGate:
    """Arm D: a label is released only after the round's audit draw selected the current item."""

    def __init__(self, store: LabelStore):
        self._store, self._allowed = store, None
        self.released = []

    def allow(self, task_id):
        self._allowed = task_id

    def revoke(self):
        self._allowed = None

    def outcome(self, task_id):
        if task_id != self._allowed:
            raise AccessError('label requested outside an audited round')
        self.released.append(task_id)
        return self._store.outcome(task_id)


def load_generator(data_dir: Path, gen: str):
    data_dir = Path(data_dir)
    items, signals, labels = [], {}, {}
    with gzip.open(data_dir/f'learner_{gen}.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            items.append(PublicItem(r['task_id'], int(r['fold']), r['group'], r['raw_r'], bool(r['has_code'])))
            signals[r['task_id']] = r['signals']
    with gzip.open(data_dir/f'labels_{gen}.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            labels[r['task_id']] = r['outcome'] if r['labeled'] else None
    return items, signals, labels
