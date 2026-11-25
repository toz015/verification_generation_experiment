"""The pre-registered grouped 5-fold split is deterministic, complete and leak-free by construction."""
import json
from pathlib import Path

import pytest

from vgx.quantum_katas import load, split

pytestmark = pytest.mark.skipif(not (load.DEFAULT_SOURCE/'dataset/qiskit_quantumkatas.jsonl').exists(),
                                reason='pinned QuantumKatas source not downloaded')
COMMITTED = Path('configs/quantum_katas_split_20261019.json')


@pytest.fixture(scope='module')
def value():
    return split.build()


def test_split_covers_every_non_development_task_once(value):
    dev = set(json.loads(load.DEFAULT_PILOT.read_text())['task_ids'])
    assert value['tasks'] == 322 == len(value['task_fold'])
    assert not dev & set(value['task_fold'])
    members = [t for group in value['groups'].values() for t in group]
    assert sorted(members) == sorted(value['task_fold'])
    assert sum(f['tasks'] for f in value['folds']) == 322
    assert max(f['tasks'] for f in value['folds']) - min(f['tasks'] for f in value['folds']) <= 2


def test_no_group_or_near_duplicate_pair_spans_folds(value):
    for members in value['groups'].values():
        assert len({value['task_fold'][t] for t in members}) == 1
    for merge in value['merges']:
        assert value['task_fold'][merge['task_a']] == value['task_fold'][merge['task_b']]


def test_collections_group_by_second_level_and_split_is_deterministic(value):
    assert split.base_group('examples/GroverSearch2Qubits') == 'examples/GroverSearch2Qubits'
    assert split.base_group('tutorials/Oracles/3') == 'tutorials/Oracles'
    assert split.base_group('QFT/1.3') == 'QFT'
    assert split.build()['split_id'] == value['split_id']
    assert value['labels_or_model_outputs_used'] is False


@pytest.mark.skipif(not COMMITTED.exists(), reason='committed split not present')
def test_committed_split_matches_rules(value):
    assert json.loads(COMMITTED.read_text()) == value
