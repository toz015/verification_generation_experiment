"""Post-pilot diagnostic probes; separate from the unchanged upstream audit.

QFT extension explicitly adopts the upstream reference's MSB-first list order.
These checks diagnose coverage; they are not a finalized benchmark oracle.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from quantum_katas_pilot import PROBES, dump, rows_to_file, run_job, sha


QFT_TEST = '''
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

def test_qft():
    # Specify list order: qubits[0] is the most significant logical bit.
    # Check the full unitary for n=1..4, not just its |0...0> column.
    for n in range(1, 5):
        N = 2**n
        j, k = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        dft = np.exp(2j*np.pi*j*k/N) / np.sqrt(N)
        reverse = [int(format(i, f"0{n}b")[::-1], 2) for i in range(N)]
        target = dft[np.ix_(reverse, reverse)]
        actual = Operator(qft(QuantumCircuit(n), list(range(n)))).data
        assert np.allclose(actual, target, atol=1e-9), f"Wrong full QFT unitary n={n}"
'''

QEC_TEST = '''
import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def test_decode_qubit():
    # Complex superpositions and every placement of the logical data qubit.
    for data, a, b in itertools.permutations(range(3)):
        for theta, phi in [(0, 0), (np.pi, 0), (np.pi/2, 0), (0.71, 1.2)]:
            before = QuantumCircuit(3)
            before.ry(theta, data)
            before.rz(phi, data)
            expected = Statevector.from_instruction(before)
            encoded = before.copy()
            encoded.cx(data, a)
            encoded.cx(data, b)
            actual = Statevector.from_instruction(decode_qubit(encoded, data, [a, b]))
            assert actual.equiv(expected), f"Wrong decoded state for {(data, a, b, theta, phi)}"
'''


def main():
    base = Path("reports/quantum_katas_pilot_20261018")
    source = Path("tmp/quantum_four_dataset_review_20261018/Qiskit-QuantumKatas/benchmark/evaluator.py").resolve()
    rows = {r["task_id"]: r for r in map(json.loads, (base / "private/tasks.jsonl").read_text().splitlines())}
    jobs = []
    tid = "CHSHGame/2.4"
    row = rows[tid]
    old = "qc.ry(-2 * angle, qubit)"
    assert row["canonical_solution"].count(old) == 1
    code = row["canonical_solution"].replace(old, "qc.ry(-2 * angle + 0.15, qubit)")
    for repeat in range(30):
        jobs.append({"task_id": tid, "kind": "same_wrong_rotation_plus_0.15", "repeat": repeat,
                     "entry_point": row["entry_point"], "code": code, "test": row["test"]})
    for tid, test in [("PhaseEstimation/2", QFT_TEST), ("QEC_BitFlipCode/6", QEC_TEST)]:
        row = rows[tid]
        mutant = next(p for p in PROBES if p[0] == tid)
        for kind, code in [("reference_extension", row["canonical_solution"]), ("mutant_extension", mutant[2])]:
            jobs.append({"task_id": tid, "kind": kind, "repeat": 0,
                         "entry_point": row["entry_point"], "code": code, "test": test})
    output = base / "diagnostics"
    output.mkdir(exist_ok=False)
    # Freeze the angle, repetition count, and extension tests before execution.
    rows_to_file(output / "planned_jobs.jsonl", jobs)
    dump(output / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
         "runner_sha256": sha(Path(__file__).read_bytes()), "evaluator_sha256": sha(source.read_bytes()),
         "plan_sha256": sha((output / "planned_jobs.jsonl").read_bytes()),
         "purpose": "Post-hoc test-coverage diagnosis; not selection or final accuracy estimation",
         "qft_convention": "qubits[0] is the most significant logical bit, matching pinned reference"})
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda job: run_job(job, source, 30), jobs))
    rows_to_file(output / "runs.jsonl", results)
    stochastic = [r for r in results if r["kind"] == "same_wrong_rotation_plus_0.15"]
    summary = {"same_candidate_native_test": {
        "task_id": "CHSHGame/2.4", "code_sha256": sha(jobs[0]["code"].encode()),
        "runs": len(stochastic), "passes": sum(r["passed"] for r in stochastic),
        "failures": sum(not r["passed"] for r in stochastic),
        "distinct_error_types": sorted({r["error_type"] for r in stochastic if not r["passed"]}),
        "note": "Fixed incorrect +0.15 radian rotation; unmodified unseeded upstream test."},
        "extension_checks": [r for r in results if r["kind"] != "same_wrong_rotation_plus_0.15"],
        "total_wall_elapsed_s": time.perf_counter() - start}
    dump(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
