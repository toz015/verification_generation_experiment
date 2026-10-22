"""Offline audit of reviewed, pinned reference code; NEVER a generated-code sandbox.

Run with the separate interpreter in configs/quantum_katas_pilot_requirements.txt.
The upstream evaluator's worker executes each unmodified test in a fresh process.
No model calls, calibration, or edits to upstream data are performed.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stderr


SOURCE_HASHES = {
    "dataset/qiskit_quantumkatas.jsonl": "03abb872a87bc62c36c9ea453659cb741d7a8f4892a4c1b52c39a08f882e7b8c",
    "benchmark/evaluator.py": "f91f4f6da9efbdb8fb88dbbd964979a9272119fb517c844a6680cedceb1b1e2e",
    "uv.lock": "d7608adaa5bbec1354bd1ffcc4a22ec5da3d779706e362b4fc63e8267c50a3c8",
    "LICENSE": "e6f27c58c270f578b26ec807c965a6aa2d3e6e6d77cc8004e34b88d47fc62dd3",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def rows_to_file(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def worker():
    payload = json.load(sys.stdin)
    spec = importlib.util.spec_from_file_location("upstream_evaluator", payload["evaluator"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    code = module.extract_code_from_response(payload["code"])
    valid, error = module.check_syntax(code)
    if not valid:
        print(json.dumps({"passed": False, "error_type": "SyntaxError", "error_message": error}))
        return

    # Same namespace/test-dispatch semantics as upstream, but the parent owns
    # the fresh subprocess and timeout instead of multiprocessing.Queue.
    class ResultCollector:
        def put(self, value):
            self.value = value

    collector = ResultCollector()
    stderr = io.StringIO()
    start = time.perf_counter()
    with redirect_stderr(stderr):
        module._run_test_worker(code, payload["test"], payload["entry_point"], collector)
    result = collector.value
    result["test_elapsed_s"] = time.perf_counter() - start
    result["stderr"] = stderr.getvalue()[:10000]
    result["stdout"] = result.get("stdout", "")[:10000]
    print(json.dumps(result, allow_nan=False))


def noop_solution(row):
    tree = ast.parse(row["canonical_solution"])
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == row["entry_point"]]
    if len(matches) != 1:
        raise ValueError(f"Expected one entry-point definition: {row['task_id']}")
    matches[0].body = [ast.Return(value=ast.Constant(value=None))]
    return ast.unparse(ast.fix_missing_locations(tree))


PROBES = [
    ("SolveSATWithGrover/3.1", "constant_true",
     "def test_grovers_with_sat():\n    return True\n",
     "Claims success without constructing or executing any Grover circuit."),
    ("PhaseEstimation/2", "hadamards_only",
     "def qft(qc, qubits):\n    for q in qubits:\n        qc.h(q)\n    return qc\n",
     "H tensor powers match QFT only on restricted inputs, omitting controlled phases."),
    ("QEC_BitFlipCode/6", "hardcoded_qubit_indices",
     "def decode_qubit(qc, data, ancillas):\n    qc.cx(0, 2)\n    qc.cx(0, 1)\n    return qc\n",
     "Ignores data/ancilla arguments; fails when the logical data qubit is 2."),
]


def make_jobs(rows, config):
    jobs = []
    for row in rows:
        count = config["stress_total_repeats"] if row["task_id"] in config["stress_task_ids"] else config["reference_repeats"]
        for repeat in range(count):
            jobs.append({"task_id": row["task_id"], "kind": "reference", "repeat": repeat,
                         "code": row["canonical_solution"], "test": row["test"], "entry_point": row["entry_point"]})
        jobs.append({"task_id": row["task_id"], "kind": "noop", "repeat": 0,
                     "code": noop_solution(row), "test": row["test"], "entry_point": row["entry_point"]})
    by_id = {r["task_id"]: r for r in rows}
    for tid, name, code, reason in PROBES:
        row = by_id[tid]
        jobs.append({"task_id": tid, "kind": name, "repeat": 0, "code": code,
                     "test": row["test"], "entry_point": row["entry_point"], "reason": reason})
    return jobs


def run_job(job, evaluator, timeout):
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vgx-qkata-") as directory:
        env = {"PATH": os.defpath, "HOME": directory, "TMPDIR": directory,
               "LANG": "en_US.UTF-8", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1", "QISKIT_PARALLEL": "FALSE"}
        try:
            process = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
                input=json.dumps({**job, "evaluator": str(evaluator)}),
                text=True, capture_output=True, timeout=timeout, cwd=directory, env=env,
            )
            if process.returncode:
                result = {"passed": False, "error_type": "ProcessError", "error_message": process.stderr[-4000:]}
            else:
                try:
                    result = json.loads(process.stdout)
                except json.JSONDecodeError:
                    result = {"passed": False, "error_type": "InvalidWorkerOutput", "error_message": process.stdout[-4000:]}
            result["process_stderr"] = process.stderr[-4000:]
        except subprocess.TimeoutExpired:
            result = {"passed": False, "error_type": "Timeout", "error_message": f"Fresh process exceeded {timeout}s"}
    return {k: v for k, v in job.items() if k not in ("code", "test")} | {
        "code_sha256": sha(job["code"].encode()), "test_sha256": sha(job["test"].encode()),
        "wall_elapsed_s": time.perf_counter() - started, **result,
    }


def summarize(results, task_ids):
    tasks = []
    for tid in task_ids:
        refs = [r for r in results if r["task_id"] == tid and r["kind"] == "reference"]
        controls = [r for r in results if r["task_id"] == tid and r["kind"] != "reference"]
        tasks.append({"task_id": tid, "reference_runs": len(refs),
                      "reference_passes": sum(r["passed"] for r in refs),
                      "reference_wall_median_s": statistics.median(r["wall_elapsed_s"] for r in refs),
                      "reference_wall_max_s": max(r["wall_elapsed_s"] for r in refs),
                      "errors": [{"repeat": r["repeat"], "error_type": r.get("error_type"),
                                  "error_message": r.get("error_message")} for r in refs if not r["passed"]],
                      "controls": [{"kind": r["kind"], "passed": r["passed"],
                                    "error_type": r.get("error_type"), "reason": r.get("reason")} for r in controls]})
    refs = [r for r in results if r["kind"] == "reference"]
    return {"tasks": tasks, "task_count": len(tasks), "reference_runs": len(refs),
            "reference_passes": sum(r["passed"] for r in refs),
            "all_reference_runs_pass_tasks": sum(t["reference_runs"] == t["reference_passes"] for t in tasks),
            "noop_passes": sum(r["passed"] for r in results if r["kind"] == "noop"),
            "noop_runs": sum(r["kind"] == "noop" for r in results),
            "targeted_probe_passes": [r for r in results if r["kind"] not in ("reference", "noop") and r["passed"]],
            "reference_wall_median_s": statistics.median(r["wall_elapsed_s"] for r in refs),
            "reference_wall_max_s": max(r["wall_elapsed_s"] for r in refs),
            "note": "Native unseeded tests. Finite pass repetitions do not prove semantic correctness, completeness, or absence of rare failures. Controls are diagnostic, not accuracy estimates."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, default=Path("configs/quantum_katas_pilot_20261018.json"))
    parser.add_argument("--source", type=Path, default=Path("tmp/quantum_four_dataset_review_20261018/Qiskit-QuantumKatas"))
    parser.add_argument("--output", type=Path, default=Path("reports/quantum_katas_pilot_20261018"))
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    for rel, expected in SOURCE_HASHES.items():
        if sha((args.source / rel).read_bytes()) != expected:
            raise ValueError(f"Pinned source hash mismatch: {rel}")
    config = json.loads(args.config.read_text())
    ids = config["task_ids"]
    if len(ids) != len(set(ids)) or config["partition"] != "development":
        raise ValueError("Pilot requires unique development task IDs")
    all_rows = [json.loads(line) for line in (args.source / "dataset/qiskit_quantumkatas.jsonl").read_text().splitlines()]
    by_id = {r["task_id"]: r for r in all_rows}
    rows = [by_id[tid] for tid in ids]
    jobs = make_jobs(rows, config)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite or mix independent runs.
    packages = {p.metadata["Name"]: p.version for p in importlib.metadata.distributions()}
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "config_sha256": sha(args.config.read_bytes()), "source_sha256": SOURCE_HASHES,
                "runner_sha256": sha(Path(__file__).read_bytes()), "python": sys.version,
                "platform": platform.platform(), "packages": packages, "jobs": len(jobs),
                "execution": "Reviewed reference/control code only; subprocess timeout and clean environment, NOT a hostile-code sandbox. Upstream test worker unchanged; one native unseeded test run per fresh process, max 2 processes and 1 BLAS/OMP thread each."}
    dump(output / "manifest.json", manifest)
    rows_to_file(output / "private" / "tasks.jsonl", rows)
    rows_to_file(output / "public" / "tasks.jsonl", [
        {"task_id": r["task_id"], "partition": "development", "prompt": r["prompt"],
         "entry_point": r["entry_point"], "revision": config["revision"]} for r in rows])
    rows_to_file(output / "private" / "control_codes.jsonl", [j for j in jobs if j["kind"] != "reference"])
    start = time.perf_counter()
    results = []
    with (output / "runs.jsonl").open("x") as stream, ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        for result in pool.map(lambda j: run_job(j, (args.source / "benchmark/evaluator.py").resolve(), config["timeout_s"]), jobs):
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if len(results) % 20 == 0 or len(results) == len(jobs):
                print(f"Completed {len(results)}/{len(jobs)} checks", flush=True)
    summary = summarize(results, ids)
    summary["total_wall_elapsed_s"] = time.perf_counter() - start
    dump(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("tasks", "targeted_probe_passes")}, indent=2))


if __name__ == "__main__":
    main()
