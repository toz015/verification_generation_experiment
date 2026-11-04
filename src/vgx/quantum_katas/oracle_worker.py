"""Standalone official-checker worker; runs under the separate quantum interpreter.

Standard library only, no vgx imports: the controller copies this file and the
pinned evaluator.py into an otherwise empty job directory. The job (code, test,
entry point, timeout and a one-time nonce) arrives on stdin. The full upstream
evaluate_solution entry point is called once, unchanged, so its extraction,
syntax check, multiprocessing execution and return fields are preserved.

The result line is prefixed with the nonce. Under the spawn start method
(macOS default) candidate code runs in a fresh evaluator child that never sees
stdin or this process's memory, so it cannot forge a result line; anything it
writes to the shared stdout is ignored. Under fork (Linux default) the child
inherits memory, so the nonce is only a guard against accidental output there.
"""
import dataclasses
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import resource
import sys
import time

PACKAGES = ('qiskit', 'qiskit-aer', 'numpy', 'scipy', 'rustworkx', 'dill', 'stevedore', 'psutil',
            'python-dateutil', 'six', 'typing_extensions')


def environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'implementation': platform.python_implementation(),
            'platform': platform.platform(), 'machine': platform.machine(),
            'executable_realpath': os.path.realpath(sys.executable), 'prefix': sys.prefix,
            'base_prefix': sys.base_prefix, 'packages': packages,
            'multiprocessing_start_method': multiprocessing.get_start_method()}


def main():
    if sys.argv[1:] == ['--probe']:
        print(json.dumps(environment(), sort_keys=True))
        return
    job = json.loads(sys.stdin.read())
    # Inherited by the evaluator's child processes; applied before any candidate code runs.
    limits = job['rlimits']
    resource.setrlimit(resource.RLIMIT_CPU, (limits['cpu_seconds'], limits['cpu_seconds']))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits['file_size_bytes'], limits['file_size_bytes']))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    here = os.path.dirname(os.path.abspath(__file__))
    evaluator_path = os.path.join(here, 'evaluator.py')
    with open(evaluator_path, 'rb') as stream:
        evaluator_sha = hashlib.sha256(stream.read()).hexdigest()
    if evaluator_sha != job['evaluator_sha256']:
        raise SystemExit('pinned evaluator hash mismatch')
    # Importable by name so multiprocessing children re-import the same module.
    sys.path.insert(0, here)
    import evaluator  # noqa: E402

    extracted = evaluator.extract_code_from_response(job['code'])
    started = time.perf_counter()
    result = evaluator.evaluate_solution(job['code'], job['test'], job['entry_point'], timeout=job['timeout'])
    elapsed = time.perf_counter() - started
    payload = {'official': dataclasses.asdict(result), 'evaluate_elapsed_s': elapsed,
               'executed_code_sha256': hashlib.sha256(extracted.encode()).hexdigest(),
               'evaluator_sha256': evaluator_sha, 'environment': environment()}
    sys.stdout.write('\n' + job['nonce'] + json.dumps(payload, sort_keys=True, allow_nan=False) + '\n')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
