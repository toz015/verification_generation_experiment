"""Resumable sequential executor with no answer keys or future observations."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import uuid

from vgx.common.api import RequestCache
from vgx.common.storage import atomic_json, digest, file_lock
from vgx.gpqa.artifacts import FrozenCandidate, load_candidates
from vgx.gpqa.planner import ImpossibleObservation, NestedPlanner
from vgx.gpqa.verifiers import VerifierSpec


def seal(value: dict, name: str) -> dict:
    return {**value, name: digest(value)}


def unseal(value: dict, name: str) -> dict:
    body = {k:v for k,v in value.items() if k != name}
    if value.get(name) != digest(body):
        raise ValueError(f'invalid {name} checksum')
    return body


def policy_parts(policy: dict):
    unseal(policy, 'policy_id')
    specs = tuple(VerifierSpec(**row) for row in policy['verifiers'])
    if len(set(s.id for s in specs)) != len(specs) or [s.identity for s in specs] != policy['verifier_identities']:
        raise ValueError('verifier identity or prompt changed since policy preparation')
    planner = NestedPlanner.from_dict(policy['planner'])
    if len(specs) != len(planner.costs):
        raise ValueError('policy verifier count differs from its value tables')
    return specs, planner


def execute_candidate(candidate: FrozenCandidate, policy: dict, query, state_path: Path) -> dict:
    """query receives exactly the selected spec and public fixed candidate.

    Provider response caching is durable before a score is checkpointed. If the
    process dies between them, resume reads that response without sending again.
    """
    specs, planner = policy_parts(policy)
    identity = {'candidate_id': candidate.item_id, 'candidate_sha256': digest(asdict(candidate)),
                'policy_id': policy['policy_id']}
    state_path = Path(state_path)
    with file_lock(str(state_path)+'.lock'):
        if state_path.exists():
            state = unseal(json.loads(state_path.read_text()), 'state_id')
            if state['identity'] != identity:
                raise ValueError('resume candidate or policy differs from checkpoint')
            observations = state['observations']
        else:
            observations = []
        valid = candidate.answer is not None and candidate.p_correct is not None
        if not valid:
            if observations:
                raise ValueError('invalid candidate has recorded observations')
            result = {**identity, 'action': 'abstain', 'posterior': candidate.p_correct,
                      'verifiers_used': 0, 'expected_verification_cost_utility': 0.,
                      'failure': 'invalid_generator_answer_or_confidence', 'trace': [], 'observed_request_keys': []}
            atomic_json(state_path, seal({'identity': identity, 'observations': [], 'result': result}, 'state_id'))
            return result
        belief, stage, trace, failure = candidate.p_correct, 0, [], None
        while True:
            decision = planner.decide(stage, belief)
            trace.append(decision)
            if decision['action'] != 'query':
                if len(observations) != stage:
                    raise ValueError('checkpoint contains observations after the policy stopped')
                break
            spec = specs[stage]
            if stage < len(observations):
                observation = observations[stage]
                if observation['stage'] != stage or observation['verifier_id'] != spec.id:
                    raise ValueError('checkpoint observation order changed')
            else:
                # Commit intent before transport. No score for a later stage is
                # requested or inspected here, even if present in the cache.
                atomic_json(state_path, seal({'identity': identity, 'observations': observations,
                    'pending': {'stage': stage, 'verifier_id': spec.id}, 'result': None}, 'state_id'))
                response = query(spec, candidate)
                observation = {'stage': stage, 'verifier_id': spec.id, **response}
                observations.append(observation)
                atomic_json(state_path, seal({'identity': identity, 'observations': observations, 'result': None}, 'state_id'))
            stage += 1
            score = observation['score']
            if score is None:
                failure = observation.get('failure') or 'missing_verifier_score'
            else:
                try:
                    belief = planner.update(stage-1, belief, score)
                    continue
                except ImpossibleObservation:
                    failure = 'impossible_verifier_observation'
            if len(observations) != stage:
                raise ValueError('checkpoint contains observations after failure')
            break
        result = {**identity, 'action': 'abstain' if failure else decision['action'],
                  'posterior': belief, 'verifiers_used': stage,
                  'expected_value_at_start': planner.decide(0, candidate.p_correct)['value'],
                  'expected_verification_cost_utility': sum(planner.costs[:stage]),
                  'failure': failure, 'trace': trace,
                  'observed_request_keys': [o['request_key'] for o in observations if o.get('request_key')]}
        atomic_json(state_path, seal({'identity': identity, 'observations': observations, 'result': result}, 'state_id'))
        return result


def execute_bundle(bundle: Path, policy: dict, cache: RequestCache, output: Path,
                   *, partition='evaluation', allow_api=False, workers=1) -> dict:
    manifest, candidates = load_candidates(bundle)  # public data only
    if policy['bundle_id'] != manifest['bundle_id']:
        raise ValueError('policy was fitted for a different frozen bundle')
    if partition not in ('calibration', 'evaluation'):
        raise ValueError('invalid execution partition')
    policy_parts(policy)
    candidates = [c for c in candidates if c.partition == partition]
    output = Path(output)
    binding = {'schema': 1, 'bundle_id': manifest['bundle_id'], 'policy_id': policy['policy_id'],
               'partition': partition, 'candidate_ids': [c.item_id for c in candidates]}
    # A single process owns this run, including its operation ID and checkpoints.
    with file_lock(output/'execution.lock'):
        path = output/'execution_manifest.json'
        if path.exists():
            execution = json.loads(path.read_text())
            unseal(execution, 'execution_id')
            if execution['binding'] != binding:
                raise ValueError('execution directory belongs to a different policy or candidate set')
        else:
            execution = seal({'binding': binding, 'operation_id': str(uuid.uuid4())}, 'execution_id')
            atomic_json(path, execution)

        def query(spec, candidate):
            request = spec.request(candidate, execution['operation_id'])
            call, cached = cache.get(spec.runner(), request, allow_api=allow_api)
            score, failure = spec.signal(call.response, candidate)
            return {'score': score, 'failure': failure, 'request_key': call.key,
                    'execution_id': call.meta.get('execution_id'), 'used_cached_response': cached}

        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError('workers must be a positive integer')
        def execute(c):
            return execute_candidate(c, policy, query, output/'items'/f'{digest(c.item_id)}.json')
        # Only separate questions run concurrently. Verifier layers within a
        # question remain strictly sequential and share no observations.
        if workers == 1:
            results = [execute(c) for c in candidates]
        else:
            from vgx.common.concurrency import parallel_map
            results = parallel_map(execute, candidates, workers)
        result = seal({**execution, 'results': results, 'logical_verifier_queries': sum(r['verifiers_used'] for r in results),
                  'note': 'Logical queries include reused responses. Incremental paid executions require the separate usage ledger.'}, 'decisions_id')
        atomic_json(output/'decisions.json', result)
        return result
