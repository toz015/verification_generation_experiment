"""Sequential accept/abstain/query over frozen code candidates, reusing NestedPlanner.

The executor sees only the public candidate (code, raw r) and the verifier
specs. A verifier signal is requested only after the planner selects that
layer; nothing after the stopping point is requested or read. Checkpoints make
resume reuse recorded observations without sending a request again.

Candidates without extractable code, or with code but an invalid r, abstain at
stage 0: there is no prior, and none is invented (no 0.5 fill).
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from vgx.common.storage import atomic_json, digest, file_lock
from vgx.gpqa.planner import ImpossibleObservation, NestedPlanner
from vgx.gpqa.sequential import seal, unseal
from vgx.quantum_katas.artifacts import FrozenCodeCandidate
from vgx.quantum_katas.runners import ModelSpec

PRIOR_MAPS = ('identity_raw_r',)


def make_policy(*, plan_id: str, verifiers, planner: NestedPlanner, kind: str, prior_map='identity_raw_r',
                note='') -> dict:
    if kind not in ('synthetic_engineering_dry_run', 'fitted_development'):
        raise ValueError('policy kind must declare whether it is synthetic or fitted')
    if prior_map not in PRIOR_MAPS:
        raise ValueError('unsupported prior map')
    return seal({'schema': 1, 'adapter': 'quantum_katas', 'kind': kind, 'plan_id': plan_id,
                 'verifiers': [asdict(v) for v in verifiers],
                 'verifier_identities': [v.identity for v in verifiers],
                 'prior_map': prior_map, 'planner': planner.to_dict(), 'note': note}, 'policy_id')


def policy_parts(policy: dict):
    unseal(policy, 'policy_id')
    specs = tuple(ModelSpec(**row) for row in policy['verifiers'])
    if [s.identity for s in specs] != policy['verifier_identities'] or len({s.id for s in specs}) != len(specs):
        raise ValueError('verifier identity changed since policy preparation')
    planner = NestedPlanner.from_dict(policy['planner'])
    if len(specs) != len(planner.costs):
        raise ValueError('policy verifier count differs from its value tables')
    return specs, planner


def prior(candidate: FrozenCodeCandidate, policy: dict) -> tuple[float | None, str | None]:
    if not candidate.has_code:
        return None, 'no_extractable_code'
    if not candidate.has_valid_confidence:
        return None, 'invalid_generator_confidence'
    return candidate.raw_p_correct, None  # identity_raw_r


def execute_candidate(candidate: FrozenCodeCandidate, policy: dict, query, state_path: Path) -> dict:
    """query(spec, candidate) is called only for the layer the planner just selected."""
    specs, planner = policy_parts(policy)
    identity = {'candidate_id': candidate.candidate_id, 'task_id': candidate.task_id,
                'candidate_sha256': digest(asdict(candidate)), 'policy_id': policy['policy_id']}
    state_path = Path(state_path)
    with file_lock(str(state_path) + '.lock'):
        if state_path.exists():
            state = unseal(json.loads(state_path.read_text()), 'state_id')
            if state['identity'] != identity:
                raise ValueError('resume candidate or policy differs from checkpoint')
            if state.get('result') is not None:
                return state['result']
            observations = state['observations']
        else:
            observations = []
        start, failure = prior(candidate, policy)
        if start is None:
            if observations:
                raise ValueError('candidate without a prior has recorded observations')
            result = {**identity, 'action': 'abstain', 'prior': None, 'raw_p_correct': candidate.raw_p_correct,
                      'posterior': None, 'verifiers_used': 0, 'expected_value_at_start': None,
                      'expected_verification_cost_utility': 0., 'failure': failure, 'trace': [],
                      'observed_request_keys': [], 'new_provider_executions': 0}
            atomic_json(state_path, seal({'identity': identity, 'observations': [], 'result': result}, 'state_id'))
            return result
        belief, stage, trace, failure = start, 0, [], None
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
                atomic_json(state_path, seal({'identity': identity, 'observations': observations,
                    'pending': {'stage': stage, 'verifier_id': spec.id}, 'result': None}, 'state_id'))
                observation = {'stage': stage, 'verifier_id': spec.id, **query(spec, candidate)}
                observations.append(observation)
                atomic_json(state_path, seal({'identity': identity, 'observations': observations, 'result': None}, 'state_id'))
            stage += 1
            if observation['score'] is None:
                failure = observation.get('failure') or 'missing_verifier_score'
            else:
                try:
                    belief = planner.update(stage - 1, belief, observation['score'])
                    continue
                except ImpossibleObservation:
                    failure = 'impossible_verifier_observation'
            if len(observations) != stage:
                raise ValueError('checkpoint contains observations after failure')
            break
        result = {**identity, 'action': 'abstain' if failure else decision['action'], 'prior': start,
                  'raw_p_correct': candidate.raw_p_correct, 'posterior': belief, 'verifiers_used': stage,
                  'expected_value_at_start': planner.decide(0, start)['value'],
                  'expected_verification_cost_utility': sum(planner.costs[:stage]),
                  'failure': failure, 'trace': trace,
                  'observed_request_keys': [o['request_key'] for o in observations if o.get('request_key')],
                  'new_provider_executions': sum(1 for o in observations if o.get('used_cached_response') is False)}
        atomic_json(state_path, seal({'identity': identity, 'observations': observations, 'result': result}, 'state_id'))
        return result


class LazySignals:
    """Replay accessor over collected signals that records which layers were read."""

    def __init__(self, scores):
        self._scores, self.accessed = list(scores), []

    def __len__(self):
        return len(self._scores)

    def __getitem__(self, index):
        self.accessed.append(index)
        return self._scores[index]


def replay(candidate: FrozenCodeCandidate, policy: dict, scores) -> dict:
    """Offline replay with the same value tables; only reports logical queries."""
    _, planner = policy_parts(policy)
    start, failure = prior(candidate, policy)
    if start is None:
        return {'action': 'abstain', 'posterior': None, 'verifiers_used': 0, 'failure': failure, 'trace': [],
                'accessed_layers': []}
    signals = LazySignals(scores)
    result = planner.replay(start, signals)
    return {**{k: result[k] for k in ('action', 'posterior', 'verifiers_used', 'failure', 'trace')},
            'accessed_layers': signals.accessed}
