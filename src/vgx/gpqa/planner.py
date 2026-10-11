"""Algorithms 1–2 for a fixed order of singleton verifier layers.

A reusable grid implements the paper's value tables. Decisions see the current
belief and stage only; observations are supplied after a query is selected.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from vgx.common.storage import digest
from vgx.gpqa.score import VerifierLikelihood


class ImpossibleObservation(ValueError):
    pass


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class NestedPlanner:
    def __init__(self, likelihoods, costs, correct_reward, incorrect_loss, *, grid_size=1001):
        self.likelihoods = tuple(likelihoods)
        self.costs = tuple(costs)
        self.reward, self.loss = correct_reward, incorrect_loss
        if len(self.costs) != len(self.likelihoods):
            raise ValueError('one cost required per verifier layer')
        if any(not _finite(c) or c < 0 for c in self.costs):
            raise ValueError('query costs must be finite and nonnegative')
        if any(not _finite(v) or v <= 0 for v in (correct_reward, incorrect_loss)):
            raise ValueError('reward and loss must be finite and positive')
        if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
            raise ValueError('grid_size must be an integer >= 2')
        for model in self.likelihoods:
            edges, p1, p0 = model.edges, model.p_bin_if_correct, model.p_bin_if_incorrect
            if (len(edges) != len(p1)+1 or len(p1) != len(p0) or not p1
                    or edges[0] != 0 or edges[-1] != 1
                    or any(not _finite(x) for x in edges)
                    or any(a >= b for a,b in zip(edges, edges[1:]))):
                raise ValueError('invalid verifier bin edges')
            for values in (p1, p0):
                if any(not _finite(v) or v < 0 for v in values) or not math.isclose(sum(values), 1., abs_tol=1e-9):
                    raise ValueError('invalid verifier probability distribution')
        self.grid = np.linspace(0., 1., grid_size)
        stop = np.maximum(0., (self.reward + self.loss) * self.grid - self.loss)
        self.j = [stop.copy() for _ in range(len(self.costs)+1)]
        self.q = [np.empty(grid_size) for _ in self.costs]
        for stage in reversed(range(len(self.costs))):
            model = self.likelihoods[stage]
            continuation = np.full(grid_size, -float(self.costs[stage]))
            for p1, p0 in zip(model.p_bin_if_correct, model.p_bin_if_incorrect):
                mass = self.grid*p1 + (1-self.grid)*p0
                updated = np.divide(self.grid*p1, mass, out=np.zeros_like(mass), where=mass > 0)
                continuation += mass * np.interp(updated, self.grid, self.j[stage+1])
            self.q[stage] = continuation
            self.j[stage] = np.maximum(stop, continuation)

    @property
    def threshold(self):
        return self.loss / (self.reward + self.loss)

    def decide(self, stage: int, belief: float) -> dict:
        if not _finite(belief) or not 0 <= belief <= 1:
            raise ValueError('belief must be finite and in [0,1]')
        if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage <= len(self.costs):
            raise ValueError('invalid verifier stage')
        assertion = (self.reward + self.loss)*belief - self.loss
        stop = max(0., assertion)
        continuation = float(np.interp(belief, self.grid, self.q[stage])) if stage < len(self.costs) else None
        query = continuation is not None and continuation > stop
        action = 'query' if query else ('assert' if belief >= self.threshold else 'abstain')
        return {'stage': stage, 'belief': belief, 'assert_value': assertion,
                'stop_value': stop, 'continuation_value': continuation,
                'value': max(stop, continuation) if continuation is not None else stop,
                'action': action, 'next_verifier_index': stage if query else None,
                'reason': 'continuation_exceeds_stop' if query else
                          ('no_remaining_verifiers' if continuation is None else 'stop_at_least_as_good')}

    def update(self, stage: int, belief: float, score: float) -> float:
        model = self.likelihoods[stage]
        index = model.bin_index(score)
        numerator = belief * model.p_bin_if_correct[index]
        mass = numerator + (1-belief) * model.p_bin_if_incorrect[index]
        if mass <= 0:
            raise ImpossibleObservation('zero predictive likelihood')
        return numerator / mass

    def replay(self, prior: float, scores) -> dict:
        if len(scores) != len(self.costs):
            raise ValueError('one potential score required per layer')
        belief, used, trace = prior, 0, []
        expected = self.decide(0, prior)['value']
        while True:
            decision = self.decide(used, belief)
            trace.append(decision)
            if decision['action'] != 'query':
                return {'action': decision['action'], 'posterior': belief, 'verifiers_used': used,
                        'expected_value_at_start': expected, 'failure': None, 'trace': trace}
            score = scores[used]  # accessed only after the policy selects this layer
            used += 1
            if score is None:
                failure = 'missing_verifier_score'
            else:
                try:
                    belief = self.update(used-1, belief, score)
                    continue
                except ImpossibleObservation:
                    failure = 'impossible_verifier_observation'
            return {'action': 'abstain', 'posterior': belief, 'verifiers_used': used,
                    'expected_value_at_start': expected, 'failure': failure, 'trace': trace}

    def to_dict(self) -> dict:
        value = {'schema': 1, 'method': 'linear_grid_tables', 'grid_size': len(self.grid),
                 'likelihoods': [asdict(m) for m in self.likelihoods], 'costs': list(self.costs),
                 'correct_reward': self.reward, 'incorrect_loss': self.loss,
                 'grid': self.grid.tolist(), 'J': [v.tolist() for v in self.j],
                 'Q': [v.tolist() for v in self.q]}
        return value

    @classmethod
    def from_dict(cls, value: dict):
        planner = cls([VerifierLikelihood(tuple(m['edges']), tuple(m['p_bin_if_correct']),
                                         tuple(m['p_bin_if_incorrect'])) for m in value['likelihoods']],
                      value['costs'], value['correct_reward'], value['incorrect_loss'], grid_size=value['grid_size'])
        # Values are recomputed from the locked inputs, not trusted as executable
        # policy data. Reject inconsistent saved tables.
        if digest(planner.to_dict()) != digest(value):
            raise ValueError('saved value tables do not match their parameters')
        return planner
