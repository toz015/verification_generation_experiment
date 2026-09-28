"""Forecast and verifier diagnostics for the GPQA pilot."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


@dataclass(frozen=True)
class BinaryForecastMetrics:
    n: int
    brier: float | None
    log_loss: float | None
    ece: float | None
    auroc: float | None
    reliability: tuple[dict[str, float | int], ...]

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "ece": self.ece,
            "auroc": self.auroc,
            "reliability": list(self.reliability),
        }


def binary_forecast_metrics(
    outcomes: list[int], probabilities: list[float], bins: int = 5
) -> BinaryForecastMetrics:
    """Score binary correctness forecasts; ECE uses fixed-width probability bins."""
    if len(outcomes) != len(probabilities):
        raise ValueError("outcomes and probabilities must have equal lengths")
    if not outcomes:
        return BinaryForecastMetrics(0, None, None, None, None, ())
    if bins < 1:
        raise ValueError("bins must be positive")
    y = np.asarray(outcomes, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("outcomes must contain only 0 and 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")

    reliability: list[dict[str, float | int]] = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (p >= lower) & ((p <= upper) if index == bins - 1 else (p < upper))
        count = int(mask.sum())
        if not count:
            continue
        confidence = float(p[mask].mean())
        accuracy = float(y[mask].mean())
        ece += count / len(y) * abs(confidence - accuracy)
        reliability.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": confidence,
                "empirical_accuracy": accuracy,
            }
        )

    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return BinaryForecastMetrics(
        n=len(y),
        brier=float(np.mean((p - y) ** 2)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        ece=float(ece),
        auroc=auc,
        reliability=tuple(reliability),
    )


def correctness_outcome(candidate: str | None, correct_index: int) -> int:
    """Return 1 if a valid answer letter is correct, else 0 (including parse failure)."""
    if candidate is None:
        return 0
    from vgx.gpqa.load import LETTERS

    return int(candidate.upper() in LETTERS and LETTERS.index(candidate.upper()) == correct_index)


@dataclass(frozen=True)
class VerifierLikelihood:
    """Smoothed binned P(score bin | fixed candidate is correct/incorrect)."""

    edges: tuple[float, ...]
    p_bin_if_correct: tuple[float, ...]
    p_bin_if_incorrect: tuple[float, ...]

    def bin_index(self, score: float) -> int:
        if not 0.0 <= score <= 1.0:
            raise ValueError("verifier score must be in [0, 1]")
        index = int(np.searchsorted(self.edges, score, side="right") - 1)
        return min(max(index, 0), len(self.p_bin_if_correct) - 1)

    def posterior(self, prior: float, score: float) -> float:
        """Bayes-update a correctness prior using this verifier score."""
        if not 0.0 <= prior <= 1.0 or not 0.0 <= score <= 1.0:
            raise ValueError("prior and verifier score must be in [0, 1]")
        index = self.bin_index(score)
        numerator = prior * self.p_bin_if_correct[index]
        denominator = numerator + (1.0 - prior) * self.p_bin_if_incorrect[index]
        return numerator / denominator if denominator else prior


def fit_verifier_likelihood(
    outcomes: list[int], scores: list[float], bins: int = 3, laplace: float = 1.0
) -> VerifierLikelihood:
    """Fit a low-capacity verifier observation model on calibration data only.

    Equal-width score bins and Laplace smoothing avoid fitting a flexible
    likelihood model to the very small initial GPQA pilot. Both correctness
    classes must occur in the calibration data.
    """
    if len(outcomes) != len(scores):
        raise ValueError("outcomes and scores must have equal lengths")
    if bins < 1 or laplace <= 0:
        raise ValueError("bins and laplace must be positive")
    if not outcomes:
        raise ValueError("cannot fit verifier likelihood on an empty sample")
    y = np.asarray(outcomes, dtype=int)
    p = np.asarray(scores, dtype=float)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("outcomes must contain only 0 and 1")
    if len(np.unique(y)) != 2:
        raise ValueError("calibration data must contain both correct and incorrect answers")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("verifier scores must be finite values in [0, 1]")

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.searchsorted(edges, p, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, bins - 1)

    def distribution(class_label: int) -> tuple[float, ...]:
        counts = np.bincount(bin_ids[y == class_label], minlength=bins).astype(float)
        counts += laplace
        return tuple((counts / counts.sum()).tolist())

    return VerifierLikelihood(
        edges=tuple(edges.tolist()),
        p_bin_if_correct=distribution(1),
        p_bin_if_incorrect=distribution(0),
    )


@dataclass(frozen=True)
class SequentialDecision:
    action: str
    posterior: float
    verifiers_used: int
    expected_value_at_start: float


def simulate_sequential_decision(
    prior: float,
    scores: list[float],
    likelihoods: list[VerifierLikelihood],
    costs: list[float],
    correct_reward: float,
    incorrect_loss: float,
) -> SequentialDecision:
    """Replay nested optimal stopping for already-collected verifier scores.

    This computes the policy's counterfactual query count. The runner currently
    collects all verifier signals to support calibration and paired policy
    comparisons; that collection cost is not itself reduced by this replay.
    """
    if not 0.0 <= prior <= 1.0:
        raise ValueError("prior must be in [0, 1]")
    if len(scores) != len(likelihoods) or len(costs) != len(likelihoods):
        raise ValueError("scores, costs, and verifier likelihoods must have equal lengths")
    if correct_reward <= 0 or incorrect_loss <= 0:
        raise ValueError("correct_reward and incorrect_loss must be positive")
    if any(cost < 0 for cost in costs):
        raise ValueError("verifier costs must be nonnegative")

    def stop_value(belief: float) -> float:
        return max(0.0, (correct_reward + incorrect_loss) * belief - incorrect_loss)

    @lru_cache(maxsize=None)
    def value(stage: int, belief_key: int) -> float:
        belief = belief_key / 1_000_000_000
        stop = stop_value(belief)
        if stage == len(likelihoods):
            return stop
        model = likelihoods[stage]
        continuation = -costs[stage]
        for p1, p0 in zip(model.p_bin_if_correct, model.p_bin_if_incorrect):
            probability = belief * p1 + (1.0 - belief) * p0
            if probability == 0:
                continue
            updated = belief * p1 / probability
            continuation += probability * value(stage + 1, round(updated * 1_000_000_000))
        return max(stop, continuation)

    expected_start = value(0, round(prior * 1_000_000_000))
    belief = prior
    used = 0
    for stage, (model, score) in enumerate(zip(likelihoods, scores)):
        stop = stop_value(belief)
        continuation = -costs[stage]
        for p1, p0 in zip(model.p_bin_if_correct, model.p_bin_if_incorrect):
            probability = belief * p1 + (1.0 - belief) * p0
            if probability > 0:
                updated = belief * p1 / probability
                continuation += probability * value(
                    stage + 1, round(updated * 1_000_000_000)
                )
        if stop >= continuation:  # prefer stopping in a tie
            break
        belief = model.posterior(belief, score)
        used += 1

    threshold = incorrect_loss / (correct_reward + incorrect_loss)
    action = "assert" if belief >= threshold else "abstain"
    return SequentialDecision(action, belief, used, expected_start)
