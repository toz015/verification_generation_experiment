"""Offline GPQA diagnostics. Synthetic inputs validate software, not calibration.

All fitted quantities use calibration records only. Intervals are descriptive
95% percentile bootstrap intervals, not evidence that a 20-item holdout is large.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from vgx.gpqa.score import (
    binary_forecast_metrics, correctness_outcome, fit_verifier_likelihood,
    simulate_sequential_decision,
)


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _interval(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return {"low": float(np.quantile(values, .025)) if values else None,
            "high": float(np.quantile(values, .975)) if values else None,
            "valid_resamples": len(values)}


def mean_interval(values, repeats=500, seed=20260928):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = (values[rng.integers(0, len(values), (repeats, len(values)))].mean(axis=1)
               if len(values) else [])
    return {"estimate": _mean(values), "ci95": _interval(samples)}


def rate_interval(values):
    """Wilson 95% intervals remain nondegenerate for all-success/failure samples."""
    n = len(values)
    if not n:
        return {"low": None, "high": None, "n": 0, "method": "Wilson"}
    rate, z = float(np.mean(values)), 1.959963984540054
    denominator = 1 + z*z/n
    center = (rate + z*z/(2*n))/denominator
    half = z*np.sqrt(rate*(1-rate)/n + z*z/(4*n*n))/denominator
    return {"low": max(0.0, float(center-half)), "high": min(1.0, float(center+half)),
            "n": n, "method": "Wilson"}


def forecast(outcomes, probabilities, repeats=500, seed=20260928):
    result = binary_forecast_metrics(outcomes, probabilities).to_dict()
    y, p = np.asarray(outcomes), np.asarray(probabilities)
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    result["ci95"] = {
        "brier": mean_interval((p-y)**2, repeats, seed)["ci95"],
        "log_loss": mean_interval(-y*np.log(clipped)-(1-y)*np.log(1-clipped), repeats, seed)["ci95"],
    }
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(repeats if len(y) else 0):
        ids = rng.integers(0, len(y), len(y))
        if len(set(y[ids])) == 2:
            aucs.append(float(roc_auc_score(y[ids], p[ids])))
    result["ci95"]["auroc"] = _interval(aucs)
    for row in result["reliability"]:
        mask = (p >= row["lower"]) & ((p <= row["upper"]) if row["upper"] == 1 else (p < row["upper"]))
        row["empirical_accuracy_ci95"] = rate_interval(y[mask])
    return result


def risk_coverage(outcomes, probabilities, total_n):
    """Release equal-score ties together; coverage uses the full partition."""
    points = []
    y, p = np.asarray(outcomes), np.asarray(probabilities)
    area, previous = 0.0, 0.0
    for threshold in sorted(set(probabilities), reverse=True):
        selected = p >= threshold
        coverage = int(selected.sum()) / total_n
        risk = float(1-y[selected].mean())
        points.append({"threshold": threshold, "coverage": coverage, "risk": risk,
                       "released_n": int(selected.sum())})
        area += (coverage-previous)*risk
        previous = coverage
    return {"points": points, "aurc_observed_coverage": area if points else None,
            "max_coverage": previous, "n_total": total_n,
            "convention": "right-step integral; ties released together; no extrapolation to missing scores"}


def _corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def dependence(records, tags):
    """Descriptive dependence checks, including dependence on generator confidence."""
    result = {}
    for left, right in combinations(["generator", *tags], 2):
        def prob(row, tag):
            if row["generator_answer"] is None:
                return None
            return (row["generator_p_correct"] if tag == "generator"
                    else row["verifiers"].get(tag, {}).get("p_correct"))
        usable = [r for r in records if prob(r, left) is not None and prob(r, right) is not None]
        y = np.asarray([_outcome(r) for r in usable])
        a = np.asarray([prob(r, left) for r in usable])
        b = np.asarray([prob(r, right) for r in usable])
        ea, eb = ((a >= .5) != y).astype(int), ((b >= .5) != y).astype(int)
        conditional = {}
        for label in (0, 1):
            mask = y == label
            aa, bb = a[mask], b[mask]
            # Total variation between empirical joint score-bin distribution
            # and product marginals. Sparse cells can inflate this diagnostic.
            joint = np.zeros((3, 3))
            for x, z in zip(aa, bb):
                joint[min(int(x*3), 2), min(int(z*3), 2)] += 1
            tv = None
            if len(aa):
                joint /= len(aa)
                tv = float(np.abs(joint-np.outer(joint.sum(1), joint.sum(0))).sum()/2)
            conditional[str(label)] = {"n": len(aa), "score_correlation": _corr(aa, bb),
                                        "joint_vs_independent_total_variation": tv}
        result[f"{left}|{right}"] = {
            "n": len(usable), "residual_correlation": _corr(a-y, b-y),
            "error_correlation_at_0.5": _corr(ea, eb),
            "error_agreement_at_0.5": _mean(ea == eb),
            "conditional_on_correctness": conditional,
        }
    return result


def _outcome(row):
    return correctness_outcome(row["generator_answer"], row["correct_index"])


def _valid(row):
    return row["generator_answer"] is not None and row["generator_p_correct"] is not None


def _paired_forecasts(y, before, after, repeats, seed):
    p, q, yy = np.asarray(before), np.asarray(after), np.asarray(y)
    brier_delta = (q-yy)**2-(p-yy)**2
    eps = np.finfo(float).eps
    p, q = np.clip(p, eps, 1-eps), np.clip(q, eps, 1-eps)
    return {
        "n": len(y), "direction": "negative favors updated forecast",
        "brier_delta": mean_interval(brier_delta, repeats, seed),
        "log_loss_delta": mean_interval(
            -yy*np.log(q)-(1-yy)*np.log(1-q)+yy*np.log(p)+(1-yy)*np.log(1-p), repeats, seed),
    }


def _stability(y, p, repeats, seed):
    rng = np.random.default_rng(seed)
    correct, incorrect = [], []
    for _ in range(repeats if y else 0):
        ids = rng.integers(0, len(y), len(y))
        try:
            fit = fit_verifier_likelihood([y[i] for i in ids], [p[i] for i in ids])
        except ValueError:
            continue
        correct.append(fit.p_bin_if_correct)
        incorrect.append(fit.p_bin_if_incorrect)
    return {"requested_resamples": repeats, "successful_resamples": len(correct),
            "failed_resamples": repeats-len(correct),
            "p_bin_if_correct_ci95": [_interval([r[i] for r in correct]) for i in range(3)],
            "p_bin_if_incorrect_ci95": [_interval([r[i] for r in incorrect]) for i in range(3)]}


def _policy_report(rows, tags, likelihoods, scenario, repeats, seed):
    reward, loss = scenario["correct_reward"], scenario["incorrect_loss"]
    costs = scenario["verifier_costs"]
    if len(costs) != len(tags) or reward <= 0 or loss <= 0 or any(c < 0 for c in costs):
        raise ValueError("invalid routing rewards, losses or verifier costs")
    threshold = loss/(reward+loss)
    names = ["always_answer", "always_abstain", "confidence_only", "query_one_verifier",
             "query_all_verifiers", "sequential_stopping"]
    stats = {name: [] for name in names}
    availability = {name: True for name in names}
    for name in ("query_one_verifier", "query_all_verifiers", "sequential_stopping"):
        required = tags[:1] if name == "query_one_verifier" else tags
        availability[name] = bool(required) and all(t in likelihoods for t in required)
    for row in rows:
        candidate, valid = row["generator_answer"] is not None, _valid(row)
        prior, outcome = row["generator_p_correct"], _outcome(row)
        scores = [row["verifiers"].get(t, {}).get("p_correct") for t in tags]
        for name in names:
            release, used, failure, available = False, 0, None, True
            if name == "always_answer":
                release = candidate
                failure = None if candidate else "invalid_answer"
            elif name == "always_abstain":
                pass
            elif not valid:
                failure = "invalid_generator_answer_or_confidence"
            elif name == "confidence_only":
                release = prior >= threshold
            else:
                required = tags[:1] if name == "query_one_verifier" else tags
                available = bool(required) and all(t in likelihoods for t in required)
                if not available:
                    failure = "likelihood_unavailable"
                elif name == "sequential_stopping":
                    decision = simulate_sequential_decision(
                        prior, scores, [likelihoods[t] for t in tags], costs, reward, loss)
                    release = decision.action == "assert"
                    used, failure = decision.verifiers_used, decision.failure
                else:
                    used = len(required)
                    if any(s is None for s in scores[:used]):
                        failure = "missing_verifier_score"
                    else:
                        posterior = prior
                        for t, s in zip(required, scores):
                            posterior = likelihoods[t].posterior(posterior, s)
                        release = posterior >= threshold
            cost = sum(costs[:used])
            utility = (reward if outcome else -loss) if release else 0.0
            stats[name].append({"release": release, "outcome": outcome, "queries": used,
                                "cost": cost, "utility": utility-cost, "failure": failure,
                                "available": available})
    report = {}
    for name, entries in stats.items():
        released = [e["outcome"] for e in entries if e["release"]]
        report[name] = {
            "n": len(rows), "released_n": len(released),
            "release_coverage": _mean([e["release"] for e in entries]),
            "release_coverage_ci95": rate_interval([e["release"] for e in entries]),
            "abstention_rate": _mean([not e["release"] for e in entries]),
            "accuracy_among_released": _mean(released),
            "accuracy_among_released_ci95": rate_interval(released),
            "mean_queries": _mean([e["queries"] for e in entries]),
            "mean_verification_cost": _mean([e["cost"] for e in entries]),
            "utility": mean_interval([e["utility"] for e in entries], repeats, seed),
            "failure_counts": dict(Counter(e["failure"] for e in entries if e["failure"])),
            "available": availability[name],
        }
    differences = {}
    for name in ("query_one_verifier", "query_all_verifiers", "sequential_stopping"):
        for baseline in ("confidence_only", "query_one_verifier"):
            if name == baseline:
                continue
            differences[f"{name}_minus_{baseline}"] = {
                "available": availability[name] and availability[baseline],
                "deltas": {
                metric: mean_interval([a[metric]-b[metric] for a, b in zip(stats[name], stats[baseline])], repeats, seed)
                for metric in ("utility", "cost")
                },
            }
    return {"n": len(rows), "policies": report, "paired_differences": differences,
            "assumes_conditional_independence": True,
            "prior": "raw generator confidence; not assumed calibrated",
            "utility_units": "normalized sensitivity analysis, not measured monetary cost",
            "failure_policy": "abstain on required missing input; charge attempted verifier calls; unfit policy marked unavailable"}


def build_report(records, config, *, bootstrap_repeats=500, seed=20260928, synthetic=False):
    """Pure offline scoring entry point; never imports or calls an inference engine."""
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    ids = [r["item_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item IDs in report")
    if any(r["partition"] not in {"calibration", "evaluation"} for r in records):
        raise ValueError("unknown partition")
    tags = [f"verifier_{i}" for i in range(1, len(config["models"]["verifiers"])+1)]
    calibration = [r for r in records if r["partition"] == "calibration"]
    valid_calibration = [r for r in calibration if _valid(r)]
    base_rate = _mean([_outcome(r) for r in valid_calibration])
    likelihoods, fits, logistics = {}, {}, {}
    for tag in tags:
        eligible = [r for r in calibration if r["generator_answer"] is not None
                    and r["verifiers"].get(tag, {}).get("p_correct") is not None]
        y = [_outcome(r) for r in eligible]
        p = [r["verifiers"][tag]["p_correct"] for r in eligible]
        fits[tag] = {"n": len(y), "correct_n": sum(y), "incorrect_n": len(y)-sum(y)}
        try:
            fitted = fit_verifier_likelihood(y, p)
            likelihoods[tag] = fitted
            logistics[tag] = LogisticRegression(C=1.0, solver="lbfgs").fit(np.asarray(p).reshape(-1, 1), y)
            fits[tag].update({"fit": True, "edges": fitted.edges,
                             "p_bin_if_correct": fitted.p_bin_if_correct,
                             "p_bin_if_incorrect": fitted.p_bin_if_incorrect,
                             "bootstrap_stability": _stability(y, p, bootstrap_repeats, seed)})
        except ValueError as error:
            fits[tag].update({"fit": False, "failure": str(error)})
    report = {
        "schema_version": 2, "synthetic": synthetic,
        "interpretation": ("Software validation only; synthetic results do not demonstrate real verifier calibration."
                           if synthetic else "Small-sample feasibility diagnostics, not a validation of incentive compatibility."),
        "planned_sample_size": config["sample_size"],
        "planned_calibration_size": config["calibration_size"],
        "planned_evaluation_size": config["sample_size"]-config["calibration_size"],
        "complete_pilot": len(records) == config["sample_size"] and len(calibration) == config["calibration_size"],
        "uncertainty": {"method": "item bootstrap, percentile 95%; binomial rates use Wilson 95%; forecast intervals conditional on fitted calibration model",
                        "repeats": bootstrap_repeats, "seed": seed,
                        "caution": "Tiny or degenerate samples may give misleadingly narrow intervals; AUROC resamples with one class are excluded."},
        "calibration": {"base_rate": base_rate, "base_rate_n": len(valid_calibration),
                        "base_rate_population": "valid generator answer and confidence, matching forecast cohort",
                        "verifier_fits": fits,
                        "logistic_specification": "score-only logistic regression, C=1, fit only on calibration"},
        "posterior_caution": "Raw generator prior and conditional independence are assumptions, not established calibration. Exact 0/1 priors cannot be updated. Dependence diagnostics do not certify independence.",
    }
    for subset in ("all", "calibration", "evaluation"):
        rows = records if subset == "all" else [r for r in records if r["partition"] == subset]
        usable = [r for r in rows if _valid(r)]
        y, p = [_outcome(r) for r in usable], [r["generator_p_correct"] for r in usable]
        outcome_all = [_outcome(r) for r in rows]
        section = {
            "n": len(rows), "subject_counts": dict(Counter(r["subject"] for r in rows)),
            "correct_n": sum(outcome_all), "incorrect_or_invalid_answer_n": len(rows)-sum(outcome_all),
            "generator_answer_accuracy_all_items": _mean(outcome_all),
            "generator_answer_accuracy_ci95": rate_interval(outcome_all),
            "generator_parse": dict(Counter(r["generator_failure"] or "parsed" for r in rows)),
            "generator_parse_failure_rate": _mean([not r["generator_ok"] for r in rows]),
            "generator_confidence_missing_n": len(rows)-len(usable),
            "generator_confidence": forecast(y, p, bootstrap_repeats, seed),
            "confidence_value_counts": dict(Counter(str(v) for v in p)),
            "endpoint_confidence_n": sum(v in (0, 1) for v in p),
            "generator_risk_coverage": risk_coverage(y, p, len(rows)),
            "dependence": dependence(rows, tags), "verifiers": {},
            "actual_verifier_calls_collected": sum(r["generator_answer"] is not None for r in rows)*len(tags),
            "actual_verifier_calls_collected_per_item": (
                sum(r["generator_answer"] is not None for r in rows)*len(tags)/len(rows) if rows else None),
        }
        if base_rate is not None:
            section["base_rate_forecast"] = forecast(y, [base_rate]*len(y), bootstrap_repeats, seed)
            section["generator_minus_base_rate"] = _paired_forecasts(y, [base_rate]*len(y), p, bootstrap_repeats, seed)
        else:
            section["base_rate_forecast"] = {"available": False, "reason": "no valid calibration forecasts"}
        for tag in tags:
            eligible = [r for r in rows if r["generator_answer"] is not None
                        and r["verifiers"].get(tag, {}).get("p_correct") is not None]
            vy = [_outcome(r) for r in eligible]
            vp = [r["verifiers"][tag]["p_correct"] for r in eligible]
            metrics = forecast(vy, vp, bootstrap_repeats, seed)
            attempted = [r for r in rows if r["generator_answer"] is not None]
            failed = [r for r in attempted if r["verifiers"].get(tag, {}).get("p_correct") is None]
            metrics.update({"missing_n": len(rows)-len(eligible),
                            "attempted_n": len(attempted), "parse_failure_n": len(failed),
                            "parse_failure_rate_among_attempted": len(failed)/len(attempted) if attempted else None,
                            "parse_counts": dict(Counter(
                                (r["verifiers"].get(tag, {}).get("failure") or "missing_response")
                                if r["verifiers"].get(tag, {}).get("p_correct") is None else "parsed" for r in rows)),
                            "risk_coverage": risk_coverage(vy, vp, len(rows))})
            if tag in likelihoods:
                paired = [r for r in eligible if _valid(r)]
                py = [_outcome(r) for r in paired]
                prior = [r["generator_p_correct"] for r in paired]
                post = [likelihoods[tag].posterior(r["generator_p_correct"], r["verifiers"][tag]["p_correct"]) for r in paired]
                metrics["posterior_from_generator_prior"] = forecast(py, post, bootstrap_repeats, seed)
                metrics["posterior_minus_generator"] = _paired_forecasts(py, prior, post, bootstrap_repeats, seed)
                metrics["posterior_risk_coverage"] = risk_coverage(py, post, len(rows))
                logistic_p = logistics[tag].predict_proba(np.asarray(vp).reshape(-1, 1))[:, 1].tolist() if vp else []
                metrics["calibrated_score_logistic"] = forecast(vy, logistic_p, bootstrap_repeats, seed)
            section["verifiers"][tag] = metrics
        if subset == "evaluation":
            section["routing_scenarios"] = {
                s["name"]: _policy_report(rows, tags, likelihoods, s, bootstrap_repeats, seed)
                for s in config.get("routing_scenarios", [])
            }
            paired = [r for r in usable if all(t in likelihoods and r["verifiers"].get(t, {}).get("p_correct") is not None for t in tags)]
            posteriors = []
            for row in paired:
                posterior = row["generator_p_correct"]
                for tag in tags:
                    posterior = likelihoods[tag].posterior(posterior, row["verifiers"][tag]["p_correct"])
                posteriors.append(posterior)
            section["sequential_independence_posterior"] = forecast([_outcome(r) for r in paired], posteriors, bootstrap_repeats, seed)
        report[f"{subset}_metrics"] = section
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="defaults to adjacent run_manifest.json config, then repository config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true", help="label software-validation fixtures explicitly")
    args = parser.parse_args()
    records_text = args.records.read_text()
    records = [json.loads(line) for line in records_text.splitlines() if line.strip()]
    manifest_path = args.records.parent/"run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.config is not None:
        config = json.loads(args.config.read_text())
    elif "config" in manifest:
        config = manifest["config"]
    else:
        config = json.loads((Path(__file__).resolve().parents[3]/"configs"/"gpqa_experiment.json").read_text())
    report = build_report(records, config, synthetic=args.synthetic)
    report["provenance"] = {
        "collection_run_id": manifest.get("run_id"),
        "records_sha256": hashlib.sha256(records_text.encode()).hexdigest(),
        "analysis_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "config_differs_from_collection": config != manifest["config"] if "config" in manifest else None,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
