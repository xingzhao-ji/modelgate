"""Sealed holdout assessment. The report cannot modify the fitted model or threshold."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from modelgate.data import prepare_dataset
from modelgate.drift import compare_drift
from modelgate.metrics import evaluate_probabilities


def block_bootstrap_intervals(y, p, repetitions=200, block_size=100) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    if repetitions < 10 or not 1 <= block_size <= len(y):
        raise ValueError("at least 10 bootstrap repetitions and valid block_size required")
    rng, values = np.random.default_rng(20261023), []
    for _ in range(repetitions):
        starts = rng.integers(0, len(y) - block_size + 1, int(np.ceil(len(y) / block_size)))
        indices = np.concatenate([np.arange(start, start + block_size) for start in starts])[
            : len(y)
        ]
        by, bp = y[indices], p[indices]
        if len(np.unique(by)) == 2:
            values.append(
                [average_precision_score(by, bp), roc_auc_score(by, bp), brier_score_loss(by, bp)]
            )
    if not values:
        return {"method": "moving_block_percentile", "error": "no resamples contain both classes"}
    intervals = np.quantile(np.array(values), [0.025, 0.975], axis=0)
    return {
        "method": "moving_block_percentile",
        "confidence": 0.95,
        "block_size": block_size,
        "repetitions": repetitions,
        "usable_repetitions": len(values),
        **{
            name: intervals[:, index].tolist()
            for index, name in enumerate(["average_precision", "roc_auc", "brier"])
        },
        "caveat": (
            "row blocks approximate temporal dependence; not a guarantee under nonstationarity"
        ),
    }


def evaluate(artifact, frame, repetitions=200) -> dict:
    partition = artifact.model.report["partitions"]["test"]
    if len(frame) != partition["stop_exclusive"]:
        raise ValueError("row count differs from training dataset")
    holdout = frame.iloc[partition["start_inclusive"] : partition["stop_exclusive"]]
    features, target = prepare_dataset(holdout)
    p = artifact.model.predict_proba(features)
    metrics = evaluate_probabilities(target, p, artifact.model.threshold)
    train_prior = artifact.model.report["development_prevalence"]["train"]
    calibration_prior = artifact.model.report["development_prevalence"]["calibration"]
    baselines = {
        "train_prior": evaluate_probabilities(target, np.full(len(target), train_prior), 0.5),
        "calibration_prior": evaluate_probabilities(
            target, np.full(len(target), calibration_prior), 0.5
        ),
    }
    slices = {}
    bands = pd.cut(
        features["age"], [0, 30, 59, 120], labels=["0-30", "31-59", "60-120"], include_lowest=True
    )
    for group_name, groups in {
        "age_band": bands,
        "job": features["job"],
        "contact": features["contact"],
    }.items():
        for group in sorted(groups.dropna().unique()):
            mask = groups.eq(group).to_numpy()
            if mask.sum() >= 50:
                slices[f"{group_name}:{group}"] = evaluate_probabilities(
                    target[mask], p[mask], artifact.model.threshold
                )
            else:
                slices[f"{group_name}:{group}"] = {
                    "rows": int(mask.sum()),
                    "suppressed": "fewer than 50 rows",
                }
    drift = compare_drift(artifact.reference, features)
    # Predeclared illustrative acceptance rules. Report only; no implicit deployment.
    checks = {
        "average_precision_lift_at_least_0.02": metrics["average_precision"]
        >= baselines["train_prior"]["average_precision"] + 0.02,
        "roc_auc_at_least_0.60": metrics["roc_auc"] >= 0.60,
        "brier_beats_calibration_prior": metrics["brier"] < baselines["calibration_prior"]["brier"],
        "ece_at_most_0.08": metrics["ece_10_bins"] <= 0.08,
        "no_covariate_drift_alert": not drift["alert"],
    }
    return {
        "model_id": artifact.model_id,
        "dataset_sha256": artifact.metadata["dataset"]["sha256"],
        "selected_model": artifact.model.report["selected_model"],
        "partition": partition,
        "metrics": metrics,
        "baselines": baselines,
        "uncertainty": block_bootstrap_intervals(target, p, repetitions, min(100, len(target))),
        "slices": slices,
        "drift": drift,
        "gate": {
            "decision": "eligible_for_review" if all(checks.values()) else "hold",
            "checks": checks,
            "automated_deployment": False,
            "rules": "illustrative criteria fixed before opening holdout; no business validation",
        },
        "evaluation_protocol": (
            "selected estimator, calibration and threshold frozen before this test report"
        ),
    }
