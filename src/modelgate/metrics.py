"""Probability quality, calibration, and explicit illustrative decision costs."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_probabilities(target, probabilities, threshold) -> dict:
    y, p = np.asarray(target), np.asarray(probabilities)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("nonempty aligned finite predictions are required")
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y)) == 2
    bins = []
    assignment = np.minimum((p * 10).astype(int), 9)
    for number in range(10):
        mask = assignment == number
        bins.append(
            {
                "lower": number / 10,
                "upper": (number + 1) / 10,
                "count": int(mask.sum()),
                "mean_probability": float(p[mask].mean()) if mask.any() else None,
                "observed_rate": float(y[mask].mean()) if mask.any() else None,
            }
        )
    ece = sum(
        b["count"] * abs(b["mean_probability"] - b["observed_rate"]) for b in bins if b["count"]
    ) / len(y)
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if two_classes else None,
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else None,
        "brier": float(brier_score_loss(y, p)),
        "ece_10_bins": float(ece),
        "mean_probability": float(p.mean()),
        "threshold": float(threshold),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "selection_rate": float((tp + fp) / len(y)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "calibration_bins": bins,
    }


def select_threshold(target, probabilities, false_positive_cost=1, false_negative_cost=3) -> dict:
    """Minimize cost on the policy window; ties prefer fewer contacts (higher threshold)."""
    y, p = np.asarray(target), np.asarray(probabilities)
    order = np.argsort(p, kind="stable")
    y, p = y[order], p[order]
    thresholds = np.unique(np.concatenate(([0.0], p, [1.0])))
    indices = np.searchsorted(p, thresholds, side="left")
    cumulative_positive = np.concatenate(([0], np.cumsum(y)))
    fn = cumulative_positive[indices]
    fp = (len(y) - indices) - (y.sum() - fn)
    costs = fp * false_positive_cost + fn * false_negative_cost
    best = np.flatnonzero(costs == costs.min())[-1]
    return {
        "threshold": float(thresholds[best]),
        "total_cost": float(costs[best]),
        "cost_per_row": float(costs[best] / len(y)),
        "false_positive_cost": false_positive_cost,
        "false_negative_cost": false_negative_cost,
        "contact_none_cost": float(y.sum() * false_negative_cost),
        "contact_all_cost": float((len(y) - y.sum()) * false_positive_cost),
        "selection_partition": "policy",
        "assumption": "illustrative units, not bank economics",
    }
