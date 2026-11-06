"""Train-derived population stability profiles; no labels or online retention."""

import numpy as np

from modelgate.data import CATEGORICAL, NUMERIC, validate_features


def _counts(series, profile):
    if profile["kind"] == "numeric":
        valid = series.dropna().to_numpy(dtype=float)
        indices = np.searchsorted(profile["cuts"], valid, side="right")
        counts = np.bincount(indices, minlength=len(profile["cuts"]) + 1).astype(float)
        return np.append(counts, series.isna().sum())
    vocabulary = profile["categories"]
    counts = series.value_counts()
    known = [float(counts.get(category, 0)) for category in vocabulary]
    return np.array(known + [float((~series.isin(vocabulary)).sum())])


def build_reference(features) -> dict:
    features = validate_features(features)
    profiles = {}
    for name in NUMERIC:
        values = features[name].dropna().to_numpy()
        cuts = np.unique(np.quantile(values, np.linspace(0, 1, 11))).tolist() if len(values) else []
        profiles[name] = {"kind": "numeric", "cuts": cuts}
    for name in CATEGORICAL:
        profiles[name] = {"kind": "categorical", "categories": sorted(features[name].unique())}
    for name, profile in profiles.items():
        counts = _counts(features[name], profile)
        profile["probabilities"] = (counts / counts.sum()).tolist()
    return {"rows": len(features), "partition": "train", "features": profiles}


def compare_drift(reference, features, threshold=0.25) -> dict:
    features = validate_features(features)
    if len(features) < 100:
        raise ValueError("drift requires at least 100 records; small windows are too noisy")
    reports = {}
    for name, profile in reference["features"].items():
        counts = _counts(features[name], profile)
        observed = counts / counts.sum()
        baseline = np.array(profile["probabilities"])
        # Same fixed epsilon for both sides: identical distributions remain exactly zero.
        a, b = np.clip(baseline, 1e-4, 1), np.clip(observed, 1e-4, 1)
        a, b = a / a.sum(), b / b.sum()
        psi = float(np.sum((b - a) * np.log(b / a)))
        reports[name] = {
            "psi": psi,
            "alert": psi > threshold,
            "missing_rate": float(features[name].isna().mean())
            if name in NUMERIC
            else float(features[name].eq("unknown").mean()),
            "unseen_rate": float(observed[-1]) if profile["kind"] == "categorical" else None,
        }
    return {
        "rows": len(features),
        "reference_rows": reference["rows"],
        "threshold": threshold,
        "max_psi": max(r["psi"] for r in reports.values()),
        "alert": any(r["alert"] for r in reports.values()),
        "features": reports,
        "interpretation": "heuristic covariate shift alert, not a significance or accuracy test",
    }
