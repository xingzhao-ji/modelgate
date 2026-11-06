"""Explicit feature contract and chronological isolation boundaries."""

from numbers import Real

import numpy as np
import pandas as pd

NUMERIC = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
FEATURES = NUMERIC + CATEGORICAL
BOUNDS = {
    "age": (0, 120),
    "campaign": (1, 1000),
    "pdays": (0, 10000),
    "previous": (0, 1000),
    "emp.var.rate": (-100, 100),
    "cons.price.idx": (0, 1000),
    "cons.conf.idx": (-1000, 1000),
    "euribor3m": (-100, 100),
    "nr.employed": (0, 100000),
}


def chronological_split(n: int) -> dict[str, slice]:
    """Row order is UCI's date order; validation has three disjoint responsibilities."""
    if n < 100:
        raise ValueError("at least 100 ordered rows are required")
    bounds = [0, int(n * 0.6), int(n * 0.7), int(n * 0.8), int(n * 0.85), n]
    names = ["train", "selection", "calibration", "policy", "test"]
    return {
        name: slice(start, stop)
        for name, start, stop in zip(names, bounds[:-1], bounds[1:], strict=True)
    }


def prepare_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if "y" not in frame or not frame["y"].isin(["yes", "no"]).all():
        raise ValueError("target y must contain only yes/no")
    # Explicit allowlist: target and duration cannot enter any candidate's pipeline.
    if set(frame.columns) - set(FEATURES) - {"duration", "y"}:
        raise ValueError("unexpected dataset columns")
    features = validate_features(frame.drop(columns=["y", "duration"], errors="ignore"))
    return features, frame["y"].eq("yes").to_numpy(dtype=np.int64)


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("records must be nonempty with unique column names")
    missing, extra = set(FEATURES) - set(frame.columns), set(frame.columns) - set(FEATURES)
    if missing or extra:
        raise ValueError(f"feature mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}")
    result = frame[FEATURES].copy()
    for column in NUMERIC:
        present = result[column].dropna()
        if not present.map(
            lambda v: isinstance(v, Real) and not isinstance(v, (bool, np.bool_))
        ).all():
            raise ValueError(f"{column} must be a JSON number or null; coercion is disabled")
        low, high = BOUNDS[column]
        if (
            not np.isfinite(present.to_numpy(dtype=float)).all()
            or not present.between(low, high).all()
        ):
            raise ValueError(f"{column} outside finite range [{low}, {high}]")
        result[column] = result[column].astype(float)
    for column in CATEGORICAL:
        present = result[column].dropna()
        if not present.map(lambda v: isinstance(v, str) and 1 <= len(v) <= 100).all():
            raise ValueError(f"{column} must be a string of 1..100 characters or null")
        result[column] = result[column].fillna("unknown").astype(object)
    return result
