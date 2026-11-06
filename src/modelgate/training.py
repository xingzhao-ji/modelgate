"""A fixed candidate set and a five-way temporal experiment, with no test tuning."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from modelgate.data import (
    CATEGORICAL,
    NUMERIC,
    chronological_split,
    prepare_dataset,
    validate_features,
)
from modelgate.metrics import evaluate_probabilities, select_threshold


@dataclass
class TrainingConfig:
    boost_iterations: int = 140
    seed: int = 20261023
    false_positive_cost: float = 1.0
    false_negative_cost: float = 3.0


STABLE_FEATURES = {
    "age",
    "campaign",
    "pdays",
    "previous",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "poutcome",
}


def make_preprocessor(stable=False) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                [name for name in NUMERIC if not stable or name in STABLE_FEATURES],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                [name for name in CATEGORICAL if not stable or name in STABLE_FEATURES],
            ),
        ],
        remainder="drop",
    )


@dataclass
class TrainedModel:
    estimator: Pipeline
    calibration_slope: float
    calibration_intercept: float
    threshold: float
    report: dict

    def predict_proba(self, features) -> np.ndarray:
        features = validate_features(features)
        with threadpool_limits(limits=1):
            raw = self.estimator.predict_proba(features)[:, 1]
        scores = logit(np.clip(raw, 1e-7, 1 - 1e-7))
        return expit(self.calibration_slope * scores + self.calibration_intercept)


def train(frame, config: TrainingConfig | None = None) -> TrainedModel:
    config = config or TrainingConfig()
    parts = chronological_split(len(frame))
    # Do not even inspect holdout values here: evaluation is a separate command.
    features, target = prepare_dataset(frame.iloc[: parts["test"].start])
    for name in ["train", "selection", "calibration", "policy"]:
        if len(np.unique(target[parts[name]])) != 2:
            raise ValueError(f"{name} must contain both target classes")
    candidates = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(C=1.0, max_iter=2000, random_state=config.seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=config.boost_iterations,
            learning_rate=0.07,
            max_leaf_nodes=15,
            l2_regularization=10.0,
            early_stopping=False,
            random_state=config.seed,
        ),
    }
    candidates["stable_logistic"] = LogisticRegression(
        C=1.0, max_iter=2000, random_state=config.seed
    )
    candidates["stable_hist_gradient_boosting"] = HistGradientBoostingClassifier(
        max_iter=config.boost_iterations,
        learning_rate=0.07,
        max_leaf_nodes=15,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=config.seed,
    )
    fitted, selection = {}, {}
    with threadpool_limits(limits=1):
        for name, estimator in candidates.items():
            pipeline = Pipeline(
                [
                    ("preprocess", make_preprocessor(stable=name.startswith("stable_"))),
                    ("classifier", estimator),
                ]
            )
            pipeline.fit(features.iloc[parts["train"]], target[parts["train"]])
            probabilities = pipeline.predict_proba(features.iloc[parts["selection"]])[:, 1]
            selection[name] = evaluate_probabilities(target[parts["selection"]], probabilities, 0.5)
            fitted[name] = pipeline
        # Dict insertion order is the deterministic tie-break; no test-based fallback.
        winner = max(selection, key=lambda name: selection[name]["average_precision"])
        estimator = fitted[winner]
        raw_calibration = estimator.predict_proba(features.iloc[parts["calibration"]])[:, 1]
        scores = logit(np.clip(raw_calibration, 1e-7, 1 - 1e-7)).reshape(-1, 1)
        sigmoid = LogisticRegression(C=10.0, max_iter=1000, random_state=config.seed)
        sigmoid.fit(scores, target[parts["calibration"]])
    report = {
        "config": asdict(config),
        "selection_metric": "average_precision",
        "selected_model": winner,
        "selection_metrics": selection,
        "partitions": {
            name: {"start_inclusive": s.start, "stop_exclusive": s.stop, "rows": s.stop - s.start}
            for name, s in parts.items()
        },
        "development_prevalence": {
            name: float(target[parts[name]].mean())
            for name in ["train", "selection", "calibration", "policy"]
        },
        "calibration": {
            "method": "logistic_on_clipped_log_odds",
            "partition": "calibration",
            "regularization_C": 10.0,
        },
        "excluded_features": ["duration", "y"],
        "candidate_expansion": (
            "Stable-feature ablations added after original three models' validation results; "
            "test still sealed."
        ),
        "stable_features": sorted(STABLE_FEATURES),
    }
    result = TrainedModel(
        estimator, float(sigmoid.coef_[0, 0]), float(sigmoid.intercept_[0]), 0.5, report
    )
    policy_y = target[parts["policy"]]
    policy_features = features.iloc[parts["policy"]]
    calibrated = result.predict_proba(policy_features)
    policy = select_threshold(
        policy_y, calibrated, config.false_positive_cost, config.false_negative_cost
    )
    result.threshold = policy["threshold"]
    report["policy"] = policy
    with threadpool_limits(limits=1):
        raw_policy = estimator.predict_proba(policy_features)[:, 1]
    report["policy_calibration_comparison"] = {
        "raw": evaluate_probabilities(policy_y, raw_policy, result.threshold),
        "calibrated": evaluate_probabilities(policy_y, calibrated, result.threshold),
    }
    return result
