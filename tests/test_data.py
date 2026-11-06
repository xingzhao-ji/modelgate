import numpy as np
import pandas as pd
import pytest

from modelgate.data import chronological_split, prepare_dataset, validate_features
from modelgate.training import make_preprocessor


def test_split_is_contiguous_ordered_disjoint_and_exhaustive():
    parts = chronological_split(1000)
    assert list(parts) == ["train", "selection", "calibration", "policy", "test"]
    assert [(s.start, s.stop) for s in parts.values()] == [
        (0, 600),
        (600, 700),
        (700, 800),
        (800, 850),
        (850, 1000),
    ]
    assert np.array_equal(
        np.concatenate([np.arange(1000)[s] for s in parts.values()]), np.arange(1000)
    )


def test_dataset_never_exposes_duration_or_outcome_as_features(toy_data):
    features, target = prepare_dataset(toy_data)
    assert "duration" not in features and "y" not in features
    assert len(features.columns) == 19
    assert set(target) == {0, 1}
    bad = toy_data.copy()
    bad.loc[0, "y"] = "maybe"
    with pytest.raises(ValueError, match="target"):
        prepare_dataset(bad)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.pop("age"),
        lambda r: r.update(duration=123),
        lambda r: r.update(y="yes"),
        lambda r: r.update(age="42"),
        lambda r: r.update(age=True),
        lambda r: r.update(age=np.inf),
        lambda r: r.update(age=-1),
        lambda r: r.update(job=55),
        lambda r: r.update(job="x" * 101),
    ],
)
def test_schema_rejects_invalid_records_without_silent_coercion(record, mutation):
    mutation(record)
    with pytest.raises(ValueError):
        validate_features(pd.DataFrame([record]))


def test_missing_values_and_unseen_categories_are_supported(record):
    record.update(age=None, job="new-occupation")
    frame = validate_features(pd.DataFrame([record]))
    assert frame.age.isna().all() and frame.job.iloc[0] == "new-occupation"


def test_preprocessing_statistics_and_vocabulary_are_train_only(record):
    train = pd.DataFrame([dict(record, age=20), dict(record, age=40)])
    future = pd.DataFrame([dict(record, age=100, job="future-only")])
    preprocessor = make_preprocessor()
    preprocessor.fit(train)
    transformed = preprocessor.transform(future)
    assert np.isfinite(transformed).all()
    numeric = preprocessor.named_transformers_["numeric"]
    assert numeric.named_steps["imputer"].statistics_[0] == 30
    assert numeric.named_steps["scale"].mean_[0] == 30
    assert (
        "future-only"
        not in preprocessor.named_transformers_["categorical"].named_steps["encode"].categories_[0]
    )


def test_split_rejects_too_little_data():
    with pytest.raises(ValueError, match="at least"):
        chronological_split(30)
