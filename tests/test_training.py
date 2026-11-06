import numpy as np
import pytest

from modelgate.metrics import evaluate_probabilities, select_threshold
from modelgate.training import TrainingConfig, train


def test_probabilistic_metrics_are_computed_against_truth():
    report = evaluate_probabilities(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.8]), 0.5)
    assert report["average_precision"] == 1.0
    assert report["roc_auc"] == 1.0
    assert report["brier"] == pytest.approx(0.045)
    assert report["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert sum(b["count"] for b in report["calibration_bins"]) == 4


def test_single_class_slice_is_explicitly_unscorable():
    report = evaluate_probabilities(np.array([0, 0]), np.array([0.2, 0.3]), 0.5)
    assert report["roc_auc"] is None and report["average_precision"] is None
    assert report["brier"] == pytest.approx(0.065)


def test_threshold_is_chosen_from_policy_labels_using_stated_cost():
    result = select_threshold(np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.3, 0.9]), 1, 3)
    assert result["threshold"] == 0.2  # Cost=1, vs threshold=.9 cost=3, .1 cost=2.
    assert result["total_cost"] == 1
    assert result["false_positive_cost"] == 1
    assert result["false_negative_cost"] == 3


def test_training_chooses_validation_winner_and_keeps_test_unread(toy_data):
    config = TrainingConfig(boost_iterations=12)
    first = train(toy_data, config)
    changed = toy_data.copy()
    changed.loc[850:, "y"] = np.where(changed.loc[850:, "y"] == "yes", "no", "yes")
    changed.loc[850:, "age"] = 100
    second = train(changed, config)
    candidates = first.report["selection_metrics"]
    assert set(candidates) == {
        "dummy",
        "logistic",
        "hist_gradient_boosting",
        "stable_logistic",
        "stable_hist_gradient_boosting",
    }
    assert first.report["selected_model"] == max(
        candidates, key=lambda n: candidates[n]["average_precision"]
    )
    assert first.report == second.report
    assert first.threshold == second.threshold
    features = toy_data.drop(columns=["y", "duration"]).iloc[:20]
    np.testing.assert_allclose(
        first.predict_proba(features), second.predict_proba(features), atol=0, rtol=0
    )
    assert "test_metrics" not in first.report


def test_calibration_changes_probability_scale(toy_data):
    fitted = train(toy_data, TrainingConfig(boost_iterations=12))
    features = toy_data.drop(columns=["y", "duration"]).iloc[-30:]
    probabilities = fitted.predict_proba(features)
    assert np.isfinite(probabilities).all() and ((probabilities > 0) & (probabilities < 1)).all()
    assert not np.allclose(probabilities, fitted.estimator.predict_proba(features)[:, 1])


def test_stable_ablation_is_invariant_to_calendar_and_macro_changes(toy_data):
    from modelgate.training import make_preprocessor

    features = toy_data.drop(columns=["y", "duration"])
    preprocessor = make_preprocessor(stable=True).fit(features.iloc[:600])
    original = features.iloc[600:700].copy()
    shifted = original.copy()
    for name in ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]:
        shifted[name] = 12345
    shifted["month"], shifted["day_of_week"] = "dec", "fri"
    np.testing.assert_array_equal(preprocessor.transform(original), preprocessor.transform(shifted))
