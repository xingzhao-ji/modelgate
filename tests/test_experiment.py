import pytest

from modelgate.artifacts import save_artifact, sha256
from modelgate.download import CSV_NAME
from modelgate.drift import build_reference
from modelgate.experiment import run_experiment
from modelgate.training import TrainingConfig, train


@pytest.mark.integration
def test_failure_drill_measures_real_http_and_recovers(tmp_path, toy_data):
    data = tmp_path / "data"
    data.mkdir()
    csv = data / CSV_NAME
    toy_data.to_csv(csv, sep=";", index=False)
    model = train(toy_data, TrainingConfig(boost_iterations=8))
    features = toy_data.drop(columns=["duration", "y"])
    artifact = save_artifact(
        model,
        build_reference(features.iloc[:600]),
        tmp_path / "model",
        {"sha256": sha256(csv.read_bytes()), "rows": 1000},
    )
    report = run_experiment(artifact, data, requests=5)
    assert report["benchmark"]["single"]["requests"] == 5
    assert report["benchmark"]["single"]["errors"] == 0
    assert report["benchmark"]["single"]["latency_ms"]["p95"] > 0
    assert report["failures"]["malformed_request_status"] == 422
    assert report["failures"]["corrupted_artifact_health_status"] == 503
    assert report["failures"]["healthy_request_after_rejection_status"] == 200
    assert report["drift"]["same_training_distribution"]["alert"] is False
    assert report["drift"]["synthetic_shift"]["alert"] is True
