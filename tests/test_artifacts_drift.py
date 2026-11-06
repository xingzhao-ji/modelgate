import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from modelgate.artifacts import ArtifactError, load_artifact, save_artifact
from modelgate.drift import build_reference, compare_drift
from modelgate.training import TrainingConfig, train


@pytest.fixture
def bundle(tmp_path, toy_data):
    model = train(toy_data, TrainingConfig(boost_iterations=8))
    features = toy_data.drop(columns=["y", "duration"])
    reference = build_reference(features.iloc[:600])
    path = save_artifact(model, reference, tmp_path / "run", {"sha256": "a" * 64, "rows": 1000})
    return path, model, features


def test_model_roundtrip_preserves_real_probabilities_and_identity(bundle):
    path, original, features = bundle
    loaded = load_artifact(path)
    np.testing.assert_allclose(
        loaded.model.predict_proba(features.iloc[:10]), original.predict_proba(features.iloc[:10])
    )
    assert len(loaded.model_id) == 16
    assert loaded.metadata["dataset"]["sha256"] == "a" * 64
    with pytest.raises(FileExistsError):
        save_artifact(original, loaded.reference, path, loaded.metadata["dataset"])


@pytest.mark.parametrize("filename", ["model.skops", "metadata.json", "reference.json"])
def test_corruption_is_detected_before_model_load(bundle, filename):
    path, _, _ = bundle
    with (path / filename).open("ab") as output:
        output.write(b"corruption")
    with pytest.raises(ArtifactError, match="integrity"):
        load_artifact(path)


def test_manifest_cannot_point_outside_artifact(bundle):
    path, _, _ = bundle
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"]["../outside"] = "a" * 64
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="file set"):
        load_artifact(path)


def test_metadata_schema_is_validated_even_if_checksum_is_recomputed(bundle):
    path, _, _ = bundle
    metadata = json.loads((path / "metadata.json").read_text())
    metadata["threshold"] = -4
    (path / "metadata.json").write_text(json.dumps(metadata))
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"]["metadata.json"] = hashlib.sha256(
        (path / "metadata.json").read_bytes()
    ).hexdigest()
    manifest["model_id"] = hashlib.sha256(
        (json.dumps(manifest["files"], sort_keys=True, indent=2) + "\n").encode()
    ).hexdigest()[:16]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError):
        load_artifact(path)


def test_invalid_reference_probabilities_fail_load_even_with_consistent_hashes(bundle):
    path, _, _ = bundle
    reference = json.loads((path / "reference.json").read_text())
    reference["features"]["age"]["probabilities"] = [-1, 2]
    (path / "reference.json").write_text(json.dumps(reference))
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"]["reference.json"] = hashlib.sha256(
        (path / "reference.json").read_bytes()
    ).hexdigest()
    manifest["model_id"] = hashlib.sha256(
        (json.dumps(manifest["files"], sort_keys=True, indent=2) + "\n").encode()
    ).hexdigest()[:16]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="reference"):
        load_artifact(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda metadata: metadata.pop("dataset"),
        lambda metadata: metadata.update(training_report={}),
        lambda metadata: metadata.update(calibration_slope=10**400),
        lambda metadata: metadata["training_report"]["partitions"]["test"].update(
            start_inclusive=850.0
        ),
        lambda metadata: metadata["dataset"].update(rows=999),
        lambda metadata: metadata["training_report"]["partitions"]["test"].update(
            start_inclusive=0
        ),
    ],
)
def test_incomplete_semantic_metadata_cannot_become_ready(bundle, mutation):
    path, _, _ = bundle
    metadata = json.loads((path / "metadata.json").read_text())
    mutation(metadata)
    (path / "metadata.json").write_text(json.dumps(metadata))
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"]["metadata.json"] = hashlib.sha256(
        (path / "metadata.json").read_bytes()
    ).hexdigest()
    manifest["model_id"] = hashlib.sha256(
        (json.dumps(manifest["files"], sort_keys=True, indent=2) + "\n").encode()
    ).hexdigest()[:16]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError):
        load_artifact(path)


def test_gradient_boosting_winner_can_be_saved_and_loaded(tmp_path, toy_data):
    nonlinear = toy_data.copy()
    nonlinear["y"] = np.where(nonlinear.age.between(40, 60), "yes", "no")
    model = train(nonlinear, TrainingConfig(boost_iterations=12))
    assert "gradient_boosting" in model.report["selected_model"]
    features = nonlinear.drop(columns=["duration", "y"])
    path = save_artifact(
        model,
        build_reference(features.iloc[:600]),
        tmp_path / "tree",
        {"sha256": "a" * 64, "rows": 1000},
    )
    loaded = load_artifact(path)
    np.testing.assert_array_equal(
        model.predict_proba(features.iloc[-25:]), loaded.model.predict_proba(features.iloc[-25:])
    )


def test_drift_detects_distribution_shift_and_unseen_categories(toy_data):
    features = toy_data.drop(columns=["y", "duration"])
    profile = build_reference(features)
    same = compare_drift(profile, features)
    assert same["alert"] is False and same["max_psi"] == pytest.approx(0.0)
    shifted = features.copy()
    shifted["age"] = 99
    shifted["job"] = "brand-new-job"
    result = compare_drift(profile, shifted)
    assert result["alert"] is True
    assert result["features"]["age"]["psi"] > 0.25
    assert result["features"]["job"]["unseen_rate"] == 1.0


def test_numeric_missingness_drift_is_visible(record):
    frame = pd.DataFrame([record] * 200)
    profile = build_reference(frame)
    missing = frame.copy()
    missing["age"] = np.nan
    result = compare_drift(profile, missing)
    assert result["features"]["age"]["missing_rate"] == 1.0
    assert result["features"]["age"]["alert"] is True


def test_small_drift_window_is_rejected(record):
    frame = pd.DataFrame([record] * 200)
    with pytest.raises(ValueError, match="100"):
        compare_drift(build_reference(frame), frame.iloc[:2])
