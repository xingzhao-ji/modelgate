import json
import socket
import subprocess
import sys
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from modelgate.api import create_app
from modelgate.artifacts import save_artifact
from modelgate.drift import build_reference
from modelgate.training import TrainingConfig, train


@pytest.fixture
def artifact(tmp_path, toy_data):
    model = train(toy_data, TrainingConfig(boost_iterations=8))
    features = toy_data.drop(columns=["y", "duration"])
    return save_artifact(
        model,
        build_reference(features.iloc[:600]),
        tmp_path / "model",
        {"sha256": "a" * 64, "rows": 1000},
    )


def test_scoring_real_trained_model_health_metadata_and_unseen_categories(artifact, record):
    with TestClient(create_app(artifact)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert client.get("/v1/model").json()["model_id"] == health.json()["model_id"]
        record["job"] = "unseen-job"
        response = client.post("/v1/predict", json={"records": [record, dict(record, age=None)]})
        assert response.status_code == 200
        result = response.json()
        assert len(result["predictions"]) == 2
        assert all(0 <= row["probability"] <= 1 for row in result["predictions"])
        assert result["model_id"] == health.json()["model_id"]
        assert "x-request-id" in response.headers


@pytest.mark.parametrize(
    "kind",
    ["missing", "extra", "mixed_missing", "empty", "oversized", "coerced", "unknown_envelope"],
)
def test_bad_requests_are_rejected_without_breaking_service(artifact, record, kind):
    records = [record.copy()]
    payload = {"records": records}
    if kind == "missing":
        records[0].pop("age")
    elif kind == "extra":
        records[0]["duration"] = 100
    elif kind == "mixed_missing":
        records.append(record.copy())
        records[1].pop("age")
    elif kind == "empty":
        payload["records"] = []
    elif kind == "oversized":
        payload["records"] = [record] * 513
    elif kind == "coerced":
        records[0]["age"] = "42"
    elif kind == "unknown_envelope":
        payload["debug"] = True
    with TestClient(create_app(artifact)) as client:
        assert client.post("/v1/predict", json=payload).status_code == 422
        assert client.get("/health").status_code == 200


def test_corrupted_artifact_leaves_health_unready_and_scoring_unavailable(artifact, record):
    (artifact / "model.skops").write_bytes(b"broken")
    with TestClient(create_app(artifact)) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/v1/predict", json={"records": [record]}).status_code == 503


def test_drift_endpoint_uses_training_reference(artifact, record):
    with TestClient(create_app(artifact)) as client:
        result = client.post("/v1/drift", json={"records": [dict(record, age=99, job="new")] * 100})
        assert result.status_code == 200
        assert result.json()["alert"] is True
        assert client.post("/v1/drift", json={"records": [record]}).status_code == 422


@pytest.mark.parametrize(
    "field,value", [("age", 10**400), ("age", float("nan")), ("job", float("nan"))]
)
def test_raw_json_extremes_do_not_bypass_schema_or_crash(artifact, record, field, value):
    record[field] = value
    with TestClient(create_app(artifact), raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/predict",
            content=json.dumps({"records": [record]}),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert client.get("/health").status_code == 200


@pytest.mark.integration
def test_actual_http_server_and_batch_cli(artifact, record, tmp_path):
    # Real TCP HTTP, independent interpreter, real on-disk serialized estimator.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "modelgate",
            "serve",
            "--artifact",
            str(artifact),
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2) as client:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(process.stdout.read())
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.05)
            else:
                pytest.fail("server did not become ready")
            response = client.post("/v1/predict", json={"records": [record]})
            assert response.status_code == 200
            body = response.json()
            assert 0 < body["predictions"][0]["probability"] < 1
        source, destination = tmp_path / "batch.json", tmp_path / "predictions.json"
        source.write_text(json.dumps({"records": [record]}))
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "modelgate",
                "predict",
                "--artifact",
                str(artifact),
                "--input",
                str(source),
                "--output",
                str(destination),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(destination.read_text())["predictions"] == body["predictions"]
    finally:
        process.terminate()
        process.communicate(timeout=10)
