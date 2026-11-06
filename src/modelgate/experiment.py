"""Actual loopback HTTP measurements and reproducible negative-path experiments."""

import platform
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

import httpx
import numpy as np
import pandas as pd

from modelgate.artifacts import load_artifact, sha256
from modelgate.data import prepare_dataset
from modelgate.download import CSV_NAME
from modelgate.drift import compare_drift


@contextmanager
def local_server(artifact, expected_health=200):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with TemporaryFile(mode="w+t") as log:
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
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            with httpx.Client(base_url=url, timeout=2) as client:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(f"server exited: {log.read()[-2000:]}")
                    try:
                        if client.get("/health").status_code == expected_health:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.05)
                else:
                    raise RuntimeError("server readiness deadline exceeded")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _benchmark(client, records, requests, concurrency=1):
    def request_once(_):
        start = time.perf_counter()
        response = client.post("/v1/predict", json={"records": records})
        return (time.perf_counter() - start) * 1000, response.status_code

    for _ in range(5):
        request_once(0)
    start = time.perf_counter()
    if concurrency == 1:
        samples = [request_once(i) for i in range(requests)]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            samples = list(executor.map(request_once, range(requests)))
    duration = time.perf_counter() - start
    latency = np.array([sample[0] for sample in samples])
    errors = sum(status != 200 for _, status in samples)
    return {
        "requests": requests,
        "rows_per_request": len(records),
        "concurrency": concurrency,
        "errors": errors,
        "elapsed_seconds": duration,
        "requests_per_second": requests / duration,
        "rows_per_second": requests * len(records) / duration,
        "latency_ms": {
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
            "p99": float(np.percentile(latency, 99)),
            "max": float(latency.max()),
        },
    }


def run_experiment(artifact_path, data_directory, requests=100) -> dict:
    if requests < 5 or requests > 10000:
        raise ValueError("requests must be between 5 and 10000")
    artifact = load_artifact(artifact_path)
    csv = Path(data_directory) / CSV_NAME
    if sha256(csv.read_bytes()) != artifact.metadata["dataset"]["sha256"]:
        raise ValueError("experiment dataset hash mismatch")
    frame = pd.read_csv(csv, sep=";")
    features, _ = prepare_dataset(frame)
    train_end = artifact.model.report["partitions"]["train"]["stop_exclusive"]
    same = compare_drift(artifact.reference, features.iloc[:train_end])
    start = artifact.model.report["partitions"]["test"]["start_inclusive"]
    test_features = features.iloc[start:]
    records = test_features.iloc[:32].to_dict(orient="records")
    shifted = test_features.iloc[:500].copy()
    shifted["age"], shifted["job"] = 99, "occupation-never-seen"
    with local_server(artifact_path) as url, httpx.Client(base_url=url, timeout=20) as client:
        benchmarks = {
            "single": _benchmark(client, records[:1], requests),
            "batch_32": _benchmark(client, records, requests),
            "concurrent_4": _benchmark(client, records[:1], requests, concurrency=4),
        }
        bad = dict(records[0], duration=300)
        rejected = client.post("/v1/predict", json={"records": [bad]}).status_code
        recovered = client.post("/v1/predict", json={"records": records[:1]}).status_code
        shifted_response = client.post(
            "/v1/drift", json={"records": shifted.to_dict(orient="records")}
        )
        shifted_response.raise_for_status()
        synthetic_shift = shifted_response.json()
        stats = client.get("/v1/stats").json()
    with TemporaryDirectory(prefix="modelgate-corrupt-") as temp:
        corrupted = Path(temp) / "artifact"
        shutil.copytree(artifact_path, corrupted)
        with (corrupted / "model.skops").open("ab") as output:
            output.write(b"injected-corruption")
        with (
            local_server(corrupted, expected_health=503) as url,
            httpx.Client(base_url=url) as client,
        ):
            corrupted_health = client.get("/health").status_code
            corrupted_score = client.post("/v1/predict", json={"records": records[:1]}).status_code
    return {
        "model_id": artifact.model_id,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "versions": artifact.metadata["versions"],
            "server_workers": 1,
            "model_threads": 1,
            "transport": "loopback HTTP/1.1 keep-alive",
        },
        "benchmark": benchmarks,
        "server_stats": stats,
        "failures": {
            "malformed_request_status": rejected,
            "healthy_request_after_rejection_status": recovered,
            "corrupted_artifact_health_status": corrupted_health,
            "corrupted_artifact_score_status": corrupted_score,
        },
        "drift": {
            "same_training_distribution": same,
            "natural_test_shift": compare_drift(artifact.reference, test_features),
            "synthetic_shift": synthetic_shift,
            "injection": "set age=99 and job=occupation-never-seen for first 500 holdout records",
        },
        "limitations": (
            "local shared laptop CPU; five warmups per workload; modest load, "
            "no saturation/SLO/cloud claim; synthetic shift does not demonstrate accuracy loss"
        ),
    }
