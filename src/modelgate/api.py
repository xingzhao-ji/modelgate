"""Local-only FastAPI demo; no raw request data is retained."""

import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from modelgate.artifacts import ArtifactError, LoadedArtifact, load_artifact
from modelgate.data import BOUNDS, CATEGORICAL, FEATURES, NUMERIC, validate_features
from modelgate.drift import compare_drift


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[dict[str, Any]] = Field(min_length=1, max_length=512)


class DriftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[dict[str, Any]] = Field(min_length=100, max_length=5000)


def records_frame(records) -> pd.DataFrame:
    for index, record in enumerate(records):
        if set(record) != set(FEATURES):
            raise ValueError(
                f"record {index}: all 19 features are required; "
                "use explicit null for missing values"
            )
        # Check raw JSON before pandas can coerce values or overflow while building
        # an array. Explicit null is distinct from nonstandard NaN.
        for name in NUMERIC:
            value = record[name]
            low, high = BOUNDS[name]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not low <= value <= high
            ):
                raise ValueError(f"record {index}: {name} must be finite within [{low}, {high}]")
        for name in CATEGORICAL:
            value = record[name]
            if value is not None and not (isinstance(value, str) and 1 <= len(value) <= 100):
                raise ValueError(f"record {index}: {name} must be a string or explicit null")
    return validate_features(pd.DataFrame(records))


def score_records(artifact: LoadedArtifact, records) -> dict:
    frame = records_frame(records)
    probabilities = artifact.model.predict_proba(frame)
    return {
        "model_id": artifact.model_id,
        "threshold": artifact.model.threshold,
        "predictions": [
            {"probability": float(p), "selected": bool(p >= artifact.model.threshold)}
            for p in probabilities
        ],
        "usage": "local research demo; not approved for real marketing decisions",
    }


class BodyLimitMiddleware:
    """Bound HTTP body before JSON parsing, including chunked bodies."""

    def __init__(self, app, max_bytes=2 * 1024 * 1024):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        messages, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                return await JSONResponse({"detail": "request body exceeds 2 MiB"}, 413)(
                    scope, receive, send
                )
            messages.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(messages)

        async def replay():
            return next(iterator, {"type": "http.request", "body": b"", "more_body": False})

        await self.app(scope, replay, send)


def create_app(path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.artifact = None
        app.state.load_error = None
        try:
            app.state.artifact = load_artifact(path)
        except ArtifactError as exc:
            app.state.load_error = str(exc)
        yield

    app = FastAPI(title="ModelGate", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware)
    app.state.counters = {"requests": 0, "rejected": 0, "scored_rows": 0}
    lock = threading.Lock()

    @app.middleware("http")
    async def observability(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        with lock:
            app.state.counters["requests"] += 1
            app.state.counters["rejected"] += int(response.status_code >= 400)
        response.headers["x-request-id"] = uuid.uuid4().hex
        response.headers["server-timing"] = f"total;dur={(time.perf_counter() - start) * 1000:.3f}"
        return response

    def ready() -> LoadedArtifact:
        if app.state.artifact is None:
            raise HTTPException(503, "model unavailable: artifact validation failed")
        return app.state.artifact

    @app.get("/health")
    def health():
        artifact = ready()
        return {"status": "ready", "model_id": artifact.model_id}

    @app.get("/v1/model")
    def metadata():
        artifact = ready()
        return {
            "model_id": artifact.model_id,
            "selected_model": artifact.model.report["selected_model"],
            "features": {"numeric": NUMERIC, "categorical": CATEGORICAL, "bounds": BOUNDS},
            "threshold": artifact.model.threshold,
            "dataset": artifact.metadata["dataset"],
            "versions": artifact.metadata["versions"],
            "promotion_status": "research_only",
        }

    @app.get("/v1/stats")
    def stats():
        with lock:
            return dict(app.state.counters)

    @app.post("/v1/predict")
    def predict(payload: PredictRequest):
        artifact = ready()
        try:
            result = score_records(artifact, payload.records)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        with lock:
            app.state.counters["scored_rows"] += len(payload.records)
        return result

    @app.post("/v1/drift")
    def drift(payload: DriftRequest):
        artifact = ready()
        try:
            return {
                "model_id": artifact.model_id,
                **compare_drift(artifact.reference, records_frame(payload.records)),
            }
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app
