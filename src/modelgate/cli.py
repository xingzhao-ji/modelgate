"""Reproducible commands, explicit paths and fail-closed exit codes."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import uvicorn

from modelgate.api import PredictRequest, create_app, score_records
from modelgate.artifacts import (
    ArtifactError,
    json_bytes,
    load_artifact,
    save_artifact,
    sha256,
    versions,
)
from modelgate.data import chronological_split, prepare_dataset
from modelgate.download import CSV_NAME, CSV_SHA256, download
from modelgate.drift import build_reference
from modelgate.evaluation import evaluate
from modelgate.training import TrainingConfig, train


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(json_bytes(value))


def parser():
    result = argparse.ArgumentParser(prog="modelgate", description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help="fetch and verify the public UCI snapshot")
    fetch.add_argument("--data-dir", default="data")
    fit = sub.add_parser(
        "train", help="train and freeze an immutable artifact; never inspect test values"
    )
    fit.add_argument("--data-dir", default="data")
    fit.add_argument("--artifact", default="artifacts/uci-v1")
    fit.add_argument("--report", default="evidence/training.json")
    assess = sub.add_parser("evaluate", help="assess the frozen model on its reserved test window")
    assess.add_argument("--data-dir", default="data")
    assess.add_argument("--artifact", default="artifacts/uci-v1")
    assess.add_argument("--output", default="evidence/evaluation.json")
    serve = sub.add_parser("serve", help="serve the local demo on loopback only")
    serve.add_argument("--artifact", default="artifacts/uci-v1")
    serve.add_argument("--port", type=int, default=8765)
    batch = sub.add_parser(
        "predict", help="score a JSON records batch with the same contract as HTTP"
    )
    batch.add_argument("--artifact", default="artifacts/uci-v1")
    batch.add_argument("--input", required=True)
    batch.add_argument("--output", required=True)
    experiment = sub.add_parser(
        "experiment", help="real HTTP latency, drift, and failure injection"
    )
    experiment.add_argument("--artifact", default="artifacts/uci-v1")
    experiment.add_argument("--data-dir", default="data")
    experiment.add_argument("--output", default="evidence/experiment.json")
    experiment.add_argument("--requests", type=int, default=100)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "download":
            print(json.dumps(download(args.data_dir), indent=2))
        elif args.command == "train":
            if Path(args.report).exists():
                raise FileExistsError(f"report already exists: {args.report}")
            provenance = download(args.data_dir)
            frame = pd.read_csv(Path(args.data_dir) / CSV_NAME, sep=";")
            start = time.perf_counter()
            model = train(frame, TrainingConfig())
            features, _ = prepare_dataset(frame.iloc[chronological_split(len(frame))["train"]])
            path = save_artifact(model, build_reference(features), args.artifact, provenance)
            artifact = load_artifact(path)
            report = {
                "model_id": artifact.model_id,
                "training_seconds": time.perf_counter() - start,
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "versions": versions(),
                },
                "dataset": provenance,
                **model.report,
            }
            write_new_json(args.report, report)
            print(
                json.dumps(
                    {
                        "artifact": str(path),
                        "report": args.report,
                        "selected_model": model.report["selected_model"],
                        "model_id": artifact.model_id,
                    }
                )
            )
        elif args.command == "evaluate":
            artifact = load_artifact(args.artifact)
            raw = (Path(args.data_dir) / CSV_NAME).read_bytes()
            if sha256(raw) != artifact.metadata["dataset"]["sha256"] or sha256(raw) != CSV_SHA256:
                raise ValueError("evaluation dataset hash differs from frozen training provenance")
            report = evaluate(artifact, pd.read_csv(Path(args.data_dir) / CSV_NAME, sep=";"))
            write_new_json(args.output, report)
            print(
                json.dumps(
                    {
                        "report": args.output,
                        "gate": report["gate"]["decision"],
                        "metrics": report["metrics"],
                    },
                    indent=2,
                )
            )
        elif args.command == "serve":
            uvicorn.run(
                create_app(args.artifact),
                host="127.0.0.1",
                port=args.port,
                access_log=False,
                log_level="warning",
                limit_concurrency=32,
                timeout_keep_alive=5,
            )
        elif args.command == "predict":
            payload = PredictRequest.model_validate_json(Path(args.input).read_bytes())
            write_new_json(
                args.output, score_records(load_artifact(args.artifact), payload.records)
            )
            print(json.dumps({"output": args.output, "rows": len(payload.records)}))
        elif args.command == "experiment":
            from modelgate.experiment import run_experiment

            write_new_json(args.output, run_experiment(args.artifact, args.data_dir, args.requests))
            print(json.dumps({"output": args.output}))
    except (ArtifactError, ValueError, OSError) as exc:
        print(f"modelgate: {exc}", file=sys.stderr)
        return 2
    return 0
