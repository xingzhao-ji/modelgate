"""Versioned, atomic local bundles with hashes and narrowly allowed skops types."""

import hashlib
import importlib.metadata
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import skops.io as sio
from sklearn.pipeline import Pipeline

from modelgate.data import FEATURES, NUMERIC, chronological_split
from modelgate.training import TrainedModel

FILES = {"model.skops", "metadata.json", "reference.json"}
# Reviewed sklearn internals needed for HistGradientBoosting. Never trust types
# merely because an artifact asks for them.
TRUSTED_TYPES = {
    "numpy.dtype",
    "sklearn._loss.link.Interval",
    "sklearn._loss.link.LogitLink",
    "sklearn._loss.loss.HalfBinomialLoss",
    "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
}


class ArtifactError(ValueError):
    """Artifact is incomplete, incompatible, or failed integrity/schema checks."""


def validate_semantic_metadata(metadata):
    dataset, report = metadata["dataset"], metadata["training_report"]
    if (
        not isinstance(dataset["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", dataset["sha256"]) is None
        or type(dataset["rows"]) is not int
        or dataset["rows"] < 100
    ):
        raise ArtifactError("invalid dataset provenance")
    candidates = {
        "dummy",
        "logistic",
        "hist_gradient_boosting",
        "stable_logistic",
        "stable_hist_gradient_boosting",
    }
    if report["selected_model"] not in candidates:
        raise ArtifactError("unsupported selected model")
    expected = chronological_split(dataset["rows"])
    if set(report["partitions"]) != set(expected):
        raise ArtifactError("invalid training partitions")
    for name, partition in expected.items():
        if not all(type(value) is int for value in report["partitions"][name].values()):
            raise ArtifactError("training partition offsets must be integers")
        if report["partitions"][name] != {
            "start_inclusive": partition.start,
            "stop_exclusive": partition.stop,
            "rows": partition.stop - partition.start,
        }:
            raise ArtifactError("training partitions do not match provenance")
    if set(report["development_prevalence"]) != set(expected) - {"test"}:
        raise ArtifactError("invalid development prevalence")
    for value in report["development_prevalence"].values():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ArtifactError("invalid development prevalence")
    if report["policy"]["threshold"] != metadata["threshold"]:
        raise ArtifactError("policy threshold differs from artifact threshold")


def validate_reference(reference):
    if (
        set(reference["features"]) != set(FEATURES)
        or reference["partition"] != "train"
        or not isinstance(reference["rows"], int)
        or reference["rows"] < 1
    ):
        raise ArtifactError("invalid reference schema")
    for name, profile in reference["features"].items():
        if name in NUMERIC:
            cuts = profile["cuts"]
            if profile["kind"] != "numeric" or not all(math.isfinite(v) for v in cuts):
                raise ArtifactError("invalid numeric reference")
            if cuts != sorted(set(cuts)):
                raise ArtifactError("invalid reference cuts")
            expected = len(cuts) + 2
        else:
            categories = profile["categories"]
            if profile["kind"] != "categorical" or not all(isinstance(v, str) for v in categories):
                raise ArtifactError("invalid categorical reference")
            if len(set(categories)) != len(categories):
                raise ArtifactError("duplicate reference categories")
            expected = len(categories) + 1
        probabilities = profile["probabilities"]
        if (
            len(probabilities) != expected
            or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8)
        ):
            raise ArtifactError("invalid reference probabilities")


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def versions() -> dict:
    return {
        name: importlib.metadata.version(name)
        for name in ["numpy", "scikit-learn", "skops", "pandas"]
    }


@dataclass
class LoadedArtifact:
    model: TrainedModel
    reference: dict
    metadata: dict
    model_id: str


def save_artifact(model, reference, path, provenance) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "feature_names": FEATURES,
        "threshold": model.threshold,
        "calibration_slope": model.calibration_slope,
        "calibration_intercept": model.calibration_intercept,
        "dataset": provenance,
        "training_report": model.report,
        "versions": versions(),
    }
    payloads = {
        "model.skops": sio.dumps(model.estimator),
        "metadata.json": json_bytes(metadata),
        "reference.json": json_bytes(reference),
    }
    hashes = {name: sha256(raw) for name, raw in payloads.items()}
    manifest = {"format_version": 1, "files": hashes, "model_id": sha256(json_bytes(hashes))[:16]}
    # Rename only after every file is written. Readers see no partially written bundle.
    with TemporaryDirectory(prefix=".modelgate-", dir=path.parent) as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        for name, raw in payloads.items():
            (staging / name).write_bytes(raw)
        (staging / "manifest.json").write_bytes(json_bytes(manifest))
        load_artifact(staging)  # Validate the exact bytes before publication.
        os.rename(staging, path)
    return path


def load_artifact(path) -> LoadedArtifact:
    path = Path(path)
    try:
        manifest_path = path / "manifest.json"
        if manifest_path.is_symlink() or manifest_path.stat().st_size > 1024 * 1024:
            raise ArtifactError("invalid manifest")
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("format_version") != 1 or set(manifest.get("files", {})) != FILES:
            raise ArtifactError("unsupported manifest version or file set")
        payloads = {}
        for name, expected in manifest["files"].items():
            source = path / name
            if source.is_symlink() or source.stat().st_size > 64 * 1024 * 1024:
                raise ArtifactError("invalid artifact file")
            raw = source.read_bytes()
            if sha256(raw) != expected:
                raise ArtifactError(f"integrity mismatch: {name}")
            payloads[name] = raw
        if sha256(json_bytes(manifest["files"]))[:16] != manifest["model_id"]:
            raise ArtifactError("manifest identity integrity mismatch")
        metadata, reference = (
            json.loads(payloads["metadata.json"]),
            json.loads(payloads["reference.json"]),
        )
        if metadata["format_version"] != 1 or metadata["feature_names"] != FEATURES:
            raise ArtifactError("unsupported feature schema")
        if metadata["versions"] != versions():
            raise ArtifactError("dependency versions differ; recreate the pinned environment")
        validate_semantic_metadata(metadata)
        for field in ["threshold", "calibration_slope", "calibration_intercept"]:
            if not isinstance(metadata[field], (int, float)) or not math.isfinite(metadata[field]):
                raise ArtifactError(f"invalid {field}")
        if not 0 <= metadata["threshold"] <= 1:
            raise ArtifactError("invalid threshold")
        validate_reference(reference)
        unknown = set(sio.get_untrusted_types(data=payloads["model.skops"]))
        if not unknown.issubset(TRUSTED_TYPES):
            raise ArtifactError(f"unapproved model types: {sorted(unknown - TRUSTED_TYPES)}")
        estimator = sio.loads(payloads["model.skops"], trusted=sorted(TRUSTED_TYPES))
        if not isinstance(estimator, Pipeline) or not np.array_equal(estimator.classes_, [0, 1]):
            raise ArtifactError("estimator must be a fitted binary pipeline")
        if list(estimator.feature_names_in_) != FEATURES:
            raise ArtifactError("estimator feature schema mismatch")
        model = TrainedModel(
            estimator,
            metadata["calibration_slope"],
            metadata["calibration_intercept"],
            metadata["threshold"],
            metadata["training_report"],
        )
        return LoadedArtifact(model, reference, metadata, manifest["model_id"])
    except ArtifactError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ArtifactError(f"invalid artifact: {type(exc).__name__}") from exc
