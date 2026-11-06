"""Fetch one immutable public UCI snapshot; never extract arbitrary ZIP paths."""

import io
import json
import urllib.request
import zipfile
from pathlib import Path

from modelgate.artifacts import json_bytes, sha256

SOURCE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
SOURCE_SHA256 = "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4"
CSV_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"
CSV_NAME = "bank-additional-full.csv"


def unpack_verified_archive(raw: bytes, destination) -> dict:
    if sha256(raw) != SOURCE_SHA256:
        raise ValueError(
            "source SHA-256 mismatch; upstream changed, review provenance before accepting"
        )
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        nested = outer.read("bank-additional.zip")
    with zipfile.ZipFile(io.BytesIO(nested)) as inner:
        csv = inner.read(f"bank-additional/{CSV_NAME}")
        description = inner.read("bank-additional/bank-additional-names.txt")
    if sha256(csv) != CSV_SHA256:
        raise ValueError("CSV SHA-256 mismatch")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / CSV_NAME).write_bytes(csv)
    (destination / "bank-additional-names.txt").write_bytes(description)
    provenance = {
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "sha256": CSV_SHA256,
        "rows": 41188,
        "bytes": len(csv),
        "file": CSV_NAME,
        "doi": "10.24432/C5K306",
        "license": "CC BY 4.0",
        "citation": (
            "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. "
            "UCI Machine Learning Repository."
        ),
        "ordering": "UCI describes file as ordered by date, May 2008 to November 2010",
    }
    (destination / "provenance.json").write_bytes(json_bytes(provenance))
    return provenance


def download(destination) -> dict:
    destination = Path(destination)
    csv = destination / CSV_NAME
    if csv.exists() and (destination / "provenance.json").exists():
        if sha256(csv.read_bytes()) != CSV_SHA256:
            raise ValueError("cached CSV SHA-256 mismatch; remove or inspect corrupted cache")
        return json.loads((destination / "provenance.json").read_bytes())
    request = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent": "ModelGate/0.1 public research demo"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read(10 * 1024 * 1024 + 1)
    return unpack_verified_archive(raw, destination)
