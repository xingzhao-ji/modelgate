import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def record():
    # Hand-written feature contract: intentionally independent of production constants.
    return {
        "age": 42,
        "job": "technician",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0,
    }


@pytest.fixture
def toy_data(record):
    # Real small models are fitted, not mocked. Signal exists in every temporal region.
    rng = np.random.default_rng(19)
    n = 1000
    frame = pd.DataFrame([record.copy() for _ in range(n)])
    frame["age"] = rng.integers(20, 80, n)
    frame["campaign"] = rng.integers(1, 8, n)
    frame["job"] = rng.choice(["technician", "admin.", "student"], n)
    frame["y"] = np.where(rng.random(n) < 1 / (1 + np.exp(-(frame.age - 55) / 12)), "yes", "no")
    frame["duration"] = np.where(frame.y == "yes", 900, 10)  # Deliberate leakage trap.
    return frame
