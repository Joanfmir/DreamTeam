import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def tabular(rng):
    """Dataset de classificació petit: una numèrica informativa, una categòrica i soroll."""
    n = 600
    x = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = ((x + (cat == "a") * 1.0 + rng.normal(0, 0.5, n)) > 0.5).astype(int)
    return pd.DataFrame({"id": np.arange(n), "x": x, "noise": rng.normal(size=n), "cat": cat, "y": y})
