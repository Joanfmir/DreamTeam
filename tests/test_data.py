import pandas as pd

from dreamteam import data


def test_summary_and_reduce_memory(tabular):
    s = data.summary(tabular)
    assert set(["dtype", "nuls", "únics"]) <= set(s.columns)
    reduced = data.reduce_memory(tabular, verbose=False)
    assert reduced.memory_usage(deep=True).sum() <= tabular.memory_usage(deep=True).sum()
    pd.testing.assert_series_equal(reduced["y"].astype(int), tabular["y"].astype(int), check_dtype=False)


def test_split_xy_drops_ids(tabular):
    X, y = data.split_xy(tabular, "y", drop=["id", "no_existeix"])
    assert "id" not in X and "y" not in X and len(y) == len(tabular)
