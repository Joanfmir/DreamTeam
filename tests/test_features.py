import numpy as np
import pandas as pd

from dreamteam import features


def test_target_encode_combination_and_test(tabular):
    train, test = tabular.iloc[:500], tabular.iloc[500:]
    tr, te = features.target_encode(train, test, ["cat"], "y")
    assert "te_cat" in tr and "te_cat" in te
    # la categoria "a" té més positius
    assert te.loc[te["cat"] == "a", "te_cat"].mean() > te.loc[te["cat"] == "b", "te_cat"].mean()


def test_count_encode_uses_train_and_test():
    tr = pd.DataFrame({"c": ["a", "a", "b"]})
    te = pd.DataFrame({"c": ["a", "z"]})
    tr2, te2 = features.count_encode(tr, te, ["c"])
    assert tr2["c_count"].tolist() == [3, 3, 1]
    assert te2["c_count"].tolist() == [3, 1]


def test_knn_exclude_self_avoids_leakage(rng):
    emb = rng.normal(size=(100, 8))
    y = rng.normal(size=100)
    with_self = features.knn_target_features(emb, y, emb, k=1, exclude_self=False)
    without = features.knn_target_features(emb, y, emb, k=1, exclude_self=True)
    assert np.allclose(with_self["knn1_mean"], y)          # es troba a si mateix
    assert not np.allclose(without["knn1_mean"], y)


def test_parse_embedding_column():
    s = pd.Series(["0.1, 0.2, 0.3", "[0.4 0.5 0.6]"])
    assert features.parse_embedding_column(s).shape == (2, 3)


def test_group_aggregates_adds_diff_column():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 3.0, 5.0]})
    out = features.group_aggregates(df, "g", ["v"], aggs=("mean",))
    assert out["v_diff_g"].tolist() == [-1.0, 1.0, 0.0]
