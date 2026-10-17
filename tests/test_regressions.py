"""Tests de regressió dels problemes detectats a la revisió amb Codex (17/10/2026)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from dreamteam import features, metrics, models, validation


def test_target_encode_own_target_never_leaks():
    # Categories úniques: canviar el target d'UNA fila no pot canviar el seu propi encoding
    df = pd.DataFrame({"c": [f"k{i}" for i in range(40)], "y": [0, 1] * 20})
    a, _ = features.target_encode(df, None, ["c"], "y", smoothing=0)
    df2 = df.copy()
    df2.loc[0, "y"] = 1 - df2.loc[0, "y"]
    b, _ = features.target_encode(df2, None, ["c"], "y", smoothing=0)
    assert a.loc[0, "te_c"] == b.loc[0, "te_c"]


def test_target_encode_handles_categorical_with_nan():
    df = pd.DataFrame({"c": pd.Categorical(["a", None, "a", "b"] * 10), "y": [1, 0, 1, 0] * 10})
    out, _ = features.target_encode(df, df.drop(columns="y"), ["c"], "y")
    assert out["te_c"].notna().all()


def test_knn_exclude_self_with_duplicate_embeddings():
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    out = features.knn_target_features(emb, y, emb, k=1, exclude_self=True)
    # el veí de cada fila ha de ser el seu duplicat, mai ella mateixa
    assert out["knn1_mean"].tolist() == [20.0, 10.0, 40.0, 30.0]


def test_knn_haversine_metric():
    rad = np.radians([[41.38, 2.17], [41.39, 2.16], [40.42, -3.70]])  # BCN, BCN, Madrid
    out = features.knn_target_features(rad, np.array([1.0, 2.0, 99.0]), rad[:1], k=1,
                                       exclude_self=False, metric="haversine")
    assert out["knn1_mean"].iloc[0] == 1.0


def test_best_scale_factor_ignores_nan_and_can_go_below_one():
    k, score = metrics.best_scale_factor([1.0, 2.0, 3.0], [np.nan, 4.0, 6.0],
                                         metric=metrics.asymmetric_error, under_weight=1, over_weight=1)
    assert np.isfinite(score) and k == pytest.approx(0.5)


def test_best_scale_factor_raises_without_valid_rows():
    with pytest.raises(ValueError):
        metrics.best_scale_factor([1.0], [np.nan])


@pytest.mark.parametrize("y", [np.zeros(20), np.full(20, 5.0)])
def test_two_stage_edge_cases(y):
    X = pd.DataFrame({"x": np.arange(20.0)})
    m = models.TwoStageRegressor(DummyClassifier(strategy="prior"), DummyRegressor()).fit(X, y)
    assert np.allclose(m.predict(X), y.mean())


def test_cross_validate_with_test_applies_fold_fn(tabular):
    seen = []

    def fold_fn(X_tr, y_tr, X_other):
        seen.append(len(X_tr))
        return X_tr.assign(m=y_tr.mean()), X_other.assign(m=y_tr.mean())

    X, y = tabular[["x"]], tabular["y"]
    oof, scores, test_pred = validation.cross_validate_with_test(
        models.lgbm_baseline("binary", n_estimators=20), X, y, X.iloc[:50],
        validation.kfold(3, stratified=True), metrics.auc, proba=True, verbose=False, fold_fn=fold_fn)
    assert len(seen) == 6  # validació + test, per a cada un dels 3 folds
    assert test_pred.shape == (50,) and not np.isnan(oof).any()
