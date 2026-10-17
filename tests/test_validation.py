import numpy as np
import pandas as pd

from dreamteam import metrics, models, validation


def test_duplicate_groups_keep_copies_in_same_fold():
    X = pd.DataFrame({"a": [1, 1, 2, 2, 3, 3, 4, 4], "b": [0, 0, 1, 1, 0, 0, 1, 1]})
    groups = validation.duplicate_groups(X)
    for tr, va in validation.group_kfold(4).split(X, groups=groups):
        assert not set(groups[tr]) & set(groups[va])


def test_duplicate_report_fraction():
    X = pd.DataFrame({"a": [1, 1, 2, 3]})
    assert validation.duplicate_report(X) == 0.5


def test_time_split_trains_on_past_only():
    t = pd.Series([1, 1, 2, 2, 3, 3])
    [(tr, va)] = validation.time_split(t, valid_values=[3])
    assert t.iloc[tr].max() < t.iloc[va].min()
    assert set(t.iloc[va]) == {3}


def test_cross_validate_fills_every_oof_prediction(tabular):
    X = tabular[["x", "noise"]]
    y = tabular["y"]
    model = models.lgbm_baseline("binary", n_estimators=30)
    oof, scores, fitted = validation.cross_validate(model, X, y, validation.kfold(3, stratified=True),
                                                    metrics.auc, proba=True, verbose=False)
    assert not np.isnan(oof).any()
    assert len(scores) == len(fitted) == 3
    assert np.mean(scores) > 0.7
