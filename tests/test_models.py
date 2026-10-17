import numpy as np
import pandas as pd

from dreamteam import models


def test_log_target_regressor_predicts_non_negative(rng):
    X = pd.DataFrame({"x": rng.uniform(0, 3, 300)})
    y = np.expm1(X["x"]) + rng.uniform(0, 0.1, 300)
    m = models.LogTargetRegressor(models.lgbm_baseline("regression", n_estimators=200, learning_rate=0.1)).fit(X, y)
    pred = m.predict(X)
    assert (pred >= 0).all()
    assert np.corrcoef(pred, y)[0, 1] > 0.9


def test_two_stage_regressor_handles_zeros(rng):
    X = pd.DataFrame({"x": rng.normal(size=500)})
    y = np.where(X["x"] > 0, 10 + X["x"], 0.0)
    m = models.TwoStageRegressor(models.lgbm_baseline("binary", n_estimators=200, learning_rate=0.1),
                                 models.lgbm_baseline("regression", n_estimators=200, learning_rate=0.1)).fit(X, y)
    pred = m.predict(X)
    assert pred[X["x"] < -1].mean() < 1 < pred[X["x"] > 1].mean()
