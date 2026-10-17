import numpy as np
import pytest

from dreamteam import metrics


def test_msle_clips_negative_predictions():
    assert metrics.msle([0, 1], [-5, 1]) == pytest.approx(0.0)


def test_asymmetric_error_penalises_underprediction_more():
    under = metrics.asymmetric_error([10], [8], under_weight=2, over_weight=1)
    over = metrics.asymmetric_error([10], [12], under_weight=2, over_weight=1)
    assert under == pytest.approx(4) and over == pytest.approx(2)


def test_best_scale_factor_inflates_when_shortfall_is_costly(rng):
    y = rng.lognormal(size=2000)
    pred = y * rng.lognormal(0, 0.3, size=2000)  # predicció sorollosa i sense biaix
    k, _ = metrics.best_scale_factor(y, pred)
    assert k > 1.0


def test_best_f1_threshold_beats_default(rng):
    y = rng.integers(0, 2, 1000)
    proba = np.clip(0.3 * y + rng.uniform(0, 0.6, 1000), 0, 1)  # positius amb probabilitats baixes
    thr, f1 = metrics.best_f1_threshold(y, proba)
    from sklearn.metrics import f1_score
    assert f1 >= f1_score(y, proba >= 0.5)
    assert 0.05 <= thr <= 0.95
