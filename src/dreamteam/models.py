"""Models baseline amb paràmetres raonables per començar ràpid."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

# Paràmetres de partida provats per a datasets tabulars de mida mitjana (10k–1M files)
LGBM_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)


def lgbm_baseline(task: str = "regression", **overrides):
    """LightGBM amb paràmetres de partida.

    task: "regression" | "binary" | "multiclass" | "quantile" | "tweedie"
    Les columnes de tipus `category` les fa servir directament, sense one-hot.
    """
    import lightgbm as lgb

    params = {**LGBM_PARAMS, **overrides}
    if task == "binary":
        return lgb.LGBMClassifier(objective="binary", **params)
    if task == "multiclass":
        return lgb.LGBMClassifier(objective="multiclass", **params)
    if task == "quantile":
        params.setdefault("alpha", 0.7)  # >0,5 → prediu per sobre de la mediana
        return lgb.LGBMRegressor(objective="quantile", **params)
    if task == "tweedie":
        return lgb.LGBMRegressor(objective="tweedie", **params)
    return lgb.LGBMRegressor(objective="regression", **params)


def catboost_baseline(task: str = "regression", cat_features=None, **overrides):
    """CatBoost: sol ser el millor quan hi ha moltes categòriques d'alta cardinalitat."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    params = dict(iterations=2000, learning_rate=0.05, depth=6, random_seed=42, verbose=0,
                  cat_features=cat_features)
    params.update(overrides)
    return CatBoostClassifier(**params) if task == "binary" else CatBoostRegressor(**params)


class LogTargetRegressor(BaseEstimator, RegressorMixin):
    """Entrena sobre log1p(y) i desfà la transformació amb expm1.

    És el truc estàndard quan la mètrica és MSLE o RMSLE (Smadex 2025) o el target és molt esbiaixat.
    """

    def __init__(self, base_model):
        self.base_model = base_model

    def fit(self, X, y, **fit_kwargs):
        self.model_ = clone(self.base_model).fit(X, np.log1p(np.clip(y, 0, None)), **fit_kwargs)
        return self

    def predict(self, X):
        return np.expm1(self.model_.predict(X)).clip(min=0)


class TwoStageRegressor(BaseEstimator, RegressorMixin):
    """Predicció = P(y > 0) · E[y | y > 0]. Per a targets amb molts zeros (ingressos, compres)."""

    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y):
        y = np.asarray(y)
        positive = y > 0
        # Casos límit: tot zeros (no hi ha res a regressar) o tot positius (no hi ha res a classificar)
        self.clf_ = clone(self.classifier).fit(X, positive.astype(int)) if 0 < positive.sum() < len(y) else None
        self.p_const_ = float(positive.mean())
        self.reg_ = clone(self.regressor).fit(X[positive], y[positive]) if positive.any() else None
        return self

    def predict(self, X):
        if self.reg_ is None:
            return np.zeros(len(X))
        if self.clf_ is None:
            p = np.full(len(X), self.p_const_)
        else:
            pos_col = list(self.clf_.classes_).index(1)  # columna de la classe positiva, sigui quina sigui
            p = self.clf_.predict_proba(X)[:, pos_col]
        return p * self.reg_.predict(X)
