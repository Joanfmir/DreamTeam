"""Esquemes de validació creuada i prediccions out-of-fold (OOF).

Regla d'or: el split de validació ha d'imitar el split train/test real.
- Files independents        → `kfold`
- IDs que es repeteixen      → `group_kfold` (p. ex. el mateix client o producte)
- Hi ha temps / temporades   → `time_split` (entrenar amb el passat, validar amb el futur)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


def kfold(n_splits: int = 5, stratified: bool = False, seed: int = 42):
    cls = StratifiedKFold if stratified else KFold
    return cls(n_splits=n_splits, shuffle=True, random_state=seed)


def group_kfold(n_splits: int = 5):
    return GroupKFold(n_splits=n_splits)


def duplicate_groups(X: pd.DataFrame) -> np.ndarray:
    """Un identificador de grup per a cada combinació única de features.

    Si hi ha files amb les mateixes features, un K-Fold aleatori posa còpies a train i a validació
    i el model les "memoritza": la CV surt inflada. Passeu aquests grups a `group_kfold`.
    Exemple real: Schneider 2025 té un 77 % de files duplicades (AUC 0,85 amb K-Fold → 0,71 agrupat).
    """
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def duplicate_report(X: pd.DataFrame) -> float:
    """Imprimeix i retorna la fracció de files amb features duplicades."""
    frac = X.duplicated(keep=False).mean()
    print(f"Files amb features duplicades: {frac:.1%}"
          + ("  ⚠️ feu servir group_kfold(duplicate_groups(X))" if frac > 0.05 else ""))
    return frac


def time_split(time_col: pd.Series, valid_values) -> list[tuple[np.ndarray, np.ndarray]]:
    """Un sol split: valida amb les files on `time_col` és a `valid_values` i entrena amb les anteriors.

    Exemple (Mango): time_split(df["id_season"], valid_values=[86])
    """
    valid_values = set(np.atleast_1d(valid_values))
    is_valid = time_col.isin(valid_values).to_numpy()
    first_valid = time_col[is_valid].min()
    is_train = (time_col < first_valid).to_numpy()
    return [(np.where(is_train)[0], np.where(is_valid)[0])]


def cross_validate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    metric: Callable,
    groups=None,
    proba: bool = False,
    fit_kwargs: dict | None = None,
    verbose: bool = True,
    fold_fn: Callable | None = None,
):
    """Entrena un clon del model a cada fold i retorna (prediccions OOF, puntuacions, models).

    Amb les prediccions OOF es poden calibrar llindars i factors i fer ensembles sense leakage.

    `fold_fn(X_tr, y_tr, X_va) -> (X_tr, X_va)`: features que depenen del target (target encoding,
    kNN de target…) calculades DINS de cada fold, només amb el train del fold. Si es fan servir,
    els models retornats esperen features transformades: per al test, `cross_validate_with_test`.
    """
    oof, scores, models, _ = _cv_loop(model, X, y, cv, metric, groups, proba, fit_kwargs,
                                      verbose, fold_fn, X_test=None)
    return oof, scores, models


def cross_validate_with_test(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cv,
    metric: Callable,
    groups=None,
    proba: bool = False,
    fit_kwargs: dict | None = None,
    verbose: bool = True,
    fold_fn: Callable | None = None,
):
    """Com `cross_validate`, però també prediu el test a cada fold amb les mateixes features del fold.

    `fold_fn(X_tr, y_tr, X_other) -> (X_tr, X_other)` s'aplica a validació i a test per separat.
    Retorna (oof, puntuacions, prediccions de test promitjades).
    """
    oof, scores, _, test_pred = _cv_loop(model, X, y, cv, metric, groups, proba, fit_kwargs,
                                         verbose, fold_fn, X_test=X_test)
    return oof, scores, test_pred


def _predict(m, X, proba):
    return m.predict_proba(X)[:, 1] if proba else m.predict(X)


def _cv_loop(model, X, y, cv, metric, groups, proba, fit_kwargs, verbose, fold_fn, X_test):
    splits = cv if isinstance(cv, list) else list(cv.split(X, y, groups))
    oof = np.full(len(X), np.nan)
    scores, models, test_preds = [], [], []
    for fold, (tr, va) in enumerate(splits):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        X_te = X_test
        if fold_fn is not None:
            X_tr_va, X_va = fold_fn(X_tr, y.iloc[tr], X_va)
            if X_test is not None:
                _, X_te = fold_fn(X_tr, y.iloc[tr], X_test)
            X_tr = X_tr_va
        m = clone(model)
        m.fit(X_tr, y.iloc[tr], **(fit_kwargs or {}))
        pred = _predict(m, X_va, proba)
        oof[va] = pred
        score = metric(y.iloc[va], pred)
        scores.append(score)
        models.append(m)
        if X_te is not None:
            test_preds.append(_predict(m, X_te, proba))
        if verbose:
            print(f"  fold {fold}: {score:.5f}")
    if verbose:
        print(f"CV: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    test_pred = np.mean(test_preds, axis=0) if test_preds else None
    return oof, scores, models, test_pred


def predict_folds(models, X_test: pd.DataFrame, proba: bool = False) -> np.ndarray:
    """Mitjana de les prediccions de tots els models dels folds sobre el test."""
    preds = [m.predict_proba(X_test)[:, 1] if proba else m.predict(X_test) for m in models]
    return np.mean(preds, axis=0)
