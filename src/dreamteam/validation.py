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
):
    """Entrena un clon del model a cada fold i retorna (prediccions OOF, puntuacions, models).

    Amb les prediccions OOF es poden calibrar llindars i factors i fer ensembles sense leakage.
    """
    splits = cv if isinstance(cv, list) else list(cv.split(X, y, groups))
    oof = np.full(len(X), np.nan)
    scores, models = [], []
    for fold, (tr, va) in enumerate(splits):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr], **(fit_kwargs or {}))
        pred = m.predict_proba(X.iloc[va])[:, 1] if proba else m.predict(X.iloc[va])
        oof[va] = pred
        score = metric(y.iloc[va], pred)
        scores.append(score)
        models.append(m)
        if verbose:
            print(f"  fold {fold}: {score:.5f}")
    if verbose:
        print(f"CV: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    return oof, scores, models


def predict_folds(models, X_test: pd.DataFrame, proba: bool = False) -> np.ndarray:
    """Mitjana de les prediccions de tots els models dels folds sobre el test."""
    preds = [m.predict_proba(X_test)[:, 1] if proba else m.predict(X_test) for m in models]
    return np.mean(preds, axis=0)
