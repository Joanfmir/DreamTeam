"""Mètriques que han aparegut a la Datathon FME i ajuts per optimitzar-les."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, mean_squared_log_error, roc_auc_score


def msle(y_true, y_pred) -> float:
    """Mean Squared Log Error (Smadex 2025). Les prediccions negatives es tallen a 0."""
    return mean_squared_log_error(y_true, np.clip(y_pred, 0, None))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def auc(y_true, y_proba) -> float:
    """ROC AUC (Accenture 2022)."""
    return roc_auc_score(y_true, y_proba)


def asymmetric_error(y_true, y_pred, under_weight: float = 2.0, over_weight: float = 1.0) -> float:
    """Error absolut que penalitza més quedar-se curt que passar-se.

    Aproxima l'esperit de la mètrica de Mango 2025: les vendes perdudes (infraprediccions)
    costen més que l'excés d'estoc. Quan es publiqui la fórmula exacta, s'ha de substituir aquí.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    diff = y_true - y_pred
    loss = np.where(diff > 0, under_weight * diff, -over_weight * diff)
    return float(loss.mean())


def best_scale_factor(y_true, y_pred, metric=asymmetric_error, grid=None, **metric_kwargs) -> tuple[float, float]:
    """Troba el factor k que minimitza metric(y, k · y_pred).

    Útil quan la mètrica és asimètrica: un equip de Mango 2025 multiplicava per ~1,08.
    El factor pot sortir per sota o per sobre d'1: depèn de la mètrica i del biaix del model.
    S'ha de calibrar sobre les prediccions OOF, mai sobre el test. Les files sense predicció
    (NaN, p. ex. fora del split temporal) s'ignoren.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if not ok.any():
        raise ValueError("No hi ha cap parella (y, predicció) finita per calibrar")
    y_true, y_pred = y_true[ok], y_pred[ok]
    grid = np.arange(0.5, 2.0001, 0.01) if grid is None else np.asarray(grid)
    scores = np.array([metric(y_true, k * y_pred, **metric_kwargs) for k in grid], dtype=float)
    if not np.isfinite(scores).any():
        raise ValueError("La mètrica no retorna cap valor finit")
    i = int(np.nanargmin(scores))
    return float(grid[i]), float(scores[i])


def best_f1_threshold(y_true, y_proba) -> tuple[float, float]:
    """Llindar que maximitza F1 (Schneider 2025 demanava F1 ≥ 0,7). Cal fer-lo servir amb les probabilitats OOF."""
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba, float)
    ok = np.isfinite(y_proba)
    y_true, y_proba = y_true[ok], y_proba[ok]
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    i = int(np.argmax(scores))
    return float(thresholds[i]), float(scores[i])
