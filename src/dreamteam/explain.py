"""Explicabilitat amb SHAP en poques línies (repte Schneider 2025).

Flux típic:
    explainer, sv = shap_values(model, X_valid)
    global_summary(sv)                         # quines variables importen i en quina direcció
    local_waterfall(sv, i)                     # per què el model ha decidit això per a la fila i
    top_reasons(sv, i)                         # el mateix, en text per a la presentació
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def shap_values(model, X: pd.DataFrame, max_rows: int = 2000, seed: int = 42):
    """Calcula els valors SHAP sobre una mostra de X (TreeExplainer per a LightGBM, CatBoost o XGBoost)."""
    import shap

    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    # Classificadors binaris: alguns models retornen una sortida per classe; ens quedem amb la positiva
    if sv.values.ndim == 3:
        sv = sv[:, :, 1]
    return explainer, sv


def global_importance(sv) -> pd.DataFrame:
    """Importància global = mitjana de |SHAP| per variable, ordenada."""
    imp = np.abs(sv.values).mean(axis=0)
    return (pd.DataFrame({"feature": sv.feature_names, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))


def global_summary(sv, max_display: int = 15, show: bool = True):
    """Gràfic beeswarm: importància (eix Y) i direcció de l'efecte (color = valor de la variable)."""
    import shap

    shap.plots.beeswarm(sv, max_display=max_display, show=show)


def local_waterfall(sv, i: int, max_display: int = 10, show: bool = True):
    """Explicació d'una sola predicció: quines variables l'han empès amunt o avall."""
    import shap

    shap.plots.waterfall(sv[i], max_display=max_display, show=show)


def top_reasons(sv, i: int, k: int = 3) -> list[str]:
    """Les k raons principals d'una predicció, en text. Es poden enganxar a les slides o passar a un LLM."""
    contrib = pd.Series(sv.values[i], index=sv.feature_names)
    data = pd.Series(sv.data[i], index=sv.feature_names)
    top = contrib.abs().sort_values(ascending=False).head(k).index

    def fmt(v):
        return f"{v:.3g}" if isinstance(v, (int, float, np.number)) else str(v)

    return [
        f"{f} = {fmt(data[f])} {'puja' if contrib[f] > 0 else 'baixa'} la predicció ({contrib[f]:+.3f})"
        for f in top
    ]
