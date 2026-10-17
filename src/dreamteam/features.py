"""Feature engineering que funciona a gairebé tots els reptes tabulars.

Totes les funcions que fan servir el target tenen una versió OOF (out-of-fold) per evitar leakage:
el valor de cada fila de train es calcula SENSE aquella fila (amb els altres folds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# ── Encodings de categòriques ────────────────────────────────────────────────

def count_encode(train: pd.DataFrame, test: pd.DataFrame | None, cols: list[str]):
    """Afegeix `<col>_count`: quantes vegades apareix cada valor (train + test junts).

    No fa servir el target, així que comptar també al test és legítim i sovint ajuda
    (p. ex. la "popularitat" d'un producte o d'una ruta).
    """
    train = train.copy()
    test = None if test is None else test.copy()
    for col in cols:
        both = train[col] if test is None else pd.concat([train[col], test[col]])
        counts = both.value_counts()
        train[f"{col}_count"] = train[col].map(counts).astype(float)
        if test is not None:
            test[f"{col}_count"] = test[col].map(counts).fillna(0).astype(float)
    return train, test


def _keys(df: pd.DataFrame, parts: list[str]) -> pd.Series:
    """Clau de text per a una o diverses columnes (funciona amb category, NaN i números)."""
    rows = df[parts].itertuples(index=False, name=None)
    return pd.Series(["|".join(map(str, r)) for r in rows], index=df.index)


def _smoothed_means(keys: pd.Series, y: pd.Series, smoothing: float, prior: float) -> pd.Series:
    stats = y.groupby(keys, observed=True).agg(["sum", "count"])
    return (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)


def target_encode(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    cols: list[str | list[str]],
    target: str,
    n_splits: int = 5,
    smoothing: float = 20.0,
    groups=None,
    seed: int = 42,
):
    """Target encoding OOF amb suavitzat bayesià.

    - Train: cada fila rep la mitjana del target de la seva categoria calculada als ALTRES folds.
    - Test: mitjana calculada amb tot el train.
    - `smoothing` empeny les categories petites cap a la mitjana global (evita sobreajust).
    - Una entrada de `cols` pot ser una llista per codificar combinacions, p. ex. ["3pl", "logistic_hub"].
    - El prior de cada fold es calcula només amb les files d'entrenament d'aquell fold.

    ⚠️ Per avaluar amb CV sense leakage, aquesta funció s'ha de cridar DINS de cada fold extern
    (train = fold d'entrenament, test = fold de validació). Vegeu `validation.cross_validate(fold_fn=...)`.
    Cridar-la un cop sobre tot el train i després fer CV amb uns altres folds infla la mètrica.

    Crea les columnes `te_<col>` (o `te_<col1>__<col2>`).
    """
    train = train.copy()
    test = None if test is None else test.copy()
    y = train[target]
    prior = float(y.mean())  # només per al test (tot el train és coneixement legítim)

    if groups is not None:
        from sklearn.model_selection import GroupKFold
        splits = list(GroupKFold(n_splits).split(train, y, groups))
    else:
        splits = list(KFold(n_splits, shuffle=True, random_state=seed).split(train))

    for col in cols:
        parts = [col] if isinstance(col, str) else list(col)
        name = "te_" + "__".join(parts)
        key_train = _keys(train, parts)

        oof = np.full(len(train), prior)
        for tr, va in splits:
            fold_prior = float(y.iloc[tr].mean())  # sense el target de les files de validació
            means = _smoothed_means(key_train.iloc[tr], y.iloc[tr], smoothing, fold_prior)
            oof[va] = key_train.iloc[va].map(means).fillna(fold_prior).to_numpy()
        train[name] = oof

        if test is not None:
            key_test = _keys(test, parts)
            means = _smoothed_means(key_train, y, smoothing, prior)
            test[name] = key_test.map(means).fillna(prior).to_numpy()
    return train, test


# ── Agregats per grup ────────────────────────────────────────────────────────

def group_aggregates(
    df: pd.DataFrame,
    by: str | list[str],
    cols: list[str],
    aggs: tuple[str, ...] = ("mean", "std", "max"),
) -> pd.DataFrame:
    """Afegeix estadístiques de `cols` per grup, p. ex. preu mitjà per família de producte.

    Fer-ho només sobre features (mai el target); si es vol amb el target, cal `target_encode`.
    També es pot afegir la diferència respecte a la mitjana del grup (`<col>_diff_<by>`).
    """
    df = df.copy()
    by = [by] if isinstance(by, str) else list(by)
    tag = "_".join(by)
    g = df.groupby(by, observed=True)[cols]
    for agg in aggs:
        out = g.transform(agg)
        for col in cols:
            df[f"{col}_{agg}_by_{tag}"] = out[col]
    for col in cols:
        if "mean" in aggs:
            df[f"{col}_diff_{tag}"] = df[col] - df[f"{col}_mean_by_{tag}"]
    return df


# ── Dates ────────────────────────────────────────────────────────────────────

def date_features(df: pd.DataFrame, col: str, prefix: str | None = None) -> pd.DataFrame:
    """Any, mes, setmana ISO, dia de la setmana, cap de setmana i dia de l'any d'una columna de data."""
    df = df.copy()
    p = prefix or col
    d = pd.to_datetime(df[col])
    df[f"{p}_year"] = d.dt.year
    df[f"{p}_month"] = d.dt.month
    df[f"{p}_week"] = d.dt.isocalendar().week.astype(int)
    df[f"{p}_dow"] = d.dt.dayofweek
    df[f"{p}_is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df[f"{p}_doy"] = d.dt.dayofyear
    return df


# ── Embeddings (imatge o text) ───────────────────────────────────────────────

def parse_embedding_column(s: pd.Series) -> np.ndarray:
    """Converteix una columna d'embeddings guardada com a text ("0.1, 0.2, ..." o "[0.1 0.2 ...]") a una matriu."""
    if isinstance(s.iloc[0], (list, np.ndarray)):
        return np.vstack(s.to_numpy()).astype(np.float32)
    cleaned = s.astype(str).str.strip("[]").str.replace(",", " ")
    return np.vstack([np.array(x.split(), dtype=np.float32) for x in cleaned])


def embedding_pca(emb_train: np.ndarray, emb_test: np.ndarray | None = None, n_components: int = 32,
                  prefix: str = "emb_pca"):
    """Redueix embeddings (p. ex. 512 dimensions de CLIP) a poques components per a un model d'arbres."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=42).fit(emb_train)
    cols = [f"{prefix}_{i}" for i in range(n_components)]
    out_train = pd.DataFrame(pca.transform(emb_train), columns=cols)
    out_test = None if emb_test is None else pd.DataFrame(pca.transform(emb_test), columns=cols)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA {emb_train.shape[1]} → {n_components} dims · variància explicada {explained:.0%}")
    return out_train, out_test


def knn_target_features(
    emb_ref: np.ndarray,
    y_ref: np.ndarray,
    emb_query: np.ndarray,
    k: int = 10,
    exclude_self: bool = False,
    prefix: str = "knn",
    metric: str = "cosine",
) -> pd.DataFrame:
    """Target dels k veïns més semblants: la clau del cold start (Mango 2025) i dels preus per zona.

    Per a un producte nou, "com es van vendre les peces més semblants de temporades anteriors?".
    - emb_ref / y_ref: productes amb target conegut (p. ex. temporades ANTERIORS).
    - emb_query: productes on volem la feature.
    - exclude_self=True quan query i ref són el mateix conjunt i en el mateix ordre: s'exclou
      explícitament la fila i-èsima de la consulta i (també si hi ha embeddings duplicats).
    - metric: "cosine" per a embeddings; per a coordenades geogràfiques, "haversine" amb
      [lat, lon] en RADIANS (veïns per distància real).

    Retorna: mitjana, mitjana ponderada per similitud, màxim, desviació i similitud mitjana.
    """
    from sklearn.neighbors import NearestNeighbors

    extra = 1 if exclude_self else 0
    algorithm = "ball_tree" if metric == "haversine" else "auto"
    nn = NearestNeighbors(n_neighbors=k + extra, metric=metric, algorithm=algorithm).fit(emb_ref)
    dist, idx = nn.kneighbors(emb_query)
    if exclude_self:
        assert len(emb_query) == len(emb_ref), "exclude_self requereix query = ref (mateix ordre)"
        drop = idx == np.arange(len(idx))[:, None]
        drop[~drop.any(axis=1), -1] = True  # si la fila no surt entre els veïns, es descarta l'últim
        keep = ~drop
        dist = dist[keep].reshape(len(idx), k)
        idx = idx[keep].reshape(len(idx), k)
    sim = np.clip(1 - dist, 1e-6, None) if metric == "cosine" else 1 / (1 + dist)
    vals = np.asarray(y_ref)[idx]
    return pd.DataFrame({
        f"{prefix}{k}_mean": vals.mean(axis=1),
        f"{prefix}{k}_wmean": (vals * sim).sum(axis=1) / sim.sum(axis=1),
        f"{prefix}{k}_max": vals.max(axis=1),
        f"{prefix}{k}_std": vals.std(axis=1),
        f"{prefix}{k}_sim": sim.mean(axis=1),
    })
