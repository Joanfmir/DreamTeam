"""Càrrega i inspecció ràpida de datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load(path: str | Path, **kwargs) -> pd.DataFrame:
    """Llegeix CSV, Parquet, JSON o Excel segons l'extensió."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Format no suportat: {suffix}")


def reduce_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Redueix la memòria baixant el tipus de les numèriques i passant objectes a `category`."""
    before = df.memory_usage(deep=True).sum() / 1024**2
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            df[col] = pd.to_numeric(s, downcast="float")
        elif s.dtype == object and s.nunique(dropna=False) < 0.5 * len(s):
            df[col] = s.astype("category")
    after = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memòria: {before:.1f} MB → {after:.1f} MB ({100 * (1 - after / before):.0f} % menys)")
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila per columna: tipus, nuls, valors únics i un exemple. Primer pas de qualsevol EDA."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nuls": df.isna().sum(),
        "% nuls": (100 * df.isna().mean()).round(1),
        "únics": df.nunique(),
        "exemple": df.iloc[0] if len(df) else np.nan,
    }).sort_values("% nuls", ascending=False)


def split_xy(df: pd.DataFrame, target: str, drop: list[str] | None = None):
    """Separa features i target, i treu les columnes que no s'han de fer servir (IDs…)."""
    drop = [c for c in (drop or []) if c in df.columns]
    X = df.drop(columns=[target, *drop])
    y = df[target]
    return X, y


def to_category(X: pd.DataFrame) -> pd.DataFrame:
    """Converteix les columnes de text a `category` (format natiu per a LightGBM)."""
    X = X.copy()
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].astype("category")
    return X
