"""Generació de submissions validades: el pitjor error possible és pujar un fitxer mal format."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def make_submission(
    ids,
    preds,
    id_col: str = "ID",
    target_col: str = "target",
    sample_path: str | Path | None = None,
    out_dir: str | Path = "submissions",
    name: str = "sub",
) -> Path:
    """Escriu `submissions/<name>_<hora>.csv` després de comprovar-ne el format.

    Si es passa `sample_path` (el sample_submission.csv oficial), comprova que les columnes i els IDs coincideixen.
    """
    sub = pd.DataFrame({id_col: np.asarray(ids), target_col: np.asarray(preds)})

    assert sub[target_col].notna().all(), "Hi ha prediccions NaN"
    if pd.api.types.is_numeric_dtype(sub[target_col]):  # les classes de text no poden ser infinites
        assert np.isfinite(sub[target_col].astype(float)).all(), "Hi ha prediccions infinites"
    assert sub[id_col].is_unique, "Hi ha IDs duplicats"

    if sample_path is not None:
        sample = pd.read_csv(sample_path)
        assert list(sample.columns) == list(sub.columns), (
            f"Columnes {list(sub.columns)} ≠ sample {list(sample.columns)}")
        assert set(sample[id_col]) == set(sub[id_col]), "Els IDs no coincideixen amb el sample"
        sub = sample[[id_col]].merge(sub, on=id_col, how="left")  # mateix ordre que el sample

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{datetime.now():%m%d_%H%M}.csv"
    sub.to_csv(path, index=False)
    print(f"✓ {path}  ({len(sub)} files)")
    return path
