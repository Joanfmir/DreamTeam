"""Pipeline de "dia D" per al repte de demanda de Mango (2025), provat amb dades sintètiques.

    python reptes/2025_mango_demanda/simulate_data.py   # una sola vegada
    python reptes/2025_mango_demanda/pipeline.py

Passos (els mateixos que hauríem de fer amb les dades reals):
  1. Agregar de setmanes a una fila per producte (target = demanda total)
  2. Validació temporal: l'última temporada de train fa de validació
  3. Cold start: demanda dels k productes més semblants (per embedding) de temporades ANTERIORS
  4. LightGBM sobre log1p(demanda)
  5. Calibrar un factor > 1 perquè la mètrica penalitza més quedar-se curt
  6. Reentrenar amb tot i generar la submission (ID, demand)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from dreamteam import features, metrics, models, submission

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/synthetic/mango2025"   # amb dades reals: canviar aquesta ruta i revisar noms de columnes
CAT_COLS = ["aggregated_family", "family", "category", "fabric", "color_name", "length_type",
            "silhouette_type", "sleeve_length_type", "print_type", "archetype", "moment"]
K = 20
NUM_COLS = ["price", "num_stores", "num_sizes", "life_cycle_length", "has_plus_size"]


def mango_score(y_true, y_pred) -> float:
    """Aproximació de la mètrica de Mango: error asimètric (quedar-se curt costa el doble). Més baix = millor."""
    return metrics.asymmetric_error(y_true, y_pred, under_weight=2.0, over_weight=1.0)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """De setmanes a productes: target = suma de weekly_demand."""
    static = df.drop_duplicates("id").set_index("id")
    if "weekly_demand" in df:
        static["demand"] = df.groupby("id")["weekly_demand"].sum()
    static["phase_in_month"] = pd.to_datetime(static["phase_in"]).dt.month
    return static.reset_index()


def add_knn(ref: pd.DataFrame, emb_ref, query: pd.DataFrame, emb_query):
    """kNN de la demanda de `ref` per a cada fila de `query` (ref ha de ser estrictament anterior)."""
    knn = features.knn_target_features(emb_ref, np.log1p(ref["demand"].to_numpy()), emb_query, k=K)
    return pd.concat([query.reset_index(drop=True), knn], axis=1)


def add_knn_past(prod: pd.DataFrame, emb) -> pd.DataFrame:
    """kNN per a les files d'entrenament fent servir NOMÉS temporades anteriors a la seva.

    Així el model aprèn amb features construïdes igual que les del test (que només veu el passat).
    La primera temporada no té passat: les seves features queden a NaN (LightGBM ho gestiona).
    """
    parts = []
    for season in sorted(prod["id_season"].unique()):
        q = (prod["id_season"] == season).to_numpy()
        ref = (prod["id_season"] < season).to_numpy()
        if ref.any():
            parts.append(add_knn(prod[ref], emb[ref], prod[q], emb[q]))
        else:
            cols = [f"knn{K}_{stat}" for stat in ("mean", "wmean", "max", "std", "sim")]
            nan = pd.DataFrame(np.nan, index=range(q.sum()), columns=cols)
            parts.append(pd.concat([prod[q].reset_index(drop=True), nan], axis=1))
    out = pd.concat(parts, ignore_index=True)
    return out.set_index("id").loc[prod["id"]].reset_index()  # mateix ordre que `prod`


def matrix(df: pd.DataFrame, emb_pca: pd.DataFrame, cats: dict | None = None):
    X = pd.concat([df[CAT_COLS + NUM_COLS + ["phase_in_month"]].reset_index(drop=True),
                   df.filter(like="knn").reset_index(drop=True), emb_pca.reset_index(drop=True)], axis=1)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=None if cats is None else cats[c])
    return X


if __name__ == "__main__":
    raw = pd.read_csv(D / "train.csv")
    test = aggregate(pd.read_csv(D / "test.csv"))
    prod = aggregate(raw)
    print(f"{len(prod):,} productes a train · {len(test):,} productes nous a test")

    emb = features.parse_embedding_column(prod["image_embedding"])
    emb_test = features.parse_embedding_column(test["image_embedding"])

    # ── 2. Validació temporal ──────────────────────────────────────────────
    valid_season = prod["id_season"].max()
    is_val = (prod["id_season"] == valid_season).to_numpy()
    tr, va = prod[~is_val], prod[is_val]

    # ── 3. Cold start: veïns només de temporades ANTERIORS a la que predim ──
    tr_f = add_knn_past(tr, emb[~is_val])
    va_f = add_knn(tr, emb[~is_val], va, emb[is_val])
    pca_tr, pca_va = features.embedding_pca(emb[~is_val], emb[is_val], n_components=16)
    X_tr = matrix(tr_f, pca_tr)
    cats = {c: X_tr[c].cat.categories for c in CAT_COLS}
    X_va = matrix(va_f, pca_va, cats)

    # ── 4. Model sobre log1p ───────────────────────────────────────────────
    base = models.lgbm_baseline("regression", n_estimators=800, learning_rate=0.03, num_leaves=31)
    model = models.LogTargetRegressor(base).fit(X_tr, tr["demand"])
    pred_va = model.predict(X_va)

    naive = np.full(len(va), tr["demand"].median())
    no_knn = models.LogTargetRegressor(base).fit(X_tr.drop(columns=X_tr.filter(like="knn").columns), tr["demand"])
    pred_no_knn = no_knn.predict(X_va.drop(columns=X_va.filter(like="knn").columns))
    print(f"\nValidació (temporada {valid_season}) · error asimètric, més baix és millor:")
    print(f"  mediana constant         {mango_score(va['demand'], naive):.6f}")
    print(f"  LightGBM sense kNN       {mango_score(va['demand'], pred_no_knn):.6f}")
    print(f"  LightGBM + kNN embedding {mango_score(va['demand'], pred_va):.6f}")

    # ── 5. Calibració asimètrica ───────────────────────────────────────────
    k, score = metrics.best_scale_factor(va["demand"], pred_va, metric=mango_score)
    print(f"  × factor {k:.2f}              {score:.6f}   ← calibrat a la mètrica asimètrica")

    # ── 6. Model final amb totes les temporades i submission ───────────────
    all_f = add_knn_past(prod, emb)
    te_f = add_knn(prod, emb, test, emb_test)
    pca_all, pca_te = features.embedding_pca(emb, emb_test, n_components=16)
    X_all = matrix(all_f, pca_all)
    cats = {c: X_all[c].cat.categories for c in CAT_COLS}
    final = models.LogTargetRegressor(base).fit(X_all, prod["demand"])
    pred_test = k * final.predict(matrix(te_f, pca_te, cats))

    path = submission.make_submission(test["id"], pred_test, id_col="ID", target_col="demand",
                                      out_dir=ROOT / "submissions", name="mango2025_synth")

    solution = D / "solution_hidden.csv"
    if solution.exists():  # només amb dades sintètiques: "Kaggle privat" simulat
        sol = pd.read_csv(solution).merge(pd.read_csv(path), on="ID", suffixes=("", "_pred"))
        print(f"\nPuntuació al test ocult: {mango_score(sol['demand'], sol['demand_pred']):.6f}")
