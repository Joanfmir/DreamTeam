"""Baseline de Mango 2024: predir 11 atributs de disseny de cada peça (mètrica: accuracy mitjana).

    bash scripts/download_past_data.sh mango2024
    python reptes/2024_mango_atributs/baseline.py

Només fa servir les metadades del producte (sense imatges). Un model LightGBM per atribut.
Les combinacions producte × atribut que no apareixen a attribute_data són "INVALID"
(l'atribut no aplica a aquella peça, p. ex. `heel_shape_type` d'una samarreta).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

from dreamteam import models, submission

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/past/2024_mango"
FEATURES = ["cod_color", "des_sex", "des_age", "des_line", "des_fabric", "des_product_category",
            "des_product_aggregated_family", "des_product_family", "des_product_type", "des_color"]
ATTRIBUTES = ["silhouette_type", "neck_lapel_type", "woven_structure", "knit_structure",
              "heel_shape_type", "length_type", "sleeve_length_type", "toecap_type", "waist_type",
              "closure_placement", "cane_height_type"]


def load():
    products = pd.read_csv(D / "product_data.csv", dtype={"cod_color": str})
    attrs = pd.read_csv(D / "attribute_data.csv", dtype=str)
    test = pd.read_csv(D / "test_data.csv", dtype={"cod_color": str})
    # product_data té una fila per imatge; ens quedem amb una per producte
    products = products.drop_duplicates("cod_modelo_color")
    # Format ample: una columna per atribut; si no hi és → INVALID
    wide = attrs.pivot_table(index="cod_modelo_color", columns="attribute_name",
                             values="des_value", aggfunc="first")
    train = products.merge(wide, left_on="cod_modelo_color", right_index=True, how="inner")
    train[ATTRIBUTES] = train[ATTRIBUTES].fillna("INVALID")
    return train, test


def model_matrix(df: pd.DataFrame, categories: dict | None = None):
    X = df[FEATURES].astype("category")
    if categories:
        for col in FEATURES:
            X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


if __name__ == "__main__":
    train, test = load()
    print(f"Productes amb atributs: {len(train):,} · files de test: {len(test):,} "
          f"({test.cod_modelo_color.nunique():,} productes × {test.attribute_name.nunique()} atributs)\n")

    # Validació agrupada pel model base (part abans de '_' del codi és la temporada; després, el model):
    # variants de color del mateix model no poden estar a train i a validació alhora.
    groups = train["cod_modelo_color"].str.split("_").str[1]
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(train, groups=groups))
    X = model_matrix(train)
    categories = {c: X[c].cat.categories for c in FEATURES}

    accs, majority = {}, {}
    preds_test = []
    for attr in ATTRIBUTES:
        y = train[attr]
        model = models.lgbm_baseline("multiclass", n_estimators=300, learning_rate=0.1,
                                     num_leaves=31, min_child_samples=10)
        model.fit(X.iloc[tr], y.iloc[tr])
        accs[attr] = accuracy_score(y.iloc[va], model.predict(X.iloc[va]))
        majority[attr] = (y.iloc[va] == y.iloc[tr].mode()[0]).mean()
        print(f"  {attr:<20} accuracy {accs[attr]:.3f}   (classe majoritària: {majority[attr]:.3f})")

        # Model final amb tot el train → prediccions de test per a aquest atribut
        model.fit(X, y)
        rows = test[test["attribute_name"] == attr]
        preds_test.append(pd.DataFrame({
            "test_id": rows["test_id"],
            "des_value": model.predict(model_matrix(rows, categories)),
        }))

    print(f"\nAccuracy mitjana (11 atributs): {np.mean(list(accs.values())):.3f}"
          f"  ·  baseline classe majoritària: {np.mean(list(majority.values())):.3f}")

    sub = pd.concat(preds_test)
    submission.make_submission(sub["test_id"], sub["des_value"], id_col="test_id",
                               target_col="des_value", out_dir=ROOT / "submissions", name="mango2024")
