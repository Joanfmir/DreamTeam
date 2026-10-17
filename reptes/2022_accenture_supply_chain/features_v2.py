"""Accenture 2022 · v2: el baseline + encodings de `dreamteam.features`.

    python reptes/2022_accenture_supply_chain/features_v2.py

Exemple de la iteració típica de la Datathon: mateixa validació, features noves i
comparació directa amb el baseline. Cada canvi s'apunta a experiments.md.
"""

from baseline import ROOT, build_features, load, run_cv, to_model_matrix
from dreamteam import features, metrics, submission

orders, test, products, cities = load()
train_f = build_features(orders, products, cities)
test_f = build_features(test, products, cities)

# Ruta completa com a categoria: port → hub → client
for df in (train_f, test_f):
    df["ruta"] = df["origin_port"] + ">" + df["logistic_hub"] + ">" + df["customer"]

# 1) Count encoding: "popularitat" de cada producte, client i ruta (càrrega del sistema)
train_f, test_f = features.count_encode(train_f, test_f, ["product_id", "customer", "ruta", "3pl"])

# 2) Target encoding OOF: % de comandes tard per categoria i per combinacions
train_f, test_f = features.target_encode(
    train_f, test_f,
    cols=["product_id", "customer", "ruta", ["3pl", "logistic_hub"], ["3pl", "customs_procedures"],
          ["origin_port", "3pl"]],
    target="late_order",
)

# 3) Agregats: unitats d'aquesta comanda vs. la mitjana del producte
train_f = features.group_aggregates(train_f, "product_id", ["units"], aggs=("mean",))
test_f = features.group_aggregates(test_f, "product_id", ["units"], aggs=("mean",))

# product_id i ruta en cru tenen massa categories; ja estan representats pels encodings
for df in (train_f, test_f):
    df.drop(columns=["product_id", "ruta"], inplace=True)

X, y, X_test = to_model_matrix(train_f, test_f)
print(f"Features: {X.shape[1]} (baseline: 12)")
oof, preds = run_cv(X, y, X_test, "v2", num_leaves=31, min_child_samples=50)
print(f"\nAUC OOF v2: {metrics.auc(y, oof):.4f}  (baseline: 0.8212)")

submission.make_submission(
    test["order_id"], preds, id_col="order_id", target_col="late_order",
    out_dir=ROOT / "submissions", name="accenture_v2",
)
