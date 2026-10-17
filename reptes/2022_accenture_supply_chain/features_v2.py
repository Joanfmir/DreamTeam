"""Accenture 2022 · v2: el baseline + encodings de `dreamteam.features`.

    python reptes/2022_accenture_supply_chain/features_v2.py

Exemple de la iteració típica de la Datathon: mateixa validació, features noves i
comparació directa amb el baseline. Cada canvi s'apunta a experiments.md.

⚠️ El target encoding es calcula DINS de cada fold (`fold_fn`): si es calculés un cop sobre tot
el train i després es fes CV amb uns altres folds, la validació veuria el seu propi target i
l'AUC sortiria inflada (ens va passar a la primera versió d'aquest script).
"""

from baseline import ROOT, build_features, load, to_model_matrix
from dreamteam import features, metrics, models, submission, validation

TARGET = "late_order"
TE_COLS = ["product_id", "customer", "ruta", ["3pl", "logistic_hub"], ["3pl", "customs_procedures"],
           ["origin_port", "3pl"]]
HIGH_CARD = ["product_id", "ruta"]  # massa categories en cru: només es fan servir via encodings

orders, test, products, cities = load()
train_f = build_features(orders, products, cities)
test_f = build_features(test, products, cities)

# Ruta completa com a categoria: port → hub → client
for df in (train_f, test_f):
    df["ruta"] = df["origin_port"] + ">" + df["logistic_hub"] + ">" + df["customer"]

# Features SENSE target: es poden calcular un cop per a tot (no hi ha leakage)
train_f, test_f = features.count_encode(train_f, test_f, ["product_id", "customer", "ruta", "3pl"])
train_f = features.group_aggregates(train_f, "product_id", ["units"], aggs=("mean",))
test_f = features.group_aggregates(test_f, "product_id", ["units"], aggs=("mean",))

X, y, X_test = to_model_matrix(train_f, test_f)


def target_encoding_fold(X_tr, y_tr, X_other):
    """Target encoding OOF calculat només amb el train del fold; després es treuen les columnes crues."""
    tr = X_tr.assign(**{TARGET: y_tr.to_numpy()})
    tr, other = features.target_encode(tr, X_other, TE_COLS, TARGET)
    return tr.drop(columns=[TARGET, *HIGH_CARD]), other.drop(columns=HIGH_CARD)


model = models.lgbm_baseline("binary", n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=50)
oof, scores, preds = validation.cross_validate_with_test(
    model, X, y, X_test, cv=validation.kfold(5, stratified=True), metric=metrics.auc, proba=True,
    fold_fn=target_encoding_fold,
)
print(f"\nAUC OOF v2: {metrics.auc(y, oof):.4f}  (baseline: 0.8212)")

submission.make_submission(
    test["order_id"], preds, id_col="order_id", target_col=TARGET,
    out_dir=ROOT / "submissions", name="accenture_v2",
)
