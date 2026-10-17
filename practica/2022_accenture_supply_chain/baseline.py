"""Baseline del repte Accenture 2022: predir quines comandes arribaran tard (mètrica: AUC).

Executar des de l'arrel del repo (abans: bash scripts/download_past_data.sh):
    python practica/2022_accenture_supply_chain/baseline.py

Mostra el flux complet de la Datathon: joins → features → CV → submission validada.
"""

from pathlib import Path

import pandas as pd

from dreamteam import data, metrics, models, submission, validation

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/past/2022_accenture"

# 1. Càrrega (atenció: orders, test i cities fan servir ';' com a separador)
orders = pd.read_csv(D / "orders.csv", sep=";")
test = pd.read_csv(D / "test.csv", sep=";")
products = pd.read_csv(D / "product_attributes.csv")
cities = pd.read_csv(D / "cities_data.csv", sep=";")
print(f"train: {orders.shape} · test: {test.shape} · % tard: {orders.late_order.mean():.1%}")

# 2. Features: pes del producte i distàncies port → hub → client
dist = cities.set_index(["city_from_name", "city_to_name"])["distance"]


def lookup_distance(a: pd.Series, b: pd.Series) -> pd.Series:
    """Distància entre dues ciutats en qualsevol sentit (el fitxer només en té un)."""
    pairs = list(zip(a, b))
    fwd = pd.Series(pairs).map(lambda p: dist.get(p))
    bwd = pd.Series(pairs).map(lambda p: dist.get(p[::-1]))
    return fwd.fillna(bwd).to_numpy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(products, on="product_id", how="left")
    df["logistic_hub"] = df["logistic_hub"].fillna("DIRECTE")  # sense hub = enviament directe
    hub = df["logistic_hub"].where(df["logistic_hub"] != "DIRECTE", df["customer"])
    df["dist_port_hub"] = lookup_distance(df["origin_port"], hub)
    df["dist_hub_client"] = lookup_distance(hub, df["customer"])
    df["dist_total"] = df["dist_port_hub"].fillna(0) + df["dist_hub_client"].fillna(0)
    df["pes_total"] = df["weight"] * df["units"]
    df["product_id"] = df["product_id"].astype(str)
    return df


train = build_features(orders)
test_f = build_features(test)

drop = ["order_id"]
X, y = data.split_xy(train, target="late_order", drop=drop)
X = data.to_category(X)
X_test = data.to_category(test_f.drop(columns=drop))
for col in X.select_dtypes("category"):  # mateixes categories a train i test
    X_test[col] = pd.Categorical(X_test[col], categories=X[col].cat.categories)

# 3. CV estratificada + AUC
model = models.lgbm_baseline("binary", n_estimators=600, learning_rate=0.05)
oof, scores, fold_models = validation.cross_validate(
    model, X, y, cv=validation.kfold(5, stratified=True), metric=metrics.auc, proba=True,
)

# 4. Submission validada (format Kaggle: order_id, late_order)
preds = validation.predict_folds(fold_models, X_test, proba=True)
submission.make_submission(
    test["order_id"], preds, id_col="order_id", target_col="late_order",
    out_dir=ROOT / "submissions", name="accenture_lgbm",
)
