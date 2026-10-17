"""Baseline del repte Accenture 2022: predir quines comandes arribaran tard (mètrica: AUC).

Executar des de l'arrel del repo (abans: bash scripts/download_past_data.sh):
    python reptes/2022_accenture_supply_chain/baseline.py

Mostra el flux complet de la Datathon: joins → features → CV → submission validada.
`features_v2.py` reutilitza aquestes funcions i hi afegeix encodings.
"""

from pathlib import Path

import pandas as pd

from dreamteam import data, metrics, models, submission, validation

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/past/2022_accenture"


def load():
    """Carrega els 4 fitxers (atenció: orders, test i cities fan servir ';' com a separador)."""
    orders = pd.read_csv(D / "orders.csv", sep=";")
    test = pd.read_csv(D / "test.csv", sep=";")
    products = pd.read_csv(D / "product_attributes.csv")
    cities = pd.read_csv(D / "cities_data.csv", sep=";")
    return orders, test, products, cities


def build_features(df: pd.DataFrame, products: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Pes del producte i distàncies port → hub → client."""
    dist = cities.set_index(["city_from_name", "city_to_name"])["distance"].to_dict()

    def lookup(a, b):  # el fitxer només té un sentit de cada parella
        return [dist.get((x, y), dist.get((y, x))) for x, y in zip(a, b, strict=True)]

    df = df.merge(products, on="product_id", how="left")
    df["logistic_hub"] = df["logistic_hub"].fillna("DIRECTE")  # sense hub = enviament directe
    hub = df["logistic_hub"].where(df["logistic_hub"] != "DIRECTE", df["customer"])
    df["dist_port_hub"] = pd.to_numeric(pd.Series(lookup(df["origin_port"], hub)), errors="coerce").to_numpy()
    df["dist_hub_client"] = pd.to_numeric(pd.Series(lookup(hub, df["customer"])), errors="coerce").to_numpy()
    df["dist_total"] = df["dist_port_hub"].fillna(0) + df["dist_hub_client"].fillna(0)
    df["pes_total"] = df["weight"] * df["units"]
    df["product_id"] = df["product_id"].astype(str)
    return df


def to_model_matrix(train: pd.DataFrame, test: pd.DataFrame, target: str = "late_order"):
    """X, y i X_test amb les mateixes categories a train i test."""
    X, y = data.split_xy(train, target=target, drop=["order_id"])
    X = data.to_category(X)
    X_test = data.to_category(test[X.columns])
    for col in X.select_dtypes("category"):
        X_test[col] = pd.Categorical(X_test[col], categories=X[col].cat.categories)
    return X, y, X_test


def run_cv(X, y, X_test, name: str, **lgbm_params):
    model = models.lgbm_baseline("binary", **{"n_estimators": 600, "learning_rate": 0.05, **lgbm_params})
    oof, scores, fold_models = validation.cross_validate(
        model, X, y, cv=validation.kfold(5, stratified=True), metric=metrics.auc, proba=True,
    )
    preds = validation.predict_folds(fold_models, X_test, proba=True)
    return oof, preds


if __name__ == "__main__":
    orders, test, products, cities = load()
    print(f"train: {orders.shape} · test: {test.shape} · % tard: {orders.late_order.mean():.1%}")

    train_f = build_features(orders, products, cities)
    test_f = build_features(test, products, cities)
    X, y, X_test = to_model_matrix(train_f, test_f)
    validation.duplicate_report(X)

    _, preds = run_cv(X, y, X_test, "baseline")
    submission.make_submission(
        test["order_id"], preds, id_col="order_id", target_col="late_order",
        out_dir=ROOT / "submissions", name="accenture_baseline",
    )
