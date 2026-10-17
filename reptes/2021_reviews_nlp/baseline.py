"""Baseline de Reviews 2021: predir les estrelles (1–5) d'una ressenya a partir del text.

    bash scripts/download_past_data.sh reviews
    python reptes/2021_reviews_nlp/baseline.py

Dos models: TF-IDF + regressió logística (segons) i, si hi ha `sentence-transformers`
instal·lat, embeddings multilingües + regressió logística. Mètriques: accuracy i MAE en estrelles.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/past/2021_reviews"


def to_frame(path: Path) -> pd.DataFrame:
    """Una fila per ressenya amb les dades del restaurant (el JSON és una llista de restaurants)."""
    rows = []
    for rest in json.load(open(path)):
        for rev in rest.get("reviews_data", []):
            rows.append({
                "restaurant": rest.get("title"), "price": rest.get("price"), "type": rest.get("type"),
                "rest_rating": rest.get("rating"), "text": rev.get("snippet") or "",
                "stars": rev.get("rating"),
            })
    df = pd.DataFrame(rows)
    return df[df["text"].str.len() > 0].dropna(subset=["stars"])


def report(name: str, y_true, y_pred):
    acc, mae = accuracy_score(y_true, y_pred), mean_absolute_error(y_true, y_pred)
    print(f"{name:<34} accuracy {acc:.3f} · MAE {mae:.3f} estrelles")


if __name__ == "__main__":
    # Train i test són restaurants diferents: així mesurem si generalitza a restaurants nous
    train, test = to_frame(D / "train_reviews.json"), to_frame(D / "test_reviews.json")
    print(f"train: {len(train):,} ressenyes · test: {len(test):,}")
    print("Distribució d'estrelles:", train["stars"].value_counts(normalize=True).sort_index().round(2).to_dict(), "\n")

    report("Sempre 5 estrelles", test["stars"], np.full(len(test), 5.0))

    tfidf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4, class_weight="balanced"),
    )
    tfidf.fit(train["text"], train["stars"].astype(int))
    report("TF-IDF + LogisticRegression", test["stars"], tfidf.predict(test["text"]))

    # Paraules que més empenyen cap a 1 i cap a 5 estrelles: material per a una slide
    clf, vocab = tfidf[-1], tfidf[0].get_feature_names_out()
    for k, label in [(0, "1★"), (-1, "5★")]:
        top = vocab[np.argsort(clf.coef_[k])[-12:]][::-1]
        print(f"  paraules {label}: {', '.join(top)}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n(sentence-transformers no instal·lat: es salta el model d'embeddings)")
    else:
        enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        E_tr = enc.encode(train["text"].tolist(), batch_size=64, show_progress_bar=False)
        E_te = enc.encode(test["text"].tolist(), batch_size=64, show_progress_bar=False)
        lr = LogisticRegression(max_iter=3000, C=1, class_weight="balanced").fit(E_tr, train["stars"].astype(int))
        report("Embeddings multilingües + LogReg", test["stars"], lr.predict(E_te))
