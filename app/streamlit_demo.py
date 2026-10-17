"""Plantilla de demo per a la presentació: prediu un cas i n'explica el perquè.

    streamlit run app/streamlit_demo.py

Ve configurada amb el repte Schneider 2025. Per a un repte nou, només cal canviar CONFIG.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dreamteam import data, explain, models

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "titol": "Guanyarem aquesta oportunitat?",
    "dades": ROOT / "data/past/2025_schneider/dataset.csv",
    "target": "target_variable",
    "drop": ["id"],
    "etiquetes": {1: "✅ Guanyada", 0: "❌ Perduda"},
}

st.set_page_config(page_title="DreamTeam · Demo", page_icon="🏆", layout="wide")
st.title(CONFIG["titol"])


@st.cache_resource
def train():
    df = data.load(CONFIG["dades"])
    X, y = data.split_xy(df, CONFIG["target"], drop=CONFIG["drop"])
    model = models.lgbm_baseline("binary", n_estimators=300, learning_rate=0.05).fit(X, y)
    return model, X


model, X = train()

# Formulari: un slider per variable, inicialitzat amb un cas real
with st.sidebar:
    st.header("Introdueix el cas")
    i = st.number_input("Partir de la fila", 0, len(X) - 1, 0)
    caso = {}
    for col in X.columns:
        lo, hi, val = float(X[col].min()), float(X[col].max()), float(X[col].iloc[i])
        caso[col] = st.slider(col, lo, hi, val) if lo < hi else val

row = pd.DataFrame([caso])[X.columns]
proba = model.predict_proba(row)[0, 1]

c1, c2 = st.columns([1, 2])
c1.metric("Probabilitat", f"{proba:.0%}")
c1.subheader(CONFIG["etiquetes"][int(proba >= 0.5)])

_, sv = explain.shap_values(model, row)
c1.markdown("**Per què?**")
for r in explain.top_reasons(sv, 0, k=3):
    c1.write("· " + r)

fig = plt.figure()
explain.local_waterfall(sv, 0, show=False)
c2.pyplot(plt.gcf(), clear_figure=True)
