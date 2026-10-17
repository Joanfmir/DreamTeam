"""Baseline del repte Schneider Electric 2025: classificació d'oportunitats + explicabilitat.

Executar des de l'arrel del repo (abans: bash scripts/download_past_data.sh):
    python reptes/2025_schneider_explicabilitat/baseline.py

Genera a reptes/2025_schneider_explicabilitat/outputs/:
    shap_beeswarm.png, shap_waterfall_guanyada.png, shap_waterfall_perduda.png, importancia.csv

⚠️ Lliçó d'aquest dataset: el 77 % de les files tenen features duplicades. Un K-Fold aleatori
dona una AUC inflada (~0,85) perquè el model memoritza còpies. Amb validació agrupada per
duplicats, l'estimació honesta és ~0,71. Aquí es mostren les dues.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sense finestra: desa els gràfics a fitxer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from dreamteam import data, explain, metrics, models, validation

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/past/2025_schneider/dataset.csv"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

# 1. Dades: el dataset ja ve net i escalat; només cal treure l'id
df = data.load(DATA)
X, y = data.split_xy(df, target="target_variable", drop=["id"])
print(f"{len(df):,} oportunitats · {X.shape[1]} variables · {y.mean():.1%} guanyades")
validation.duplicate_report(X)
groups = validation.duplicate_groups(X)

# Holdout agrupat: cap combinació de features és alhora a train i a test
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
g_train = groups[tr]

# 2. Model (el "15 %" del temps): LightGBM
model = models.lgbm_baseline("binary", n_estimators=400, learning_rate=0.05, num_leaves=31)

print("\nCV amb K-Fold aleatori (INFLADA per duplicats):")
validation.cross_validate(model, X_train, y_train, cv=validation.kfold(5, stratified=True),
                          metric=metrics.auc, proba=True)

print("\nCV agrupada per duplicats (HONESTA):")
oof, scores, fold_models = validation.cross_validate(
    model, X_train, y_train, cv=validation.group_kfold(5), groups=g_train,
    metric=metrics.auc, proba=True,
)

# Llindar òptim per a F1 triat sobre OOF (no sobre el test)
threshold, f1_oof = metrics.best_f1_threshold(y_train, oof)
proba_test = validation.predict_folds(fold_models, X_test, proba=True)
f1_test = f1_score(y_test, proba_test >= threshold)
print(f"\nLlindar òptim (OOF): {threshold:.2f} · F1 OOF = {f1_oof:.3f} · F1 test = {f1_test:.3f}"
      f"  {'✓ supera el' if f1_test >= 0.7 else '✗ per sota del'} 0,7 que demanaven")
print(f"AUC test (agrupat): {metrics.auc(y_test, proba_test):.3f}\n")

# 3. Explicabilitat (el "85 %" del temps)
final_model = fold_models[0]
X_expl = X_test.sample(min(2000, len(X_test)), random_state=42)  # mostra per a SHAP
_, sv = explain.shap_values(final_model, X_expl)

imp = explain.global_importance(sv)
imp.to_csv(OUT / "importancia.csv", index=False)
print("Variables més importants (mitjana de |SHAP|):")
print(imp.head(8).to_string(index=False))

plt.figure()
explain.global_summary(sv, show=False)
plt.tight_layout()
plt.savefig(OUT / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close("all")

# Explicacions locals: l'oportunitat més clarament guanyada i la més clarament perduda
pred = final_model.predict_proba(X_expl)[:, 1]
for name, i in {"guanyada": int(np.argmax(pred)), "perduda": int(np.argmin(pred))}.items():
    plt.figure()
    explain.local_waterfall(sv, i, show=False)
    plt.savefig(OUT / f"shap_waterfall_{name}.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\nPer què el model diu '{name}' (prob = {pred[i]:.2f}):")
    for reason in explain.top_reasons(sv, i):
        print("  ·", reason)

print(f"\n✓ Gràfics desats a {OUT.relative_to(ROOT)}/")
