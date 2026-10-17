<div align="center">

# 🏆 DreamTeam · Datathon FME 2026

**Preparació, reptes anteriors resolts i plantilles de ML per a la 6a Datathon FME**
Facultat de Matemàtiques i Estadística (UPC) · Barcelona · **14–15 de novembre de 2026**

[![CI](https://github.com/Joanfmir/DreamTeam/actions/workflows/ci.yml/badge.svg)](https://github.com/Joanfmir/DreamTeam/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-tabular-9ACD32)
![SHAP](https://img.shields.io/badge/SHAP-explainability-FF0051)
![CLIP](https://img.shields.io/badge/CLIP-embeddings-412991)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

La Datathon FME és una competició de **48 hores** amb **3 reptes reals d'empreses**. Els reptes no es publiquen fins a l'inici, però **el tipus de problema es repeteix cada any**. Aquest repo recull els **15 reptes de 2021–2025** amb el seu enunciat, la mètrica i l'estratègia guanyadora, i hi afegeix **codi provat** per arrencar el dia D amb el pipeline ja fet.

## Què hi ha

| | |
|---|---|
| 📚 **[`reptes/`](reptes/README.md)** | **14 enunciats** (context, dades, mètrica, format d'entrega, estratègia guanyadora) i **7 solucions executables** amb resultats |
| 🧭 **[`docs/`](docs/)** | Format de la Datathon, **patrons i predicció per al 2026**, estratègia de 48 h, playbook de ML, **mètriques**, recursos i plantilla de presentació |
| 🧰 **[`src/dreamteam/`](src/dreamteam/)** | Llibreria pròpia amb tests: validació (amb detecció de duplicats), feature engineering, models, mètriques, SHAP, embeddings CLIP/text, optimització i submissions validades |
| 📓 **[`notebooks/`](notebooks/)** | Plantilles: EDA en 45 min i de CSV a submission en 10 min |
| 🖥️ **[`app/`](app/streamlit_demo.py)** | Demo de Streamlit amb predicció i explicació, per a la presentació |

## Resultats dels reptes que hem resolt

| Repte | Què fa el nostre codi | Resultat |
|---|---|---|
| [Mango 2025 · demanda](reptes/2025_mango_demanda/ENUNCIAT.md) | Agregació, validació temporal, **kNN per embedding** (cold start), `log1p`, **calibració asimètrica** | error 0,524 → **0,220** (dades sintètiques) |
| [Mango 2024 · atributs](reptes/2024_mango_atributs/ENUNCIAT.md) | Un LightGBM per atribut, validació agrupada per model | accuracy **0,801** (majoritària 0,640) |
| [Schneider 2025 · XAI](reptes/2025_schneider_explicabilitat/ENUNCIAT.md) | LightGBM, llindar F1, SHAP global i local, **detecció de leakage per duplicats** | AUC 0,85 (inflada) → **0,71 honesta** |
| [Accenture 2022 · retards](reptes/2022_accenture_supply_chain/ENUNCIAT.md) | Joins i distàncies → **target encoding OOF** i count encoding | AUC 0,821 → **0,836** |
| [Qualcomm 2022 · xips](reptes/2022_qualcomm_power_switches/ENUNCIAT.md) | K-means equilibrat, hongarès, serp + 2-opt (250.000 pins en 2 s) | **8,4×** millor que l'exemple oficial |
| [Reviews 2021 · NLP](reptes/2021_reviews_nlp/ENUNCIAT.md) | TF-IDF vs. embeddings multilingües | MAE **0,33** estrelles (baseline 0,56) |

## Començar

```bash
git clone https://github.com/Joanfmir/DreamTeam.git && cd DreamTeam
python -m venv .venv && source .venv/bin/activate

make setup        # dependències i paquet dreamteam
make data         # dades públiques dels reptes anteriors (~125 MB) → data/past/
make test         # 28 tests de la llibreria
make baselines    # executa totes les solucions de reptes/
make app          # demo de Streamlit

make setup-dl     # (opcional, ~2 GB) torch, CLIP i MiniLM a la caché per a imatges i text
```

## La llibreria en 10 línies

```python
from dreamteam import data, features, validation, models, metrics, explain, submission

df = data.load("train.csv")
X, y = data.split_xy(df, "target", drop=["id"])
validation.duplicate_report(X)                                    # ⚠️ hi ha files duplicades?
train, test = features.target_encode(train, test, ["client", ["port", "hub"]], "target")  # OOF, sense leakage
oof, scores, fold_models = validation.cross_validate(
    models.lgbm_baseline("binary"), X, y, validation.kfold(5, stratified=True), metrics.auc, proba=True)
thr, f1 = metrics.best_f1_threshold(y, oof)                       # llindar òptim sobre OOF
_, sv = explain.shap_values(fold_models[0], X); explain.top_reasons(sv, 0)   # per què?
submission.make_submission(ids, preds, sample_path="sample_submission.csv")  # validada
```

## Lliçons de 5 anys de Datathon

> **Model simple ben validat + ajustat a la mètrica + convertit en producte + ben explicat.** Detall a [`docs/02_patrons_guanyadors.md`](docs/02_patrons_guanyadors.md).

1. **Mango ha posat repte 3 anys seguits** (outfits → atributs → demanda) i torna el 2026. És l'escenari més probable.
2. **Gairebé cada any hi ha un repte amb Kaggle**, i **tots tenen part de presentació** (del 20 % al 100 %).
3. Els canvis més rendibles són d'**ajust a la mètrica**: `log1p` per a MSLE, quantil o factor per a pèrdues asimètriques, llindar per a F1 ([`docs/05_metriques.md`](docs/05_metriques.md)).
4. **3 de 6 guanyadors analitzats van fer un producte** (API, recomanador, app).
5. **Validació honesta**: a Schneider 2025 la CV aleatòria donava AUC 0,85 i la real era 0,71.

## Docs

| | |
|---|---|
| [01 · La Datathon FME](docs/01_datathon_fme.md) | Dates, format, equips, premis, com s'avalua |
| [02 · Patrons i estratègia guanyadora](docs/02_patrons_guanyadors.md) | Què es repeteix, què fan els guanyadors, **escenaris 2026** |
| [03 · Estratègia de 48 h](docs/03_estrategia_48h.md) | Rols, pla hora a hora, tria de repte, git |
| [04 · Playbook de ML](docs/04_playbook_ml.md) | Què fer segons el tipus de problema |
| [05 · Mètriques](docs/05_metriques.md) | AUC, F1, accuracy, MSLE, asimètrica, MAE/RMSE i com optimitzar cadascuna |
| [06 · Recursos](docs/06_recursos.md) | Enllaços, llibres, datasets oberts |
| [07 · Plantilla de presentació](docs/07_plantilla_presentacio.md) | Slide a slide i preguntes del jurat |

## Equip DreamTeam

| Membre | Rol principal | Rol de suport |
|---|---|---|
| Joan Fernández ([@Joanfmir](https://github.com/Joanfmir)) | *per decidir* | |
| *—* | | |
| *—* | | |
| *—* | | |

Rols (detall a [`docs/03_estrategia_48h.md`](docs/03_estrategia_48h.md)): **Modelatge** · **Dades i features** · **Visualització i demo** · **Història i presentació**.
Pla d'entrenament setmanal i checklist: [`reptes/README.md`](reptes/README.md#pla-dentrenament-1710--1311). Experiments: [`experiments.md`](experiments.md).

---

<sub>Fonts: [datathon.cat](https://www.datathon.cat/) · enunciats i dades de l'[Associació d'Estudiants de Dades](https://github.com/data-students) · repos públics d'equips participants (citats a cada enunciat).</sub>
