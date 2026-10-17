# Member 1 — Modeling & Optimization Roadmap

**Role:** Modeling & Optimization (Member 1)  
**Event:** 14–15 November 2026  
**Preparation:** 19 October–13 November · **3–5 hours/week**  
**Focus:** Tabular ML (classification and demand forecasting). Prioritize **2025 challenges** and practical exercises.

## Setup — Before Week 1 (30 min)

- [ ] Clone [DreamTeam](https://github.com/Joanfmir/DreamTeam) and run `make setup`, `make data`, `make test` (Linux/WSL).
- [ ] Check the [team strategy](https://github.com/Joanfmir/DreamTeam/blob/main/docs/03_estrategia_48h.md) and [experiment log](https://github.com/Joanfmir/DreamTeam/blob/main/experiments.md).

## Week 1 · 19–25 Oct — Baselines & Classification (3–5 h)

**Foundations (1–2 h)**
- [ ] [Kaggle Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning): Model Validation, Underfitting/Overfitting, Random Forests.
- [ ] [Kaggle Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning): Cross-Validation, Data Leakage.

**Specialization / hands-on (2–3 h)**
- [ ] Read [Schneider 2025](https://github.com/Joanfmir/DreamTeam/blob/main/reptes/2025_schneider_explicabilitat/ENUNCIAT.md) and run `python reptes/2025_schneider_explicabilitat/baseline.py`.
- [ ] Record **F1 and ROC-AUC**; compare random and **duplicate-aware/grouped** validation with Member 3.

**Deliverable:** A reproducible classifier + trustworthy baseline scores.

## Week 2 · 26 Oct–1 Nov — Preprocessing & Better Models (3–5 h)

**Foundations (1–2 h)**
- [ ] [Kaggle Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning): Missing Values, Categorical Variables, Pipelines, XGBoost.
- [ ] Understand training/validation separation and why the **same folds** are required for fair model comparison.

**Specialization / hands-on (2–3 h)**
- [ ] Compare **LightGBM and CatBoost** on Schneider (or tune one LightGBM) using **identical grouped CV**.
- [ ] Test one hypothesis at a time; coordinate new features with Member 2 and log results in [`experiments.md`](https://github.com/Joanfmir/DreamTeam/blob/main/experiments.md).

**Deliverable:** Model comparison table + one validated improvement.

## Week 3 · 2–8 Nov — Regression & Demand Forecasting (3–5 h)

**Foundations (1–2 h)**
- [ ] [Kaggle Time Series](https://www.kaggle.com/learn/time-series): Trend, Seasonality, Time Series as Features.
- [ ] Read the [metrics guide](https://github.com/Joanfmir/DreamTeam/blob/main/docs/05_metriques.md): regression losses, temporal splits and asymmetric penalties.

**Specialization / hands-on (2–3 h)**
- [ ] Read [Mango 2025](https://github.com/Joanfmir/DreamTeam/blob/main/reptes/2025_mango_demanda/ENUNCIAT.md).
- [ ] Run `python reptes/2025_mango_demanda/simulate_data.py` and `python reptes/2025_mango_demanda/pipeline.py` (**synthetic data**).
- [ ] Compare baseline vs improved predictions with **temporal validation** and the challenge's asymmetric metric.

**Deliverable:** Reproducible demand model + metric comparison.

## Week 4 · 9–13 Nov — Team Simulation (4–5 h)

**Shared preparation (1 h)**
- [ ] Review the [48-hour strategy](https://github.com/Joanfmir/DreamTeam/blob/main/docs/03_estrategia_48h.md) and agree on interfaces, metric, splits and submission format.

**Specialization / hands-on (3–4 h)**
- [ ] Simulate one **unseen past challenge** as a team; produce the first baseline **before** tuning.
- [ ] Own model training, fair CV comparisons and final predictions; give validated results to Member 4 for the pitch.
- [ ] Rehearse the final five-minute presentation together.

**Deliverable:** Working team submission + reproducible training command.

## Datathon Checklist

- [ ] Choose a challenge after reading all briefs; confirm **target, metric and submission format**.
- [ ] Agree on **leakage-safe validation** with Member 3 before comparing models.
- [ ] Build a simple baseline first; then improve one factor at a time.
- [ ] Log experiments and save final model, predictions and reproduction steps.
- [ ] Reserve time for business insights, demo and presentation.

**Rule:** Spend at least **70% of preparation time doing exercises**. Use AI to accelerate implementation, but personally verify validation, metrics and conclusions. After the Datathon, return to your longer-term probability, Kaggle and quantitative-finance roadmap.
