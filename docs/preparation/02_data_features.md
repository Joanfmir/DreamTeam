# Member 2 — Data & Feature Engineering Roadmap

**Role:** Data & Feature Engineering (Member 2)  
**Event:** 14–15 November 2026  
**Preparation:** 19 October–13 November · **3–5 hours/week**  
**Focus:** Reliable data pipelines and useful features for recent tabular ML challenges.

## Setup — Before Week 1 (30 min)

- [ ] Clone [DreamTeam](https://github.com/Joanfmir/DreamTeam); in Linux/WSL run `make setup`, `make data`, `make test`.
- [ ] Open the [EDA templates](https://github.com/Joanfmir/DreamTeam/tree/main/notebooks) and [experiment log](https://github.com/Joanfmir/DreamTeam/blob/main/experiments.md).

## Week 1 · 19–25 Oct — Data Audit & First Baseline (3–5 h)

**Foundations (1–2 h)**
- [ ] [Kaggle Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning): Basic Data Exploration, Model Validation.
- [ ] [Kaggle Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning): Missing Values, Data Leakage.

**Specialization / hands-on (2–3 h)**
- [ ] Read [Schneider 2025](https://github.com/Joanfmir/DreamTeam/blob/main/reptes/2025_schneider_explicabilitat/ENUNCIAT.md); run its existing baseline after `make data`.
- [ ] Audit missing values, column types, identifiers, duplicate rows and target leakage. Share potential duplicate groups with Member 3.

**Deliverable:** One-page audit + reusable train/validation feature table.

## Week 2 · 26 Oct–1 Nov — Preprocessing & Feature Engineering (3–5 h)

**Foundations (1–2 h)**
- [ ] [Kaggle Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning): Categorical Variables, Pipelines.
- [ ] [Kaggle Feature Engineering](https://www.kaggle.com/learn/feature-engineering): Mutual Information, Creating Features.

**Specialization / hands-on (2–3 h)**
- [ ] Build a reproducible imputation + encoding pipeline; fit transformations **only on training folds**.
- [ ] Create 2–3 justified Schneider features; have Member 1 assess improvements using Member 3's **same grouped folds**. Log results in [`experiments.md`](https://github.com/Joanfmir/DreamTeam/blob/main/experiments.md).

**Deliverable:** Reusable preprocessing/feature functions + baseline vs improved CV comparison.

## Week 3 · 2–8 Nov — Forecasting Features (3–5 h)

**Foundations (1–2 h)**
- [ ] [Kaggle Time Series](https://www.kaggle.com/learn/time-series): Trend, Seasonality, Time Series as Features.
- [ ] [DreamTeam ML Playbook](https://github.com/Joanfmir/DreamTeam/blob/main/docs/04_playbook_ml.md): tabular and time-series sections.

**Specialization / hands-on (2–3 h)**
- [ ] Read [Mango 2025](https://github.com/Joanfmir/DreamTeam/blob/main/reptes/2025_mango_demanda/ENUNCIAT.md); run `python reptes/2025_mango_demanda/simulate_data.py` (**synthetic exercise data**).
- [ ] Create product/season aggregations and 2 time-aware features; verify **no future information** enters earlier rows. Coordinate scoring with Members 1 and 3.

**Deliverable:** Leak-free demand features + documented feature availability.

## Week 4 · 9–13 Nov — Team Simulation (4–5 h)

**Shared preparation (1 h)**
- [ ] Review the [48-hour strategy](https://github.com/Joanfmir/DreamTeam/blob/main/docs/03_estrategia_48h.md) and agree on data format and handoff boundaries.

**Specialization / hands-on (3–4 h)**
- [ ] Solve an **unseen past challenge** with the team. Own fast EDA, missing-value handling, joins and feature generation.
- [ ] Deliver clean features early so Member 1 can train; review the final input schema and rerun the preprocessing pipeline.

**Deliverable:** Reproducible `features.parquet`/CSV + feature definitions.

## Datathon Checklist

- [ ] Identify target, IDs, missingness, duplicates and train/test schema immediately.
- [ ] Check feature availability at prediction time; avoid target leakage.
- [ ] Fit preprocessing within training folds and version the data pipeline.
- [ ] Share the first usable feature table before trying advanced features.

**Rule:** At least **70% hands-on work**. Learn concepts well enough to audit AI-generated preprocessing code; skip unrelated NLP/CV until the challenge is known.
