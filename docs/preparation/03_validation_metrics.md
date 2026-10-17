# Member 3 — Validation & Metrics Roadmap

**Role:** Validation, Metrics & Statistical Evaluation (Member 3)  
**Event:** 14–15 November 2026  
**Preparation:** 19 October–13 November · **3–5 hours/week**  
**Focus:** Choose trustworthy validation and optimize the exact challenge metric.

## Setup — Before Week 1 (30 min)

- [ ] Clone [DreamTeam](https://github.com/Joanfmir/DreamTeam); in Linux/WSL run `make setup`, `make data`, `make test`.
- [ ] Read [DreamTeam Metrics](https://github.com/Joanfmir/DreamTeam/blob/main/docs/05_metriques.md) and open the [experiment log](https://github.com/Joanfmir/DreamTeam/blob/main/experiments.md).

## Week 1 · 19–25 Oct — Validation & Leakage (3–5 h)

**Foundations (1–2 h)**
- [ ] [Kaggle Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning): Model Validation, Underfitting and Overfitting.
- [ ] [Kaggle Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning): Cross-Validation, Data Leakage.

**Specialization / hands-on (2–3 h)**
- [ ] Read [Schneider 2025](https://github.com/Joanfmir/DreamTeam/blob/main/reptes/2025_schneider_explicabilitat/ENUNCIAT.md) and run its existing baseline.
- [ ] Compare random CV with duplicate-aware/**grouped** CV; compute F1 and ROC-AUC and explain any score inflation to Members 1–2.

**Deliverable:** Trusted Schneider folds, metrics and a short leakage explanation.

## Week 2 · 26 Oct–1 Nov — F1 Thresholds & Score Reliability (3–5 h)

**Foundations (1–2 h)**
- [ ] [scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html): precision, recall, F1, ROC-AUC, confusion matrices.
- [ ] [DreamTeam Metrics](https://github.com/Joanfmir/DreamTeam/blob/main/docs/05_metriques.md): metric selection and thresholding.

**Specialization / hands-on (2–3 h)**
- [ ] Generate out-of-fold Schneider probabilities on the **same grouped folds**. Compare threshold 0.5 with a selected F1 threshold.
- [ ] Report fold variability and explain that a threshold tuned on the same OOF predictions needs **separate held-out confirmation** before claiming an unbiased gain.

**Deliverable:** Reusable scoring script + validation/threshold memo.

## Week 3 · 2–8 Nov — Temporal Validation & Business Loss (3–5 h)

**Foundations (1–2 h)**
- [ ] [Kaggle Time Series](https://www.kaggle.com/learn/time-series): Time Series as Features, Hybrid Models (focus on temporal leakage concepts).
- [ ] [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html): correct ordering of training and validation.

**Specialization / hands-on (2–3 h)**
- [ ] Read [Mango 2025](https://github.com/Joanfmir/DreamTeam/blob/main/reptes/2025_mango_demanda/ENUNCIAT.md); run its **synthetic** practice pipeline.
- [ ] Compare random vs temporal splits and untuned vs asymmetry-aware predictions. Consult [Smadex 2025](https://github.com/Joanfmir/DreamTeam/blob/main/reptes/2025_smadex_ingressos/ENUNCIAT.md) for MSLE differences; original Smadex data may be unavailable.

**Deliverable:** One-page decision tree for metric + CV scheme by problem type.

## Week 4 · 9–13 Nov — Team Simulation (4–5 h)

**Shared preparation (1 h)**
- [ ] Read the [48-hour strategy](https://github.com/Joanfmir/DreamTeam/blob/main/docs/03_estrategia_48h.md); agree on target, metric and leakage checks before the team rehearsal.

**Specialization / hands-on (3–4 h)**
- [ ] During an **unseen team challenge**, define splits early, approve baseline scoring, monitor CV stability and audit final predictions.
- [ ] Provide Member 4 with the exact approved score and a limitation the jury should know.

**Deliverable:** Reproducible evaluation code + validated score and concise caveats.

## Datathon Checklist

- [ ] Confirm organizer metric and available features at prediction time.
- [ ] Choose group/time/random CV to match the hidden-test design.
- [ ] Compare against a trivial baseline; report score variability.
- [ ] Never tune or choose a model using the untouched final test set.

**Rule:** At least **70% hands-on work**. AI may generate metric code, but manually test it against small known examples.
