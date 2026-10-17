# Member 4 — Explainability, Demo & Story Roadmap

**Role:** Explainability, Business Insights & Presentation (Member 4)  
**Event:** 14–15 November 2026  
**Preparation:** 19 October–13 November · **3–5 hours/week**  
**Focus:** Transform validated ML into understandable business value and a convincing demo.

## Setup — Before Week 1 (30 min)

- [ ] Clone [DreamTeam](https://github.com/Joanfmir/DreamTeam); in Linux/WSL run `make setup`, `make data`, `make test`.
- [ ] Run `make app`; inspect the [Streamlit demo](https://github.com/Joanfmir/DreamTeam/tree/main/app) and [presentation template](https://github.com/Joanfmir/DreamTeam/blob/main/docs/07_plantilla_presentacio.md).

## Week 1 · 19–25 Oct — Prediction Explanations (3–5 h)

**Foundations (1–2 h)**
- [ ] [Kaggle Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning): Model Validation, Random Forests (understand what is being explained).
- [ ] [Kaggle ML Explainability](https://www.kaggle.com/learn/machine-learning-explainability): Permutation Importance, SHAP Values.

**Specialization / hands-on (2–3 h)**
- [ ] Read [Schneider 2025](https://github.com/Joanfmir/DreamTeam/blob/main/reptes/2025_schneider_explicabilitat/ENUNCIAT.md) and run the existing baseline.
- [ ] Create one global feature-importance plot and one local SHAP explanation, using the model approved by Member 3.

**Deliverable:** Two clear explanations, each with one plain-language takeaway.

## Week 2 · 26 Oct–1 Nov — Business Interpretation (3–5 h)

**Foundations (1–2 h)**
- [ ] [Kaggle Data Visualization](https://www.kaggle.com/learn/data-visualization): Distributions, Bar Charts, Heatmaps.
- [ ] [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/): SHAP overview and caveats; understand **correlation is not causation**.

**Specialization / hands-on (2–3 h)**
- [ ] Translate three Schneider findings into clear actions for a sales team; describe model failure modes and uncertainty.
- [ ] Verify each quantitative claim with Member 3; draft three slides: problem, evidence, business impact.

**Deliverable:** Three evidence-backed insights + one limitations statement.

## Week 3 · 2–8 Nov — Forecast Demo & Pitch (3–5 h)

**Foundations (1–2 h)**
- [ ] [Streamlit Getting Started](https://docs.streamlit.io/get-started): widgets, charts and app execution.
- [ ] [DreamTeam Presentation Template](https://github.com/Joanfmir/DreamTeam/blob/main/docs/07_plantilla_presentacio.md): narrative and judging questions.

**Specialization / hands-on (2–3 h)**
- [ ] Adapt the team's existing Streamlit app: input → prediction → explanation. Use **Mango 2025 synthetic forecasts** or Schneider predictions.
- [ ] Build one chart showing the **organizer's metric** or a meaningful business cost and include a fallback screenshot.

**Deliverable:** Working 30-second demo + five-slide presentation outline.

## Week 4 · 9–13 Nov — Team Simulation (4–5 h)

**Shared preparation (1 h)**
- [ ] Review the [48-hour strategy](https://github.com/Joanfmir/DreamTeam/blob/main/docs/03_estrategia_48h.md); align the pitch structure with the team's chosen challenge.

**Specialization / hands-on (3–4 h)**
- [ ] During an **unseen team challenge**, own the storyline, visualization, five-minute pitch and demo integration.
- [ ] Confirm score and limitations with Member 3; rehearse twice with teammates and save screenshots as backup.

**Deliverable:** Five-minute rehearsal presentation + working demo/fallback.

## Datathon Checklist

- [ ] State the business problem and decision/use case clearly.
- [ ] Compare reliable model performance against a baseline.
- [ ] Show only a few useful and defensible findings.
- [ ] Bring slides and a fallback for any live demo.

**Rule:** At least **70% hands-on work**. A clear, defensible story beats flashy, unverified visualizations.
