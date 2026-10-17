# Smadex 2025 · Quant valdrà aquest usuari?

> **2025** · Smadex · Regressió amb target molt esbiaixat (ad-tech, dades grans) · **MSLE** + velocitat + simplicitat · Dades privades (Kaggle)

## Context

Smadex és una plataforma de publicitat mòbil (DSP) que participa en **subhastes en temps real**: per a cada impressió d'anunci té **pocs mil·lisegons** per decidir quant pujar. Si sap quins usuaris generaran ingressos un cop instal·lin l'app, pot pujar més per aquests i menys per la resta.

## La tasca

Predir **`iap_revenue_d7`**: els ingressos per compres dins de l'app (*in-app purchases*) que genera cada instal·lació durant els **primers 7 dies**.

## Dades

Dades reals d'instal·lacions, anonimitzades, amb característiques de l'usuari, el dispositiu, l'app i la campanya. El volum era **gran** (els equips feien servir Dask o mostrejaven el 20 %). El detall de columnes era al Kaggle privat de l'esdeveniment.

Característiques del target:
- **La majoria de valors són 0**: la gran majoria d'usuaris no compra res.
- Pocs usuaris "balena" concentren gran part dels ingressos, amb una cua llarga.

## Avaluació

- **Mètrica principal: MSLE** (Mean Squared Log Error), `mean((log(1+y) − log(1+ŷ))²)`.
- També valoraven:
  - la **velocitat d'inferència** (en producció compten els mil·lisegons);
  - l'**elegància i simplicitat** de la solució;
  - la **capacitat d'identificar els usuaris d'alt valor**.

## Format d'entrega

CSV amb l'identificador de la instal·lació i la predicció de `iap_revenue_d7` (format exacte a `sample_submission.csv` del Kaggle).

## Estratègia guanyadora

**Repo públic de referència:** [paulaa2/datathon-fme-2025](https://github.com/paulaa2/datathon-fme-2025):
- **LightGBM sobre `log(1 + revenue)`**;
- Dask per carregar les dades i una mostra del 20 % per iterar;
- imputació per mediana o moda i ordinal encoding;
- **Optuna** (50 proves, CV de 3 folds) i ensemble dels 5 millors models.

**El nostre pla:**
1. **MSLE equival a MSE sobre `log1p(y)`**: entrenar amb `models.LogTargetRegressor(models.lgbm_baseline())`. És el canvi que més puntua.
2. **Inflació de zeros**: provar `models.TwoStageRegressor` (P(compra) × E[ingrés | compra]) i `objective="tweedie"`, i comparar-los amb CV.
3. **Dades grans**: `pyarrow` o `polars`, `data.reduce_memory`, iterar amb una mostra del 10–20 % i entrenar el model final amb tot.
4. **Velocitat com a argument**: mesurar el temps d'inferència per fila i mostrar la corba "nombre d'arbres vs MSLE". Un model 5× més ràpid amb un 1 % més d'error pot guanyar.
5. **Gràfic de negoci**: % dels ingressos reals capturats pel top-k % de prediccions (corba de *lift*). Respon directament a "identificar usuaris d'alt valor".

## Com practicar-lo

- [Google Analytics Customer Revenue Prediction](https://www.kaggle.com/competitions/ga-customer-revenue-prediction) (Kaggle): és **el mateix problema** (ingressos per usuari, la majoria zeros, RMSE sobre el log).
- Comparar `LogTargetRegressor` i `TwoStageRegressor` amb la mateixa CV.

## Fonts
- [data-students/datathon2025-challenges/Smadex](https://github.com/data-students/datathon2025-challenges/tree/main/Smadex) (enllaç al Kaggle privat)
- [paulaa2/datathon-fme-2025](https://github.com/paulaa2/datathon-fme-2025)
