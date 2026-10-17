# Pràctica · Smadex 2025: ingressos per instal·lació

**Repte 2025:** predir `iap_revenue_d7`, els ingressos per compres in-app als 7 dies d'una instal·lació, amb dades d'ad-tech anonimitzades i **grans**.
**Mètrica:** MSLE. També valoren la **velocitat d'inferència** (en producció hi ha mil·lisegons per a cada puja), la simplicitat i la detecció d'usuaris d'alt valor.
Smadex torna a ser patrocinador el 2026.

## Claus

1. **MSLE → entrenar sobre `log1p(y)`** amb MSE: `models.LogTargetRegressor(models.lgbm_baseline())`.
2. **Gairebé tots els valors són 0** (la majoria no compra res). Es pot provar `models.TwoStageRegressor` (P(compra) × ingrés) o `objective="tweedie"`.
3. **Dades grans:** `pyarrow` o `polars`, `data.reduce_memory`, iterar amb una mostra del 10–20 % i entrenar el final amb tot.
4. **Velocitat:** un LightGBM amb pocs arbres i poca profunditat és un argument a la presentació. Cal mesurar el temps d'inferència per fila.

Solució publicada: [paulaa2/datathon-fme-2025](https://github.com/paulaa2/datathon-fme-2025) (LightGBM sobre log1p, Dask, Optuna amb 50 proves).

## Tasques de la setmana 3 (dades públiques semblants)

- [ ] [Google Analytics Customer Revenue Prediction](https://www.kaggle.com/competitions/ga-customer-revenue-prediction): és **el mateix problema** (ingressos per usuari, la majoria zeros, mètrica RMSE sobre log). Muntar el pipeline amb `LogTargetRegressor` i `TwoStageRegressor` i comparar-los.
- [ ] Mesurar el temps d'inferència de 10.000 files i provar quants arbres es poden treure sense perdre MSLE.
- [ ] Fer un gràfic de "captura dels usuaris d'alt valor": % d'ingressos reals dins del top-k % de prediccions.
