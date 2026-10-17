# 05 · Mètriques que es repeteixen

Quina mètrica ha sortit, a quins reptes, què vol dir, **com s'optimitza** i on és el codi.

> **Regla 0:** abans d'escriure cap model, implementeu la mètrica de l'enunciat a `src/dreamteam/metrics.py` i comproveu-la contra el `sample_submission` o els exemples. Optimitzar la mètrica equivocada és l'error més car.

## Resum

| Mètrica | Reptes | Tipus | Truc clau | Codi |
|---|---|---|---|---|
| **ROC AUC** | Accenture 2022 | Classificació (probabilitats) | Només importa l'**ordre**: no cal calibrar. Fer ensemble amb mitjana de **rangs** | `metrics.auc` |
| **F1** | Schneider 2025 (mínim 0,7) | Classificació (etiquetes) | **Triar el llindar** sobre les prediccions OOF (rarament és 0,5) | `metrics.best_f1_threshold` |
| **Accuracy** | Mango 2024, Reviews 2021 | Classificació multiclasse | Comparar sempre amb la **classe majoritària**. Amb desbalanceig, enganya | `sklearn.metrics.accuracy_score` |
| **MSLE / RMSLE** | Smadex 2025, (Restb.ai) | Regressió esbiaixada | **Entrenar sobre `log1p(y)`** amb MSE i desfer amb `expm1` | `metrics.msle`, `models.LogTargetRegressor` |
| **Asimètrica** (penalitza quedar-se curt) | Mango 2025 | Regressió de quantitats | Predir **per sobre** de la mitjana: quantil > 0,5 o factor > 1 calibrat | `metrics.asymmetric_error`, `metrics.best_scale_factor` |
| **MAE / RMSE / MAPE** | Restb.ai 2024, Reviews 2021 (MAE d'estrelles) | Regressió | MAE → mediana (L1); RMSE → mitjana (L2); MAPE → errors relatius (log) | `metrics.rmse` |
| **Longitud de ruta** + temps | Qualcomm 2022 | Optimització | Reformular l'objectiu (mitjana = total / k) i heurístiques que escalin | `optim.route_length` |
| **Rúbrica** (presentació, insights, creativitat) | Gairebé tots: Mango 20 %, Schneider 75 %, Mango 2023 100 % | Qualitativa | Història, demo i repo net. Vegeu la plantilla de presentació | [`07_plantilla_presentacio.md`](07_plantilla_presentacio.md) |

**Patró:** **cada any hi ha almenys un repte amb leaderboard** (AUC, accuracy, MSLE o una de pròpia) i **tots tenen una part avaluada per persones**. La mètrica decideix si entres al grup de dalt; la presentació decideix qui guanya.

---

## ROC AUC
**Què mesura:** la probabilitat que el model posi un positiu aleatori per sobre d'un negatiu aleatori. 0,5 = atzar, 1 = perfecte.

**Implicacions:**
- No depèn del llindar ni de la calibració. **Només l'ordre** de les prediccions.
- Ensemble: `rankdata(p1) + rankdata(p2)` en lloc de la mitjana de probabilitats (models amb escales diferents).
- Desbalanceig: l'AUC no s'hi immuta gaire. Si importa la classe rara, mireu també la PR-AUC.

## F1
`F1 = 2 · precision · recall / (precision + recall)`

**Implicacions:**
- Depèn del **llindar**. El llindar òptim es tria sobre les prediccions **OOF**, mai sobre el test: `metrics.best_f1_threshold(y, oof)`.
- Amb classes balancejades (Schneider: 47 % positius) el llindar òptim sovint és **< 0,5**, perquè el F1 premia el recall.
- Multiclasse: cal saber si és *macro* (cada classe pesa igual) o *micro* o *weighted*.

## Accuracy
**Implicacions:**
- **Baseline obligatori:** la classe majoritària. Reviews 2021: "sempre 5★" ja treu 0,69. Mango 2024: la majoritària fa 0,64 de mitjana.
- Multi-sortida (Mango 2024, 11 atributs): l'accuracy global és la mitjana. **Els atributs fàcils (calçat, ~0,99) amaguen els difícils (silueta, 0,53)**: millorar els difícils és on hi ha punts.
- Classes "no aplica" (`INVALID`): són la classe més fàcil d'encertar amb regles.

## MSLE / RMSLE
`MSLE = mean( (log(1 + y) − log(1 + ŷ))² )`

**Implicacions:**
- Mesura **errors relatius**: equivocar-se 10 € en un usuari d'1 € pesa molt; en un de 1.000 €, gairebé res.
- **El truc:** MSLE sobre `y` = MSE sobre `log1p(y)`. Entrenar amb `LogTargetRegressor` és **la decisió que més puntua** en aquests reptes.
- Les prediccions negatives no tenen sentit: `clip(0)`.
- Molts zeros (Smadex): model en dues etapes o `tweedie`.

## Pèrdua asimètrica (Mango 2025)
Quedar-se curt (vendes perdudes) costa més que passar-se (estoc sobrant, que es pot liquidar).

**Implicacions:**
- La predicció òptima **no és la mitjana**, sinó un **quantil alt**. Si quedar-se curt costa *a* i passar-se *b*, el quantil òptim és `a / (a + b)` (p. ex. 2:1 → quantil 0,67).
- Dues maneres de fer-ho:
  1. `models.lgbm_baseline("quantile", alpha=0.67)`;
  2. un model normal i **multiplicar per un factor k > 1** calibrat sobre la validació (`metrics.best_scale_factor`). Un equip de 2025 feia servir ×1,08; al nostre pipeline sintètic surt ×1,14.
- Quan es publiqui la fórmula exacta, cal implementar-la i calibrar contra ella.

## MAE vs. RMSE vs. MAPE
| | Penalitza | Predicció òptima | Quan surt |
|---|---|---|---|
| **MAE** | errors per igual | **mediana** (`objective="l1"`) | outliers que no volem perseguir |
| **RMSE** | errors grans molt més | **mitjana** (`objective="regression"`) | quan un error gran és molt dolent |
| **MAPE** | error relatiu | ≈ treballar en log | preus (Restb.ai): l'error del 10 % importa igual a totes les escales |

## Optimització (Qualcomm 2022)
- Llegir bé la funció objectiu: "longitud **mitjana**" = total / nombre de cadenes → **fer servir el màxim de cadenes** i minimitzar el total.
- També puntuaven el **temps** d'execució i la **visualització**. Una heurística O(n log n) que escala guanya a una d'exacta que no acaba.

## Rúbriques (la part humana)
Pesos que hem vist:
- Mango 2025: 20 % presentació.
- Schneider 2025: 75 % entre tècniques, insights i creativitat, més un 5 % extra.
- Mango 2023: 100 % pitch i repo.
- Accenture 2022: cas de negoci.

Què s'hi repeteix:
1. **Insights entenedors per a no tècnics.**
2. **Creativitat / producte** (demo, API, app).
3. **Repo**: reproductibilitat, README i git.
4. **"Què faríeu amb més temps"** (Schneider ho puntuava a part).
