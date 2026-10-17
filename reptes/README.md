# Reptes de la Datathon FME (2021–2025)

Tots els reptes publicats, cadascun amb el seu **`ENUNCIAT.md`**: context, tasca, diccionari de dades, avaluació, format d'entrega, **estratègia guanyadora** i com practicar-lo. On hi ha dades públiques, hi ha també **codi que funciona** i els resultats que n'hem tret.

```bash
bash scripts/download_past_data.sh          # totes les dades públiques (~125 MB) a data/past/
```

## Catàleg

| Any | Repte | Tipus | Mètrica | Dades | Codi nostre | Resultat |
|---|---|---|---|---|---|---|
| 2025 | [**Mango** · demanda](2025_mango_demanda/ENUNCIAT.md) | Forecasting + embeddings | Asimètrica 0–100 (Kaggle 80 %) | Sintètiques | [`pipeline.py`](2025_mango_demanda/pipeline.py) | error 0,524 → **0,220** |
| 2025 | [**Smadex** · ingressos](2025_smadex_ingressos/ENUNCIAT.md) | Regressió amb zeros | **MSLE** + velocitat | Privades | — | — |
| 2025 | [**Schneider** · explicabilitat](2025_schneider_explicabilitat/ENUNCIAT.md) | Classificació + XAI | Rúbrica (F1 ≥ 0,7) | ✅ | [`baseline.py`](2025_schneider_explicabilitat/baseline.py) | AUC **0,71** (honesta) |
| 2024 | [**Mango** · atributs](2024_mango_atributs/ENUNCIAT.md) | Multi-sortida, visió | **Accuracy** | ✅ (sense imatges) | [`baseline.py`](2024_mango_atributs/baseline.py) | **0,801** (majoritària 0,640) |
| 2024 | [**Restb.ai** · habitatge](2024_restbai_habitatge/ENUNCIAT.md) | Regressió de preus | MAE / RMSE | Privades | — | — |
| 2024 | [**AED** · equips](2024_aed_equips/ENUNCIAT.md) | Matching + NLP + optimització | Oberta | ✅ | — | — |
| 2023 | [**Mango** · outfits](2023_mango_outfits/ENUNCIAT.md) | Recomanació visual | Pitch + **repo** | ❌ (404) | — | — |
| 2023 | [**NTT DATA** · hospitals](2023_nttdata_hospitals/ENUNCIAT.md) | Forecasting + segmentació | Oberta | ✅ | — | — |
| 2023 | [**UPC Sostenible** · mobilitat](2023_upc_mobilitat/ENUNCIAT.md) 🥇 | Factors + mapes + propostes | Oberta | ✅ | — | — |
| 2022 | [**Accenture** · supply chain](2022_accenture_supply_chain/ENUNCIAT.md) 🥇 | Classificació tabular | **ROC AUC** | ✅ | [`baseline.py`](2022_accenture_supply_chain/baseline.py), [`features_v2.py`](2022_accenture_supply_chain/features_v2.py) | AUC 0,821 → **0,836** |
| 2022 | [**Qualcomm** · power switches](2022_qualcomm_power_switches/ENUNCIAT.md) | Optimització (TSP múltiple) | Longitud mitjana | ✅ | [`solver.py`](2022_qualcomm_power_switches/solver.py) | **8,4×** millor que l'exemple |
| 2022 | [**AED** · envelliment](2022_aed_envelliment/ENUNCIAT.md) | Impacte social | Oberta | ✅ obertes | — | — |
| 2021 | [**Reviews** · NLP](2021_reviews_nlp/ENUNCIAT.md) 🥇 | Text | Oberta (acc / MAE) | ✅ | [`baseline.py`](2021_reviews_nlp/baseline.py) | MAE **0,334** estrelles |
| 2021 | [**Housing + Mobility**](2021_habitatge_mobilitat/ENUNCIAT.md) | Forecasting, impacte | Oberta | ✅ obertes | — | — |

🥇 = el repo del guanyador és públic i està analitzat a l'enunciat.
Anàlisi transversal (què es repeteix i què fan els guanyadors): [`docs/02_patrons_guanyadors.md`](../docs/02_patrons_guanyadors.md) · Mètriques: [`docs/05_metriques.md`](../docs/05_metriques.md).

---

## Pla d'entrenament (17/10 → 13/11)

Quatre setmanes per arribar a la Datathon **sabent treballar junts** i amb les eines provades.

| Setmana | Reptes | Què entrenem | Objectiu concret |
|---|---|---|---|
| **1** · 19–25 oct | Schneider 2025 + Reviews 2021 | Setup, validació, **leakage per duplicats**, SHAP, NLP bàsic | Tothom executa els baselines. F1 ≥ 0,7 **amb CV agrupada** a Schneider. 1 insight per persona |
| **2** · 26 oct–1 nov | Accenture 2022 + Mango 2024 | **Feature engineering**, encodings, multi-sortida | Superar AUC 0,836 i accuracy 0,801. Afegir imatges (CLIP) a Mango 2024 si podem aconseguir-les |
| **3** · 2–8 nov | Mango 2025 (sintètic) + Smadex (proxy de Kaggle) + Qualcomm | **Validació temporal, cold start, pèrdua asimètrica, `log1p`**, optimització | Entendre i modificar `pipeline.py`. Millorar el solver de Qualcomm |
| **4** · 9–13 nov | **Simulacre de 6 h** amb AED 2024 o UPC 2023 | El dia D en petit | Triar repte, baseline en 1 h, demo, **pitch de 5 min cronometrat** |

### Regles de l'entrenament

1. **Una branca per persona i experiment** (`joan/target-encoding`). Es fa merge a `main` quan millora la CV.
2. Cada experiment s'apunta a [`experiments.md`](../experiments.md): qui, idea, CV.
3. El codi que fem servir dues vegades **puja a `src/dreamteam/`**, amb el seu test.
4. Al final de cada setmana, **10 minuts de "pitch"** per persona: què he provat i què he après.

### Checklist abans del 14/11

- [ ] Tots tenim el repo clonat i `make setup` funcionant **al portàtil que portarem**.
- [ ] `make data` i `make baselines` sense errors.
- [ ] Tenim compte de Kaggle i de Devpost.
- [ ] Rols decidits ([`docs/03_estrategia_48h.md`](../docs/03_estrategia_48h.md)).
- [ ] Plantilla de slides preparada ([`docs/07_plantilla_presentacio.md`](../docs/07_plantilla_presentacio.md)).
- [ ] Models preentrenats (CLIP, MiniLM) **baixats a la caché** (la wifi del dia D pot fallar).
- [ ] Carregadors, regleta i auriculars a la motxilla 🔌
