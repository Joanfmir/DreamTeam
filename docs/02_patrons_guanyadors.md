# 02 · Patrons dels reptes i estratègia guanyadora

Anàlisi transversal dels **15 reptes de 2021–2025**. El detall de cada repte és a [`reptes/`](../reptes/README.md).

## 1. Què es repeteix

### Empreses que repeteixen
| Empresa | 2021 | 2022 | 2023 | 2024 | 2025 | **2026 (patrocinador)** |
|---|---|---|---|---|---|---|
| **Mango** | | | outfits | atributs | demanda | ✅ |
| Accenture | (organitza) | supply chain | | | | |
| **Qualcomm** | | power switches | | | | ✅ |
| **Restb.ai** | | | | habitatge | | ✅ |
| **Smadex** | | | | | ingressos | ✅ |
| **Schneider** | | | | | explicabilitat | ✅ |
| AED / organització | reviews, housing, mobility | ageing | | equips | | |
| NTT DATA, UPC Sostenible | | | ✅ | | | |
| **CaixaBank, Novartis, T2C** | | | | | | ✅ **(nous)** |

Mango ha posat repte **3 anys seguits**, sempre amb dades de producte (atributs, imatges i embeddings) i cada any amb una tasca diferent: compatibilitat → atributs → demanda.

### Tipus de problema
| Tipus | Vegades | Reptes |
|---|---|---|
| **Tabular supervisat** (classificació o regressió) | 5 | Accenture, Restb.ai, Schneider, Smadex, Mango 2025 |
| **Moda / visió / embeddings** | 3 | Mango 2023, 2024, 2025 |
| **Obert de ciutat o impacte social** | 5 | Housing, Mobility, Ageing, UPC Mobilitat, NTT |
| **NLP** | 2 | Reviews, AED 2024 (text lliure) |
| **Optimització** | 2 | Qualcomm, AED 2024 (formació d'equips) |
| **Forecasting** | 3 | Mango 2025, NTT, Housing |

### Forma de l'avaluació
- **Leaderboard (Kaggle)**: Accenture 2022, Mango 2024 i 2025, Smadex 2025. **Gairebé sempre hi ha un repte amb Kaggle.**
- **Rúbrica humana**: tots. Fins i tot els de Kaggle tenen presentació (Mango 2025: 20 %).
- **El repo també compta**: Mango 2023 avaluava explícitament el codi, la reproductibilitat, la documentació i l'ús de git i branques.

## 2. Què fan els guanyadors

Sis repos de guanyadors o de primers classificats analitzats:

| Guanyador | Què va fer | Patró |
|---|---|---|
| Accenture 2022 🥇 | Model i **API d'inferència** | Producte |
| Reviews 2021 🥇 | **Recomanador de restaurants per a grups** a partir de les ressenyes | Producte, reformular el problema |
| UPC Mobilitat 2023 🥇 | **Scraping de Google Maps** per tenir temps de trajecte reals | Dades pròpies |
| Mango 2024 (2n Kaggle) | Un model per atribut **només amb metadades** i **GUI de Streamlit** | Simple + demo |
| Mango 2025 (repo públic) | Stacking LGBM + CatBoost, ensemble i **factor ×1,08** per la mètrica asimètrica | Ajustar-se a la mètrica |
| Smadex 2025 (repo públic) | LightGBM sobre **log1p**, Optuna | Ajustar-se a la mètrica |

### Les 7 regles que se'n desprenen

1. **Model simple ben validat > model complex.** El 2n de Mango 2024 va fer servir Random Forests sobre 6 columnes. El 1r d'UPC 2023, pandas i matplotlib.
2. **Ajustar-se a la mètrica real**: `log1p` per a MSLE, quantil o factor per a pèrdues asimètriques, llindar per a F1. Són els canvis més barats i que més puntuen.
3. **Convertir el model en un producte** (Streamlit, API, recomanador). Tres dels sis l'han fet.
4. **Reformular la pregunta** si el repte és obert: "predir estrelles" → "on anem a sopar en grup".
5. **Dades que ningú més té** (scraping, APIs, dades obertes) als reptes oberts.
6. **Validació honesta**: temporal si hi ha temps, agrupada si hi ha duplicats. A Schneider 2025, la CV aleatòria inflava l'AUC de 0,71 a 0,85.
7. **Presentació per a no tècnics**, amb un número de negoci (euros, minuts, CO₂).

## 3. Predicció per al 2026 i preparació

Segons els patrocinadors (CaixaBank, Smadex, Qualcomm, Restb.ai, DoraHacks, Mango, Schneider, Novartis, T2C), els escenaris més probables són:

| Escenari | Probabilitat | Per què | Estem preparats amb |
|---|---|---|---|
| **Mango: dades de producte** (demanda, atributs, recomanació) | Molt alta | 3 anys seguits | `reptes/2025_mango_demanda/pipeline.py`, `2024_mango_atributs/`, `features.knn_target_features`, `embeddings` |
| **Smadex: ad-tech** (ingressos, CTR, conversió) | Alta | Repeteix com a patrocinador | `LogTargetRegressor`, `TwoStageRegressor`, `metrics.msle` |
| **Schneider: classificació + XAI** | Alta | Repeteix | `explain`, app de Streamlit, detecció de duplicats |
| **CaixaBank: banca** (frau, risc de crèdit, churn, segmentació de clients) | Mitjana (nou) | Patrocinador nou | Tabular amb molt desbalanceig → AUC/PR-AUC, `best_f1_threshold`, SHAP (en banca l'explicabilitat és obligatòria) |
| **Novartis: salut / farma** (assajos clínics, forecasting de demanda de fàrmacs, NLP mèdic) | Mitjana (nou) | Patrocinador nou | Forecasting (NTT 2023), NLP (Reviews 2021) |
| **Restb.ai: immobiliari amb imatges** | Mitjana | Repeteix | `LogTargetRegressor`, kNN geogràfic, `embeddings.image_embeddings` |
| **Qualcomm: optimització** | Mitjana | Repeteix | `optim`, `reptes/2022_qualcomm_power_switches/solver.py` |

> **Conclusió per a la tria del dia D:** un repte amb leaderboard que encaixi amb el nostre codi (Mango, Smadex o Schneider) ens dona avantatge perquè **arrenquem amb el pipeline ja fet**. Els de Mango són els més concorreguts. Si n'hi ha un de nou (CaixaBank o Novartis) amb menys equips, pot ser una oportunitat.
