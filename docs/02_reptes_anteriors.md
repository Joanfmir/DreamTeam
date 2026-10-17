# 02 · Reptes anteriors (2021–2025)

Recull de **tots els reptes publicats** i de les solucions destacades que hem trobat. Fonts: repos de l'AED ([data-students](https://github.com/data-students)) i repos públics d'equips participants.

## Resum

| Any | Empresa | Tipus de problema | Avaluació | Practicable? |
|---|---|---|---|---|
| 2025 | **Mango** | Previsió de demanda / producció per producte nou (tabular + embeddings d'imatge) | 80 % Kaggle (pèrdua asimètrica) + 20 % pitch | Kaggle (dades privades) |
| 2025 | **Smadex** | Regressió dels ingressos IAP a 7 dies per instal·lació (ad-tech) | MSLE + velocitat + simplicitat | Kaggle (dades privades) |
| 2025 | **Schneider Electric** | Classificació d'oportunitats guanyades o perdudes + **explicabilitat** | Rúbrica (25/30/30/15 + 5) | ✅ `dataset.csv` públic |
| 2024 | **Mango** | Predir **atributs de disseny** de roba a partir d'imatges (multi-output) | Kaggle | Parcial |
| 2024 | **Restb.ai** | Predicció del preu d'habitatges (>100k cases, Chicago) | Model + app | — |
| 2024 | **AED** | Formació d'equips: matching de participants (dades JSON) | Obert + visualització | ✅ repo `AEDChallenge` |
| 2023 | **Mango** | Compatibilitat d'outfits ("fill in the blank"), recomanació visual | Pitch + repo (codi, docs, git) | ✅ zip públic |
| 2023 | **NTT Data** | Consum de material (Excel) | Obert | ✅ |
| 2023 | **UPC Sostenible** | Mobilitat dels estudiants: factors, visualització, propostes | Obert | ✅ |
| 2022 | **Accenture** | Supply chain: predir comandes que arriben tard | AUC a Kaggle + cas de negoci | ✅ CSV públics |
| 2022 | **Qualcomm** | Optimització: agrupar i encadenar *power switches* d'un xip (tipus TSP) | Mètrica de solució | ✅ testcases |
| 2022 | **AED (Ageing)** | Impacte social: envelliment, dades obertes | Obert | ✅ |
| 2021 | Housing | Assequibilitat de l'habitatge a BCN, forecasting | Obert | ✅ Open Data BCN |
| 2021 | Mobility | Mobilitat i contaminació (ZBE, carrils bici) | Obert | ✅ Open Data BCN |
| 2021 | Reviews | NLP sobre ressenyes de restaurants de BCN | Obert / model | ✅ JSON públic |

**Patró clar:** cada any hi ha **1 repte tabular amb leaderboard**, gairebé sempre de **Mango** (retail/moda), **1 repte d'empresa amb component de negoci o explicabilitat** i **1 repte més obert**. Mango ha estat present els anys 2023, 2024 i 2025, i és als patrocinadors del 2026: **cal preparar-se per a un repte de moda**.

---

## 2025

### Mango: previsió de demanda per a productes nous
- **Tasca:** decidir la **quantitat a produir** per a cada producte de la temporada nova. Target: `demand`, la demanda potencial "amb estoc infinit", calculada per un algorisme intern de Mango.
- **Dades:** 4 temporades d'històric (2 anys: Primavera-Estiu i Tardor-Hivern) i una de validació. Variables escalades a [0,1]. Inclou:
  - **embeddings d'imatge** de la peça (útils per enllaçar amb productes semblants de temporades passades);
  - família, atributs (màniga, coll, silueta…), color RGB, preu;
  - `phase_in` / `phase_out` / `life_cycle_length`, `num_stores`, `num_sizes`, `has_plus_size`;
  - vendes setmanals, producció i demanda setmanal.
- **Trampa:** les vendes estan **censurades per l'estoc** (no pots vendre més del que produeixes). Per això el target és la demanda i no les vendes.
- **Mètrica:** puntuació 0–100 que **penalitza més les vendes perdudes (infraprediccions) que l'excés d'estoc**. Avaluació: 80 % Kaggle i 20 % presentació.
- **Solucions publicades:**
  - [manelcomiche/datathon-fme](https://github.com/manelcomiche/datathon-fme): **stacking LightGBM + CatBoost** amb un meta-model LightGBM, i després un super-ensemble (v3 0,55 + stacking 0,35 + Ridge sobre PCA d'embeddings 0,10) amb un **factor global ×1,08** per calibrar cap amunt, perquè la mètrica penalitza quedar-se curt.
  - Altres: [TheSpaceHub](https://github.com/TheSpaceHub/DatathonFME2025Mango), [frozono9](https://github.com/frozono9/Mango-DatathonFME25), [SDRC137](https://github.com/SDRC137/Datathon-FME2025-Mango).
- **Lliçons:** agregar les dades setmanals a nivell de producte; afegir features de similitud per embeddings (demanda de les k peces més semblants de temporades anteriors); entrenar amb una pèrdua asimètrica o **multiplicar la predicció per un factor >1** calibrat amb validació.

### Smadex: ingressos per instal·lació
- **Tasca:** predir `iap_revenue_d7`, els ingressos per compres in-app als 7 dies d'una instal·lació, amb dades d'ad-tech anonimitzades. Són **dades grans**.
- **Mètrica:** **MSLE**. També valoren la velocitat d'inferència (en producció tenen mil·lisegons per a cada puja), la simplicitat i la detecció d'usuaris d'alt valor.
- **Solució publicada:** [paulaa2/datathon-fme-2025](https://github.com/paulaa2/datathon-fme-2025): **LightGBM** sobre `log1p(revenue)`, Dask per carregar les dades, mostra del 20 % per iterar, Optuna (50 proves, CV de 3 folds).
- **Lliçons:** amb MSLE cal **entrenar sobre `log1p(y)`** amb MSE i desfer amb `expm1`. El target és molt esbiaixat (la majoria són zeros), així que es pot provar un **model en dues etapes**: P(compra) × E[ingrés | compra].

### Schneider Electric: explicabilitat
- **Tasca:** classificació binària d'oportunitats de venda (guanyada o perduda) i, sobretot, **explicar-la**. Unes 15 variables, i el dataset ja està net.
- **Mètrica:** F1 mínim de 0,7. Després, rúbrica: rendiment 25 %, tècniques 30 %, insights per a no tècnics 30 %, creativitat 15 %, i un extra del 5 %.
- **Tècniques que suggereixen:** SHAP, LIME, PDP, ALE, importància de features. **Idea seva:** fer servir un LLM per traduir els gràfics de SHAP a text.
- **⚠️ Trampa que hem trobat:** el 77 % de les files tenen features duplicades. Amb un K-Fold aleatori, la CV surt inflada (AUC 0,85, o 0,91 amb un model gran); amb validació agrupada queda en ~0,71. El F1 mínim de 0,7 és més difícil del que sembla.
- **Practica-ho:** [`practica/2025_schneider_explainability/`](../practica/2025_schneider_explainability/).

## 2024

### Mango: atributs de disseny a partir d'imatges
- **Tasca:** predir atributs de roba (`length_type`, `neck_lapel_type`, `heel_shape_type`, `closure_placement`, `knit_structure`…) a partir de la imatge i les metadades. És un problema de **classificació multi-sortida**.
- **Solució del 2n lloc al Kaggle:** [bielnebot/datathon_fme_2024](https://github.com/bielnebot/datathon_fme_2024): un model per atribut (`model_<atribut>.pkl`), una EDA a part i una **GUI de Streamlit** per fer inferència.
- **Lliçons:** features d'un **CNN o CLIP preentrenat** (congelat) i un classificador lleuger per atribut. El valor "INVALID" o "no aplica" és una classe més. Una demo visual suma molt.

### Restb.ai: preu d'habitatges
- Més de 100.000 cases de Chicago i rodalies. Regressió del preu.
- Solució publicada: [victorbrull/DatathonFME2024](https://github.com/victorbrull/DatathonFME2024), amb pipeline, **MLflow** per fer el seguiment dels experiments i app de **Streamlit**.

### AED: formació d'equips
- Dades JSON de participants (habilitats, objectius, rols, disponibilitat, text lliure). Cal crear equips equilibrats i complementaris, amb visualitzacions.
- Enfocaments: embeddings del text lliure, clustering i optimització amb restriccions (amics junts, mida ≤ 4).
- Repo: [data-students/AEDChallenge](https://github.com/data-students/AEDChallenge).

## 2023

### Mango: compatibilitat d'outfits
- Dades: `product_data.csv` (atributs i imatge de cada producte) i `outfit_data.csv` (quins productes formen cada outfit).
- Tasca "fill in the blank": donat un outfit incomplet, triar la peça que hi encaixa. Al final, **generar outfits complets** a partir d'una peça.
- Avaluació **sense mètrica**: pitch de 10 minuts amb un mínim de 5 outfits, i el repo (qualitat del codi, reproductibilitat, docs, git i branques).
- Enfocaments: embeddings d'imatge (CLIP o ResNet), aprenentatge de compatibilitat (siamese o triplet loss) i regles per categoria (top + bottom + sabates).

### UPC Sostenible: mobilitat
- Quins factors fan que els estudiants triïn transport públic o privat, una visualització dels fluxos i propostes per reduir CO₂. Dades GTFS i Open Data.
- **1r premi:** [JordinaGR/datathon-FME-2023](https://github.com/JordinaGR/datathon-FME-2023) ("Optimització de Rodalies"): **web scraping de Google Maps** (Selenium) per obtenir temps de trajecte, i després pandas i matplotlib. **Un projecte simple però amb dades pròpies i una història clara.**

## 2022

### Accenture: resiliència de la cadena de subministrament
- Dades: `orders.csv` (port, operador logístic, hub, client, producte, unitats → `late_order`), `product_attributes.csv` i `cities_data.csv`.
- Mètrica: **AUC** a Kaggle i un cas de negoci amb els drivers dels retards.
- **Guanyador:** [JNADatathon2022/Datathon](https://github.com/JNADatathon2022/Datathon) ("Metis Business Intelligence"): model amb Keras/TF **i, a més, una API d'inferència** ([ModelInferenceApp](https://github.com/JNADatathon2022/ModelInferenceApp)).
- Altres: [KensingtonOscupant](https://github.com/KensingtonOscupant/datathon-fme2022) (ensemble).
- **Practica-ho:** [`practica/2022_accenture_supply_chain/`](../practica/2022_accenture_supply_chain/).

### Qualcomm: optimització combinatòria
- Agrupar els *power switches* d'un xip en cadenes independents per minimitzar el temps d'encesa. És un **TSP o VRP amb múltiples viatjants**.
- Enfocament publicat ([marc-herrero](https://github.com/marc-herrero/Qualcomm-Challenge-)): **clustering per àrea, TSP per clúster, i proves amb algorisme genètic i simulated annealing**.
- **Lliçó:** no tot és ML. Cal tenir a mà OR-Tools, heurístiques (nearest neighbour + 2-opt) i simulated annealing.

## 2021

- **Reviews** (ressenyes de Google de 150 restaurants de BCN): **guanyador** [megaelius/Delfos-datathon-fme](https://github.com/megaelius/Delfos-datathon-fme), un **sistema de recomanació de restaurants per a grups**. Van convertir un dataset de ressenyes en un producte.
- **Housing** i **Mobility**: anàlisi i forecasting amb Open Data BCN.

---

## Què fan els guanyadors (resum)

1. **Converteixen el model en un producte**: Streamlit, API, recomanador. Ho fan el guanyador d'Accenture 2022, el de Reviews 2021 i el 2n de Mango 2024.
2. **Dades extra o originals** quan el repte és obert (scraping de Google Maps el 2023).
3. **Gradient boosting** (LightGBM o CatBoost) a tots els reptes tabulars, amb ensembles al final.
4. **Ajusten a la mètrica**: `log1p` per a MSLE, calibratge cap amunt per a les pèrdues asimètriques.
5. **Un README i una presentació cuidats.** Diversos jurats avaluen el repo.
