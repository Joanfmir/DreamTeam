# 04 · Playbook de ML segons el tipus de problema

Receptes ràpides per als tipus de repte que han sortit a la Datathon FME. Per a cadascun hi ha: **baseline**, **model fort**, **trucs** i **errors**.

---

## 1. Tabular: classificació o regressió
*(Accenture 2022, Schneider 2025, Restb.ai 2024, Smadex 2025)*

- **Baseline (15 min):** `dreamteam.models.lgbm_baseline` amb features crues i les categòriques com a `category`.
- **Model fort:** LightGBM i CatBoost. CatBoost sol guanyar si hi ha moltes categòriques d'alta cardinalitat. Al final, un **ensemble** de 2–3 models (mitjana o rank average).
- **Feature engineering que sol funcionar:**
  - Comptatges i freqüències de cada categòrica (`count encoding`).
  - **Target encoding fet dins la CV** (out-of-fold). Fet sobre tot el train és leakage.
  - Agregats per grup: mitjana, desviació i màxim del target o de numèriques per client, producte o ciutat.
  - Interaccions de negoci: preu/unitat, distància/temps, ràtios.
  - Dates: dia de la setmana, mes, setmana ISO, festius, temps des de l'últim esdeveniment.
- **Tuning:** Optuna amb 30–50 proves, **només quan les features estiguin congelades**.
- **Errors:** files duplicades repartides entre train i validació (CV inflada), IDs a les features, desbalanceig sense ajustar el llindar (per optimitzar F1 cal **triar el llindar sobre les prediccions OOF**).

## 2. Previsió de demanda o forecasting
*(Mango 2025, Housing 2021)*

- **Reformular com a tabular**: una fila per producte i temporada, amb el target agregat.
- **Validació temporal**: entrenar amb les temporades 1–3 i validar amb la 4. Mai un K-Fold aleatori.
- **Productes nous (cold start):** buscar els **k veïns més propers per embedding d'imatge o atributs** a les temporades passades i fer servir la seva demanda com a feature.
- **Mètriques asimètriques:** si quedar-se curt costa més, entrenar amb una pèrdua *quantile* (`objective="quantile"`, `alpha=0,6–0,8`) o **multiplicar per un factor** triat amb validació (`dreamteam.metrics.best_scale_factor`).
- **Censura:** les vendes estan limitades per l'estoc. Cal fer servir la demanda si la donen, o marcar els casos de *stockout*.

## 3. Target molt esbiaixat (revenue, preus)
*(Smadex 2025, Restb.ai 2024)*

- Si la mètrica és **MSLE o RMSLE**, entrenar sobre `log1p(y)` amb MSE i desfer amb `expm1` (`dreamteam.models.LogTargetRegressor`).
- **Molts zeros:** model en dues etapes, classificador P(y>0) × regressor E[y | y>0]. O bé un objectiu `tweedie` a LightGBM.
- **Dades grans:** fer servir `pyarrow` o `polars`, reduir el tipus de dades (`dreamteam.data.reduce_memory`) i iterar amb una mostra del 10–20 %.

## 4. Explicabilitat
*(Schneider 2025)*

- **Global:** SHAP *beeswarm* (importància i direcció), importància per permutació i PDP o ALE de les 3–5 variables principals.
- **Local:** SHAP *waterfall* d'un cas guanyat, d'un cas perdut i d'un cas "frontera" (prob ≈ 0,5).
- **Per a no tècnics:** traduir cada gràfic a una frase del tipus *"Si el client ja ens havia comprat el producte A, la probabilitat de guanyar puja de ~X punts"*. Es pot generar amb un LLM a partir dels valors SHAP (idea que suggeria Schneider).
- **Creativitat:** contrafactuals ("què hauria de canviar perquè aquesta oportunitat es guanyi?") i una app on el comercial introdueix una oportunitat i veu el perquè.
- Codi: `dreamteam.explain.global_summary`, `dreamteam.explain.local_waterfall`.

## 5. Imatges i embeddings
*(Mango 2023 i 2024; embeddings a Mango 2025)*

- **No entrenar CNNs des de zero.** Extreure features amb un model preentrenat congelat (**CLIP** `openai/clip-vit-base-patch32`, ResNet50, EfficientNet) i entrenar un classificador lleuger (LogReg o LightGBM) a sobre.
- Multi-atribut: un cap per atribut o un model per atribut. "No aplica" és una classe més.
- **Similitud i recomanació:** similitud cosinus entre embeddings, i FAISS si n'hi ha molts.
- Compatibilitat (outfits): els bons outfits combinen peces **complementàries**, no semblants. Cal filtrar per categoria (top + bottom + sabates + accessori) i puntuar parelles amb un model entrenat sobre els outfits reals.
- Si donen embeddings, **reduir amb PCA (16–64 components)** abans de passar-los a un model d'arbres.

## 6. NLP (text lliure, ressenyes)
*(Reviews 2021, AED 2024)*

- Baseline: TF-IDF + Logistic Regression.
- Fort: `sentence-transformers` (`all-MiniLM-L6-v2`, multilingüe: `paraphrase-multilingual-MiniLM-L12-v2`), embeddings i un model a sobre, o bé clustering.
- Aspectes i temes: BERTopic, o un LLM per etiquetar una mostra.

## 7. Optimització combinatòria
*(Qualcomm 2022, formació d'equips de l'AED 2024)*

- Formular-ho primer: **funció objectiu i restriccions**.
- Eines: **OR-Tools** (routing, TSP i VRP, CP-SAT per a assignacions), heurístiques (nearest neighbour + 2-opt), **simulated annealing**.
- Problemes grans: clustering per descompondre i resoldre cada clúster per separat.

## 8. Reptes oberts (visualització o impacte)
*(UPC Sostenible 2023, Ageing 2022, Mobility 2021)*

- El 1r premi del 2023 va **generar dades pròpies** fent scraping de Google Maps. Cal pensar quina dada falta per respondre la pregunta.
- **Una pregunta clara**, 3 insights i una proposta accionable.
- Mapes: `folium` o `plotly` (`scatter_mapbox`), GeoPandas i GTFS per al transport.
- El lliurable és una **demo interactiva** (Streamlit) i una història.

---

## Checklist de validació (abans de confiar en un número)

- [ ] El split de validació imita el split train/test real (temps, grup o aleatori).
- [ ] He mirat si hi ha **files duplicades** (`validation.duplicate_report`). A Schneider 2025 n'hi havia un 77 %, i la CV aleatòria passava de 0,71 a 0,85.
- [ ] Cap feature fa servir el target o informació del futur.
- [ ] La CV i el leaderboard es mouen en la mateixa direcció.
- [ ] La mètrica que optimitzo és la de l'enunciat, implementada a `metrics.py`.
- [ ] La submission passa `dreamteam.submission.make_submission` (columnes, IDs i NaN).
