# Pràctica · Mango 2025: previsió de demanda

> Mango ha tingut repte el 2023, el 2024 i el 2025, i és patrocinador el 2026. **És el repte més probable.**

**Repte 2025:** decidir quantes unitats produir de cada producte de la temporada nova. Target: `demand` (demanda potencial amb estoc infinit).
**Avaluació:** 80 % Kaggle i 20 % presentació. La mètrica **penalitza més quedar-se curt** (vendes perdudes) que passar-se (excés d'estoc).
Enunciat: `data/past/2025_mango/enunciat.pdf`. Les dades eren al Kaggle privat de l'esdeveniment i no són públiques.

## Què el fa difícil

1. **Cold start:** els productes del test són **nous**. No tenen històric propi.
2. **Censura:** les vendes estan limitades per la producció, i per això el target és la demanda.
3. **Dades setmanals** (`weekly_sales`, `weekly_demand`) però la predicció és **per producte**: cal agregar.
4. **Embeddings d'imatge** de cada peça: és la clau per trobar productes semblants del passat.

## Pipeline que hem de saber fer de memòria

```python
# 1. Agregar a una fila per producte i temporada
prod = df.groupby(["id", "id_season"]).agg(demand=("weekly_demand", "sum"), ...)

# 2. Validació temporal: l'última temporada de train fa de validació
splits = validation.time_split(prod["id_season"], valid_values=[ultima_temporada])

# 3. Features de veïns per embedding: la demanda dels k productes més semblants del passat
#    (NearestNeighbors(metric="cosine") sobre PCA(64) dels embeddings)

# 4. Model: LightGBM i CatBoost (categòriques natives), o un objectiu quantile amb alpha≈0,7

# 5. Calibrar un factor > 1 per a la pèrdua asimètrica, sobre la validació
k, score = metrics.best_scale_factor(y_valid, pred_valid)
```

Solució publicada de referència: [manelcomiche/datathon-fme](https://github.com/manelcomiche/datathon-fme). Fan un stacking LightGBM + CatBoost, un ensemble amb Ridge sobre PCA d'embeddings i un factor ×1,08.

## Tasques de la setmana 3 (amb dades públiques semblants)

- [ ] **Validació temporal i cold start:** [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) (articles amb atributs i imatges i transaccions per data). Agregar vendes per article i temporada i predir les dels articles **nous** de la darrera temporada.
- [ ] **Embeddings d'imatge:** extreure embeddings CLIP de les imatges d'H&M (o del zip de [Mango 2023](https://mng-datathon-upc.s3.eu-west-1.amazonaws.com/datathon.zip)) i fer la feature de "demanda dels veïns".
- [ ] Implementar a `metrics.py` una versió de la mètrica asimètrica i provar `best_scale_factor`.
- [ ] Preparar un notebook de "dia D" amb el pipeline de dalt, buit i a punt per omplir.
