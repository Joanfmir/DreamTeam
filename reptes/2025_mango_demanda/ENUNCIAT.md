# Mango 2025 · Quant hem de produir?

> **2025** · Mango · Previsió de demanda (tabular + embeddings d'imatge) · **Mètrica asimètrica 0–100** · 80 % Kaggle + 20 % presentació · Dades privades → [simulador](simulate_data.py)

## Context

Mango dissenya cada temporada milers de peces noves. Abans de començar a vendre, l'equip de compres ha de decidir **quantes unitats en fabrica el proveïdor**. Si en fabrica poques, perd vendes; si en fabrica massa, li queda estoc. L'estoc sobrant encara es pot vendre més endavant (rebaixes, outlet), però una venda perduda ja no es recupera.

## La tasca

Per a cada producte de la **temporada nova** (test), predir `demand`: la **demanda potencial total**, és a dir, el que es vendria amb estoc infinit. Aquesta és la quantitat que s'encarregaria al proveïdor.

> ⚠️ Els productes del test són **nous**: no tenen històric propi. Cal aprendre de productes **semblants** de temporades anteriors.

## Dades

Quatre temporades d'històric (2 anys: Primavera-Estiu i Tardor-Hivern) i una de validació. Hi ha una fila per **producte i setmana**. Vendes, demanda i producció estan **escalades a [0, 1]** per anonimitzar-les, però conserven les proporcions.

| Columna | Descripció |
|---|---|
| `id` | Identificador del model |
| `id_season` | Temporada |
| `aggregated_family`, `family`, `category` | Jerarquia de producte (vestits, abrics, texans… / tops, bottoms…) |
| `fabric`, `color_name`, `color_rgb` | Teixit i color |
| `image_embedding` | **Embedding precalculat de la imatge** de la peça. Serveix per trobar productes semblants |
| `length_type`, `silhouette_type`, `waist_type`, `neck_lapel_type`, `sleeve_length_type`, `heel_shape_type`, `toecap_type`, `woven_structure`, `knit_structure`, `print_type` | Atributs de disseny |
| `archetype`, `moment`, `ocassion` | Per a qui i per a quin moment està pensada la peça |
| `phase_in`, `phase_out`, `life_cycle_length` | Dilluns de la primera i l'última setmana de venda, i setmanes entre totes dues |
| `num_stores`, `num_sizes`, `has_plus_size` | Distribució: més botigues i més talles volen dir més vendes potencials |
| `price` | Preu de venda |
| `year`, `num_week_iso` | Setmana |
| `weekly_sales` | Vendes de la setmana. **Censurades per l'estoc**: mai superen la producció |
| `production` | Unitats que va encarregar l'equip de Mango (un indicador de l'"èxit esperat") |
| `weekly_demand` | **Target setmanal**: vendes potencials calculades per un algorisme intern de Mango |

Fitxers: `train.csv`, `test.csv` (sense vendes, demanda ni producció) i `sample_submission.csv`.

## Avaluació

- **Kaggle (80 %)**: una puntuació de 0 a 100 que mesura com de prop és la predicció de la demanda real de la temporada de validació. **Penalitza més les vendes perdudes (infraprediccions) que l'excés d'estoc.** Hi havia una part del test reservada com a leaderboard públic.
- **Presentació (20 %)**.

## Format d'entrega

```csv
ID,demand
2,0.2
5,0.45
```

## Estratègia guanyadora

**Què van fer els equips amb repo públic:**
- [manelcomiche/datathon-fme](https://github.com/manelcomiche/datathon-fme):
  - **stacking LightGBM + CatBoost** amb un meta-model LightGBM;
  - un "super-ensemble" amb pesos v3 0,55 + stacking 0,35 + Ridge sobre PCA d'embeddings 0,10;
  - un **factor global ×1,08** per compensar l'asimetria de la mètrica.

**El nostre pla** (implementat a [`pipeline.py`](pipeline.py)):
1. **Agregar** de setmanes a una fila per producte, amb target = Σ `weekly_demand`.
2. **Validació temporal**: l'última temporada de train fa de validació. Mai un K-Fold aleatori.
3. **Cold start amb embeddings**: per a cada producte, la demanda dels **k veïns més semblants** de temporades anteriors (`features.knn_target_features`), més la PCA de l'embedding.
4. **LightGBM sobre `log1p(demanda)`** (`models.LogTargetRegressor`), perquè la demanda és molt esbiaixada.
5. **Calibrar un factor > 1** sobre la validació (`metrics.best_scale_factor`). Si la fórmula exacta de la mètrica es publica, cal implementar-la a `metrics.py` i calibrar contra ella.
6. Variants per provar: objectiu `quantile` (alpha 0,6–0,75), CatBoost, ensemble.

**Errors a evitar:** fer servir `weekly_sales` com a target (està censurat), barrejar temporades al K-Fold, o fer servir `production` del test (no hi és).

## Com practicar-lo

```bash
python reptes/2025_mango_demanda/simulate_data.py   # dades sintètiques amb el mateix esquema
python reptes/2025_mango_demanda/pipeline.py
```

Resultat amb dades sintètiques (error asimètric, més baix és millor):

| Model | Error |
|---|---|
| Mediana constant | 0,524 |
| LightGBM sense veïns | 0,242 |
| LightGBM + kNN per embedding | 0,232 |
| + factor ×1,14 | **0,218** |

Per a pràctica amb dades reals de moda: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations). Té articles amb atributs i imatges i transaccions amb data, així que es poden agregar les vendes per article i temporada i predir les dels articles nous.

## Fonts
- Enunciat oficial: [data-students/datathon2025-challenges/Mango](https://github.com/data-students/datathon2025-challenges/tree/main/Mango) (`statement.pdf`)
