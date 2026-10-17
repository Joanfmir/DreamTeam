# Registre d'experiments

Una línia per experiment. **CV** és la de la validació correcta (agrupada o temporal si cal), no la més alta.

| data/hora | qui | repte | experiment | CV | LB | notes |
|---|---|---|---|---|---|---|
| 17/10 | Joan | Schneider 2025 | LGBM baseline, K-Fold aleatori | 0,850 AUC | — | ⚠️ inflada: 77 % de duplicats |
| 17/10 | Joan | Schneider 2025 | LGBM baseline, GroupKFold per duplicats | 0,695 AUC | — | estimació honesta |
| 17/10 | Joan | Accenture 2022 | LGBM + distàncies + pes | 0,821 AUC | — | baseline a superar |
| 17/10 | Joan | Accenture 2022 | + count encoding + target encoding OOF (producte, client, ruta, 3PL×hub…) | 0,837 AUC | — | +0,016 vs. baseline |
| 17/10 | Joan | Mango 2024 | LightGBM per atribut, només metadades, CV agrupada per model | 0,801 acc | — | majoritària: 0,640 |
| 17/10 | Joan | Reviews 2021 | TF-IDF 1–2-grames + LogReg | MAE 0,334 | — | embeddings MiniLM: 0,443 |
| 17/10 | Joan | Mango 2025 (sintètic) | LGBM log1p + kNN embedding + factor ×1,14 | 0,218 err | 0,208 | mediana: 0,524 |
| 17/10 | Joan | Qualcomm 2022 | k-means equilibrat + hongarès + serp + 2-opt | 8,78 M | — | exemple oficial: 73,4 M |
