# Registre d'experiments

Una línia per experiment. **CV** és la de la validació correcta (agrupada o temporal si cal), no la més alta.

| data/hora | qui | repte | experiment | CV | LB | notes |
|---|---|---|---|---|---|---|
| 17/10 | Joan | Schneider 2025 | LGBM baseline, K-Fold aleatori | 0,850 AUC | — | ⚠️ inflada: 77 % de duplicats |
| 17/10 | Joan | Schneider 2025 | LGBM baseline, GroupKFold per duplicats | 0,695 AUC | — | estimació honesta |
| 17/10 | Joan | Accenture 2022 | LGBM + distàncies + pes | 0,821 AUC | — | baseline a superar |
