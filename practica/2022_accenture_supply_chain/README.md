# Pràctica · Accenture 2022: comandes que arriben tard

**Repte:** predir la probabilitat que una comanda arribi tard (`late_order`) i explicar-ne les causes en un cas de negoci.
**Mètrica:** ROC AUC. Avaluació: model + presentació de negoci.
Guanyador: [JNADatathon2022/Datathon](https://github.com/JNADatathon2022/Datathon), que va acompanyar el model d'una API d'inferència.

## Dades (`data/past/2022_accenture/`)

| Fitxer | Contingut | Separador |
|---|---|---|
| `orders.csv` | 114.276 comandes: port, operador logístic (`3pl`), duanes, hub, client, producte, unitats → `late_order` (23,8 % tard) | `;` |
| `test.csv` | 28.563 comandes sense target | `;` |
| `product_attributes.csv` | pes i tipus de manipulació de cada producte | `,` |
| `cities_data.csv` | distàncies entre parelles de ciutats | `;` |

## Baseline (ja funciona)

```bash
python practica/2022_accenture_supply_chain/baseline.py
```

Joins, distàncies port → hub → client, pes total, LightGBM amb CV de 5 folds: **AUC 0,821**. Escriu una submission validada a `submissions/`.

## Tasques de la setmana 2: superar 0,821

- [ ] **Target encoding OOF** de `product_id`, `customer` i les parelles (`3pl`, `logistic_hub`).
- [ ] **Agregats**: % de comandes tard per ruta (port → hub → client) calculat dins de cada fold.
- [ ] Comptatges: quantes comandes té cada producte o client (càrrega del sistema).
- [ ] CatBoost amb les categòriques natives i comparar amb LightGBM.
- [ ] Ensemble LGBM + CatBoost (mitjana de rangs).
- [ ] **Cas de negoci**: quin operador logístic o hub causa més retards? Fer un gràfic per a l'slide.
