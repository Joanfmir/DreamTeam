# Accenture 2022 · Arribarà tard aquesta comanda?

> **2022** · Accenture · Classificació binària + cas de negoci · **ROC AUC** (Kaggle) + presentació · ✅ Dades públiques · 🥇 Guanyador amb repo públic

## Context

Una empresa amb una cadena de subministrament internacional importa productes per diversos **ports**, els distribueix amb **operadors logístics (3PL)** a través de **hubs** i els entrega a **clients** de tot Europa. Els retards costen diners i clients. Accenture vol **visibilitat de la resiliència de la cadena**: entendre què causa els retards i anticipar-los.

## La tasca

1. **Insights descriptius**: quines són les causes principals dels retards i com es tradueixen en accions de negoci.
2. **Model predictiu**: la **probabilitat que una comanda arribi tard**.

## Dades

| Fitxer | Columnes |
|---|---|
| `orders.csv` (114.276, sep `;`) | `order_id`, `origin_port` (5 ports), `3pl` (4 operadors), `customs_procedures` (3 tipus), `logistic_hub` (9 hubs, nul si l'enviament és directe), `customer` (28 ciutats), `product_id` (770), `units`, **`late_order`** (23,8 % tard) |
| `test.csv` (28.563, sep `;`) | El mateix sense `late_order` |
| `product_attributes.csv` | `product_id`, `weight` (g/unitat), `material_handling` (classe de risc: fràgil, tòxic, inflamable…) |
| `cities_data.csv` (sep `;`) | `city_from_name`, `city_to_name`, coordenades, `distance` (km) |

## Avaluació

- **Model:** **ROC AUC** sobre el test en un [Kaggle](https://www.kaggle.com/t/187d01ee269e404ca7267e8caddf0eae).
- **Cas de negoci:** presentació amb els drivers dels retards, com es converteixen en accions i quina proposta de valor tenen.

## Format d'entrega

```csv
order_id,late_order
366c7a3d298f,0.31
```

## Estratègia guanyadora

**🥇 Guanyador:** [JNADatathon2022/Datathon](https://github.com/JNADatathon2022/Datathon), "Metis Business Intelligence" ([Devpost](https://devpost.com/software/metis-business-intelligence)):
- model amb **Keras/TensorFlow**;
- a més, una **API d'inferència** desplegable ([ModelInferenceApp](https://github.com/JNADatathon2022/ModelInferenceApp)). **El model es va convertir en un producte.**

**El nostre pla (implementat):**
1. Joins: pes del producte i distàncies port → hub → client ([`baseline.py`](baseline.py)) → **AUC 0,821**.
2. **Encodings** ([`features_v2.py`](features_v2.py)): count encoding de producte, client, ruta i 3PL, **target encoding OOF** de producte, client, ruta i combinacions (3PL × hub, 3PL × duanes, port × 3PL), i unitats respecte a la mitjana del producte → **AUC 0,836** (+0,015).
   ⚠️ El target encoding es calcula **dins de cada fold** (`validation.cross_validate_with_test(fold_fn=...)`). La primera versió el calculava un cop sobre tot el train i feia la CV amb uns altres folds: la validació veia el seu propi target. Aquí la inflació va ser petita (0,8366 vs. 0,8363), però en datasets petits pot ser enorme.
3. Següents passos: CatBoost amb categòriques natives, ensemble per rangs, Optuna.
4. **Negoci**: taxa de retard per 3PL × hub, i "quin canvi d'operador en una ruta reduiria més els retards?".

## Com practicar-lo

```bash
bash scripts/download_past_data.sh accenture
python reptes/2022_accenture_supply_chain/baseline.py      # AUC 0,821
python reptes/2022_accenture_supply_chain/features_v2.py   # AUC 0,836
```

**Repte per a l'equip:** superar 0,836 i apuntar cada intent a `experiments.md`.

## Fonts
- [data-students/datathon2022/accenture_challenge](https://github.com/data-students/datathon2022/tree/main/accenture_challenge)
- Altres: [KensingtonOscupant/datathon-fme2022](https://github.com/KensingtonOscupant/datathon-fme2022) (ensemble), [thomas-gonzalez-saito](https://github.com/thomas-gonzalez-saito/DatathonFME_2022_Accenture)
