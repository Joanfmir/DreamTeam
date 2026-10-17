# Mango 2024 · Quins atributs de disseny té aquesta peça?

> **2024** · Mango · Classificació **multi-sortida** (11 atributs) a partir d'imatge i metadades · **Accuracy** a Kaggle · ✅ Metadades públiques (sense imatges)

## Context

Cada peça de Mango s'etiqueta amb desenes d'**atributs de disseny** (silueta, tipus de coll, llargada, màniga, tancament…). Aquestes etiquetes alimenten el cercador, els filtres de la web, les recomanacions i l'anàlisi de tendències. Etiquetar a mà milers de productes cada temporada és lent i inconsistent.

## La tasca

Per a cada producte del test, predir el valor dels **11 atributs**: `silhouette_type`, `neck_lapel_type`, `woven_structure`, `knit_structure`, `heel_shape_type`, `length_type`, `sleeve_length_type`, `toecap_type`, `waist_type`, `closure_placement` i `cane_height_type`.

Si un atribut **no aplica** a la peça (p. ex. `heel_shape_type` d'una samarreta), el valor correcte és **`INVALID`**.

## Dades

| Fitxer | Contingut |
|---|---|
| `product_data.csv` | 61.484 files (una per imatge) i 33.443 productes amb atributs: `cod_modelo_color`, color, sexe, edat, línia, teixit, categoria, família, tipus i **nom del fitxer d'imatge** |
| `attribute_data.csv` | Format llarg: `cod_modelo_color`, `attribute_name`, `cod_value`, `des_value`. **Només hi ha els atributs que apliquen** |
| `test_data.csv` | 6.529 productes nous × 11 atributs = 71.819 files, amb `test_id = <producte>_<atribut>` |
| `images/` | Foto frontal de cada producte (no són al repo públic) |

Nombre de classes per atribut: silueta 33, coll/solapa 33, llargada 12, taló 11, punt 8, màniga 6, tancament 6, canya 6, cintura 4, puntera 4, teixit pla 4, més `INVALID`.

## Avaluació

**Accuracy** sobre les 71.819 prediccions del test (cada atribut pesa 1/11). Kaggle amb leaderboard. El 2n classificat també presentava una GUI de Streamlit.

## Format d'entrega

```csv
test_id,des_value
88_49711373_cane_height_type,INVALID
88_49711373_silhouette_type,Slim
```

## Estratègia guanyadora

**2n lloc del Kaggle:** [bielnebot/datathon_fme_2024](https://github.com/bielnebot/datathon_fme_2024):
- **un Random Forest per atribut** entrenat només amb metadades (`des_sex`, `des_age`, `des_line`, `des_fabric`, `des_product_family`, `des_product_type`);
- una EDA a part;
- una **GUI de Streamlit** per fer inferència sobre productes nous.

**Lliçó: les metadades soles ja donen moltíssim.** El tipus de producte determina quins atributs apliquen: unes botes sempre tenen `INVALID` a la màniga.

Altres enfocaments ([marcserranos](https://github.com/marcserranos/DATATHON_FME_mango_2024)): **CLIP zero-shot** sobre la imatge ("a photo of a slim dress" vs. "oversize").

**El nostre pla:**
1. Baseline amb metadades i un LightGBM per atribut ([`baseline.py`](baseline.py)) → submission en menys d'1 hora.
2. Afegir **embeddings CLIP de la imatge** (`dreamteam.embeddings.image_embeddings`) amb PCA, i fer-los servir com a features.
3. Atributs "visuals" (silueta, coll, llargada) amb imatge; atributs "estructurals" (`INVALID` sí o no) amb metadades i regles.
4. **Regles de negoci** sobre la predicció: si el tipus és calçat, la màniga és `INVALID`.
5. Validació **agrupada pel model** (les variants de color d'un model no poden estar a train i a validació alhora).

## Com practicar-lo

```bash
bash scripts/download_past_data.sh mango2024
python reptes/2024_mango_atributs/baseline.py
```

Resultat (només metadades, validació agrupada):

| Atribut | Accuracy | Classe majoritària |
|---|---|---|
| silhouette_type | 0,533 | 0,366 |
| neck_lapel_type | 0,630 | 0,361 |
| length_type | 0,637 | 0,418 |
| woven_structure | 0,711 | 0,387 |
| sleeve_length_type | 0,776 | 0,388 |
| closure_placement | 0,782 | 0,556 |
| **Mitjana dels 11** | **0,801** | 0,640 |

Els atributs més fluixos (silueta, coll, llargada) són **visuals**: és on la imatge hauria d'aportar més.

## Fonts
- Dades: repo públic del 2n classificat, [bielnebot/datathon_fme_2024](https://github.com/bielnebot/datathon_fme_2024)
- [DoraHacks · Datathon FME 2024](https://dorahacks.io/hackathon/datathon-fme-2024/buidl)
