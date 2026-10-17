# Mango 2023 · Quina peça completa aquest outfit?

> **2023** · Mango · Recomanació visual / compatibilitat de moda (imatges + metadades) · **Pitch + repo** (sense mètrica) · ⚠️ L'enllaç a les dades ja no funciona

## Context

Fer un bon outfit no és juntar peces **semblants**, sinó peces **complementàries**: un vestit de setí lila, unes sandàlies blanques i una bossa platejada combinen o no? Les metadades no ho diuen. **Cal mirar les imatges.** Mango té outfits reals creats per estilistes.

## La tasca

Crear un model que, a partir d'**una peça inicial**, generi **recomanacions d'outfit**. El pas intermedi suggerit és la tasca **"fill in the blank"**: donat un outfit amb una peça que falta i una llista de candidates, triar la que hi encaixa.

## Dades

| Fitxer | Contingut |
|---|---|
| `product_data.csv` | Una fila per producte (`cod_modelo_color`): color, sexe, edat, línia, teixit, categoria, família, tipus i **imatge** (foto frontal sense model) |
| `outfit_data.csv` | Relació `cod_outfit` ↔ `cod_modelo_color`: quines peces formen cada outfit |
| `images/` | Imatges dels productes |

L'enllaç oficial de baixada (S3 de Mango) **dona 404**. El [fork oficial del repte](https://github.com/data-science-mango/datathon-2023-fashion-compatibility) conserva l'estructura.

## Avaluació

Sense mètrica numèrica:
1. **Pitch de 10 minuts** davant d'experts en dades, ML i moda. Cal ensenyar **com a mínim 5 outfits nous** generats a partir de productes del dataset. Es valora la creativitat, la complexitat, l'aplicabilitat i la comunicació.
2. **El repo de GitHub**: **qualitat del codi i reproductibilitat**, **documentació** i **ús de control de versions i branques**.

Recomanaven fer una demo (Streamlit) i podien enriquir les dades amb fonts externes, respectant les llicències.

## Estratègia guanyadora

**Enfocament que funciona:**
1. **Embeddings visuals** de cada peça amb un model preentrenat (CLIP, ResNet) → `dreamteam.embeddings.image_embeddings`.
2. **Compatibilitat ≠ similitud**: aprendre de les parelles que apareixen juntes als outfits reals. Opcions:
   - **co-ocurrència** de tipus i colors (estadística simple, molt explicable);
   - model de parelles (siamès o *triplet loss*): peces del mateix outfit a prop, peces aleatòries lluny;
   - mitjana dels embeddings de l'outfit parcial i triar la candidata més compatible.
3. **Regles de categoria**: un outfit = top + bottom (o vestit) + calçat + accessori, sense repetir categoria.
4. **Demo** visual amb Streamlit: tries una peça i surten 3 outfits amb fotos. És el que guanya el pitch.
5. **Repo impecable**: README, `requirements`, branques i PRs. Aquí puntuava explícitament.

**Equips amb repo públic:** [theWandroid](https://github.com/theWandroid/FME_Datathon_Mango), [vascoG](https://github.com/vascoG/datathon-fme-2023-mango), [PauMayench](https://github.com/PauMayench/datathon-FME-2023-fashion-compatibility-X2PJ).

## Com practicar-lo

- [Polyvore Outfits](https://github.com/mvasil/fashion-compatibility) (dataset acadèmic de compatibilitat, amb la tasca "fill in the blank").
- Si Mango torna amb imatges, cal tenir **CLIP** provat al portàtil i saber calcular embeddings per a 10.000 imatges (≈ minuts amb CPU i model base).

## Fonts
- [data-students/datathon2023/mango](https://github.com/data-students/datathon2023/tree/main/mango)
