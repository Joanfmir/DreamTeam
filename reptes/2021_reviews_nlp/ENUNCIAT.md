# 2021 · Reviews Challenge: NLP sobre ressenyes de restaurants

> **2021** · Organització (dades de Google Maps via SerpApi) · **NLP** (classificació, generació, anàlisi) · Avaluació oberta · ✅ Dades públiques · 🥇 Guanyador amb repo públic

## Context

Les ressenyes de Google Maps tenen informació molt valuosa sobre els restaurants de Barcelona: qualitat, servei, preu, ambient… però en **text lliure**.

## La tasca

Crear una **pipeline de NLP** amb les ressenyes. Exemples que donaven:
- **Predir les estrelles** que ha posat un usuari a partir del text.
- Generar ressenyes noves amb un model de llenguatge.
- Anàlisi i visualització: relació entre la ubicació, el preu i la valoració.

Encoratjaven a fer tasques d'NLP/ML, però l'enfocament era lliure.

## Dades

`train_reviews.json` (270 restaurants, ~38.000 ressenyes) i `test_reviews.json` (30 restaurants diferents, ~4.000 ressenyes). Cada restaurant té:

```json
{"title", "place_id", "gps_coordinates": {"latitude", "longitude"}, "rating", "reviews",
 "price", "type", "address", "hours", "service_options": {"dine_in", "takeout", "delivery"},
 "reviews_data": [{"user": {...}, "rating", "date", "snippet"}, ...]}
```

Distribució d'estrelles: **65 % són 5★**, 19 % 4★, 7 % 3★, 4 % 2★ i 6 % 1★ (molt desbalancejada).

## Avaluació

Oberta. Si es feia un model, calia reportar mètriques sobre `test_reviews.json` perquè tothom fos comparable.

## Estratègia guanyadora

**🥇 Guanyador:** [megaelius/Delfos-datathon-fme](https://github.com/megaelius/Delfos-datathon-fme), un **sistema de recomanació de restaurants per a grups**. **No es van quedar en "predir estrelles"**: van convertir el dataset en un **producte** que resol un problema real (on anem a sopar 6 amics amb gustos diferents?).

**El nostre baseline ([`baseline.py`](baseline.py))**, validat sobre restaurants nous:

| Model | Accuracy | MAE (estrelles) |
|---|---|---|
| Sempre 5★ | 0,688 | 0,556 |
| **TF-IDF (1–2-grames) + LogisticRegression** | **0,719** | **0,334** |
| Embeddings multilingües (MiniLM) + LogReg | 0,653 | 0,443 |

**Lliçons:**
- Amb classes tan desbalancejades, l'**accuracy enganya** (sempre 5★ ja treu 0,69). Cal mirar el **MAE** o el F1 macro.
- **TF-IDF guanya els embeddings genèrics** en aquest cas. Sempre cal tenir el baseline clàssic: és ràpid i explicable (les paraules de 1★ són *rude, worst, horrible*; les de 5★, *amazing, delicious, perfect*).
- Per guanyar: **aspectes** (menjar, servei, preu, ambient) amb un LLM o zero-shot, i un **producte** a sobre.

## Com practicar-lo

```bash
bash scripts/download_past_data.sh reviews
python reptes/2021_reviews_nlp/baseline.py
```

## Fonts
- [data-students/datathon2021/Reviews_Challenge](https://github.com/data-students/datathon2021/tree/main/Reviews_Challenge)
