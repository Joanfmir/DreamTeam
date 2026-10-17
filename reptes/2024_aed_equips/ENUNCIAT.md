# AED 2024 · Troba el teu equip ideal

> **2024** · Associació d'Estudiants de Dades · Matching i optimització amb restriccions + NLP + visualització · Avaluació oberta · ✅ Dades públiques

## Context

Cada any molts participants arriben sense equip i el busquen a Discord. L'organització vol eines que ajudin a **trobar companys compatibles**, tant en habilitats tècniques com en objectius (guanyar, aprendre, fer portfoli, conèixer gent). No volen un assignador automàtic, sinó **eines que facilitin la connexió**.

## La tasca

Crear un sistema de formació d'equips (de **fins a 4 persones**) **equilibrats i complementaris**, que tingui en compte la compatibilitat tècnica i l'alineació d'objectius, amb **visualitzacions i dashboards** que expliquin la composició de cada equip.

## Dades

`datathon_participants.json`: una llista de participants (dataclass `Participant` a `participant.py`).

| Camp | Tipus |
|---|---|
| `id`, `name`, `email`, `age`, `university`, `shirt_size`, `dietary_restrictions` | Dades personals |
| `year_of_study` | 1r–4t, Màster o PhD |
| `programming_skills` | Diccionari llenguatge → nivell 1–5 |
| `experience_level`, `hackathons_done` | Principiant, intermedi o avançat, i nombre de hackathons |
| `interests`, `interest_in_challenges`, `preferred_languages` | Llistes |
| `preferred_role` | Analysis, Visualization, Development o Design |
| `objective` | **Text lliure**: "què busques a la Datathon?" |
| `friend_registration` | Llista d'IDs d'amics (han d'anar junts) |
| `preferred_team_size` | 1–5 |
| `availability` | Diccionari franja → disponible |
| `introduction`, `technical_project`, `future_excitement`, `fun_fact` | Text lliure |

## Avaluació

Oberta: qualitat del matching, ús de les dades (sobretot el **text lliure**), visualitzacions, usabilitat i presentació.

## Format d'entrega

Repo, demo i presentació.

## Estratègia guanyadora

**Equips amb repo públic:**
- [Camoraz/Datathon-FME](https://github.com/Camoraz/Datathon-FME), amb vídeo:
  - **Sentence-BERT** (`all-MiniLM-L6-v2`) per al text dels objectius, amb tractament de negacions;
  - similitud combinada: euclidiana per a les numèriques, Jaccard per a les llistes i cosinus per als embeddings, amb **pesos dinàmics**;
  - recomana els *n* participants més compatibles.
- [jordi04/aed_grouping_webapp](https://github.com/jordi04/aed_grouping_webapp): app de Streamlit + Firebase amb registre, agrupació i gestió visual dels equips.
- [DidacDV/Datathon-FME-Connect](https://github.com/DidacDV/Datathon-FME-Connect): React + Django + PostgreSQL, amb un algorisme d'equips amb paràmetres de diversitat i compatibilitat.

**El nostre pla:**
1. Vector per participant: habilitats (one-hot amb nivell), rol, experiència, disponibilitat i **embedding del text lliure** (`dreamteam.embeddings.text_embeddings`).
2. Puntuació de parella = **similitud en objectius i disponibilitat** + **complementarietat en rols i habilitats** (diversitat, no semblança).
3. Restriccions dures: els amics junts i mida ≤ 4 → **optimització** (voraç + millora local, o CP-SAT d'OR-Tools).
4. Visualització: UMAP o t-SNE dels participants i un "radar" d'habilitats per equip.

## Com practicar-lo

```bash
bash scripts/download_past_data.sh aed
```

És un bon simulacre de 6 hores per a la setmana 4: hi ha text, optimització, visualització i demo.

## Fonts
- [data-students/AEDChallenge](https://github.com/data-students/AEDChallenge)
