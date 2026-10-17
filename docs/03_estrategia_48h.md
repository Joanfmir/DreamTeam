# 03 · Estratègia per a les 48 hores

## Rols (4 persones)

| Rol | Responsabilitat | Durant el hackathon |
|---|---|---|
| **🧠 Modelatge** | Validació, models i leaderboard | És el propietari del `train.py` i de les submissions |
| **🔧 Dades i features** | Neteja, joins, feature engineering, dades externes | Prepara `features.parquet` perquè modelatge només hagi d'importar |
| **📊 Visualització i demo** | EDA, gràfics i app de Streamlit | Fa els gràfics "de slide" des de la primera hora |
| **🎤 Història i presentació** | Entendre el negoci, parlar amb els mentors, slides i README | Escriu la narrativa en paral·lel, no al final |

Els rols són flexibles, però **cada cosa ha de tenir un responsable**. Tothom fa commits.

## Pla hora a hora

> Els horaris exactes es publiquen a la web. Ajusteu-ho a l'hora real d'obertura i d'entrega.

### Dia 1
| Temps | Què | Sortida |
|---|---|---|
| **T+0:00** | Presentació dels reptes | Notes de cada repte |
| **T+0:30** | **Triar el repte** amb la matriu de decisió de sota | Decisió, sense tornar-hi |
| **T+1:00** | Setup: fork o branca, `data/`, fer servir els notebooks plantilla | Repo a punt |
| **T+1:30** | EDA ràpida (`00_eda_template.ipynb`) i **parlar amb els mentors** | 5 insights i preguntes resoltes |
| **T+2:30** | **Baseline** (`01_baseline_tabular.ipynb`) i **primera submission** | Puntuació al leaderboard |
| **T+3–10** | Iteració: features → validació → model. Una idea per branca | Registre d'experiments |
| **Nit** | Dormir per torns (mínim 4 h cadascú). Deixar entrenaments llargs en marxa | |

### Dia 2
| Temps | Què | Sortida |
|---|---|---|
| **T+−8 h** | **Congelar les features.** Només tuning i ensemble a partir d'aquí | |
| **T+−6 h** | Començar les slides i la demo. **Ningú no obre cap idea nova** | Esborrany de slides |
| **T+−3 h** | Submission final triada amb CV (no amb el leaderboard públic) | `submission_final.csv` |
| **T+−2 h** | README final, neteja del repo, assaig del pitch ×2 | Repo i pitch |
| **T+−1 h** | **Entregar** a Devpost o la plataforma que toqui. No esperar l'últim minut | ✅ |

## Matriu per triar el repte

Puntueu cada repte d'1 a 5 en cada criteri:

| Criteri | Pes |
|---|---|
| Hi ha mètrica objectiva o leaderboard (sabem quan anem bé) | ×2 |
| Encaixa amb el que sabem fer (tabular > visió > NLP > optimització) | ×2 |
| Menys equips previsibles (els de Mango s'omplen) | ×1 |
| El pes de la presentació ens afavoreix (tenim bona demo i bona història) | ×1 |
| Ens motiva | ×1 |

## Workflow de git

- `main` sempre ha de funcionar. Treballeu en branques curtes (`feat/target-encoding`, `exp/catboost`).
- Les dades **no van a git** (`data/` és a `.gitignore`). Compartiu-les per USB o Drive.
- Notebooks: **una persona per notebook**, per evitar conflictes. El codi compartit va a `src/`.
- Registre d'experiments a `experiments.md`: data, qui, idea, CV, LB.

```
| hora  | qui  | experiment                    | CV     | LB     |
|-------|------|-------------------------------|--------|--------|
| 11:40 | Joan | baseline LGBM, sense features | 0.712  | 0.705  |
```

## Presentació (5–10 min)

Plantilla slide a slide: [`07_plantilla_presentacio.md`](07_plantilla_presentacio.md).

1. **El problema en una frase**, en llenguatge de negoci, no de dades.
2. **Un insight sorprenent** de l'EDA, amb un sol gràfic.
3. **La solució:** diagrama simple del pipeline i per què funciona.
4. **Resultats:** mètrica vs. baseline, i què vol dir en euros o unitats per a l'empresa.
5. **Demo** en directe o en vídeo de 30 s (Streamlit). Tingueu el vídeo per si falla la wifi.
6. **Limitacions i següents passos** (Schneider ho puntuava a part).

Checklist:
- [ ] Màxim un missatge per slide i gràfics llegibles des del fons de la sala.
- [ ] Noms de l'equip i enllaç al repo a la primera i a l'última slide.
- [ ] Assaig cronometrat ×2.
- [ ] README amb: problema, com executar, resultats i estructura.

## Errors típics a evitar

- ❌ Passar 3 hores triant repte o fent EDA sense baseline.
- ❌ **Leakage**: features calculades amb el target o amb dades del futur, o IDs que es repeteixen entre train i test.
- ❌ Perseguir el leaderboard públic (overfitting). Cal fiar-se de la CV.
- ❌ Començar les slides 2 hores abans de l'entrega.
- ❌ Instal·lar llibreries el mateix dia. **Tot instal·lat i provat abans** (`make setup`, `make setup-dl`, `make baselines`).
