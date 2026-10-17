<div align="center">

# 🏆 DreamTeam · Datathon FME 2026

**Preparació, plantilles i entrenament per a la 6a Datathon FME**
Facultat de Matemàtiques i Estadística (UPC) · Barcelona · **14–15 de novembre de 2026**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-tabular-9ACD32)
![CatBoost](https://img.shields.io/badge/CatBoost-categorical-FFCC00)
![SHAP](https://img.shields.io/badge/SHAP-explainability-FF0051)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## Què és aquest repo

La Datathon FME és una competició de **48 hores** amb **3 reptes reals** proposats per empreses (Mango, Smadex, Schneider Electric, Accenture, Qualcomm, Restb.ai…). Els reptes no es publiquen fins a l'inici, però **el tipus de problema es repeteix cada any**: predicció tabular, previsió de demanda, classificació amb explicabilitat, imatges/embeddings i un repte "obert" de visualització o d'impacte social.

Aquest repo serveix perquè el dia de l'esdeveniment **no perdem temps muntant res**:

1. **Coneixement**: què s'ha demanat els anys anteriors, com s'avalua i què van fer els guanyadors.
2. **Codi reutilitzable** (`src/dreamteam/`): càrrega de dades, validació (amb detecció de duplicats), models baseline, mètriques, explicabilitat i generació de la submission. Tot ja està provat.
3. **Entrenament**: reptes antics amb dades públiques per practicar com a equip abans del 14 de novembre.
4. **Estratègia**: pla hora a hora, rols i checklist de la presentació.

## Mapa del repo

```
DreamTeam/
├── docs/
│   ├── 01_datathon_fme.md        # Format, dates, regles, premis, com s'avalua
│   ├── 02_reptes_anteriors.md    # Tots els reptes 2021–2025 + solucions guanyadores
│   ├── 03_estrategia_48h.md      # Pla hora a hora, rols, git, presentació
│   ├── 04_playbook_ml.md         # Què fer segons el tipus de problema
│   └── 05_recursos.md            # Enllaços, cursos, llibres, datasets
├── src/dreamteam/                # Llibreria pròpia (pip install -e .)
│   ├── data.py                   # Càrrega, reducció de memòria, resum del dataset
│   ├── validation.py             # K-Fold / Group / temporal + prediccions OOF
│   ├── models.py                 # Baselines LightGBM, CatBoost, lineals
│   ├── metrics.py                # MSLE, pèrdua asimètrica (tipus Mango), F1, AUC…
│   ├── explain.py                # SHAP global i local, en un parell de línies
│   └── submission.py             # Submission validada contra sample_submission
├── notebooks/
│   ├── 00_eda_template.ipynb     # EDA ràpida per a qualsevol dataset nou
│   └── 01_baseline_tabular.ipynb # De CSV a submission en 10 minuts
├── practica/                     # Reptes antics per entrenar-nos
│   ├── README.md                 # Pla setmanal fins al 14/11
│   ├── 2025_schneider_explainability/
│   ├── 2022_accenture_supply_chain/
│   ├── 2025_mango_demand/
│   └── 2025_smadex_revenue/
├── app/streamlit_demo.py         # Plantilla de demo per a la presentació
├── scripts/download_past_data.sh # Baixa les dades públiques dels reptes antics
├── data/                         # (ignorat per git) dades locals
├── experiments.md                # Registre d'experiments: qui, què, CV, LB
├── pyproject.toml
└── requirements.txt
```

## Començar

```bash
git clone https://github.com/Joanfmir/DreamTeam.git
cd DreamTeam
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # instal·la el paquet dreamteam

bash scripts/download_past_data.sh     # baixa les dades dels reptes antics a data/
python practica/2025_schneider_explainability/baseline.py   # prova que tot funciona
```

## El repte en una línia (lliçons dels anys anteriors)

> **Model decent ràpid + molt bona història.** Gairebé tots els reptes avaluen la presentació (del 20 % al 100 % de la nota), i molts també el repo: codi, documentació i ús de git. Un LightGBM ben validat i ben explicat guanya a un model brillant que ningú entén.

- Tingueu una **submission vàlida en menys de 2 hores**, encara que sigui una mitjana.
- **Validació correcta** abans que tuning: split temporal si hi ha temps, per grup si hi ha IDs repetits o **files duplicades**. A Schneider 2025, el 77 % de les files eren duplicades i un K-Fold aleatori donava AUC 0,85 en lloc de la real, ~0,71.
- **Optimitzeu la mètrica real**: Mango 2025 penalitzava més les vendes perdudes que l'excés d'estoc, i Smadex feia servir MSLE i també valorava la velocitat d'inferència.
- **Reserveu 6 hores** per a la presentació i la demo.

## Equip DreamTeam

| Membre | Rol principal | Rol de suport |
|---|---|---|
| Joan Fernández ([@Joanfmir](https://github.com/Joanfmir)) | *per decidir* | |
| *—* | | |
| *—* | | |
| *—* | | |

Rols suggerits (detall a [`docs/03_estrategia_48h.md`](docs/03_estrategia_48h.md)): **Modelatge**, **Dades i features**, **Visualització i demo**, **Història i presentació**.

## Dates clau

| Data | Què |
|---|---|
| Fins al 13/11 | Entrenament amb reptes antics (vegeu [`practica/README.md`](practica/README.md)) |
| **14/11/2026** | Inici, presentació dels reptes, tria del repte |
| **15/11/2026** | Entrega, presentacions i premis |

Web oficial: [datathon.cat](https://www.datathon.cat/) · Organitza l'[Associació d'Estudiants de Dades (AED)](https://github.com/data-students) amb la FME-UPC.
