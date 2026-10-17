# Pla d'entrenament (17/10 → 13/11)

Quatre setmanes per arribar a la Datathon **sabent treballar junts** i amb les eines provades.
Cada setmana hi ha un repte antic, un objectiu concret i una petita entrega al repo.

| Setmana | Repte | Tipus | Objectiu de la setmana | Dades |
|---|---|---|---|---|
| **1** · 19–25 oct | [Schneider 2025](2025_schneider_explainability/) | Classificació + explicabilitat | Tothom executa el repo. Entendre el **leakage per duplicats** (77 % de files!). Cada persona porta 1 insight per a no tècnics amb SHAP | ✅ `download_past_data.sh` |
| **2** · 26 oct–1 nov | [Accenture 2022](2022_accenture_supply_chain/) | Tabular + AUC + negoci | Superar el baseline (AUC CV 0,821) amb **feature engineering**. Registre a `experiments.md` | ✅ `download_past_data.sh` |
| **3** · 2–8 nov | [Mango 2025](2025_mango_demand/) i [Smadex 2025](2025_smadex_revenue/) | Forecasting, embeddings, MSLE | Practicar **validació temporal, pèrdua asimètrica i `log1p`** amb datasets públics semblants | Kaggle (vegeu cada carpeta) |
| **4** · 9–13 nov | **Simulacre de 6 h** | Repte sorpresa | Fer el dia D en petit: triar, baseline en 1 h, pitch de 5 min | AED 2024 o Reviews 2021 |

## Regles de l'entrenament

1. **Una branca per persona i experiment** (`joan/target-encoding`). Es fa merge a `main` quan millora la CV.
2. Cada experiment s'apunta a `experiments.md` a l'arrel: qui, idea, CV.
3. El codi que fem servir dues vegades **puja a `src/dreamteam/`**.
4. Al final de cada setmana, **10 minuts de "pitch"** per persona: què he provat i què he après.

## Checklist abans del 14/11

- [ ] Tots tenim el repo clonat, el `.venv` creat i `pip install -r requirements.txt` funcionant **al portàtil que portarem**.
- [ ] Tots hem executat `python practica/2025_schneider_explainability/baseline.py` sense errors.
- [ ] Tenim compte de Kaggle i de Devpost.
- [ ] Hem decidit rols ([`docs/03_estrategia_48h.md`](../docs/03_estrategia_48h.md)).
- [ ] Plantilla de slides preparada (Google Slides compartit).
- [ ] Carregadors, regleta i auriculars a la motxilla 🔌
