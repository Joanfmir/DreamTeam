# 2021 · Housing i Mobility: els reptes de Barcelona

> **2021** (1a edició, "Barcelona's present and future challenges") · Accenture + Oficina Municipal de Dades · Forecasting, regressió, anàlisi d'impacte · Avaluació oberta · ✅ Open Data BCN

## Housing · Serà assequible viure a Barcelona?

**Preguntes:**
- Quin percentatge de residents es podrà permetre els preus futurs de Barcelona?
- Migraran a altres barris o als voltants?
- Com afectarà la **nova llei del lloguer** als preus?

**Enfocaments suggerits:** forecasting i models de regressió.

**Dades (Open Data BCN):**

| Dataset | Període | Freqüència |
|---|---|---|
| Renda per família i barri | 2008–2017 | Anual |
| Compravendes d'habitatges | 2013–2021 | Trimestral |
| Preus de lloguer | 2014–2021 | Trimestral |
| Renda neta mitjana per secció censal | 2015–2018 | Anual |
| Padró d'habitants | 2001–2021 | Anual |
| Atur registrat | 2011–2021 | Mensual |

## Mobility · Barcelona tindrà una mobilitat més verda?

**Preguntes:**
- Com afecta la mobilitat a la **contaminació de l'aire i acústica**?
- Els carrils bici nous o la **ZBE** (Zona de Baixes Emissions) milloren la qualitat de l'aire o del soroll?
- Quines mesures s'haurien d'aplicar per reduir la contaminació i quin efecte tindrien?

**Enfocaments suggerits:** ANOVA, models de regressió i **anàlisi d'impacte**.

**Dades:**
- Qualitat de l'aire per zones (horària, des del 2008).
- Informació de la ZBE (AMB).
- Enquesta de mobilitat en dia feiner (2011–2020).
- Parc de vehicles (2016–2020).
- Trànsit i soroll (des del 2020).
- Transport públic (2010–2020).
- Bicing i carrils bici (2007–2020).

## Estratègia guanyadora (reptes oberts de ciutat)

1. **Indicador d'assequibilitat** = lloguer mitjà / renda per barri, amb una sèrie temporal i una projecció. És un sol número que s'entén.
2. **Anàlisi d'impacte causal** de la ZBE o d'un carril bici: **diferències en diferències** (zona afectada vs. zona control, abans vs. després) o *regression discontinuity* en el temps. Molt més potent que una correlació.
3. Controlar l'**estacionalitat** i la **COVID** (2020) en qualsevol sèrie de mobilitat o contaminació.
4. **Mapa** per barri i una proposta concreta.

**Equips amb repo públic:** [MarcLlopart/DatathonFME](https://github.com/MarcLlopart/DatathonFME) (Housing), [EdgarMM19/DatathonFME2021](https://github.com/EdgarMM19/DatathonFME2021).

## Fonts
- [data-students/datathon2021](https://github.com/data-students/datathon2021) (`Challenges Datathon FME 2021.pdf`, `get_data_opendata.ipynb`, `python_useful_functions.ipynb`)
