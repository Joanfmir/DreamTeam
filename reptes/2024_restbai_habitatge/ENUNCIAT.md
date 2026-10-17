# Restb.ai 2024 · Quant val aquesta casa?

> **2024** · Restb.ai · Regressió de preus immobiliaris (tabular, potencialment amb imatges) · Model + app · Dades privades

## Context

Restb.ai és una empresa de Barcelona que fa **visió per computador per al sector immobiliari**: extreu de les fotos d'un anunci l'estat de la cuina, els acabats o les vistes. El seu negoci és enriquir els anuncis amb dades que serveixin per **valorar habitatges automàticament** (AVM, *Automated Valuation Model*).

## La tasca

Predir el **preu de venda** d'habitatges a partir de les seves característiques.

## Dades

Segons els equips participants:
- Més de **100.000 habitatges de Chicago i rodalies**.
- **10 mesos d'històric** d'anuncis.
- Ubicació, superfície, habitacions, banys i altres atributs.
- Valors nuls i **outliers** que calia tractar.

Restb.ai sol aportar **features extretes de les fotos**: estat, qualitat i tipus d'estança. Si torna el 2026, és probable que n'hi hagi.

## Avaluació

Model de regressió i aplicació demostrativa. Mètriques típiques en valoració immobiliària: **MAE, RMSE i MAPE** (error percentual, perquè 10.000 $ no pesen igual en una casa de 100.000 que en una d'1 milió).

## Format d'entrega

Repo, model i app o presentació.

## Estratègia guanyadora

**Equips amb repo públic:**
- [victorbrull/DatathonFME2024](https://github.com/victorbrull/DatathonFME2024): pipeline amb **MLflow** per fer el seguiment d'experiments i app de **Streamlit** on l'usuari introdueix els paràmetres i obté el preu.
- [GenisLopez5/DatathonFME2024](https://github.com/GenisLopez5/DatathonFME2024): Random Forest, eliminació d'outliers i imputació per la mitjana.

**El nostre pla:**
1. **Target en log** (`log1p(preu)`): els preus són esbiaixats i l'error relatiu és el que importa (`models.LogTargetRegressor`).
2. **Features geogràfiques**: lat/long, distància al centre, **target encoding OOF del barri o codi postal** (`features.target_encode`) i preu mitjà dels **k veïns geogràfics** (`features.knn_target_features` amb coordenades).
3. Ràtios: preu per m² del barri, m² per habitació, antiguitat.
4. **Temps**: si hi ha 10 mesos, validar amb l'últim mes (`validation.time_split`), perquè el mercat es mou.
5. Outliers: retallar el target per percentils **només al train**, mai al test.
6. Demo: app on canvies m² o habitacions i veus com canvia el preu, amb una explicació SHAP de cada valoració.

## Com practicar-lo

- [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (Kaggle, RMSE sobre el log): el clàssic, petit i ràpid.
- Dades obertes de BCN: preus de lloguer per barri a [Open Data BCN](https://opendata-ajuntament.barcelona.cat/).

## Fonts
- [DoraHacks · Datathon FME 2024](https://dorahacks.io/hackathon/datathon-fme-2024/buidl)
- Repos d'equips citats a dalt.
