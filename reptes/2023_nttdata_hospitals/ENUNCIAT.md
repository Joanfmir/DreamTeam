# NTT DATA 2023 · Cadena de subministrament sanitària

> **2023** · NTT DATA · Forecasting de compres + segmentació + visualització · Avaluació oberta · ✅ Dades públiques

## Context

La logística sanitària a Espanya és **reactiva**: es compra quan un producte s'esgota o es demana. Cada unitat compra pel seu compte, amb poca coordinació i sense una visió global. NTT DATA proposa avançar cap a un **sistema predictiu**: comprar per avançat i de manera unificada redueix els trencaments d'estoc, el cost i les **emissions de CO₂** del transport.

## La tasca

Objectiu principal: **establir un pla de compres per al 2023**. Idees que proposaven:
- **Predir la despesa** de l'any següent per producte.
- Detectar **patrons de compra** i combinacions de productes.
- Avaluar l'**impacte ambiental** d'un canvi d'estratègia de compra (comandes agrupades).

Enfocaments suggerits: **forecasting**, **segmentació** i **visualització**.

## Dades

`consumo_material_clean.xlsx`: 15.698 línies de compra de material sanitari d'un grup d'hospitals des del 2015.

| Columna | Descripció |
|---|---|
| `CODIGO` | Codi del producte (la primera lletra n'indica la classificació) |
| `PRODUCTO` | Descripció del producte |
| `FECHAPEDIDO` | Data de la comanda (dd/mm/aa) |
| `NUMERO` | Número de comanda/any |
| `REFERENCIA` | Referència |
| `CANTIDADCOMPRA` | Unitats de producte comprades |
| `UNIDADESCONSUMOCONTENIDAS` | Unitats de consum que conté cada producte |
| `PRECIO` | Preu unitari (€) |
| `IMPORTELINEA` | Cost total de la línia (€) |
| `TIPOCOMPRA` | Tipus de compra pública (compra menor, contracte…) |
| `ORIGEN` | Origen (codi d'hospital o unitat) |
| `TGL` | Tipus de gestió logística (`ALMACENABLE`, `TRANSITO`) |

## Avaluació

Oberta: valor del pla de compres, qualitat del forecasting, insights i presentació.

## Estratègia guanyadora

1. **Agregar a sèrie temporal mensual** per producte o família (primera lletra de `CODIGO`).
2. **Forecasting jeràrquic**: global → família → producte. Començar amb models simples (naive estacional, ETS) i després LightGBM amb *lags* i mitjanes mòbils. Validar amb els últims 12 mesos.
3. **Segmentació ABC/XYZ**: importància per despesa (ABC) × regularitat de la demanda (XYZ). Diu on val la pena predir i on n'hi ha prou amb un estoc de seguretat.
4. **Pla de compres**: quantitat mensual = previsió + estoc de seguretat, i agrupar comandes → menys enviaments → **estimació de CO₂ estalviat**. És la part que lliga amb el "per què" de l'empresa.
5. **Dashboard**: despesa per família i any, productes amb més variabilitat, calendari de compres proposat.

## Com practicar-lo

```bash
bash scripts/download_past_data.sh ntt
```

```python
import pandas as pd
df = pd.read_excel("data/past/2023_ntt/consumo_material_clean.xlsx")
df["FECHA"] = pd.to_datetime(df["FECHAPEDIDO"], format="%d/%m/%y")
mensual = df.groupby([df["CODIGO"].str[0], pd.Grouper(key="FECHA", freq="MS")])["IMPORTELINEA"].sum()
```

## Fonts
- [data-students/datathon2023/ntt-data](https://github.com/data-students/datathon2023/tree/main/ntt-data) (`challenge.pptx`, `consumo_material_clean.xlsx`)
