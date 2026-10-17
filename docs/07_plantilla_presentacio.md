# 07 · Plantilla de presentació (5–10 min)

Construïda a partir de les rúbriques dels anys anteriors: negoci, insights per a no tècnics, creativitat, demo i "què faríem amb més temps". Copieu-la a Google Slides **el dia 1 al matí** i l'aneu omplint, no el diumenge a última hora.

| # | Slide | Contingut | Temps |
|---|---|---|---|
| 1 | **Títol** | Nom del projecte (no el del repte), equip, enllaç al repo i QR | 10 s |
| 2 | **El problema** | Una frase en llenguatge de l'empresa: *"Mango perd vendes quan fabrica poc i…"*. Amb un número: quants diners o unitats hi ha en joc | 40 s |
| 3 | **Insight sorpresa** | **Un sol gràfic** de l'EDA que ningú esperava. Títol = la conclusió, no la descripció ("Les oportunitats antigues es perden 3× més") | 60 s |
| 4 | **La nostra solució** | Diagrama de 4–5 caixes: dades → features clau → model → ajust a la mètrica → producte | 60 s |
| 5 | **Resultats** | Taula: baseline vs. el nostre model, amb la mètrica oficial i la **validació honesta**. Traduïda a negoci ("−18 % de vendes perdudes") | 60 s |
| 6 | **Per què funciona** | Explicabilitat: 3 factors principals (SHAP) en frases, no en gràfics tècnics | 45 s |
| 7 | **Demo** | En directe (Streamlit) **o vídeo de 30 s per si falla la wifi** | 90 s |
| 8 | **Limitacions i què faríem amb més temps** | 3 punts honestos (Schneider ho puntuava a part) | 30 s |
| 9 | **Gràcies** | Equip, repo i QR | 10 s |

## Regles

- **Un missatge per slide.** Si cal llegir-la, té massa text.
- Gràfics amb **lletra gran** (≥ 18 pt als eixos) i colors consistents.
- **Res de codi a les slides.** El codi és al repo, i el repo és a l'última slide.
- Números rodons: "0,837 → +2 punts d'AUC", no "0,836572".
- Assajar **dues vegades amb cronòmetre**. Qui presenta no és qui ha programat el model: així s'explica més clar.
- Preparar 3 respostes per a les preguntes típiques del jurat:
  1. *Com sabeu que no hi ha overfitting o leakage?* → validació (temporal o agrupada).
  2. *Com ho posaríeu en producció?* → la demo i el temps d'inferència.
  3. *Què és el que menys us convenç?* → les limitacions de la slide 8.

## README del repo (també puntua)

Estructura mínima que ha de tenir el README del repo de la Datathon el diumenge:

```markdown
# <Nom del projecte> · Datathon FME 2026 · Repte <Empresa>
<Una frase: què fa i per què importa>
![demo](docs/demo.gif)

## Resultats      ← taula amb la mètrica (validació i leaderboard)
## Com funciona   ← diagrama del pipeline i 3 decisions clau
## Com executar-ho ← 3 comandes: install, train, app
## Estructura del repo
## Equip
```
