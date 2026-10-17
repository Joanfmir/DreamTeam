# UPC Sostenible 2023 · Com venen els estudiants a la UPC?

> **2023** · UPC Sostenible · Anàlisi de factors + visualització de fluxos + propostes · Avaluació oberta · ✅ Dades públiques · 🥇 Guanyador amb repo públic

## Context

UPC Sostenible és l'oficina de sostenibilitat de la UPC: acció climàtica, transició energètica i **mobilitat sostenible**. La UPC té 9 campus (Diagonal Nord, Diagonal Sud, Besòs, Nàutica, Sant Cugat, Terrassa, Castelldefels, Vilanova i Manresa), i cadascun té un context de mobilitat diferent.

Categories de mobilitat: **activa** (a peu, bici), **pública** (metro, bus, tren, tram) i **privada** (cotxe, moto).

## La tasca

1. **(Obligatori)** Explicar **quins factors determinen** si els estudiants fan servir transport públic o privat (i actiu).
2. **(Triar 2, 3 o tots dos)**
   - **Visualització dels fluxos** diaris d'estudiants per categoria de transport. Ha de respondre preguntes com: quines línies fan servir, quant triguen, quines zones estan mal servides, on hi ha discontinuïtats als carrils bici, si el Bicing o l'AMBici tenen prou capacitat…
   - **Propostes** per **reduir emissions de CO₂**: noves línies, freqüències, millores de la xarxa o del bicing. Han de ser atractives per als usuaris: cal analitzar què els fa descartar el transport públic.

## Dades

| Fitxer | Contingut |
|---|---|
| `enquesta_mobilitat_2022.xlsx` | 1.448 respostes de l'enquesta de mobilitat: perfil, campus, curs, dies per setmana, **codi postal d'origen**, **modes de transport per etapes** (anada i tornada, fins a 3 etapes) i **motius** (més ràpid, més barat, més còmode, única opció, menys impacte, més saludable, necessito el vehicle…) |
| `estudiants_residencia_22-23.xlsx` | 4.814 files: estudiants matriculats per centre UPC i municipi o codi postal de residència durant el curs |

**Fonts externes recomanades:**
- Open Data BCN (transport), Observatori de la Mobilitat de Catalunya.
- Dades obertes de Renfe (Rodalies, horaris i estacions), TMB (API), FGC, TRAM.
- Enquesta EMEF, fitxers **GTFS** ([gtfs.org](https://gtfs.org)).

## Avaluació

Oberta: rigor de l'anàlisi de factors, qualitat i utilitat de la visualització, viabilitat i impacte de les propostes, i presentació.

## Estratègia guanyadora

**🥇 1r premi:** [JordinaGR/datathon-FME-2023](https://github.com/JordinaGR/datathon-FME-2023), "Optimització de Rodalies" ([Devpost](https://devpost.com/software/optimitzacio-de-rodalies)):
- **Van generar dades pròpies**: un script amb **Selenium i PyAutoGUI** que feia **scraping de Google Maps** per obtenir els temps de trajecte reals en transport públic des de cada origen.
- Després neteja amb pandas i NumPy, gràfics amb matplotlib i conclusions sobre Rodalies.
- **Lliçó: no cal un model complex.** Una pregunta concreta, una dada que ningú més tenia i una proposta clara.

**El nostre pla:**
1. **Factors (part 1)**: model interpretable (regressió logística o LightGBM + SHAP) de "públic vs. privat" segons el campus, la distància (calculada amb el codi postal), els dies per setmana, el curs i els motius. Amb taules creuades simples n'hi ha prou per començar.
2. **Temps de trajecte** amb la [Google Directions API](https://developers.google.com/maps/documentation/directions) o OpenTripPlanner amb GTFS (més net que fer scraping), per comparar públic vs. cotxe per codi postal.
3. **Mapa**: fluxos origen → campus (folium o kepler.gl), amb el gruix proporcional als estudiants i el color segons el mode.
4. **Proposta quantificada**: "si la línia X tingués freqüència Y, Z estudiants guanyarien N minuts → W tones de CO₂ menys a l'any".

## Com practicar-lo

```bash
bash scripts/download_past_data.sh upc
```

## Fonts
- [data-students/datathon2023/upc-sostenible](https://github.com/data-students/datathon2023/tree/main/upc-sostenible) (`challenge.pdf` i els dos Excel)
