# Qualcomm 2022 · "One driver to rule them all"

> **2022** · Qualcomm · **Optimització combinatòria** (TSP amb diversos viatjants) · Longitud mitjana de les cadenes + eficiència + visualització · ✅ Dades públiques

## Context

Per alimentar un xip cal repartir l'energia per tot el disseny amb **power switches**. Tots s'han d'**encadenar**: la sortida de cada un es connecta a l'entrada del següent. Com més ràpid s'encenen i s'apaguen, menys energia es consumeix, i això depèn de la **longitud de les cadenes**.

## La tasca

Agrupar els power switches en **cadenes independents** per reduir el temps d'encesa i apagada:
- Cada cadena **comença en un driver pin d'entrada i acaba en un de sortida**. Hi ha 32 drivers (16 + 16), i cadascun es pot fer servir **com a màxim un cop** → entre 2 i 16 cadenes.
- **Tots** els pins s'han de connectar **exactament un cop**.
- Objectiu: **minimitzar la longitud mitjana de les cadenes** amb distància **Manhattan**.

## Dades

Fitxers `.def` (format de disseny de xips):
```
- DRIVERPIN_0 + NET DRIVERPIN_0 + DIRECTION INPUT + USE SIGNAL
  + LAYER CIA ( -38 0 ) ( 38 790 )
  + FIX ( 0 199711 ) E ;                      ← coordenades del driver
im_psyched_..._0 come_and_join_..._0 + FIXED ( 2177143 631847 ) N;   ← un pin a encadenar
```

| Fitxer | Pins |
|---|---|
| `example_input.def` | 100 |
| `priv_testcase0..3.def` | 1.000 · 10.000 · 100.000 · 250.000 |

Hi ha 10 testcases privats per a l'avaluació final.

## Avaluació

1. Validesa: tots els pins encaminats un cop i totes les cadenes de driver d'entrada a driver de sortida.
2. **Longitud mitjana** de les cadenes (Manhattan).
3. **Eficiència** del programa: ha d'escalar a 250.000 pins.
4. **Bonus** per una visualització de la solució.

Script oficial: `solution_metrics.py input.def output.def`.

## Format d'entrega

```
- NET
  ( DRIVERPIN_0 conn_in )
  ( im_psyched_..._0 conn_out )
;
```

## Estratègia guanyadora

**Clau:** longitud mitjana = longitud total / nombre de cadenes *k*. Més cadenes reparteixen el total entre més, però cada cadena afegeix el tram fins als seus drivers, així que **no és automàtic que 16 sigui el millor**. El nostre solver prova k ∈ {16, 12, 8, 4}: amb aquestes dades, **k = 16 guanya sempre** (p. ex. testcase 1: 35,3 M amb k = 16 vs. 42,8 M amb k = 12).

**Equip amb repo públic:** [marc-herrero/Qualcomm-Challenge-](https://github.com/marc-herrero/Qualcomm-Challenge-):
- **clustering per àrea** per reduir la mida de cada subproblema;
- **TSP per clúster**;
- primer van provar un **algorisme genètic** (lent i subòptim) i després **simulated annealing**.

**El nostre solver ([`solver.py`](solver.py)), validat amb el script oficial:**
1. **K-means equilibrat** en k clústers (`optim.balanced_kmeans_labels`).
2. Tria dels k drivers i assignació clúster ↔ driver amb l'**algorisme hongarès** (`scipy.optimize.linear_sum_assignment`, matriu k × 16).
3. Ordre dins de cada clúster: **"serp"** O(n log n), que escala a 250.000 pins en ~2 s (`optim.serpentine_order`), més **2-opt** amb extrems fixos si el clúster és petit (`optim.two_opt`).

| Testcase | Pins | Longitud mitjana | Temps |
|---|---|---|---|
| example | 100 | **8,78 M** (la solució d'exemple oficial: 73,4 M → **8,4× millor**) | 0,1 s |
| 0 | 1.000 | 16,4 M | 2,3 s |
| 1 | 10.000 | 35,3 M | 19 s |
| 2 | 100.000 | 146 M | 2,7 s |
| 3 | 250.000 | 222 M | 5,9 s |

(Els temps inclouen provar els 4 valors de k.)

**Millores possibles:** Or-opt, 2-opt amb llistes de veïns (KD-tree) per als casos grans, una millor assignació de drivers (provar permutacions), i una **visualització** amb matplotlib (punts + línies per cadena) per al bonus.

## Com practicar-lo

```bash
bash scripts/download_past_data.sh qualcomm
python reptes/2022_qualcomm_power_switches/solver.py
```

## Fonts
- [data-students/datathon2022/qualcomm_challenge](https://github.com/data-students/datathon2022/tree/main/qualcomm_challenge)
