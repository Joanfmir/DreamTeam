"""Genera un dataset SINTÈTIC amb el mateix esquema que el de Mango 2025.

Les dades reals eren privades (Kaggle de l'esdeveniment). Aquest simulador reprodueix les
columnes de l'enunciat i els fenòmens importants perquè puguem entrenar el pipeline:
  · productes nous cada temporada (cold start) → cal trobar peces semblants per embedding
  · demanda que depèn de família, preu, botigues, talles, cicle de vida i "estil" (embedding)
  · vendes censurades per la producció (no es pot vendre més del que es fabrica)
  · una fila per producte i setmana (weekly_sales, weekly_demand)

    python reptes/2025_mango_demanda/simulate_data.py      # → data/synthetic/mango2025/{train,test}.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/synthetic/mango2025"
rng = np.random.default_rng(2025)

SEASONS = {86: "SS23", 87: "FW23", 88: "SS24", 89: "FW24", 90: "SS25"}  # 90 = temporada de test
FAMILIES = {  # família: (categoria, demanda base, temporada preferida: 0 = PV, 1 = TH, None = totes)
    "Dresses": ("Dresses", 1.2, 0), "T-shirts": ("Tops", 1.5, None), "Shirts": ("Tops", 0.9, None),
    "Jeans": ("Bottoms", 1.1, None), "Trousers": ("Bottoms", 0.8, None), "Coats": ("Outerwear", 0.9, 1),
    "Jackets": ("Outerwear", 0.7, None), "Sweaters": ("Tops", 1.0, 1), "Swimwear": ("Beachwear", 0.8, 0),
    "Sandals": ("Shoes", 0.6, 0), "Boots": ("Shoes", 0.6, 1), "Bags": ("Accessories", 0.5, None),
}
ATTRS = {
    "fabric": ["Cotton", "Denim", "Synthetic Leather", "Wool", "Linen", "Polyester"],
    "color_name": ["Black", "White", "Blue", "Beige", "Red", "Green", "Pink", "Brown"],
    "length_type": ["Midi", "Crop", "Regular", "Long", "Mini"],
    "silhouette_type": ["Slim", "Oversize", "Regular", "Straight", "Flared"],
    "sleeve_length_type": ["Long", "Short", "Sleeveless", "Three-quarter"],
    "print_type": ["Plain", "Striped", "Floral", "Check", "Animal"],
    "archetype": ["Young", "Classic", "Trendy", "Basic"],
    "moment": ["Casual", "Formal Work", "Party", "Beach"],
}
EMB_DIM = 32
N_STYLES = 15  # "estils" latents: alguns venen molt més que d'altres


def simulate(n_per_season: int = 1500) -> pd.DataFrame:
    style_centers = rng.normal(size=(N_STYLES, EMB_DIM))
    style_effect = rng.normal(0, 0.45, N_STYLES)          # el que fa que un estil sigui "hit"
    rows, pid = [], 0
    for season, code in SEASONS.items():
        is_fw = code.startswith("FW")
        year = 2000 + int(code[2:])
        fams = [f for f, (_, _, pref) in FAMILIES.items() if pref is None or pref == int(is_fw)]
        for _ in range(n_per_season):
            pid += 1
            fam = rng.choice(fams)
            cat, base, _ = FAMILIES[fam]
            style = rng.integers(N_STYLES)
            emb = style_centers[style] + rng.normal(0, 0.6, EMB_DIM)
            price = round(float(np.exp(rng.normal(np.log(40), 0.45))), 2)
            stores = int(rng.integers(50, 1200))
            sizes = int(rng.integers(1, 8))
            life = int(rng.integers(6, 20))
            attrs = {k: rng.choice(v) for k, v in ATTRS.items()}
            log_d = (np.log(base * 2000) + style_effect[style] + 0.55 * np.log(stores / 400)
                     + 0.15 * sizes - 0.6 * np.log(price / 40) + 0.04 * life
                     + (0.25 if attrs["color_name"] in ("Black", "White") else 0) + rng.normal(0, 0.35))
            demand_total = float(np.exp(log_d))
            production = demand_total * rng.uniform(0.6, 1.4)  # els compradors de Mango també s'equivoquen
            # Perfil setmanal: pujada, pic i baixada
            w = np.arange(life)
            profile = np.exp(-0.5 * ((w - life * 0.35) / (life * 0.3)) ** 2)
            weekly_demand = demand_total * profile / profile.sum()
            stock, weekly_sales = production, []
            for d in weekly_demand:
                s = min(d * rng.uniform(0.9, 1.1), stock)
                weekly_sales.append(s)
                stock -= s
            start = pd.Timestamp(f"{year}-{'09' if is_fw else '02'}-01") + pd.Timedelta(weeks=int(rng.integers(0, 6)))
            start -= pd.Timedelta(days=start.dayofweek)  # dilluns
            common = dict(
                id=pid, id_season=season, aggregated_family=cat, family=fam, category=cat,
                color_rgb=",".join(map(str, rng.integers(0, 256, 3))),
                image_embedding=",".join(f"{x:.4f}" for x in emb),
                phase_in=start.date(), phase_out=(start + pd.Timedelta(weeks=life - 1)).date(),
                life_cycle_length=life, num_stores=stores, num_sizes=sizes,
                has_plus_size=int(rng.random() < 0.2), price=price, production=production, **attrs,
            )
            for i in range(life):
                date = start + pd.Timedelta(weeks=i)
                rows.append({**common, "year": date.year, "num_week_iso": date.isocalendar().week,
                             "weekly_sales": weekly_sales[i], "weekly_demand": weekly_demand[i]})
    df = pd.DataFrame(rows)
    # Com a l'enunciat: vendes, demanda i producció escalades a [0, 1] mantenint proporcions
    scale = df["weekly_demand"].max()
    for col in ("weekly_sales", "weekly_demand", "production"):
        df[col] = df[col] / scale
    return df


if __name__ == "__main__":
    df = simulate()
    OUT.mkdir(parents=True, exist_ok=True)
    test_season = max(SEASONS)
    train = df[df["id_season"] != test_season]
    test = (df[df["id_season"] == test_season]
            .drop(columns=["weekly_sales", "weekly_demand", "production", "year", "num_week_iso"])
            .drop_duplicates("id"))
    # Solució oculta del test (per poder avaluar-nos com faria el Kaggle)
    solution = df[df["id_season"] == test_season].groupby("id", as_index=False)["weekly_demand"].sum()
    solution = solution.rename(columns={"id": "ID", "weekly_demand": "demand"})
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    solution.to_csv(OUT / "solution_hidden.csv", index=False)
    print(f"✓ train: {len(train):,} files ({train.id.nunique():,} productes) · "
          f"test: {len(test):,} productes nous → {OUT.relative_to(ROOT)}/")
