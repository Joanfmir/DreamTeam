"""Solver de Qualcomm 2022: encadenar tots els power switches minimitzant la longitud mitjana.

    bash scripts/download_past_data.sh qualcomm
    python reptes/2022_qualcomm_power_switches/solver.py                    # tots els testcases
    python reptes/2022_qualcomm_power_switches/solver.py data/past/2022_qualcomm/priv_testcase0.def

Idea: longitud mitjana = longitud total / k. Més cadenes reparteixen el total entre més, però cada
cadena afegeix el tram d'anada i tornada al seu driver: és un compromís, no una regla. Per això
provem diversos k (2–16) i ens quedem el millor:
  1. Agrupar els pins en k clústers equilibrats (k-means)
  2. Triar quins drivers fer servir i assignar-los als clústers amb l'algorisme hongarès
  3. Ordenar cada clúster: "serp" (ràpid, qualsevol mida) + 2-opt si és petit
  4. Escriure la sortida i validar-la amb el script oficial `solution_metrics.py`
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from dreamteam import optim

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/past/2022_qualcomm"
OUT = ROOT / "submissions/qualcomm"


def parse_def(path: Path):
    """Retorna (drivers d'entrada, drivers de sortida, pins) com a llistes de (nom, x, y)."""
    inputs, outputs, pins = [], [], []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("- DRIVERPIN_"):
            name = line.split()[1]
            x, y = (int(t) for t in lines[i + 2].split() if t.lstrip("-").isdigit())
            (inputs if "DIRECTION INPUT" in line else outputs).append((name, x, y))
        elif line.startswith("im_psyched"):
            parts = line.split()
            x, y = (int(t) for t in parts if t.isdigit())
            pins.append((parts[0], x, y))
    return inputs, outputs, pins


def solve(inputs, outputs, pins, k: int = 16, two_opt_max: int = 1500):
    xy = np.array([(x, y) for _, x, y in pins], dtype=float)
    k = min(k, len(inputs), len(outputs), len(pins))
    labels = optim.balanced_kmeans_labels(xy, k)
    centers = np.array([xy[labels == c].mean(axis=0) for c in range(k)])

    # Cost clúster ↔ driver (Manhattan al centre). Matriu k × 16: l'hongarès tria els k millors drivers
    ins = np.array([(x, y) for _, x, y in inputs], dtype=float)
    outs = np.array([(x, y) for _, x, y in outputs], dtype=float)
    _, in_of = linear_sum_assignment(np.abs(centers[:, None] - ins[None]).sum(-1))
    _, out_of = linear_sum_assignment(np.abs(centers[:, None] - outs[None]).sum(-1))

    chains = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        pts = np.vstack([ins[in_of[c]], xy[idx], outs[out_of[c]]])
        inner = optim.serpentine_order(pts[1:-1])
        # orientar la serp: començar per l'extrem més proper al driver d'entrada
        if len(inner) and np.abs(pts[1:-1][inner[0]] - pts[0]).sum() > np.abs(pts[1:-1][inner[-1]] - pts[0]).sum():
            inner = inner[::-1]
        order = np.concatenate([[0], inner + 1, [len(pts) - 1]])
        if len(pts) <= two_opt_max:
            order = optim.two_opt(pts, order, fixed_ends=True)
        chain = [inputs[in_of[c]][0]] + [pins[idx[j - 1]][0] for j in order[1:-1]] + [outputs[out_of[c]][0]]
        chains.append((chain, optim.route_length(pts, order)))
    return chains


def write_output(chains, path: Path):
    """Format oficial: cada net connecta (pin conn_in) → (pin següent conn_out)."""
    with open(path, "w") as f:
        for chain, _ in chains:
            for a, b in zip(chain[:-1], chain[1:], strict=True):
                f.write(f"- NET\n  ( {a} conn_in )\n  ( {b} conn_out )\n;\n")


if __name__ == "__main__":
    files = [Path(p) for p in sys.argv[1:]] or sorted(D.glob("*testcase*.def")) + [D / "example_input.def"]
    OUT.mkdir(parents=True, exist_ok=True)
    for path in files:
        inputs, outputs, pins = parse_def(path)
        t0 = time.perf_counter()
        candidates = {k: solve(inputs, outputs, pins, k=k) for k in (16, 12, 8, 4)}
        best_k = min(candidates, key=lambda k: np.mean([length for _, length in candidates[k]]))
        chains = candidates[best_k]
        dt = time.perf_counter() - t0
        tried = " · ".join(f"k={k}: {np.mean([le for _, le in c]) / 1e6:.1f}M" for k, c in candidates.items())
        out = OUT / f"{path.stem}_solution.def"
        write_output(chains, out)
        avg = np.mean([length for _, length in chains])
        print(f"{path.name:<24} {len(pins):>7,} pins · {len(chains)} cadenes · "
              f"longitud mitjana {avg:,.0f} · {dt:.1f} s   ({tried})")
        official = D / "solution_metrics.py"
        if official.exists() and len(pins) <= 10_000:  # validació oficial (lenta en els casos grans)
            res = subprocess.run([sys.executable, str(official), str(path), str(out)],
                                 capture_output=True, text=True)
            summary = [line for line in res.stdout.splitlines() if "verage" in line or "rror" in line]
            print("   script oficial →", " | ".join(summary) or res.stdout.strip()[-200:] or res.stderr[-300:])
