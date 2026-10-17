"""Heurístiques d'optimització combinatòria (rutes, TSP, agrupacions).

No tot és ML: Qualcomm 2022 era un TSP amb diversos viatjants i AED 2024 una assignació
amb restriccions. Aquestes funcions donen una solució vàlida ràpida; després es millora.
"""

from __future__ import annotations

import numpy as np


def route_length(points: np.ndarray, order, metric: str = "manhattan") -> float:
    """Longitud d'una ruta que visita `points[order]` en ordre (sense tornar a l'inici)."""
    p = points[np.asarray(order)]
    d = np.diff(p, axis=0)
    return float(np.abs(d).sum() if metric == "manhattan" else np.sqrt((d ** 2).sum(axis=1)).sum())


def serpentine_order(points: np.ndarray, n_strips: int | None = None, axis: int = 0) -> np.ndarray:
    """Ordre en "serp" (boustrofèdon): franges verticals recorregudes amunt i avall alternativament.

    O(n log n): serveix per a milers o milions de punts i dona rutes raonables d'entrada.
    """
    n = len(points)
    if n == 0:
        return np.array([], dtype=int)
    n_strips = n_strips or max(1, int(np.sqrt(n / 2)))
    a, b = points[:, axis], points[:, 1 - axis]
    edges = np.quantile(a, np.linspace(0, 1, n_strips + 1)[1:-1])
    strip = np.searchsorted(edges, a)
    # dins de cada franja, ordre creixent o decreixent segons la paritat
    key_b = np.where(strip % 2 == 0, b, -b)
    return np.lexsort((key_b, strip))


def nearest_neighbour(points: np.ndarray, start: int = 0, metric: str = "manhattan") -> np.ndarray:
    """Ruta voraç pel veí més proper. O(n²): fer-la servir fins a uns quants milers de punts."""
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    p = metric == "manhattan"
    for _ in range(n - 1):
        diff = points - points[order[-1]]
        dist = np.abs(diff).sum(axis=1) if p else np.sqrt((diff ** 2).sum(axis=1))
        dist[visited] = np.inf
        nxt = int(np.argmin(dist))
        order.append(nxt)
        visited[nxt] = True
    return np.array(order)


def two_opt(points: np.ndarray, order, metric: str = "manhattan", max_passes: int = 20,
            fixed_ends: bool = True) -> np.ndarray:
    """Millora una ruta invertint trams mentre escurci (2-opt).

    Amb `fixed_ends=True` el primer i l'últim punt no es mouen (p. ex. drivers d'entrada i sortida).
    Vectoritzat per fila: viable fins a ~2.000 punts.
    """
    order = np.asarray(order).copy()
    n = len(order)
    if n < 4:
        return order

    def dist(a, b):
        d = points[a] - points[b]
        return np.abs(d).sum(axis=-1) if metric == "manhattan" else np.sqrt((d ** 2).sum(axis=-1))

    lo = 1 if fixed_ends else 0
    for _ in range(max_passes):
        improved = False
        for i in range(lo, n - 2):
            a, b = order[i - 1] if i > 0 else None, order[i]
            if a is None:
                continue
            js = np.arange(i + 1, n - 1)
            c, d = order[js], order[js + 1]
            delta = dist(a, c) + dist(b, d) - dist(a, b) - dist(c, d)
            k = int(np.argmin(delta))
            if delta[k] < -1e-9:
                j = js[k]
                order[i:j + 1] = order[i:j + 1][::-1]
                improved = True
        if not improved:
            break
    return order


def balanced_kmeans_labels(points: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """K-means i després reequilibri voraç perquè tots els grups tinguin ~n/k punts.

    Útil quan cada grup té una capacitat (equips de 4, rutes amb el mateix nombre de parades…).
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(points)
    centers = km.cluster_centers_
    cap = int(np.ceil(len(points) / k))
    d = np.abs(points[:, None, :] - centers[None, :, :]).sum(axis=2)
    labels = np.full(len(points), -1)
    counts = np.zeros(k, dtype=int)
    # assignar primer els punts amb més "pressa" (gran diferència entre millor i segona opció)
    srt = np.sort(d, axis=1)
    urgency = srt[:, 1] - srt[:, 0] if k > 1 else np.zeros(len(points))
    for i in np.argsort(-urgency):
        for c in np.argsort(d[i]):
            if counts[c] < cap:
                labels[i] = c
                counts[c] += 1
                break
    return labels
