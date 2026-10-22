"""Kern: Zufallsgraph G(n, p), Komponenten per Union-Find, Riesenkomponente, Theorie (Fixpunkt, Zusammenhangsschwelle).

Jede Regel ist eine eigene kleine Funktion (DEMO-PLAYBOOK §1b); Zufall nur über einen übergebenen Generator."""

import math

import numpy as np


# --- Kanten erzeugen -----------------------------------------------------------------------------------------------------------------------


def n_pairs(n: int) -> int:
    """Zahl der möglichen Kanten eines einfachen Graphen mit n Knoten."""
    return n * (n - 1) // 2


def pair_from_index(k: int) -> tuple[int, int]:
    """Die k-te Kante (0-basiert) in der Reihenfolge (0,1), (0,2), (1,2), (0,3), ...; Rückgabe (u, v) mit u < v."""
    v = (1 + math.isqrt(1 + 8 * k)) // 2
    u = k - v * (v - 1) // 2
    return u, v


def row_estimate(ks: np.ndarray) -> np.ndarray:
    """Schätzung des größeren Knotens v je Kantenindex aus der Gleitkomma-Wurzel (kann an den Grenzen um eins danebenliegen)."""
    return ((1 + np.sqrt(1 + 8 * ks.astype(np.float64))) // 2).astype(np.int64)


def row_correct(v: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Korrigiert eine um höchstens eins falsche Schätzung auf die ganzzahlige Grenze: v(v−1)/2 <= k < (v+1)v/2."""
    v = np.where(v * (v - 1) // 2 > ks, v - 1, v)
    return np.where((v + 1) * v // 2 <= ks, v + 1, v)


def pairs_from_indices(ks: np.ndarray) -> np.ndarray:
    """Vektorisierte Umkehrung von `pair_from_index`: (m, 2)-Feld mit u < v."""
    ks = np.asarray(ks, dtype=np.int64)
    v = row_correct(row_estimate(ks), ks)
    u = ks - v * (v - 1) // 2
    return np.stack([u, v], axis=1)


def sample_edges(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """m verschiedene Kanten gleichverteilt aus allen möglichen, in zufälliger Reihenfolge, als (m, 2)-Feld."""
    total = n_pairs(n)
    m = max(0, min(int(m), total))
    if m == 0:
        return np.zeros((0, 2), dtype=np.int64)
    ks = rng.choice(total, size=m, replace=False)
    rng.shuffle(ks)
    return pairs_from_indices(ks)


def gnp_edges(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """G(n, p): jede der möglichen Kanten unabhängig mit Wahrscheinlichkeit p. Die Zahl der Kanten ist binomial, die Kanten selbst sind gleichverteilt."""
    p = min(1.0, max(0.0, float(p)))
    m = int(rng.binomial(n_pairs(n), p)) if n >= 2 else 0
    return sample_edges(n, m, rng)


def edge_probability(n: int, c: float) -> float:
    """Kantenwahrscheinlichkeit p mit erwartetem mittleren Grad c: p = c / (n - 1) (höchstens 1)."""
    return 0.0 if n < 2 else min(1.0, max(0.0, c) / (n - 1))


# --- Komponenten ---------------------------------------------------------------------------------------------------------------------------


def find(parent: list[int], x: int) -> int:
    """Wurzel von x mit Pfadhalbierung."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent: list[int], size: list[int], a: int, b: int) -> int:
    """Vereinigt die Komponenten von a und b nach Größe; Rückgabe: Größe der entstandenen (oder bestehenden) Komponente."""
    ra, rb = find(parent, a), find(parent, b)
    if ra == rb:
        return size[ra]
    if size[ra] < size[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    size[ra] += size[rb]
    return size[ra]


def component_labels(n: int, edges) -> list[int]:
    """Komponentenwurzel je Knoten."""
    parent = list(range(n))
    size = [1] * n
    for u, v in edges:
        union(parent, size, int(u), int(v))
    return [find(parent, x) for x in range(n)]


def component_sizes(n: int, edges) -> list[int]:
    """Größen aller Komponenten, absteigend sortiert (Einzelknoten zählen als Komponente der Größe 1)."""
    labels = component_labels(n, edges)
    counts: dict[int, int] = {}
    for r in labels:
        counts[r] = counts.get(r, 0) + 1
    return sorted(counts.values(), reverse=True)


def giant_curve(n: int, edges) -> list[int]:
    """Größe der größten Komponente nach jeder Kante: Eintrag t gilt nach den ersten t Kanten (Eintrag 0: ein Einzelknoten, bei n = 0 leer)."""
    if n <= 0:
        return []
    parent = list(range(n))
    size = [1] * n
    best = 1
    out = [best]
    for u, v in edges:
        best = max(best, union(parent, size, int(u), int(v)))
        out.append(best)
    return out


def is_connected(n: int, edges) -> bool:
    """Ist der Graph zusammenhängend? (Ein leerer Graph mit einem Knoten ja, mit mehreren Knoten nein.)"""
    return n >= 1 and component_sizes(n, edges)[0] == n


def mean_finite_cluster(sizes: list[int]) -> float:
    """Größen-gewichtete mittlere Größe der Komponenten OHNE die größte: Σ s² / Σ s (0 ohne weitere Komponenten)."""
    rest = sizes[1:]
    total = sum(rest)
    return 0.0 if total == 0 else sum(s * s for s in rest) / total


# --- Theorie -------------------------------------------------------------------------------------------------------------------------------


def giant_theory(c: float) -> float:
    """Anteil der Riesenkomponente im Grenzfall n → ∞ bei mittlerem Grad c: die größte Lösung von S = 1 − exp(−c S) (0 für c ≤ 1)."""
    if c <= 1.0:
        return 0.0
    lo, hi = 1e-12, 1.0
    for _ in range(200):  # Bisektion von f(S) = S − 1 + exp(−c S) auf (0, 1]; f < 0 dicht über 0 und f(1) > 0
        mid = (lo + hi) / 2
        if mid - 1.0 + math.exp(-c * mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def finite_cluster_theory(c: float) -> float:
    """Erwartete Größe der Komponente eines zufälligen Knotens außerhalb der Riesenkomponente im Grenzfall: 1/(1 − c(1 − S)) (1/(1 − c) unterhalb der Schwelle)."""
    s = giant_theory(c)
    eff = c * (1.0 - s)
    return math.inf if eff >= 1.0 else 1.0 / (1.0 - eff)


def connectivity_theory(n: int, c: float) -> float:
    """Wahrscheinlichkeit, dass G(n, c/n) zusammenhängend ist, für große n: exp(−exp(−x)) mit x = c − ln n."""
    return math.exp(-math.exp(-(c - math.log(n))))


def critical_scale(n: int) -> float:
    """Größenordnung der größten Komponente am kritischen Punkt c = 1: n^(2/3)."""
    return float(n) ** (2.0 / 3.0)
