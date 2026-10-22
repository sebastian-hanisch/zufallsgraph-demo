"""Szenario: Zufallsgraphen G(n, p) mit festem Zufalls-Seed und das schrittweise Wachsen eines kleinen Netzes (G(n, m)-Prozess)."""

from dataclasses import dataclass

import numpy as np

import zgr_algorithm as A
import zgr_constants as C


@dataclass(eq=False)
class Graph:
    n: int
    c: float
    edges: np.ndarray  # (m, 2), u < v

    @property
    def m(self) -> int:
        return int(len(self.edges))


def make_graph(n: int, c: float, seed: int) -> Graph:
    """Deterministisch aus (n, c, seed): gleiche Eingabe, gleiche Kanten."""
    rng = np.random.default_rng(seed)
    return Graph(n, c, A.gnp_edges(n, A.edge_probability(n, c), rng))


def make_process(n: int, c_max: float, seed: int) -> Graph:
    """Kanten in zufälliger Reihenfolge bis zum mittleren Grad c_max: die ersten m Kanten sind ein G(n, m)."""
    rng = np.random.default_rng(seed)
    m = int(round(c_max * n / 2))
    return Graph(n, c_max, A.sample_edges(n, m, rng))


def edges_for_mean_degree(n: int, c: float) -> int:
    """Kantenzahl m, bei der ein Netz mit n Knoten den mittleren Grad c hat (2m / n = c), gerundet."""
    return int(round(c * n / 2))
