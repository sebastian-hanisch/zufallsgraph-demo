"""Orakel-Tests: Komponenten gegen scipy und networkx, exakte Aufzählung aller Graphen mit 5 Knoten, Fixpunkt gegen brentq, Messung gegen die Theorie."""

import itertools
import math

import numpy as np
import pytest
from scipy import optimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import zgr_algorithm as A


def scipy_sizes(n, edges):
    edges = np.asarray(edges).reshape(-1, 2)
    a = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    k, lab = connected_components(a, directed=False)
    return sorted(np.bincount(lab).tolist(), reverse=True)


def test_component_sizes_agree_with_scipy_on_random_graphs():
    rng = np.random.default_rng(100)
    for n, p in [(2, 0.5), (10, 0.1), (10, 0.5), (50, 0.02), (50, 0.05), (200, 0.01), (200, 0.008), (200, 0.03), (500, 0.003), (500, 0.002)] * 4:
        edges = A.gnp_edges(n, p, rng)
        assert A.component_sizes(n, edges) == scipy_sizes(n, edges), (n, p)


def test_component_sizes_agree_with_networkx():
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(101)
    for _ in range(25):
        n = int(rng.integers(5, 120))
        edges = A.gnp_edges(n, float(rng.uniform(0.005, 0.06)), rng)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(map(tuple, edges.tolist()))
        assert A.component_sizes(n, edges) == sorted((len(c) for c in nx.connected_components(g)), reverse=True)


def all_graphs(n):
    pairs = list(itertools.combinations(range(n), 2))
    for mask in range(1 << len(pairs)):
        yield [pairs[i] for i in range(len(pairs)) if mask >> i & 1]


def test_exact_enumeration_matches_the_sampler_for_five_nodes():
    """Alle 1024 Graphen mit 5 Knoten: exakte Wahrscheinlichkeit für „zusammenhängend“ und exakter Erwartungswert der größten Komponente bei p = 0,4;
    die gezogenen Graphen müssen innerhalb von vier Standardfehlern liegen."""
    n, p = 5, 0.4
    exact_conn, exact_giant, exact_sq = 0.0, 0.0, 0.0
    for edges in all_graphs(n):
        w = p ** len(edges) * (1 - p) ** (10 - len(edges))
        sizes = scipy_sizes(n, edges)
        exact_conn += w * (sizes[0] == n)
        exact_giant += w * sizes[0]
        exact_sq += w * sizes[0] ** 2
    sd_giant = math.sqrt(exact_sq - exact_giant ** 2)
    reps = 20000
    rng = np.random.default_rng(7)
    conn = giant = 0
    for _ in range(reps):
        sizes = A.component_sizes(n, A.gnp_edges(n, p, rng))
        conn += sizes[0] == n
        giant += sizes[0]
    se_conn = math.sqrt(exact_conn * (1 - exact_conn) / reps)
    assert abs(conn / reps - exact_conn) < 4 * se_conn
    assert abs(giant / reps - exact_giant) < 4 * sd_giant / math.sqrt(reps)


def test_sample_edges_are_uniform_over_all_pairs():
    """Zwei Kanten aus fünf Knoten: jedes der 45 Paare von Kanten gleich wahrscheinlich (Chi-Quadrat gegen 44 Freiheitsgrade, Schwelle weit im Rauschen)."""
    rng = np.random.default_rng(3)
    counts = {}
    reps = 22500
    for _ in range(reps):
        e = A.sample_edges(5, 2, rng)
        key = frozenset(map(tuple, e.tolist()))
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 45
    expected = reps / 45
    chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
    assert chi2 < 80  # 44 Freiheitsgrade: Mittel 44, die 99,9-%-Grenze liegt bei etwa 78


def test_binomial_edge_count_has_the_right_mean():
    rng = np.random.default_rng(11)
    n, p = 300, 0.01
    ms = [len(A.gnp_edges(n, p, rng)) for _ in range(300)]
    mean = A.n_pairs(n) * p
    sd = math.sqrt(A.n_pairs(n) * p * (1 - p) / len(ms))
    assert abs(np.mean(ms) - mean) < 4 * sd


@pytest.mark.parametrize("c", [1.001, 1.01, 1.1, 1.5, 2.0, 3.0, 5.0, 8.0])
def test_giant_theory_agrees_with_brentq(c):
    # Die positive Wurzel von s − 1 + exp(−c s) liegt zwischen (c − 1) und 2 (c − 1) / c^2 ... grob; sicher ist die Klammer [1e-9, 1]: f < 0 nahe 0 für c > 1.
    f = lambda s: s - 1 + math.exp(-c * s)  # noqa: E731
    lo = 1e-9
    assert f(lo) < 0 < f(1.0)
    root = optimize.brentq(f, lo, 1.0, xtol=1e-14)
    assert A.giant_theory(c) == pytest.approx(root, abs=1e-8)


@pytest.mark.parametrize("c", [0.5, 2.0, 3.0])
def test_large_random_graph_matches_the_theory(c):
    n = 20000
    rng = np.random.default_rng(int(c * 10))
    sizes = A.component_sizes(n, A.gnp_edges(n, A.edge_probability(n, c), rng))
    assert sizes[0] / n == pytest.approx(A.giant_theory(c), abs=0.02 if c > 1 else 0.01)


@pytest.mark.parametrize("c", [0.5, 1.5, 3.0])
def test_mean_finite_cluster_matches_the_theory(c):
    n = 20000
    vals = []
    for seed in range(6):
        rng = np.random.default_rng(1000 + seed)
        vals.append(A.mean_finite_cluster(A.component_sizes(n, A.gnp_edges(n, A.edge_probability(n, c), rng))))
    assert np.mean(vals) == pytest.approx(A.finite_cluster_theory(c), rel=0.08)
