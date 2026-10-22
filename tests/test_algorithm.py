"""Kern von Hand an der Mini-Instanz und an Sonderfällen: Kantenindex, Ziehen, Union-Find, Kurve, Theorie."""

import itertools
import math

import numpy as np
import pytest

import zgr_algorithm as A


def test_pair_index_follows_the_documented_order():
    assert [A.pair_from_index(k) for k in range(10)] == [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4)]


def test_pair_index_is_the_inverse_of_the_enumeration():
    n = 14
    expected = sorted(itertools.combinations(range(n), 2), key=lambda e: (e[1], e[0]))
    assert [A.pair_from_index(k) for k in range(A.n_pairs(n))] == expected


def test_vectorised_pair_index_matches_the_scalar_one_including_the_float_boundaries():
    ks = list(range(0, 3000)) + [v * (v - 1) // 2 + d for v in (1000, 5000, 30000, 30001, 60000) for d in (-1, 0, 1)]
    got = A.pairs_from_indices(np.array(ks))
    assert [tuple(map(int, row)) for row in got] == [A.pair_from_index(k) for k in ks]


def test_n_pairs():
    assert [A.n_pairs(n) for n in (0, 1, 2, 3, 4, 10)] == [0, 0, 1, 3, 6, 45]


def test_sample_edges_are_distinct_valid_and_deterministic():
    e1 = A.sample_edges(30, 100, np.random.default_rng(5))
    e2 = A.sample_edges(30, 100, np.random.default_rng(5))
    assert e1.shape == (100, 2) and (e1 == e2).all()
    assert len({tuple(r) for r in e1}) == 100 and (e1[:, 0] < e1[:, 1]).all() and e1.min() >= 0 and e1.max() < 30


def test_sample_edges_sizes_at_the_limits():
    rng = np.random.default_rng(1)
    assert A.sample_edges(10, 0, rng).shape == (0, 2)
    assert A.sample_edges(10, -3, rng).shape == (0, 2)
    full = A.sample_edges(6, 999, rng)  # mehr als möglich: alle 15 Paare
    assert sorted(map(tuple, full.tolist())) == sorted(itertools.combinations(range(6), 2))


def test_gnp_extremes():
    rng = np.random.default_rng(2)
    assert len(A.gnp_edges(12, 0.0, rng)) == 0
    assert len(A.gnp_edges(12, 1.0, rng)) == A.n_pairs(12)
    assert len(A.gnp_edges(1, 0.7, rng)) == 0
    assert len(A.gnp_edges(12, 5.0, rng)) == A.n_pairs(12)  # p wird auf 1 begrenzt


def test_edge_probability():
    assert A.edge_probability(101, 2.0) == pytest.approx(0.02)
    assert A.edge_probability(1, 3.0) == 0.0
    assert A.edge_probability(5, 99.0) == 1.0
    assert A.edge_probability(50, -1.0) == 0.0


def test_component_sizes_of_the_mini_instance(mini):
    n, edges = mini
    assert A.component_sizes(n, edges) == [3, 3, 1]
    labels = A.component_labels(n, edges)
    assert labels[0] == labels[1] == labels[2] and labels[3] == labels[4] == labels[5]
    assert len({labels[0], labels[3], labels[6]}) == 3


def test_union_returns_the_size_and_ignores_repeated_edges():
    parent, size = list(range(5)), [1] * 5
    assert A.union(parent, size, 0, 1) == 2
    assert A.union(parent, size, 1, 0) == 2  # zweite Vereinigung derselben Komponenten ändert nichts
    assert A.union(parent, size, 2, 3) == 2
    assert A.union(parent, size, 1, 3) == 4
    assert size[A.find(parent, 0)] == 4


def test_find_halves_the_path_and_keeps_the_root():
    parent = [0, 0, 1, 2, 3]  # Kette 4 → 3 → 2 → 1 → 0
    assert A.find(parent, 4) == 0
    assert parent[4] == 2  # Pfadhalbierung: 4 zeigt jetzt auf seinen Großvater


def test_is_connected(mini):
    n, edges = mini
    assert not A.is_connected(n, edges)
    assert A.is_connected(3, [(0, 1), (1, 2)])
    assert A.is_connected(1, [])
    assert not A.is_connected(2, [])


def test_mean_finite_cluster_by_hand(mini):
    n, edges = mini
    assert A.mean_finite_cluster(A.component_sizes(n, edges)) == pytest.approx(2.5)  # ohne die größte: Größen 3 und 1 → (9 + 1) / 4
    assert A.mean_finite_cluster([5]) == 0.0
    assert A.mean_finite_cluster([4, 1, 1, 1]) == pytest.approx(1.0)


def test_giant_curve_by_hand(mini):
    n, edges = mini
    assert A.giant_curve(n, edges) == [1, 2, 3, 3, 3, 3]
    assert A.giant_curve(0, []) == []
    assert A.giant_curve(4, []) == [1]


def test_giant_curve_agrees_with_recomputing_every_prefix():
    rng = np.random.default_rng(9)
    for _ in range(10):
        edges = A.sample_edges(40, 60, rng)
        curve = A.giant_curve(40, edges)
        assert curve == [A.component_sizes(40, edges[:t])[0] for t in range(len(edges) + 1)]


def test_giant_theory_known_values():
    assert A.giant_theory(0.5) == 0.0 and A.giant_theory(1.0) == 0.0
    assert A.giant_theory(2.0) == pytest.approx(0.7968121300, abs=1e-9)
    assert A.giant_theory(3.0) == pytest.approx(0.9404798, abs=1e-6)


@pytest.mark.parametrize("c", [1.01, 1.1, 1.5, 2.0, 4.0, 10.0])
def test_giant_theory_solves_its_fixed_point_equation(c):
    s = A.giant_theory(c)
    assert 0 < s < 1
    assert s == pytest.approx(1 - math.exp(-c * s), abs=1e-9)


def test_giant_theory_is_increasing():
    values = [A.giant_theory(c) for c in (1.05, 1.2, 1.5, 2.0, 3.0, 5.0)]
    assert values == sorted(values)


def test_finite_cluster_theory_by_hand():
    assert A.finite_cluster_theory(0.5) == pytest.approx(2.0)  # 1 / (1 − 0,5)
    assert A.finite_cluster_theory(0.8) == pytest.approx(5.0)
    assert A.finite_cluster_theory(1.0) == math.inf
    s = A.giant_theory(2.0)
    assert A.finite_cluster_theory(2.0) == pytest.approx(1 / (1 - 2 * (1 - s)))


def test_connectivity_theory_and_critical_scale():
    assert A.connectivity_theory(1000, math.log(1000)) == pytest.approx(math.exp(-1))  # x = 0
    assert A.connectivity_theory(1000, math.log(1000) + 20) == pytest.approx(1.0)
    assert A.critical_scale(1000) == pytest.approx(100.0)
    assert A.critical_scale(8) == pytest.approx(4.0)


def test_row_correct_repairs_an_estimate_that_is_off_by_one_in_either_direction():
    ks = np.array(list(range(0, 400)) + [v * (v - 1) // 2 + d for v in (50, 1000, 30000) for d in (-1, 0, 1) if v * (v - 1) // 2 + d >= 0], dtype=np.int64)
    truth = np.array([A.pair_from_index(int(k))[1] for k in ks], dtype=np.int64)
    assert (A.row_correct(truth, ks) == truth).all()
    assert (A.row_correct(truth + 1, ks) == truth).all()
    assert (A.row_correct(np.maximum(truth - 1, 1), ks)[truth > 1] == truth[truth > 1]).all()


def test_row_estimate_is_exact_for_small_indices_and_never_further_off_than_one():
    ks = np.arange(0, 5000, dtype=np.int64)
    truth = np.array([A.pair_from_index(int(k))[1] for k in ks], dtype=np.int64)
    assert (A.row_estimate(ks) == truth).all()
    big = np.array([v * (v - 1) // 2 + d for v in (5000, 30000, 60000) for d in (-1, 0, 1)], dtype=np.int64)
    big_truth = np.array([A.pair_from_index(int(k))[1] for k in big], dtype=np.int64)
    assert (np.abs(A.row_estimate(big) - big_truth) <= 1).all()


def test_the_smallest_networks_have_the_documented_edge_probability_and_edges():
    assert A.edge_probability(2, 0.5) == 0.5 and A.edge_probability(2, 3.0) == 1.0
    assert len(A.gnp_edges(2, 1.0, np.random.default_rng(1))) == 1 and len(A.gnp_edges(2, 0.0, np.random.default_rng(1))) == 0


def test_is_connected_of_the_empty_graph_is_false_and_does_not_raise():
    assert A.is_connected(0, []) is False
