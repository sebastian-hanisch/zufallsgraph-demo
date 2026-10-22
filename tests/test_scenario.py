"""Szenario: deterministisch aus dem Zufalls-Seed, Kantenzahl im erwarteten Bereich, wachsendes Netz als G(n, m)-Prozess."""

import numpy as np

import zgr_algorithm as A
import zgr_scenario as S


def test_make_graph_is_deterministic_and_seed_dependent():
    a, b, c = S.make_graph(200, 1.5, 7), S.make_graph(200, 1.5, 7), S.make_graph(200, 1.5, 8)
    assert (a.edges == b.edges).all() and a.m == b.m
    assert not (a.m == c.m and (a.edges == c.edges).all())


def test_edge_count_is_near_the_expected_mean_degree():
    g = S.make_graph(2000, 3.0, 35)  # erwartet n·c/2 = 3000 Kanten, Standardabweichung etwa 55
    assert abs(g.m - 3000) < 5 * 55
    assert (g.edges[:, 0] < g.edges[:, 1]).all()


def test_process_has_the_rounded_number_of_edges_and_a_random_order():
    p = S.make_process(60, 3.0, 5)
    assert p.m == 90 and len({tuple(r) for r in p.edges}) == 90
    sorted_edges = sorted(map(tuple, p.edges.tolist()), key=lambda e: (e[1], e[0]))
    assert list(map(tuple, p.edges.tolist())) != sorted_edges  # nicht in der Aufzählungsreihenfolge


def test_process_prefix_is_a_uniform_graph_with_m_edges():
    """Die ersten m Kanten des Prozesses sind verschieden und liegen im Bereich: an der Kante-für-Kante-Ansicht hängt die Behauptung „G(n, m)“."""
    p = S.make_process(40, 2.0, 9)
    for m in (0, 1, 17, 40):
        prefix = p.edges[:m]
        assert len({tuple(r) for r in prefix}) == m
    assert A.giant_curve(p.n, p.edges)[0] == 1


def test_edges_for_mean_degree():
    assert S.edges_for_mean_degree(100, 3.0) == 150
    assert S.edges_for_mean_degree(60, 1.0) == 30
    assert S.edges_for_mean_degree(7, 1.0) == 4  # 3,5 gerundet
    assert S.edges_for_mean_degree(100, 0.0) == 0


def test_graph_without_edges_for_zero_mean_degree():
    g = S.make_graph(50, 0.0, 1)
    assert g.m == 0 and isinstance(g.edges, np.ndarray)
