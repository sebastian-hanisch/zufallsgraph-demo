"""Randfälle vor dem ersten Mutationstest: n = 1 und 2, c = 0, vollständiger Graph, Gleichstände der Komponentengrößen, Grenzen der Phasen."""

import numpy as np

import zgr_algorithm as A
import zgr_evaluation as ev
import zgr_visualization as V


def test_single_node_and_two_nodes():
    assert A.component_sizes(1, []) == [1] and A.giant_curve(1, []) == [1]
    a = ev.analyse(ev.Settings(n=1, c=3.0, seed=1))
    assert a.sizes == (1,) and a.giant == 1.0 and a.connected and a.second == 0
    two = A.component_sizes(2, [(0, 1)])
    assert two == [2] and A.is_connected(2, [(0, 1)])


def test_no_edges_gives_only_singletons():
    a = ev.analyse(ev.Settings(n=100, c=0.0, seed=1))
    assert a.sizes == (1,) * 100 and a.giant == 0.01 and not a.connected and a.second == 1
    assert a.mean_finite == 1.0  # lauter Einzelknoten außerhalb der „größten“: Σ 1 / Σ 1


def test_complete_graph_is_one_component():
    a = ev.analyse(ev.Settings(n=30, c=1000.0, seed=1))
    assert a.connected and a.graph.m == A.n_pairs(30)


def test_equal_sized_components_break_ties_by_order_only():
    sizes = A.component_sizes(6, [(0, 1), (2, 3), (4, 5)])
    assert sizes == [2, 2, 2] and A.mean_finite_cluster(sizes) == 2.0
    assert A.giant_curve(6, [(0, 1), (2, 3), (4, 5)]) == [1, 2, 2, 2]


def test_repeated_and_reversed_edges_do_not_change_the_components():
    assert A.component_sizes(4, [(0, 1), (1, 0), (0, 1), (2, 3)]) == [2, 2]


def test_giant_theory_just_above_the_threshold_is_small_but_positive():
    assert 0 < A.giant_theory(1.0001) < 0.001
    assert A.giant_theory(1.0) == 0.0


def test_the_figure_of_an_empty_edge_list_has_no_edge_segments():
    state = {"labels": list(range(5)), "edges": np.zeros((0, 2), dtype=int), "sizes": [1] * 5}
    fig = V.build_process(state, 5)
    assert len(fig.data[0].x) == 0 and len(fig.data[1].x) == 5


def test_the_size_chart_handles_a_connected_graph():
    fig = V.build_sizes([40], 40, 9.0)
    assert [int(x) for x in fig.data[0].x] == [40] and list(fig.data[0].y) == [1]
