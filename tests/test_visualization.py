"""Abbildungen: Kreislayout, Komponentenreihenfolge, Zählung der Größen, Spuren je Figur; alle Achsen gesperrt."""

import math

import numpy as np

import zgr_constants as C
import zgr_evaluation as ev
import zgr_visualization as V
from zgr_scenario import make_process


def test_size_counts_by_hand():
    assert V.size_counts([5, 1, 1, 2, 1, 2]) == [(1, 3), (2, 2), (5, 1)]
    assert V.size_counts([]) == []


def test_circle_layout_by_hand():
    pos = V.circle_layout([2, 0, 1, 3])
    assert pos[2] == (1.0, 0.0)
    assert math.isclose(pos[0][0], 0.0, abs_tol=1e-12) and math.isclose(pos[0][1], 1.0)
    assert math.isclose(pos[1][0], -1.0) and math.isclose(pos[3][1], -1.0)


def test_component_order_puts_larger_components_first_and_keeps_members_together():
    labels = [5, 9, 5, 7, 9, 5]  # Komponenten: {0, 2, 5} (Größe 3), {1, 4} (2), {3} (1)
    assert V.component_order(labels) == [0, 2, 5, 1, 4, 3]


def test_size_chart_marks_the_giant_only_when_it_holds_a_tenth():
    fig = V.build_sizes([60, 3, 3, 1, 1, 1], 100, 1.5)
    colors = list(fig.data[0].marker.color)
    xs = list(fig.data[0].x)
    assert xs == ["1", "3", "60"] and colors[xs.index("60")] == C.COLOR_GIANT and colors[xs.index("1")] == C.COLOR_MEASURED
    fig2 = V.build_sizes([9, 3, 1], 100, 0.5)  # die größte hält weniger als ein Zehntel
    assert C.COLOR_GIANT not in list(fig2.data[0].marker.color)


def test_process_figure_draws_every_edge_and_colours_the_giant():
    g = make_process(60, 3.0, 4)
    state = ev.process_state(g, 80)
    fig = V.build_process(state, 60)
    assert fig.data[0].x.count(None) == 80  # eine Strecke je Kante
    colors = list(fig.data[1].marker.color)
    assert colors.count(C.COLOR_GIANT) == state["sizes"][0] and len(colors) == 60


def test_process_figure_at_zero_edges_has_no_giant():
    g = make_process(60, 3.0, 4)
    fig = V.build_process(ev.process_state(g, 0), 60)
    assert C.COLOR_GIANT not in list(fig.data[1].marker.color)


def test_curve_figure_contains_measurement_theory_band_and_the_marker():
    rows = ev.sweep_c(300, grid=(0.5, 1.5, 3.0), seeds=(1, 2))
    fig = V.build_curve(rows, 1.5)
    assert [t.name for t in fig.data] == ["Minimum bis Maximum", "Theorie (n → ∞)", "gemessen (Median)"]
    assert len(fig.layout.shapes) == 1  # die gestrichelte Linie beim gewählten c
    assert len(V.build_curve(rows, 8.0).layout.shapes) == 0  # außerhalb der Achse keine Linie


def test_series_figures_use_the_rows():
    finite = ev.series_finite(sizes=(100, 400), cs=(0.8, 1.4), seeds=(1, 2))
    f = V.build_finite(finite, (0.8, 1.4))
    assert [t.name for t in f.data] == ["Theorie (n → ∞)", "n = 100", "n = 400"]
    connect = ev.series_connect(sizes=(50, 100), xs=(-1, 1), reps=20)
    c = V.build_connect(connect)
    assert [t.name for t in c.data] == ["Theorie exp(−exp(−x))", "n = 50", "n = 100"]
    critical = ev.series_critical(sizes=(300, 1200), reps=5)
    k = V.build_critical(critical)
    assert list(k.data[2].x) == [300, 1200]
    chi = V.build_chi(ev.sweep_c(300, grid=(0.5, 1.5), seeds=(1,)))
    assert chi.data[-1].name == "gemessen (Median)"


def test_process_curve_marks_the_current_state():
    g = make_process(60, 3.0, 4)
    curve = ev.process_curve(g)
    fig = V.build_process_curve(curve, 60, 45)
    assert list(fig.data[2].x) == [2.0 * 45 / 60] and list(fig.data[2].y) == [curve[45]]


def test_all_axes_are_locked():
    rows = ev.sweep_c(200, grid=(0.5, 1.5), seeds=(1,))
    g = make_process(60, 3.0, 4)
    figs = [
        V.build_sizes([5, 1], 6, 1.0), V.build_curve(rows, 1.0), V.build_chi(rows),
        V.build_finite(ev.series_finite(sizes=(100,), cs=(0.8, 1.4), seeds=(1,)), (0.8, 1.4)),
        V.build_connect(ev.series_connect(sizes=(50,), xs=(0,), reps=5)),
        V.build_critical(ev.series_critical(sizes=(200,), reps=3)),
        V.build_process(ev.process_state(g, 10), 60), V.build_process_curve(ev.process_curve(g), 60, 10),
    ]
    for fig in figs:
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


def test_figure_heights_are_set_and_the_background_is_transparent():
    fig = V.build_sizes([3, 1], 4, 1.0)
    assert fig.layout.height == 360 and np.isclose(len(fig.data), 1)
