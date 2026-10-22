"""Auswertung: Analyse, Phasengrenzen, Messreihen (Struktur), Zusehen (Zustand und Kurve)."""

import math

import pytest

import zgr_algorithm as A
import zgr_constants as C
import zgr_evaluation as ev
from zgr_scenario import make_process


def test_regime_thresholds():
    assert ev.regime(0.89) == "unterkritisch" and ev.regime(0.9) == "kritisch"
    assert ev.regime(1.0) == "kritisch" and ev.regime(1.1) == "kritisch" and ev.regime(1.11) == "überkritisch"


def test_analyse_is_consistent_with_the_kernel():
    a = ev.analyse(ev.Settings(n=500, c=2.0, seed=3))
    assert sum(a.sizes) == 500 and list(a.sizes) == sorted(a.sizes, reverse=True)
    assert a.giant == a.sizes[0] / 500 and a.second == a.sizes[1]
    assert a.giant_theory == A.giant_theory(2.0) and a.regime == "überkritisch"
    assert a.connected == (len(a.sizes) == 1)


def test_analyse_has_no_second_component_when_connected():
    a = ev.analyse(ev.Settings(n=60, c=10.0, seed=1))
    assert a.connected and a.second == 0 and a.giant == 1.0 and a.mean_finite == 0.0


def test_deviation_and_num():
    assert ev.deviation(0.31, 0.4) == pytest.approx(0.09) and ev.deviation(0.5, 0.4) == pytest.approx(0.1)
    assert ev.num(1.25, 1) == "1,2" or ev.num(1.25, 1) == "1,3"  # Rundung auf gerade bzw. auf; wichtig ist das Komma
    assert ev.num(3.14159, 2) == "3,14" and ev.num(10, 0) == "10"


def test_sweep_c_structure_and_band():
    rows = ev.sweep_c(300, grid=(0.5, 1.5, 3.0), seeds=(1, 2, 3))
    assert [r["c"] for r in rows] == [0.5, 1.5, 3.0]
    for r in rows:
        assert 0 < r["low"] <= r["giant"] <= r["high"] <= 1
        assert r["theory"] == A.giant_theory(r["c"]) and r["chi_theory"] == A.finite_cluster_theory(r["c"])
    assert rows[0]["giant"] < rows[1]["giant"] < rows[2]["giant"]


def test_series_finite_structure():
    rows = ev.series_finite(sizes=(100, 400), cs=(0.8, 1.4), seeds=(1, 2, 3))
    assert [r["n"] for r in rows] == [100, 400] and set(rows[0]["giant"]) == {0.8, 1.4}
    assert rows[0]["giant"][0.8] <= rows[0]["giant"][1.4]


def test_series_connect_structure():
    rows = ev.series_connect(sizes=(50, 100), xs=(-1, 3), reps=40)
    assert [(r["n"], r["x"]) for r in rows] == [(50, -1), (50, 3), (100, -1), (100, 3)]
    for r in rows:
        assert 0 <= r["share"] <= 1 and r["reps"] == 40
        assert r["theory"] == pytest.approx(math.exp(-math.exp(-r["x"])))
    assert rows[1]["share"] > rows[0]["share"]  # mehr Kanten, öfter verbunden


def test_series_critical_structure():
    rows = ev.series_critical(sizes=(300, 1200), reps=8)
    for r in rows:
        assert r["low"] <= r["median"] <= r["high"] and r["ratio"] == pytest.approx(r["median"] / A.critical_scale(r["n"]))


def test_process_state_at_the_start_and_the_end():
    g = make_process(60, 3.0, 4)
    s0, s1 = ev.process_state(g, 0), ev.process_state(g, 10 ** 6)
    assert s0["m"] == 0 and s0["sizes"] == [1] * 60 and s0["giant"] == pytest.approx(1 / 60) and s0["c"] == 0.0
    assert s1["m"] == g.m and s1["c"] == pytest.approx(3.0) and sum(s1["sizes"]) == 60
    assert ev.process_state(g, -5)["m"] == 0


def test_process_curve_is_monotone_and_matches_the_state():
    g = make_process(60, 3.0, 6)
    curve = ev.process_curve(g)
    assert len(curve) == g.m + 1 and curve == sorted(curve)
    for m in (0, 13, 45, g.m):
        assert curve[m] == pytest.approx(ev.process_state(g, m)["giant"])


def test_the_fixed_plans_are_the_documented_ones():
    assert C.SWEEP_SEEDS == tuple(range(100000, 100005)) and len(C.SWEEP_SEEDS) == 5
    assert C.FINITE_N == (300, 3000, 30000) and C.CONNECT_REPS == 150 and C.CRITICAL_N == (500, 2000, 8000, 32000) and C.CRITICAL_REPS == 20
