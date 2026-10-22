"""Jede Zahl der README wird hier nachgerechnet (DEMO-PLAYBOOK §4): über dieselben Auswertungsfunktionen wie in der App, mit den festen Plänen.

Die Zufallszahlen eines numpy-Generators sind über Versionen nicht stabil: die Messwerte gelten deshalb als Bänder um die in der README genannten Zahlen.
Die Theoriewerte und die Zählung am Orakel sind exakt."""

import math

import pytest

import zgr_algorithm as A
import zgr_constants as C
import zgr_evaluation as ev


@pytest.fixture(scope="module")
def curve():
    return {round(r["c"], 2): r for r in ev.sweep_c(2000)}


@pytest.fixture(scope="module")
def finite():
    return {r["n"]: r["giant"] for r in ev.series_finite()}


@pytest.fixture(scope="module")
def connect():
    return ev.series_connect()


@pytest.fixture(scope="module")
def critical():
    return ev.series_critical()


def test_the_series_use_the_documented_fixed_plans():
    assert C.SWEEP_SEEDS == tuple(range(100000, 100005)) and C.FINITE_N == (300, 3000, 30000) and C.FINITE_C == (0.8, 0.9, 1.0, 1.1, 1.2, 1.4)
    assert C.CONNECT_N == (100, 400, 1600) and C.CONNECT_X == (-2, -1, 0, 1, 2) and C.CONNECT_REPS == 150
    assert C.CRITICAL_N == (500, 2000, 8000, 32000) and C.CRITICAL_REPS == 20 and (C.N_MIN, C.N_MAX, C.DEFAULT_N, C.DEFAULT_C) == (100, 5000, 2000, 1.5) and C.PROCESS_N == 60


def test_theory_values():
    assert A.giant_theory(1.5) == pytest.approx(0.583, abs=5e-4) and A.giant_theory(2.0) == pytest.approx(0.797, abs=5e-4) and A.giant_theory(3.0) == pytest.approx(0.940, abs=5e-4)
    assert A.giant_theory(1.4) == pytest.approx(0.511, abs=5e-4) and A.giant_theory(1.1) == pytest.approx(0.176, abs=5e-4)
    assert A.finite_cluster_theory(0.5) == pytest.approx(2.0) and A.connectivity_theory(1000, math.log(1000) + 1) == pytest.approx(0.692, abs=5e-4)


def test_the_curve_follows_the_theory_away_from_the_threshold(curve):
    assert curve[1.4]["giant"] == pytest.approx(0.496, abs=0.03) and curve[2.0]["giant"] == pytest.approx(0.790, abs=0.03)
    assert curve[3.0]["giant"] == pytest.approx(0.936, abs=0.03) and curve[4.0]["giant"] == pytest.approx(0.980, abs=0.02)
    worst = max(ev.deviation(r["giant"], r["theory"]) for c, r in curve.items() if c >= 1.4)
    assert worst < 0.04  # README: 1,9 Prozentpunkte


def test_the_threshold_is_blurred_at_finite_size(curve):
    assert curve[1.1]["giant"] == pytest.approx(0.089, abs=0.06) and curve[1.1]["giant"] < curve[1.1]["theory"] - 0.04  # deutlich unter 17,6 %
    assert 0.02 < curve[1.0]["giant"] < 0.2  # statt 0
    assert curve[0.2]["chi"] == pytest.approx(1.22, abs=0.1) and curve[4.0]["chi"] == pytest.approx(1.06, abs=0.1)
    assert 4.5 < curve[1.0]["chi"] < 11 and 10 < curve[1.1]["chi"] < 13
    assert curve[1.1]["chi"] > curve[1.0]["chi"] > curve[0.8]["chi"] > curve[0.4]["chi"]  # unterhalb der Schwelle steigt sie bis zum Übergang


def test_finite_size_smears_the_threshold(finite):
    assert finite[300][0.8] == pytest.approx(0.043, abs=0.03) and finite[3000][0.8] == pytest.approx(0.012, abs=0.012) and finite[30000][0.8] == pytest.approx(0.003, abs=0.004)
    assert finite[300][0.8] > finite[3000][0.8] > finite[30000][0.8]
    assert finite[300][1.4] == pytest.approx(0.450, abs=0.06) and finite[3000][1.4] == pytest.approx(0.505, abs=0.04) and finite[30000][1.4] == pytest.approx(0.509, abs=0.03)
    assert finite[300][1.4] < A.giant_theory(1.4) - 0.02 and abs(finite[30000][1.4] - A.giant_theory(1.4)) < 0.03


def test_connectivity_follows_the_double_exponential(connect):
    assert len(connect) == 15 and {r["reps"] for r in connect} == {150}
    small = max(ev.deviation(r["share"], r["theory"]) for r in connect if r["n"] == 100)
    big = max(ev.deviation(r["share"], r["theory"]) for r in connect if r["n"] != 100)
    assert small < 0.09 and big < 0.12  # README: höchstens 5 bzw. 8 Prozentpunkte, Zufallsfehler 4
    cell = next(r for r in connect if r["n"] == 1600 and r["x"] == 1)
    assert cell["share"] == pytest.approx(0.707, abs=0.1) and cell["theory"] == pytest.approx(0.692, abs=5e-4)
    assert next(r for r in connect if r["n"] == 1600 and r["x"] == -2)["share"] < 0.04
    assert 0.5 / math.sqrt(150) == pytest.approx(0.0408, abs=1e-3)


def test_the_critical_component_grows_like_n_to_the_two_thirds(critical):
    by = {r["n"]: r for r in critical}
    assert by[500]["median"] == pytest.approx(64.5, rel=0.5) and by[500]["low"] == pytest.approx(21, rel=0.7) and by[500]["high"] == pytest.approx(166, rel=0.7)
    assert by[32000]["median"] == pytest.approx(721, rel=0.5)
    assert all(0.5 < r["ratio"] < 1.5 for r in critical)  # README: 0,72 bis 1,02
    first, last = by[500], by[32000]
    assert first["median"] / 500 > 0.08 and last["median"] / 32000 < 0.05  # Anteil fällt von 12,9 % auf 2,3 %
    assert all(5 < r["high"] / r["low"] < 14 for r in critical)  # README: Faktor 7 bis 9 zwischen Minimum und Maximum


def test_presets_hold_what_their_help_texts_say():
    by = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    std, sub, crit, small, near = (by[name] for name in C.PRESETS)
    assert abs(std.giant - std.giant_theory) < 0.04 and std.regime == "überkritisch" and std.giant_theory == pytest.approx(0.583, abs=5e-4)
    assert sub.sizes[0] < 60 and sub.giant < 0.03 and sub.regime == "unterkritisch"
    assert crit.regime == "kritisch" and 0.02 < crit.giant < 0.2 and crit.second > 20
    assert small.graph.n == 300 and small.giant_theory == pytest.approx(0.314, abs=5e-4)
    spread = [ev.analyse(ev.Settings(n=300, c=1.2, seed=s)).giant for s in C.SWEEP_SEEDS]
    assert max(spread) - min(spread) > 0.15 and min(spread) == pytest.approx(0.107, abs=0.06) and max(spread) == pytest.approx(0.42, abs=0.1)  # README: 10,7 % bis 42,0 %
    assert near.giant > 0.99 and near.graph.n == 2000 and abs(C.PRESETS["Nahe am Zusammenhang (c = ln n + 1)"]["c"] - (math.log(2000) + 1)) < 0.05


def test_the_measurement_series_run_in_a_few_seconds():
    import time

    t0 = time.perf_counter()
    ev.series_finite(), ev.series_connect(), ev.series_critical()
    assert time.perf_counter() - t0 < 120  # weite Obergrenze (gemessen etwa 6 s); keine Zeit-Assertion im engeren Sinn
