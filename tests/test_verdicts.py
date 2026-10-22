"""Verdikte (reine Funktionen): Art und jede Zahl im Text an handgesetzten Werten, einschließlich der Schwellen."""

import numpy as np

import zgr_algorithm as A
import zgr_evaluation as ev
from zgr_scenario import Graph


def make(c, n=1000, giant=0.4, theory=0.3, sizes=None, second=7, connected=False):
    sizes = sizes or (int(giant * n), second, 1)
    return ev.Analysis(graph=Graph(n, c, np.zeros((0, 2), dtype=int)), sizes=tuple(sizes), giant=giant, giant_theory=theory, second=second,
                       mean_finite=2.0, mean_finite_theory=2.5, connected=connected, regime=ev.regime(c))


def test_regime_verdict_below_the_threshold():
    kind, text = ev.verdict_regime(make(0.5, giant=0.012, theory=0.0, sizes=(12, 8, 1)))
    assert kind == "info" and "unterkritisch" in text
    assert "c = 0,5" in text and "12 von 1000 Knoten" in text and "1,2 %" in text and "Theorie: 0,0 %" in text


def test_regime_verdict_in_the_transition():
    kind, text = ev.verdict_regime(make(1.0, giant=0.062, theory=0.0, sizes=(62, 40, 1)))
    assert kind == "warning" and "Übergangsbereich" in text and "c = 1,0" in text and "62 Knoten" in text and "6,2 %" in text and "0,0 %" in text


def test_regime_verdict_above_the_threshold():
    kind, text = ev.verdict_regime(make(1.5, giant=0.594, theory=0.583, sizes=(594, 13, 1), second=13))
    assert kind == "success" and "überkritisch" in text and "c = 1,5" in text and "594 von 1000 Knoten" in text and "59,4 %" in text and "58,3 %" in text and "nur 13 Knoten" in text


def test_theory_verdict_names_the_far_deviation_and_the_transition_value():
    rows = [
        {"c": 1.1, "giant": 0.116, "theory": 0.176},
        {"c": 1.4, "giant": 0.50, "theory": 0.51},
        {"c": 2.0, "giant": 0.79, "theory": 0.797},
        {"c": 3.0, "giant": 0.95, "theory": 0.94},
    ]
    kind, text = ev.verdict_theory(rows, 5000)
    assert kind == "info"
    assert "n = 5000" in text and "höchstens um 1,0 Prozentpunkte" in text and "bei c = 1,1" in text and "11,6 %" in text and "17,6 %" in text


def test_theory_verdict_without_the_transition_point():
    rows = [{"c": 1.4, "giant": 0.50, "theory": 0.51}, {"c": 2.0, "giant": 0.79, "theory": 0.797}]
    kind, text = ev.verdict_theory(rows, 300)
    assert kind == "success" and "höchstens um 1,0 Prozentpunkte" in text and "bei c = 1,1" not in text


def test_finite_verdict_uses_the_first_and_the_last_size():
    rows = [{"n": 300, "giant": {0.8: 0.043, 1.4: 0.45}}, {"n": 30000, "giant": {0.8: 0.003, 1.4: 0.509}}]
    kind, text = ev.verdict_finite(rows, cs=(0.8, 1.4))
    assert kind == "info" and "Bei c = 0,8" in text and "n = 300" in text and "4,3 %" in text and "n = 30000" in text and "0,3 %" in text
    assert "bei c = 1,4 sind es 45,0 % und 50,9 %" in text and f"Theorie {ev.num(A.giant_theory(1.4) * 100, 1)} %" in text and "verschmiert" in text


def test_connect_verdict_separates_the_small_size():
    rows = [
        {"n": 100, "x": 0, "share": 0.457, "theory": 0.368, "reps": 150},
        {"n": 100, "x": 1, "share": 0.70, "theory": 0.692, "reps": 150},
        {"n": 400, "x": 0, "share": 0.417, "theory": 0.368, "reps": 150},
        {"n": 1600, "x": 1, "share": 0.707, "theory": 0.692, "reps": 150},
    ]
    kind, text = ev.verdict_connect(rows)
    assert kind == "info" and "n = 100" in text and "um höchstens 9 Prozentpunkte" in text and "um höchstens 5" in text and "150 feste Zufalls-Seeds" in text and "bis zu 4 Prozentpunkte" in text


def test_critical_verdict_gives_the_ratio_range_and_the_share():
    rows = [
        {"n": 500, "median": 64.5, "ratio": 1.02, "low": 21, "high": 166},
        {"n": 2000, "median": 122.5, "ratio": 0.77, "low": 46, "high": 398},
        {"n": 32000, "median": 721.0, "ratio": 0.72, "low": 335, "high": 2497},
    ]
    kind, text = ev.verdict_critical(rows)
    assert kind == "info" and "64" in text and "721" in text and "0,72 bis 1,02" in text and "12,9 %" in text and "2,3 %" in text
