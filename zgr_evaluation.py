"""Kennzahlen, Messreihen und Verdikte (reine Funktionen: die App ruft st.<Art>(Text), die Tests prüfen jede Zahl im Text)."""

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

import zgr_algorithm as A
import zgr_constants as C
from zgr_scenario import Graph, make_graph, make_process


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    c: float = C.DEFAULT_C
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    graph: Graph
    sizes: tuple[int, ...]       # alle Komponentengrößen, absteigend
    giant: float                 # Anteil der größten Komponente
    giant_theory: float          # Grenzwert für n → ∞
    second: int                  # Größe der zweitgrößten Komponente (0 ohne zweite)
    mean_finite: float           # größen-gewichtete mittlere Größe der übrigen Komponenten
    mean_finite_theory: float
    connected: bool
    regime: str


def regime(c: float) -> str:
    """Phase nach dem mittleren Grad: unter 0,9 unterkritisch, bis 1,1 kritisch (Übergangsbereich), darüber überkritisch."""
    if c < C.CRITICAL_LOW:
        return "unterkritisch"
    if c <= C.CRITICAL_HIGH:
        return "kritisch"
    return "überkritisch"


def analyse(settings: Settings) -> Analysis:
    g = make_graph(settings.n, settings.c, settings.seed)
    sizes = A.component_sizes(g.n, g.edges)
    return Analysis(
        graph=g, sizes=tuple(sizes), giant=sizes[0] / g.n, giant_theory=A.giant_theory(settings.c),
        second=sizes[1] if len(sizes) > 1 else 0, mean_finite=A.mean_finite_cluster(sizes),
        mean_finite_theory=A.finite_cluster_theory(settings.c), connected=len(sizes) == 1, regime=regime(settings.c),
    )


def num(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma (für Texte)."""
    return f"{x:.{digits}f}".replace(".", ",")


def deviation(measured: float, theory: float) -> float:
    """Abstand der Messung von der Theorie (Betrag)."""
    return abs(measured - theory)


# --- Messreihen ----------------------------------------------------------------------------------------------------------------------------


def sweep_c(n: int, grid=C.C_GRID, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Riesenkomponente und mittlere Größe der übrigen Komponenten über den mittleren Grad: Median über die festen Pläne (nie ein einzelner Seed)."""
    rows = []
    for c in grid:
        runs = [analyse(Settings(n, c, seed)) for seed in seeds]
        giants = [r.giant for r in runs]
        rows.append({
            "c": c, "giant": median(giants), "low": min(giants), "high": max(giants), "theory": A.giant_theory(c),
            "chi": median(r.mean_finite for r in runs), "chi_theory": A.finite_cluster_theory(c),
        })
    return rows


def series_finite(sizes=C.FINITE_N, cs=C.FINITE_C, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Verschmierung der Schwelle: Riesenkomponente je Knotenzahl n und mittlerem Grad c (Median über die Pläne) neben der Theorie."""
    rows = []
    for n in sizes:
        cells = {c: median(analyse(Settings(n, c, s)).giant for s in seeds) for c in cs}
        rows.append({"n": n, "giant": cells})
    return rows


def series_connect(sizes=C.CONNECT_N, xs=C.CONNECT_X, reps=C.CONNECT_REPS) -> list[dict]:
    """Zusammenhang bei c = ln n + x: Anteil der verbundenen Netze unter `reps` festen Zufalls-Seeds, neben exp(−exp(−x))."""
    rows = []
    for n in sizes:
        for x in xs:
            c = math.log(n) + x
            ok = sum(analyse(Settings(n, c, 100000 + r)).connected for r in range(reps))
            rows.append({"n": n, "x": x, "share": ok / reps, "theory": A.connectivity_theory(n, c), "reps": reps})
    return rows


def series_critical(sizes=C.CRITICAL_N, reps=C.CRITICAL_REPS) -> list[dict]:
    """Größte Komponente am kritischen Punkt c = 1: Median und Verhältnis zu n^(2/3) über `reps` feste Zufalls-Seeds."""
    rows = []
    for n in sizes:
        vals = [analyse(Settings(n, 1.0, 100000 + r)).sizes[0] for r in range(reps)]
        med = median(vals)
        rows.append({"n": n, "median": med, "ratio": med / A.critical_scale(n), "low": min(vals), "high": max(vals)})
    return rows


# --- Zusehen: Kante für Kante ----------------------------------------------------------------------------------------------------------------


def process_state(graph: Graph, m: int) -> dict:
    """Zustand des wachsenden Netzes nach den ersten m Kanten."""
    m = max(0, min(int(m), graph.m))
    edges = graph.edges[:m]
    labels = A.component_labels(graph.n, edges)
    sizes = A.component_sizes(graph.n, edges)
    c = 2.0 * m / graph.n
    return {"m": m, "c": c, "labels": labels, "sizes": sizes, "giant": sizes[0] / graph.n, "theory": A.giant_theory(c), "edges": edges}


def process_curve(graph: Graph) -> list[float]:
    """Anteil der größten Komponente nach jeder Kante (Länge m + 1)."""
    return [s / graph.n for s in A.giant_curve(graph.n, graph.edges)]


# --- Verdikte --------------------------------------------------------------------------------------------------------------------------------


def verdict_regime(a: Analysis) -> tuple[str, str]:
    """Kernfrage: liegt die größte Komponente beim gewählten mittleren Grad vor oder hinter der Theorie, und in welcher Phase ist das Netz?"""
    n, c = a.graph.n, a.graph.c
    if a.regime == "unterkritisch":
        return "info", (f"Bei c = {num(c, 1)} ist das Netz unterkritisch: Die größte Komponente hat {a.sizes[0]} von {n} Knoten ({num(a.giant * 100, 1)} %), "
                        f"eine Riesenkomponente gibt es in der Theorie nicht (Theorie: {num(a.giant_theory * 100, 1)} %).")
    if a.regime == "kritisch":
        return "warning", (f"Bei c = {num(c, 1)} liegt das Netz im Übergangsbereich: Die größte Komponente hat {a.sizes[0]} Knoten ({num(a.giant * 100, 1)} %); "
                           f"die Theorie ergibt {num(a.giant_theory * 100, 1)} %, bei endlicher Größe weicht die Messung davon deutlich ab.")
    return "success", (f"Bei c = {num(c, 1)} ist das Netz überkritisch: Die Riesenkomponente hat {a.sizes[0]} von {n} Knoten ({num(a.giant * 100, 1)} %), "
                       f"die Theorie ergibt {num(a.giant_theory * 100, 1)} %. Die zweitgrößte Komponente hat nur {a.second} Knoten.")


def verdict_theory(rows: list[dict], n: int) -> tuple[str, str]:
    """Experiment Kurve: größte Abweichung von der Theorie ab c = 1,4 und am Übergang (c = 1,1)."""
    far = [r for r in rows if r["c"] >= 1.4]
    near = next((r for r in rows if abs(r["c"] - 1.1) < 1e-9), None)
    worst = max(deviation(r["giant"], r["theory"]) for r in far)
    text = f"Ab c = 1,4 weicht die gemessene Riesenkomponente bei n = {n} höchstens um {num(worst * 100, 1)} Prozentpunkte von der Theorie ab"
    if near is None:
        return "success", text + "."
    return "info", text + f"; bei c = 1,1 liegt sie bei {num(near['giant'] * 100, 1)} % statt {num(near['theory'] * 100, 1)} %."


def verdict_finite(rows: list[dict], cs=C.FINITE_C) -> tuple[str, str]:
    """Experiment Endlichkeit: die Riesenkomponente unter der Schwelle (c = 0,8) schrumpft mit n, über der Schwelle (c = 1,4) nähert sie sich der Theorie."""
    small, large = rows[0], rows[-1]
    low, high = cs[0], cs[-1]
    return "info", (f"Bei c = {num(low, 1)} liegt die größte Komponente bei n = {small['n']} noch bei {num(small['giant'][low] * 100, 1)} % der Knoten, bei n = {large['n']} nur noch bei "
                    f"{num(large['giant'][low] * 100, 1)} %; bei c = {num(high, 1)} sind es {num(small['giant'][high] * 100, 1)} % und {num(large['giant'][high] * 100, 1)} % (Theorie {num(A.giant_theory(high) * 100, 1)} %). "
                    "Die Schwelle ist bei kleinen Netzen verschmiert.")


def verdict_connect(rows: list[dict]) -> tuple[str, str]:
    """Experiment Zusammenhang: größte Abweichung der gemessenen Anteile von exp(−exp(−x)), getrennt nach kleinem n = 100 und den größeren Netzen."""
    n_small = min(r["n"] for r in rows)
    big = [r for r in rows if r["n"] != n_small]
    worst_small = max(deviation(r["share"], r["theory"]) for r in rows if r["n"] == n_small)
    worst_big = max(deviation(r["share"], r["theory"]) for r in big)
    return "info", (f"Bei n = {n_small} weicht der gemessene Anteil verbundener Netze um höchstens {num(worst_small * 100, 0)} Prozentpunkte von exp(−exp(−x)) ab, "
                    f"bei den größeren Netzen um höchstens {num(worst_big * 100, 0)} (jeweils {rows[0]['reps']} feste Zufalls-Seeds je Zelle; der Zufallsfehler eines solchen Anteils "
                    f"beträgt bis zu {num(50 / math.sqrt(rows[0]['reps']), 0)} Prozentpunkte).")


def verdict_critical(rows: list[dict]) -> tuple[str, str]:
    """Experiment kritischer Punkt: die größte Komponente wächst etwa wie n^(2/3); Spanne der Verhältnisse."""
    ratios = [r["ratio"] for r in rows]
    first, last = rows[0], rows[-1]
    return "info", (f"Am kritischen Punkt c = 1 liegt die größte Komponente im Median bei {num(first['median'], 0)} (n = {first['n']}) bis {num(last['median'], 0)} (n = {last['n']}). "
                    f"Geteilt durch n^(2/3) ergeben sich Werte von {num(min(ratios), 2)} bis {num(max(ratios), 2)}; ein fester Anteil von n wäre es nicht ({num(first['median'] / first['n'] * 100, 1)} % gegen "
                    f"{num(last['median'] / last['n'] * 100, 1)} %).")
