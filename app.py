"""Zufallsgraph – Phasenübergang der Riesenkomponente - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 1 der Linie „Netzmodelle“: Wann entsteht in einem Zufallsnetz eine Riesenkomponente, wann ist es zusammenhängend, und wie scharf ist das
bei endlicher Größe? Der Kern liegt in zgr_algorithm.py (Union-Find, Kantenziehung, Theorie).

Lauffähig mit: streamlit run app.py
"""

import math
import time

import streamlit as st

import zgr_algorithm as A
import zgr_constants as C
import zgr_evaluation as ev
from zgr_evaluation import Settings, analyse, num, regime, sweep_c, verdict_connect, verdict_critical, verdict_finite, verdict_regime, verdict_theory
from zgr_layout import dunkles_design, kopf_css
from zgr_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from zgr_scenario import make_process
from zgr_visualization import (
    build_chi,
    build_connect,
    build_critical,
    build_curve,
    build_finite,
    build_process,
    build_process_curve,
    build_sizes,
)

st.set_page_config(page_title="Zufallsgraph – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _curve(n):
    return sweep_c(n)


@st.cache_data(show_spinner=False)
def _process(seed):
    g = make_process(C.PROCESS_N, C.PROCESS_C_MAX, seed)
    return g, ev.process_curve(g)


@st.cache_data(show_spinner=False)
def _series():
    """Die drei Messreihen (festes Programm, keine Eingabe): Endlichkeit, Zusammenhang, kritischer Punkt."""
    return ev.series_finite(), ev.series_connect(), ev.series_critical()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in zgr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🌐 Zufallsgraph – Phasenübergang der Riesenkomponente")
st.markdown(
    """
Ein **Zufallsgraph** entsteht, indem jedes Paar von Knoten mit gleicher Wahrscheinlichkeit durch eine Kante verbunden wird. Wächst die Zahl der Kanten, passiert etwas Plötzliches:
Ab einem **mittleren Grad von 1** (im Mittel eine Kante je Knoten) bildet sich eine **Riesenkomponente**, die einen festen Anteil aller Knoten enthält, und ab etwa dem
**natürlichen Logarithmus der Knotenzahl** hängt das ganze Netz zusammen. Die Demo rechnet das nach, stellt Messung und Theorie nebeneinander und zeigt, wie **verschmiert**
der Übergang bei endlicher Größe ist. Erstes Stück der Linie „Netzmodelle“; sie erzeugt Netze, statt ein gegebenes Netz zu vermessen.

Weiter unten: 🔬 **Kante für Kante wachsen lassen**, 🔬 **Riesenkomponente gegen die Theorie**, 🔬 **Messreihen** (Verschmierung, Zusammenhang, kritischer Punkt), 🚧 **Wo die Annahmen enden** und der Expander
**📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert ein Zufallsgraph", expanded=True):
    st.markdown(
        """
1. **Kanten würfeln:** Jedes der n·(n−1)/2 möglichen Paare bekommt mit Wahrscheinlichkeit p = c / (n − 1) eine Kante; c ist der erwartete **mittlere Grad**.
2. **Komponenten zählen:** Ein **Union-Find** vereinigt die Knoten Kante für Kante; die Größen der Komponenten sind das Ergebnis.
3. **Vergleichen:** Die **Theorie** sagt für große Netze voraus, welcher Anteil der Knoten in der Riesenkomponente liegt (die Lösung von S = 1 − exp(−c·S)); die Demo zeigt, wie nah ein endliches Netz herankommt.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    n = st.slider("Knoten n", *bounds("n_slider"), key="n_slider", help="Wie viele Knoten das Zufallsnetz hat.")
    c = st.slider("Mittlerer Grad c", *bounds("c_slider"), key="c_slider", step=0.1,
                  help="Erwartete Zahl der Kanten je Knoten. Die Kantenwahrscheinlichkeit ist p = c / (n − 1); die Schwelle der Riesenkomponente liegt bei c = 1.")
    seed = st.number_input("Zufalls-Seed des Netzes", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "c_slider": round(float(c), 1), "seed_input": int(seed)})

settings = Settings(n=int(n), c=round(float(c), 1), seed=int(seed))
a = _analysis(settings)
g = a.graph

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Knoten / Kanten", f"{g.n} / {g.m}", delta=f"mittlerer Grad {num(2 * g.m / g.n, 2)} (erwartet {num(settings.c)})", delta_color="off")
m2.metric("Größte Komponente", f"{num(a.giant * 100)} %", delta=f"{a.sizes[0]} Knoten · Theorie {num(a.giant_theory * 100)} %", delta_color="off")
m3.metric("Zweitgrößte Komponente", f"{a.second} Knoten", delta=f"{len(a.sizes)} Komponenten insgesamt", delta_color="off")
m4.metric("Phase", a.regime, delta="zusammenhängend" if a.connected else "nicht zusammenhängend", delta_color="off")
st.plotly_chart(build_sizes(a.sizes, g.n, settings.c), width="stretch", key="main_chart")
st.caption(
    "Wie viele Komponenten welcher Größe das Netz hat (Zahl der Komponenten auf einer Log-Achse). "
    "Rot: die größte Komponente, sobald sie mehr als ein Zehntel der Knoten hält; ein Balken ganz rechts weit abseits der übrigen ist die Riesenkomponente."
)

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Gibt es bei diesem mittleren Grad schon eine Riesenkomponente?")
kind, text = verdict_regime(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Zusehen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Kante für Kante wachsen lassen")
st.caption(
    f"Ein kleines Netz mit {C.PROCESS_N} Knoten (Zufalls-Seed wie in der Seitenleiste); die Kanten kommen in zufälliger Reihenfolge dazu. "
    "Rot: die größte Komponente. Die Kurve zeigt den Anteil der größten Komponente über dem mittleren Grad 2m / n."
)
proc, curve = _process(settings.seed)
m_now = st.slider("Zahl der Kanten m", 0, proc.m, min(proc.m, C.PROCESS_N), key="process_m", help="Wie viele der Kanten schon eingefügt sind.")
state = ev.process_state(proc, m_now)
pc1, pc2 = st.columns(2)
pc1.plotly_chart(build_process(state, proc.n), width="stretch", key="process_net")
pc2.plotly_chart(build_process_curve(curve, proc.n, state["m"]), width="stretch", key="process_curve")
st.markdown(
    f"Nach **{state['m']}** Kanten (mittlerer Grad {num(state['c'], 2)}, {regime(state['c'])}): größte Komponente **{state['sizes'][0]}** von {proc.n} Knoten "
    f"({num(state['giant'] * 100)} %), Theorie für n → ∞: {num(state['theory'] * 100)} %; {len(state['sizes'])} Komponenten."
)

st.markdown("---")

# --- Experiment: Kurve ---------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Riesenkomponente gegen die Theorie")
st.caption(
    f"Median über {len(C.SWEEP_SEEDS)} feste Zufalls-Seeds bei n = {settings.n} (wie in der Seitenleiste), Band: Minimum bis Maximum. Die Theorie gilt für n → ∞; "
    "der gewählte mittlere Grad ist rot gestrichelt markiert."
)
rows = _curve(settings.n)
st.plotly_chart(build_curve(rows, settings.c), width="stretch", key="curve_chart")
kind, text = verdict_theory(rows, settings.n)
getattr(st, kind)(text)
st.markdown("**Die kleineren Komponenten:** Unterhalb der Schwelle wachsen sie bis zum Übergang an und werden **größer**, oberhalb wieder kleiner: die mittlere Größe divergiert bei c = 1 (in der Theorie).")
st.plotly_chart(build_chi(rows), width="stretch", key="chi_chart")
st.caption("Mittlere Größe der kleineren Komponenten (größen-gewichtet, ohne die größte); Messung im Median gegen die Theorie 1 / (1 − c(1 − S)) mit S der Riesenkomponente.")

st.markdown("---")

# --- Messreihen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Messreihen: Verschmierung, Zusammenhang und kritischer Punkt")
st.caption(
    f"Drei feste Programme (unabhängig von den Reglern): Netze mit {C.FINITE_N[0]} bis {C.FINITE_N[-1]} Knoten, {C.CONNECT_REPS} Zufalls-Seeds je Zelle für den Zusammenhang und "
    f"{C.CRITICAL_REPS} je Größe am kritischen Punkt. Das Rechnen dauert einige Sekunden."
)
if st.button("Messreihen rechnen", key="series_button"):
    t0 = time.perf_counter()
    with st.spinner("Rechne die Messreihen (einige Sekunden) …"):
        _series()
    st.session_state["series_done"] = time.perf_counter() - t0
if "series_done" in st.session_state:
    finite, connect, critical = _series()
    tab1, tab2, tab3 = st.tabs(["Wie scharf ist der Übergang?", "Zusammenhang", "Kritischer Punkt c = 1"])
    with tab1:
        st.plotly_chart(build_finite(finite, C.FINITE_C), width="stretch", key="finite_chart")
        kind, text = verdict_finite(finite)
        getattr(st, kind)(text)
    with tab2:
        st.plotly_chart(build_connect(connect), width="stretch", key="connect_chart")
        kind, text = verdict_connect(connect)
        getattr(st, kind)(text)
        st.caption(f"Die Theorie exp(−exp(−x)) gilt für große Netze. Der Anteil aus {C.CONNECT_REPS} Zufalls-Seeds hat selbst eine Standardabweichung von höchstens etwa {num(0.5 / math.sqrt(C.CONNECT_REPS), 2)}: kleinere Abweichungen liegen im Rauschen der Stichprobe.")
    with tab3:
        st.plotly_chart(build_critical(critical), width="stretch", key="critical_chart")
        kind, text = verdict_critical(critical)
        getattr(st, kind)(text)
    st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['series_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Alle Paare gleich wahrscheinlich** | Echte Netze haben Hubs und Gemeinschaften: bei vorgegebenen Graden bestimmt das Verhältnis aus mittlerem Quadrat und Mittelwert der Grade die Schwelle, nicht der mittlere Grad allein. | Konfigurationsmodell (Folgestück) |
| **Keine Nachbarschaft, kein Raum** | Straßennetze verbinden Nachbarn, Zufallsnetze verbinden beliebige Paare: als Modell eines Straßennetzes taugt ein Zufallsnetz nicht. | Netze im Raum (Folgestück) |
| **Große Netze (n → ∞)** | Bei kleinen Netzen ist die Schwelle verschmiert und die Theorie trifft nur grob; die Messreihen oben zeigen, wie stark. | diese Demo |
| **Kein Wachstumsmechanismus** | Die Kanten entstehen alle gleichzeitig und unabhängig; erklärt wird nicht, woher Hubs kommen. | Wachstumsmodelle (Folgestück) |
| **Nur Zusammenhang, keine Wege** | Die Demo misst Komponentengrößen, nicht die Länge der Wege in der Riesenkomponente. | Strukturkennzahlen-Demo (Linie „Graphen und Netzwerke“) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Knoten; jede der $\binom{n}{2}$ möglichen Kanten kommt unabhängig mit Wahrscheinlichkeit $p$ vor. Gesucht: wie groß die größte Komponente wird, wenn der mittlere Grad $c$ wächst.

| Symbol | Bedeutung |
|---|---|
| $n$ | Zahl der Knoten |
| $p$ | Wahrscheinlichkeit je möglicher Kante, hier $p = c / (n-1)$ |
| $c$ | erwarteter mittlerer Grad |
| $S$ | Anteil der Knoten in der Riesenkomponente (Grenzwert für $n \to \infty$) |
| $\chi$ | größen-gewichtete mittlere Größe der kleineren Komponenten |

**Riesenkomponente.** Für $c \le 1$ ist $S = 0$; für $c > 1$ ist $S$ die positive Lösung von

$$S = 1 - e^{-cS}.$$

**Kleinere Komponenten.** Mit der Lösung $S$ gilt für die mittlere Größe der Komponente eines Knotens außerhalb der Riesenkomponente

$$\chi = \frac{1}{1 - c\,(1 - S)},$$

unterhalb der Schwelle also $\chi = 1/(1-c)$; sie divergiert bei $c \to 1$.

**Zusammenhang.** Bei mittlerem Grad $c = \ln n + x$ ist das Netz mit Wahrscheinlichkeit

$$P(\text{zusammenhängend}) \to e^{-e^{-x}}$$

verbunden (Grenzwert für $n \to \infty$).

**Kritischer Punkt.** Bei $c = 1$ ist die größte Komponente von der Größenordnung $n^{2/3}$, nicht ein fester Anteil von $n$.

**Verfahren.** Union-Find mit Pfadhalbierung und Vereinigung nach Größe; die Kanten werden als gleichverteilte Teilmenge aller Paare gezogen (Zahl binomialverteilt).

**Literatur** (aus dem Gedächtnis, nicht im Original geprüft): Erdős und Rényi, *On the evolution of random graphs*, Publ. Math. Inst. Hung. Acad. Sci. 5 (1960); Bollobás, *Random Graphs*, 2. Auflage, Cambridge University Press 2001.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
