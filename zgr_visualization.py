"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math
from collections import Counter

import plotly.graph_objects as go

import zgr_algorithm as A
import zgr_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def size_counts(sizes) -> list[tuple[int, int]]:
    """(Größe, Zahl der Komponenten dieser Größe), nach Größe aufsteigend."""
    return sorted(Counter(sizes).items())


def build_sizes(sizes, n, c):
    """Komponentengrößen eines Netzes: wie viele Komponenten welcher Größe (Log-Achse); die größte ist rot, wenn sie mehr als ein Zehntel der Knoten hält."""
    counts = size_counts(sizes)
    biggest = max(sizes)
    colors = [C.COLOR_GIANT if (s == biggest and biggest > n / 10) else C.COLOR_MEASURED for s, _ in counts]
    fig = go.Figure(go.Bar(x=[str(s) for s, _ in counts], y=[k for _, k in counts], marker_color=colors,
                           hovertemplate="Größe %{x}: %{y} Komponenten<extra></extra>"))
    fig.update_xaxes(title_text="Größe der Komponente (Knoten)", type="category")
    fig.update_yaxes(title_text="Zahl der Komponenten", type="log")
    return _base(fig)


def build_curve(rows, c_now):
    """Riesenkomponente über dem mittleren Grad: Messung (Median, Band Minimum bis Maximum) gegen die Theorie; der gewählte Wert ist markiert."""
    cs = [r["c"] for r in rows]
    fine = [i / 50 for i in range(1, 4 * 50 + 1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cs + cs[::-1], y=[r["high"] for r in rows] + [r["low"] for r in rows][::-1], fill="toself", fillcolor="rgba(47,107,101,0.15)",
                             line=dict(width=0), hoverinfo="skip", name="Minimum bis Maximum"))
    fig.add_trace(go.Scatter(x=fine, y=[A.giant_theory(c) for c in fine], mode="lines", line=dict(color=C.COLOR_THEORY, width=2, dash="dash"), name="Theorie (n → ∞)",
                             hovertemplate="c = %{x:.2f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=cs, y=[r["giant"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_MEASURED, width=3), name="gemessen (Median)",
                             hovertemplate="c = %{x}: %{y:.3f}<extra></extra>"))
    if c_now <= 4.0:
        fig.add_vline(x=c_now, line=dict(color=C.COLOR_GIANT, width=1, dash="dot"))
    fig.update_xaxes(title_text="mittlerer Grad c", range=[0, 4.05])
    fig.update_yaxes(title_text="Anteil der Riesenkomponente", range=[-0.02, 1.02])
    return _base(fig)


def build_chi(rows):
    """Mittlere Größe der kleineren Komponenten: Messung gegen Theorie (die Theorie wird bei c = 1 unendlich und dort nicht gezeichnet)."""
    fine_lo = [i / 100 for i in range(10, 98)]
    fine_hi = [1.03 + i / 100 for i in range(0, 300)]
    fig = go.Figure()
    for xs, name in ((fine_lo, "Theorie (n → ∞)"), (fine_hi, None)):
        fig.add_trace(go.Scatter(x=xs, y=[A.finite_cluster_theory(c) for c in xs], mode="lines", line=dict(color=C.COLOR_THEORY, width=2, dash="dash"),
                                 name=name or "Theorie (n → ∞)", showlegend=name is not None, hovertemplate="c = %{x:.2f}: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r["c"] for r in rows], y=[r["chi"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_MEASURED, width=3), name="gemessen (Median)",
                             hovertemplate="c = %{x}: %{y:.1f}<extra></extra>"))
    fig.update_xaxes(title_text="mittlerer Grad c", range=[0, 4.05])
    fig.update_yaxes(title_text="mittlere Größe der kleineren Komponenten", type="log")
    return _base(fig)


def build_finite(rows, cs):
    """Riesenkomponente je Netzgröße über dem mittleren Grad (nahe der Schwelle) neben der Theorie."""
    fig = go.Figure()
    fine = [cs[0] + i * (cs[-1] - cs[0]) / 60 for i in range(61)]
    fig.add_trace(go.Scatter(x=fine, y=[A.giant_theory(c) for c in fine], mode="lines", line=dict(color=C.COLOR_THEORY, width=2, dash="dash"), name="Theorie (n → ∞)"))
    palette = [C.COLOR_THIRD, C.COLOR_MEASURED, C.COLOR_GIANT]
    for i, r in enumerate(rows):
        fig.add_trace(go.Scatter(x=list(cs), y=[r["giant"][c] for c in cs], mode="lines+markers", line=dict(color=palette[i % len(palette)], width=2.5), name=f"n = {r['n']}",
                                 hovertemplate="c = %{x}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="mittlerer Grad c")
    fig.update_yaxes(title_text="Anteil der Riesenkomponente", range=[-0.02, 0.65])
    return _base(fig)


def build_connect(rows):
    """Anteil zusammenhängender Netze über x = c − ln n, je Netzgröße, neben exp(−exp(−x))."""
    xs_fine = [-2.5 + i * 5.0 / 50 for i in range(51)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_fine, y=[math.exp(-math.exp(-x)) for x in xs_fine], mode="lines", line=dict(color=C.COLOR_THEORY, width=2, dash="dash"), name="Theorie exp(−exp(−x))"))
    palette = [C.COLOR_THIRD, C.COLOR_MEASURED, C.COLOR_GIANT]
    for i, n in enumerate(sorted({r["n"] for r in rows})):
        sel = [r for r in rows if r["n"] == n]
        fig.add_trace(go.Scatter(x=[r["x"] for r in sel], y=[r["share"] for r in sel], mode="lines+markers", line=dict(color=palette[i % len(palette)], width=2.5), name=f"n = {n}",
                                 hovertemplate="x = %{x}: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="x = c − ln n")
    fig.update_yaxes(title_text="Anteil zusammenhängender Netze", range=[-0.02, 1.02])
    return _base(fig)


def build_critical(rows):
    """Größte Komponente bei c = 1 über n (beide Achsen logarithmisch) mit den Vergleichslinien n^(2/3) und n."""
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[A.critical_scale(n) for n in ns], mode="lines", line=dict(color=C.COLOR_THEORY, width=2, dash="dash"), name="n hoch 2/3"))
    fig.add_trace(go.Scatter(x=ns, y=ns, mode="lines", line=dict(color=C.COLOR_OTHER, width=2, dash="dot"), name="n (fester Anteil)"))
    fig.add_trace(go.Scatter(x=ns, y=[r["median"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_MEASURED, width=3), name="größte Komponente (Median)",
                             hovertemplate="n = %{x}: %{y:.0f}<extra></extra>"))
    fig.update_xaxes(title_text="Knotenzahl n", type="log")
    fig.update_yaxes(title_text="Größe der größten Komponente", type="log")
    return _base(fig)


def circle_layout(order: list[int]) -> dict[int, tuple[float, float]]:
    """Knoten gleichmäßig auf einem Kreis, in der Reihenfolge `order` (Knoten der gleichen Komponente liegen so nebeneinander)."""
    n = len(order)
    return {v: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i, v in enumerate(order)}


def component_order(labels: list[int]) -> list[int]:
    """Knotenreihenfolge für den Kreis: Komponenten nach Größe absteigend, innerhalb gleich große nach kleinstem Knoten."""
    groups: dict[int, list[int]] = {}
    for v, r in enumerate(labels):
        groups.setdefault(r, []).append(v)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    return [v for g in ordered for v in g]


def build_process(state, n):
    """Das wachsende Netz nach m Kanten: die größte Komponente rot (sobald sie mehr als einen Knoten hat), Kanten grau."""
    labels, edges = state["labels"], state["edges"]
    order = component_order(labels)
    pos = circle_layout(order)
    sizes = state["sizes"]
    big_root = max(set(labels), key=lambda r: (labels.count(r), -r))
    ex, ey = [], []
    for u, v in edges:
        ex += [pos[int(u)][0], pos[int(v)][0], None]
        ey += [pos[int(u)][1], pos[int(v)][1], None]
    colors = [C.COLOR_GIANT if (labels[v] == big_root and sizes[0] > 1) else C.COLOR_OTHER for v in range(n)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=C.COLOR_EDGE, width=1), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[pos[v][0] for v in range(n)], y=[pos[v][1] for v in range(n)], mode="markers", marker=dict(size=9, color=colors),
                             hovertemplate="Knoten %{customdata}<extra></extra>", customdata=list(range(n)), showlegend=False))
    # gleicher Maßstab nur mit Autorange (DEMO-PLAYBOOK §3): den Rand erzwingen zwei unsichtbare Marker an den Ecken
    fig.add_trace(go.Scatter(x=[-1.15, 1.15], y=[-1.15, 1.15], mode="markers", marker=dict(opacity=0, size=1), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, scaleanchor="y")
    fig.update_yaxes(visible=False)
    return _base(fig, height=420)


def build_process_curve(curve, n, m_now):
    """Anteil der größten Komponente über dem mittleren Grad 2m/n beim Wachsen; der aktuelle Stand ist markiert, daneben die Theorie."""
    xs = [2.0 * t / n for t in range(len(curve))]
    fine = [i / 50 for i in range(0, int(C.PROCESS_C_MAX * 50) + 1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fine, y=[A.giant_theory(c) for c in fine], mode="lines", line=dict(color=C.COLOR_THEORY, width=2, dash="dash"), name="Theorie (n → ∞)"))
    fig.add_trace(go.Scatter(x=xs, y=curve, mode="lines", line=dict(color=C.COLOR_MEASURED, width=2.5), name="dieses Netz"))
    fig.add_trace(go.Scatter(x=[xs[m_now]], y=[curve[m_now]], mode="markers", marker=dict(size=12, color=C.COLOR_GIANT), name="jetzt", hovertemplate="c = %{x:.2f}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="mittlerer Grad 2m / n", range=[0, C.PROCESS_C_MAX])
    fig.update_yaxes(title_text="Anteil der größten Komponente", range=[0, 1.02])
    return _base(fig, height=300)
