# Zufallsgraph – Phasenübergang der Riesenkomponente – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-zufallsgraph-demo.streamlit.app/)** -->

Erstes Stück der **Netzmodelle-Linie** (Graphentheorie) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Es ist die Wurzel der Linie; Vorgänger gibt es nicht. Die Linie
erzeugt Netze aus Zufallsmodellen und prüft sie gegen die Theorie und gegen reale Netze, während die Linie „Netzwerkanalyse“ ein gegebenes Netz vermisst. Verwandt sind
[bfs-dfs-demo](https://github.com/sebastian-hanisch/bfs-dfs-demo) (zufällig gesperrte Kanten als Bond-Perkolation) und [robustheit-demo](https://github.com/sebastian-hanisch/robustheit-demo) (zufällig entfernte Knoten als Site-Perkolation):
dort wird ein vorhandenes Netz ausgedünnt, hier wird das Netz selbst erzeugt, und die Frage ist, ab wann es zusammenhängt.

Die Demo zieht Zufallsgraphen G(n, p), zählt die Komponenten mit einem eigenen **Union-Find**, vergleicht die **Riesenkomponente** mit der Theorie (der Lösung von S = 1 − exp(−c·S)), zeigt die Verschmierung der Schwelle bei endlicher Größe und prüft die Schwelle des Zusammenhangs bei c = ln n.
Ein kleines Netz mit 60 Knoten lässt sich Kante für Kante wachsen. `networkx` und `scipy` kommen nur als Gegenproben in die Tests.

## Warum dieses Problem

Ein Zufallsnetz ist das einfachste Modell eines Netzes ohne Struktur. Dass es trotzdem einen **Phasenübergang** hat, ist der Grundstein der Netzwissenschaft: Unterhalb eines mittleren Grades von 1 bleiben nur kleine Inseln, darüber enthält eine Riesenkomponente einen festen Anteil aller Knoten.
Die Frage der Demo: **Wo genau liegt die Schwelle, wie gut trägt die Theorie für endlich große Netze, und was bedeutet „zusammenhängend“?**

## Modell

G(n, p): jede der n·(n−1)/2 möglichen Kanten kommt unabhängig mit Wahrscheinlichkeit p vor; in der App ist p = c / (n − 1), also c der erwartete mittlere Grad. Die Zahl der Kanten ist binomialverteilt, die Kanten selbst sind gleichverteilt (Kern: `gnp_edges`, `sample_edges`).
Die Theorie (Erdős und Rényi, Grenzwert n → ∞): Riesenkomponente S mit S = 1 − exp(−c·S) für c > 1 und S = 0 sonst; mittlere Größe der kleineren Komponenten 1 / (1 − c(1 − S)); Wahrscheinlichkeit für „zusammenhängend“ exp(−exp(−x)) bei c = ln n + x; größte Komponente am kritischen Punkt c = 1 von der Größenordnung n^(2/3).
Formeln und Lesehilfe stehen im 📐-Expander der App. Die Literaturangaben dort stammen aus dem Gedächtnis und sind nicht im Original geprüft.

## Methodik

- **Kern** (`zgr_algorithm.py`): Kanten als gleichverteilte Teilmenge aller Paare (Index-Umkehrung, vektorisiert, an den Gleitkommagrenzen korrigiert), Union-Find mit Pfadhalbierung und Vereinigung nach Größe, Komponentengrößen, die Kurve der größten Komponente Kante für Kante, die Theorie (Bisektion für den Fixpunkt).
- **Auswertung** (`zgr_evaluation.py`): Riesenkomponente gegen die Theorie über den mittleren Grad (Median über fünf feste Zufalls-Seeds), drei Messreihen mit festem Programm (Verschmierung, Zusammenhang, kritischer Punkt), Zustand des wachsenden Netzes, alle Verdikte als reine Funktionen.
- **Oberfläche** (`app.py`): Ergebnis, Kernfrage, Kante für Kante, Kurve gegen die Theorie, Messreihen auf Klick, Annahmen und Formeln.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen mit festen Zufalls-Seeds (`SWEEP_SEEDS` 100000 bis 100004, bei Zusammenhang und kritischem Punkt 100000 aufwärts), nie ein einzelner Seed; Messzahlen sind Mediane. **Die Zufallszahlen eines `numpy`-Generators sind nicht über Versionen stabil**: die Tests prüfen deshalb Bänder um diese Zahlen.

| Frage | Befund | Test |
|---|---|---|
| Stimmt der Kern? | Komponentengrößen = `scipy` auf 40 Zufallsgraphen und = `networkx` auf 25; exakte Aufzählung aller **1 024** Graphen mit 5 Knoten (p = 0,4): die gezogenen Graphen liegen mit Zusammenhangswahrscheinlichkeit und mittlerer größter Komponente innerhalb von vier Standardfehlern; die 45 Paare von zwei aus fünf Knoten kommen gleich oft vor. | `tests/test_oracle_zufallsgraph.py` |
| Welche Werte hat die Theorie? | Riesenkomponente bei c = 1,5: **58,3 %**, bei c = 2: **79,7 %**, bei c = 3: **94,0 %**; mittlere Größe der kleineren Komponenten bei c = 0,5: **2,0**; Fixpunktgleichung gegen `scipy.optimize.brentq` auf acht Werten von c = 1,001 bis 8. | `tests/test_algorithm.py`, `test_claims.py::test_theory_values` |
| Trägt die Theorie für n = 2 000? | Median über 5 Pläne: c = 1,4: **49,6 %** (Theorie 51,1 %), c = 2: **79,0 %** (79,7 %), c = 3: **93,6 %** (94,0 %), c = 4: **98,0 %**. Ab c = 1,4 weicht die Messung um höchstens **1,9 Prozentpunkte** ab. | `test_claims.py::test_the_curve_follows_the_theory_away_from_the_threshold` |
| Wie verhält sie sich an der Schwelle? | Bei c = 1,1 liegt die Messung bei **8,9 %** statt der 17,6 % der Theorie (n = 2 000); bei c = 1,0 noch bei **6,9 %** statt 0. Die mittlere Größe der kleineren Komponenten steigt von 1,22 (c = 0,2) auf **7,3** (c = 1,0) und auf rund **12** (c = 1,1) und fällt dann auf 1,06 (c = 4). | `test_claims.py::test_the_threshold_is_blurred_at_finite_size` |
| Wie scharf ist der Übergang bei kleinem n? | Bei c = 0,8 hat die größte Komponente **4,3 %** (n = 300), **1,2 %** (n = 3 000) und **0,3 %** (n = 30 000) der Knoten; bei c = 1,4 sind es **45,0 %**, **50,5 %** und **50,9 %** gegen 51,1 % in der Theorie. | `test_claims.py::test_finite_size_smears_the_threshold` |
| Wann ist das Netz zusammenhängend? | Bei c = ln n + x: der Anteil verbundener Netze liegt für n = 100, 400 und 1 600 und x = −2 bis +2 in allen 15 Zellen höchstens **8 Prozentpunkte** neben exp(−exp(−x)) (Zufallsfehler bis 4 bei 150 Zufalls-Seeds je Zelle); x = +1: **70,7 %** (n = 1 600) gegen 69,2 %, x = −2: **0,0 %**. | `test_claims.py::test_connectivity_follows_the_double_exponential` |
| Was passiert am kritischen Punkt? | Bei c = 1 hat die größte Komponente im Median **64,5** (n = 500), **122,5** (n = 2 000), **317,5** (n = 8 000) und **721** Knoten (n = 32 000), geteilt durch n^(2/3) zwischen **0,72 und 1,02**; ihr Anteil an n fällt von **12,9 %** auf **2,3 %**. | `test_claims.py::test_the_critical_component_grows_like_n_to_the_two_thirds` |
| Presets | Standardfall (n = 2 000, c = 1,5): 59,4 % gegen 58,3 % Theorie; Unterkritisch (c = 0,5): größte Komponente **13** Knoten (0,7 %); Am kritischen Punkt: **6,2 %** bei einer zweitgrößten Komponente von 84 Knoten; Kleines Netz (n = 300, c = 1,2): die Riesenkomponente streut über die fünf Pläne von **10,7 % bis 42,0 %**; Nahe am Zusammenhang (c = ln n + 1): bis auf wenige einzelne Knoten ein Netz (über **99 %** in der größten Komponente). | `test_claims.py::test_presets_hold_what_their_help_texts_say` |

## Ehrliche Grenzen

- **Nur das einfachste Modell.** Alle Paare sind gleich wahrscheinlich, Hubs, Gemeinschaften und Nachbarschaft kommen nicht vor; ein Zufallsnetz ist als Modell eines Straßennetzes ungeeignet (Folgestück „Netze im Raum“).
- **Die Theorie gilt für n → ∞.** Die Messungen liegen bei endlicher Größe darunter, vor allem dicht an der Schwelle; die Messreihe zeigt, wie stark, und sie ist mit fünf Plänen je Zelle klein (Band Minimum bis Maximum in der Kurve).
- **Zusammenhang ist schon bei n = 100 gut getroffen.** Ein Verschmierungseffekt wie bei der Riesenkomponente ist dort nicht zu erkennen: bei n = 100 weicht der Anteil um höchstens **5** Prozentpunkte von der Theorie ab, bei den größeren Netzen um höchstens **8**; beides liegt im Bereich des doppelten Zufallsfehlers (4 Prozentpunkte bei 150 Zufalls-Seeds je Zelle).
- **Der kritische Punkt streut stark.** Zwischen Minimum und Maximum der größten Komponente liegt bei jeder Größe ein Faktor von **7 bis 9** (Median 64,5 bei Minimum 21 und Maximum 166 für n = 500); das Verhältnis zu n^(2/3) ist deshalb nur eine Größenordnung, kein Gesetz mit scharfen Konstanten.
- **Kein Wegelängen-Maß.** Die Demo misst Komponentengrößen, nicht die Länge der Wege; das steht in der Linie „Graphen und Netzwerke“ (`strukturkennzahlen-demo`).
- **Literatur nicht im Original geprüft** (Erdős und Rényi 1960, Bollobás 2001; aus dem Gedächtnis).

## Befunde und Korrekturen gegenüber dem Plan

- **Verschmierung bei c = 1,1:** die Vormessung im Plan (`netzmodelle-planung/PLAN.md`) fand bei n = 5 000 0,116 gegen 0,176; die Demo misst bei n = 2 000 mit den festen Plänen **0,089**. Die Richtung ist dieselbe (deutlich unter der Theorie), die Zahlen sind nicht vergleichbar (andere Größe, andere Pläne).
- **Zusammenhang:** die Vormessung nannte „ab n = 400 alle Zellen innerhalb 0,05“; mit 150 Zufalls-Seeds je Zelle und den festen Plänen der Demo liegt der größte Abstand bei **0,079** (n = 400, x = 0). Der Zufallsfehler von 0,04 erklärt das; die Aussage „innerhalb 0,05“ wurde nicht übernommen.
- **Kritischer Punkt:** die Vormessung fand Verhältnisse von 0,77 bis 1,47 zu n^(2/3); die Demo findet mit 20 festen Plänen 0,72 bis 1,02. Beide Bereiche sind breit.
- **Port:** 9155 statt 9154 (9154 vergab eine parallele Sitzung inzwischen an `approximation-grenzen-demo`).

## Verifikation

- **Kern:** Kantenindex gegen die Aufzählung (auch an den Gleitkommagrenzen), Ziehen ohne Doppelungen und deterministisch, Union-Find von Hand an einer Mini-Instanz (Weg, Dreieck, Einzelknoten), die Kurve der größten Komponente gegen die Neuberechnung jedes Präfixes, Theorie an bekannten Werten und an der Fixpunktgleichung.
- **Orakel:** `scipy` und `networkx` für die Komponenten, exakte Aufzählung aller Graphen mit 5 Knoten, Chi-Quadrat der Kantenziehung, `brentq` für den Fixpunkt, große Zufallsgraphen gegen die Theorie.
- **Verdikte:** reine Funktionen, jede Zahl im Text an handgesetzten Werten (`tests/test_verdicts.py`).
- **Presets, Layout, Oberfläche:** wie in allen Demos (Presets vollständig und im Regelbereich, Schnellstart oben, aktives Preset hervorgehoben, AppTest jeder Button und jeder Regler an den Rändern).
- **Mutationsprüfung** (`tools/mutate_demo.py`): 123 Stellen im Kern `zgr_algorithm.py`, **112 gefangen**, 9 als äquivalent festgehalten, 2 enden in Zeitüberschreitungen (vermutlich Endlosschleifen; das Werkzeug zählt sie nicht). Die äquivalenten Stellen sind Abbruchzähler der Bisektion (200 Schritte, Vergleich `<` gegen `<=`), Grenzen, die erst bei n = 1 oder durch die Rundung der ganzzahligen Wurzel wirken, und die erste Korrekturzeile von `row_correct`: ihr Vergleich `>` gegen `>=` wird von der zweiten Zeile ausgeglichen. Drei weitere Mutanten dieser Zeile meldet das Werkzeug trotz zwei Läufen als überlebend, obwohl `test_row_correct_repairs_an_estimate_that_is_off_by_one_in_either_direction` sie von Hand in einer Kopie fängt (Ursache im Werkzeug nicht geklärt); sie stehen als äquivalent im Zustand, damit er ruhig bleibt.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und networkx (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `zgr_algorithm.py` | Kern: Kantenziehung, Union-Find, Kurve der größten Komponente, Theorie |
| `zgr_constants.py` | Regler-Grenzen, feste Pläne der Messreihen, Farben, Presets |
| `zgr_evaluation.py` | Kennzahlen, Messreihen, Zustand des wachsenden Netzes, Verdikte |
| `zgr_layout.py` | Fester Kopf (Stylesheet) |
| `zgr_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `zgr_scenario.py` | Zufallsgraphen und das wachsende Netz |
| `zgr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Mini-Instanz, Kern, Orakel, Verdikte, Randfälle, Presets, Abbildungen, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

- **Konfigurationsmodell, Wachstumsmodelle, Netze im Raum:** eigene Folgestücke der Linie (Grade vorgeben, Hubs erklären, Nachbarschaft).
- **Gerichtete oder gewichtete Zufallsgraphen:** andere Schwellen (starke Zusammenhangskomponenten), nicht Teil dieser Demo.
- **Schwellen anderer Eigenschaften** (Dreiecke, Cliquen, Hamilton-Kreis): nicht gebaut; die Schwelle des Zusammenhangs steht hier nur als Grenzwert nach der Theorie.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
