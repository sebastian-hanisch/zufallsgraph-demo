"""Konstanten: Regler-Grenzen, Rechengrenzen, feste Pläne der Messreihen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 100, 5000, 2000    # Knoten; Komplexität am Maximum: etwa n·c/2 Kanten, Union-Find fast linear (5 000 Knoten bei c = 10: 25 000 Kanten)
C_MIN, C_MAX, DEFAULT_C = 0.2, 10.0, 1.5     # erwarteter mittlerer Grad c; die Kantenwahrscheinlichkeit ist p = c / (n − 1)
SEED_MAX, DEFAULT_SEED = 999999, 35

CRITICAL_LOW, CRITICAL_HIGH = 0.9, 1.1       # Übergangsbereich um c = 1: dort heißt die Phase „kritisch“, darunter „unterkritisch“, darüber „überkritisch“

SWEEP_SEEDS = tuple(range(100000, 100005))   # feste Pläne für alle Messreihen: nie ein einzelner Seed als Befund
C_GRID = (0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0)  # Raster der Kurve S(c)

FINITE_N = (300, 3000, 30000)                # Messreihe „Wie scharf ist der Übergang?“: n · Plan, Median über SWEEP_SEEDS
FINITE_C = (0.8, 0.9, 1.0, 1.1, 1.2, 1.4)
CONNECT_N = (100, 400, 1600)                 # Messreihe „Zusammenhang“: c = ln n + x
CONNECT_X = (-2, -1, 0, 1, 2)
CONNECT_REPS = 150
CRITICAL_N = (500, 2000, 8000, 32000)        # Messreihe „kritischer Punkt“: c = 1
CRITICAL_REPS = 20

PROCESS_N = 60                               # kleines Netz zum Zusehen: Kanten kommen eine nach der anderen dazu
PROCESS_C_MAX = 3.0                          # das Zusehen reicht bis mittlerer Grad 3 (90 Kanten)

COLOR_GIANT = "#d62728"
COLOR_OTHER = "rgba(120,120,120,0.85)"
COLOR_EDGE = "rgba(150,150,150,0.55)"
COLOR_MEASURED = "#2F6B65"
COLOR_THEORY = "#4c78a8"
COLOR_THIRD = "#e8a33d"

PRESETS = {
    "Standardfall (c = 1,5)": {"n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED},
    "Unterkritisch (c = 0,5)": {"n": DEFAULT_N, "c": 0.5, "seed": DEFAULT_SEED},
    "Am kritischen Punkt (c = 1)": {"n": DEFAULT_N, "c": 1.0, "seed": DEFAULT_SEED},
    "Kleines Netz knapp darüber": {"n": 300, "c": 1.2, "seed": DEFAULT_SEED},
    "Nahe am Zusammenhang (c = ln n + 1)": {"n": DEFAULT_N, "c": 8.6, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (c = 1,5)": "2 000 Knoten, im Mittel 1,5 Kanten je Knoten: über der Schwelle bei 1 liegt der größte Teil des Netzes in einer Riesenkomponente.",
    "Unterkritisch (c = 0,5)": "Im Mittel 0,5 Kanten je Knoten: nur kleine Inseln, die größte Komponente wächst höchstens langsam mit n.",
    "Am kritischen Punkt (c = 1)": "Genau auf der Schwelle: die größte Komponente ist weder winzig noch ein fester Anteil und wächst etwa wie n hoch zwei Drittel.",
    "Kleines Netz knapp darüber": "300 Knoten bei c = 1,2: bei endlicher Größe ist die Schwelle verschmiert, die Riesenkomponente streut stark von Netz zu Netz.",
    "Nahe am Zusammenhang (c = ln n + 1)": "2 000 Knoten mit mittlerem Grad ln n + 1: bis auf einzelne Inseln ist das Netz verbunden, nach der Theorie ganz verbunden mit rund 69 % Wahrscheinlichkeit.",
}
