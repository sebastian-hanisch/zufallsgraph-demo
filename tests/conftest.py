"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """7 Knoten, 5 Kanten: ein Weg 0-1-2, ein Dreieck 3-4-5 und der Einzelknoten 6.
    Handrechnung: Komponentengrößen 3, 3, 1; größte Komponente nach jeder Kante 1, 2, 3, 3, 3, 3 (nach 0, 1, ..., 5 Kanten)."""
    return 7, [(0, 1), (1, 2), (3, 4), (4, 5), (3, 5)]
