"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset ändert mehr als einen Regler gegenüber dem Standard
oder liegt nicht auf einem Rand; die Beschreibungen sagen nichts, was die Rechnung nicht trägt."""

import zgr_constants as C
import zgr_evaluation as ev
import zgr_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.C_MIN <= preset["c"] <= C.C_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (c = 1,5)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_no_preset_sits_on_a_slider_bound():
    """Die Layout-Tests setzen jeden Regler auf seinen Rand: ein Preset dort würde die Hervorhebung des Standard-Presets verfälschen."""
    for name, preset in C.PRESETS.items():
        assert preset["n"] not in (C.N_MIN, C.N_MAX) and preset["c"] not in (C.C_MIN, C.C_MAX), name


def test_presets_are_pairwise_different():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_the_connectivity_preset_uses_the_natural_log_plus_one():
    p = C.PRESETS["Nahe am Zusammenhang (c = ln n + 1)"]
    import math

    assert abs(p["c"] - (math.log(p["n"]) + 1)) < 0.06  # 8,6 gegen ln 2000 + 1 = 8,60
