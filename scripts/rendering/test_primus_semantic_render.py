"""Tests for PrIMuS semantic rendering."""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve()
while _SCRIPTS.name != "scripts":
    _SCRIPTS = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from _bootstrap import setup  # noqa: E402

setup(__file__)

from primus_semantic_render import (
    _metric_beam_groups_for_measure,
    _resolve_note_accid,
    semantic_to_mei,
)


def _measure_accidentals() -> dict[tuple[str, str], str]:
    return {}


def test_ab_major_e_natural_then_e_flat_shows_both_accidentals() -> None:
    measure = _measure_accidentals()
    key = "4f"  # Ab major

    assert _resolve_note_accid("e", "4", None, key, measure) == "n"
    assert measure[("e", "4")] == "n"

    assert _resolve_note_accid("e", "4", "f", key, measure) == "f"
    assert measure[("e", "4")] == "f"


def test_ab_major_first_e_flat_omits_redundant_accidental() -> None:
    measure = _measure_accidentals()
    key = "4f"

    assert _resolve_note_accid("e", "4", "f", key, measure) is None
    assert measure[("e", "4")] == "f"


def test_ab_major_second_e_natural_omits_redundant_accidental() -> None:
    measure = _measure_accidentals()
    key = "4f"
    measure[("e", "4")] = "n"

    assert _resolve_note_accid("e", "4", None, key, measure) is None


def test_gs1_measure_three_emits_reinstatement_flat() -> None:
    text = (
        "clef-F4 keySignature-AbM timeSignature-4/4 "
        "note-F4_quarter note-E4_quarter note-Eb4_eighth barline"
    )
    mei = semantic_to_mei(text, autobeam=False)
    assert mei.count('accid="n"') == 1
    assert mei.count('accid="f"') == 1


def test_4_4_eighths_beam_in_beat_pairs() -> None:
    tokens = ["note-C4_eighth"] * 8
    assert _metric_beam_groups_for_measure(tokens, "4", "4", False) == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]


def test_3_4_has_three_beat_groups() -> None:
    tokens = ["note-C4_eighth"] * 6
    assert _metric_beam_groups_for_measure(tokens, "3", "4", False) == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]


def test_6_8_beams_groups_of_three() -> None:
    tokens = ["note-C4_eighth"] * 6
    assert _metric_beam_groups_for_measure(tokens, "6", "8", False) == [
        (0, 2),
        (3, 5),
    ]


def test_2_2_beams_by_half_note() -> None:
    tokens = ["note-C4_eighth"] * 8
    assert _metric_beam_groups_for_measure(tokens, "2", "2", False) == [
        (0, 3),
        (4, 7),
    ]


def test_rest_within_beat_does_not_break_beam() -> None:
    tokens = [
        "note-C4_sixteenth",
        "note-D4_sixteenth",
        "rest-sixteenth",
        "note-E4_sixteenth",
    ]
    assert _metric_beam_groups_for_measure(tokens, "4", "4", False) == [(0, 3)]


def test_quarter_note_splits_beams_across_beats() -> None:
    tokens = [
        "note-C4_eighth",
        "note-D4_eighth",
        "note-E4_quarter",
        "note-F4_eighth",
        "note-G4_eighth",
    ]
    assert _metric_beam_groups_for_measure(tokens, "4", "4", False) == [
        (0, 1),
        (3, 4),
    ]


def test_single_eighth_in_beat_is_not_beamed() -> None:
    tokens = ["note-C4_quarter", "note-D4_eighth", "note-E4_quarter", "note-F4_quarter"]
    assert _metric_beam_groups_for_measure(tokens, "4", "4", False) == []


def test_gs1_measure_two_uses_four_beat_groups() -> None:
    semantic = Path(
        r"C:\alpha_ai\small-language-model\golden_set\manual\ground_truth\gs1.semantic"
    ).read_text(encoding="utf-8")
    measure_tokens = semantic.split()[9:17]
    assert _metric_beam_groups_for_measure(measure_tokens, "4", "4", False) == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]


def test_gs0_musicxml_diatonic_flats_have_pitch_alter() -> None:
    from primus_semantic_to_musicxml import semantic_to_musicxml

    semantic = Path(
        r"C:\alpha_ai\small-language-model\golden_set\manual\ground_truth\gs0.semantic"
    ).read_text(encoding="utf-8")
    xml = semantic_to_musicxml(semantic)
    assert "<attributes><divisions>16</divisions><key>" in xml
    assert "<step>A</step><alter>-1</alter><octave>4</octave>" in xml
    assert "<step>D</step><alter>-1</alter><octave>4</octave>" in xml
    assert "accidental>natural</accidental>" in xml


def test_gs1_musicxml_has_measures_beams_and_accidentals() -> None:
    from primus_semantic_to_musicxml import semantic_to_musicxml

    semantic = Path(
        r"C:\alpha_ai\small-language-model\golden_set\manual\ground_truth\gs1.semantic"
    ).read_text(encoding="utf-8")
    xml = semantic_to_musicxml(semantic)
    assert "<score-partwise" in xml
    assert xml.count("<measure") == 4
    assert xml.count('beam number="1">begin</beam>') == 7
    assert "accidental>natural</accidental>" in xml
    assert "accidental>flat</accidental>" in xml

