"""Pitch parsing and generation constraints for PrIMuS semantic note tokens."""

from __future__ import annotations

from music_lm.timing import split_note_or_rest

_PITCH_CLASS = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_CLEF_PITCH_LIMITS = {
    "clef-F4": ("B2", "F4"),
    "clef-G2": ("G3", "D6"),
}

MAX_JUMP_SEMITONES = 12


def pitch_to_midi(pitch: str) -> int:
    """Convert a pitch like C4, F#3, or Bb5 to a MIDI note number."""
    if len(pitch) < 2:
        raise ValueError(f"invalid pitch: {pitch!r}")

    letter = pitch[0]
    if letter not in _PITCH_CLASS:
        raise ValueError(f"invalid pitch letter: {pitch!r}")

    accidental = 0
    idx = 1
    while idx < len(pitch) and pitch[idx] in "#b":
        accidental += 1 if pitch[idx] == "#" else -1
        idx += 1

    octave = int(pitch[idx:])
    semitone = _PITCH_CLASS[letter] + accidental
    octave_adjust, semitone = divmod(semitone, 12)
    return (octave + octave_adjust + 1) * 12 + semitone


def midi_to_pitch(midi: int) -> str:
    """Convert a MIDI note number back to a pitch name with octave."""
    octave = midi // 12 - 1
    semitone = midi % 12
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[semitone]}{octave}"


def pitch_limits_for_clef(clef_token: str) -> tuple[int, int]:
    """Return inclusive MIDI bounds for the given clef token."""
    if clef_token not in _CLEF_PITCH_LIMITS:
        raise ValueError(f"unsupported clef for pitch limits: {clef_token!r}")
    low_name, high_name = _CLEF_PITCH_LIMITS[clef_token]
    return pitch_to_midi(low_name), pitch_to_midi(high_name)


def note_pitch_midi(token: str) -> int | None:
    """Return MIDI pitch for a note token, or None for non-note tokens."""
    if not token.startswith("note-"):
        return None
    pitch, _suffix = split_note_or_rest(token)
    return pitch_to_midi(pitch)


def last_note_pitch_midi(tokens: list[str]) -> int | None:
    """Return the most recent sounded pitch, skipping ties and structural tokens."""
    for token in reversed(tokens):
        if token == "tie":
            continue
        if token.startswith("rest-"):
            return None
        midi = note_pitch_midi(token)
        if midi is not None:
            return midi
    return None


def note_passes_pitch_constraints(
    token: str,
    clef_token: str,
    prev_pitch_midi: int | None,
    *,
    max_jump_semitones: int = MAX_JUMP_SEMITONES,
) -> bool:
    """True if a note token is inside the clef range and within jump distance."""
    if not token.startswith("note-"):
        return True

    midi = note_pitch_midi(token)
    if midi is None:
        return False

    low, high = pitch_limits_for_clef(clef_token)
    if not (low <= midi <= high):
        return False

    if prev_pitch_midi is not None:
        if abs(midi - prev_pitch_midi) > max_jump_semitones:
            return False

    return True


def clamp_pitch_to_clef(pitch: str, clef_token: str) -> str:
    """Clamp a pitch name to the inclusive range for a clef."""
    low, high = pitch_limits_for_clef(clef_token)
    midi = max(low, min(high, pitch_to_midi(pitch)))
    return midi_to_pitch(midi)
