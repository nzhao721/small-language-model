"""Detect and limit repetitive pitch and duration runs during generation."""

from __future__ import annotations

from music_lm.pitch import note_pitch_midi
from music_lm.timing import duration_group_key, split_note_or_rest


def _note_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token.startswith("note-")]


def consecutive_same_pitch_run(
    ids: list[int],
    id_to_token: dict[int, str],
) -> tuple[int, int | None]:
    """Return (run_length, midi) for the trailing chain of the same pitch."""
    run = 0
    target: int | None = None
    for tid in reversed(ids):
        token = id_to_token[tid]
        if token == "tie":
            continue
        if token.startswith("rest-"):
            break
        midi = note_pitch_midi(token)
        if midi is None:
            break
        if target is None:
            target = midi
            run = 1
        elif midi == target:
            run += 1
        else:
            break
    return run, target


def consecutive_same_duration_run(
    ids: list[int],
    id_to_token: dict[int, str],
) -> tuple[int, str | None]:
    """Return (run_length, duration_key) for the trailing chain of the same duration."""
    run = 0
    target: str | None = None
    for tid in reversed(ids):
        token = id_to_token[tid]
        if token == "tie":
            continue
        if not token.startswith("note-"):
            break
        key = duration_group_key(token)
        if target is None:
            target = key
            run = 1
        elif key == target:
            run += 1
        else:
            break
    return run, target


def longest_same_pitch_run(tokens: list[str]) -> int:
    """Longest run of consecutive notes with the same pitch."""
    pitches = [split_note_or_rest(token)[0] for token in _note_tokens(tokens)]
    if not pitches:
        return 0

    best = 1
    run = 1
    for index in range(1, len(pitches)):
        if pitches[index] == pitches[index - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def longest_same_duration_run(tokens: list[str]) -> int:
    """Longest run of consecutive notes with the same duration value."""
    durations = [duration_group_key(token) for token in _note_tokens(tokens)]
    if not durations:
        return 0

    best = 1
    run = 1
    for index in range(1, len(durations)):
        if durations[index] == durations[index - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def is_overly_repetitive(
    tokens: list[str],
    *,
    max_same_pitch_run: int = 3,
    max_same_duration_run: int = 6,
) -> bool:
    """True when pitch or duration repeats too many times in a row."""
    return (
        longest_same_pitch_run(tokens) > max_same_pitch_run
        or longest_same_duration_run(tokens) > max_same_duration_run
    )


def filter_allowed_avoid_pitch(
    allowed: list[int],
    id_to_token: dict[int, str],
    avoid_midi: int,
    *,
    tie_id: int | None,
) -> list[int]:
    """Drop note tokens on *avoid_midi*, keeping ties and other options."""
    filtered = []
    for tid in allowed:
        if tie_id is not None and tid == tie_id:
            filtered.append(tid)
            continue
        token = id_to_token[tid]
        midi = note_pitch_midi(token)
        if midi is None or midi != avoid_midi:
            filtered.append(tid)
    return filtered


def filter_allowed_avoid_duration(
    allowed: list[int],
    id_to_token: dict[int, str],
    avoid_duration_key: str,
    *,
    tie_id: int | None,
) -> list[int]:
    """Drop note tokens with *avoid_duration_key*, keeping ties and other options."""
    filtered = []
    for tid in allowed:
        if tie_id is not None and tid == tie_id:
            filtered.append(tid)
            continue
        token = id_to_token[tid]
        if not token.startswith("note-"):
            filtered.append(tid)
            continue
        if duration_group_key(token) != avoid_duration_key:
            filtered.append(tid)
    return filtered
