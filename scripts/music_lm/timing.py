"""Deterministic rhythmic accounting for PrIMuS semantic tokens.

All durations are expressed in integer "ticks" on a grid where a quarter note =
QUARTER_TICKS. This avoids any floating-point equality comparisons when deciding
whether a measure is exactly full. With QUARTER_TICKS = 64, every duration that
can appear in the PrIMuS vocabulary (base durations down to a 64th note, with up
to two augmentation dots) maps to an exact integer, so measure-fill checks are
always exact.
"""

from __future__ import annotations

import re

QUARTER_TICKS = 64
WHOLE_TICKS = QUARTER_TICKS * 4  # 256

# Base (undotted) duration -> ticks. Covers every duration name the PrIMuS
# tooling knows about, not just the five that happen to occur in this corpus,
# so an unusual checkpoint/vocab can't produce an unparseable token.
_BASE_TICKS = {
    "quadruple_whole": WHOLE_TICKS * 4,
    "double_whole": WHOLE_TICKS * 2,
    "whole": WHOLE_TICKS,
    "half": WHOLE_TICKS // 2,
    "quarter": WHOLE_TICKS // 4,
    "eighth": WHOLE_TICKS // 8,
    "sixteenth": WHOLE_TICKS // 16,
    "thirty_second": WHOLE_TICKS // 32,
    "sixty_fourth": WHOLE_TICKS // 64,
}

_NOTE_RE = re.compile(r"^([A-G](?:#|b)?)(-?\d)_(.+)$")

# Common-time / cut-time aliases -> (numerator, denominator).
_TIME_ALIASES = {"C": (4, 4), "C/": (2, 2)}


def _dotted_ticks(base_ticks: int, dots: int) -> int:
    """Apply augmentation dots. Each dot adds half of the previous increment:
    dots=0 -> x1, dots=1 -> x1.5, dots=2 -> x1.75. Computed in integer ticks:
    base * (2^(dots+1) - 1) / 2^dots, which is exact for every base in this grid.
    """
    numerator = base_ticks * ((1 << (dots + 1)) - 1)
    denominator = 1 << dots
    if numerator % denominator != 0:
        raise ValueError(f"non-integer dotted duration: base={base_ticks} dots={dots}")
    return numerator // denominator


def duration_ticks(token: str) -> int | None:
    """Return the rhythmic length in ticks for a note/rest token, or None for a
    token that carries no duration (barline, tie, clef, key, time signature, or
    a special token). Grace notes carry 0 ticks.
    """
    if token.startswith("note-"):
        body = token[len("note-") :]
    elif token.startswith("rest-"):
        body = token[len("rest-") :]
    else:
        return None

    if body.endswith("_fermata"):
        body = body[: -len("_fermata")]

    grace = body.startswith("grace_")
    if grace:
        body = body[len("grace_") :]

    if token.startswith("note-"):
        m = _NOTE_RE.match(body)
        if not m:
            raise ValueError(f"unparseable note token: {token}")
        dur_field = m.group(3)
    else:
        dur_field = body  # rest: the whole remaining body is the duration field

    dots = dur_field.count(".")
    base_word = dur_field.replace(".", "")
    if base_word not in _BASE_TICKS:
        raise ValueError(f"unknown duration '{base_word}' in token: {token}")

    if grace:
        return 0
    return _dotted_ticks(_BASE_TICKS[base_word], dots)


def measure_ticks(time_signature_token: str) -> int:
    """Ticks per full measure for a timeSignature-* token (or a bare 'C'/'C/')."""
    body = time_signature_token
    if body.startswith("timeSignature-"):
        body = body[len("timeSignature-") :]

    if body in _TIME_ALIASES:
        num, den = _TIME_ALIASES[body]
    else:
        if "/" not in body:
            raise ValueError(f"unrecognized time signature: {time_signature_token}")
        num_s, den_s = body.split("/", 1)
        num, den = int(num_s), int(den_s)

    if WHOLE_TICKS % den != 0:
        raise ValueError(f"time signature denominator {den} not representable on tick grid")
    return num * (WHOLE_TICKS // den)


def validate_vocab(id_to_token: list[str]) -> int:
    """Assert every note/rest token in a vocab parses to a positive integer tick
    count. Returns the smallest note/rest tick value (used as the measure-fill
    safety threshold). Raises ValueError on the first unparseable token.
    """
    smallest = None
    for tok in id_to_token:
        if tok.startswith(("note-", "rest-")):
            ticks = duration_ticks(tok)
            if ticks is None or ticks < 0:
                raise ValueError(f"bad duration for {tok}: {ticks}")
            if ticks > 0 and (smallest is None or ticks < smallest):
                smallest = ticks
    if smallest is None:
        raise ValueError("vocab contains no note/rest tokens")
    return smallest
