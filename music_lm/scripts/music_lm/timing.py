"""Duration/measure arithmetic for PrIMuS `.semantic` tokens.

Ticks are scaled so a whole note = 64 ticks, keeping durations down to a
double-dotted sixteenth note (7 ticks) integral.
"""

TICKS_PER_WHOLE = 64

BASE_TICKS = {
    "sixteenth": TICKS_PER_WHOLE // 16,
    "eighth": TICKS_PER_WHOLE // 8,
    "quarter": TICKS_PER_WHOLE // 4,
    "half": TICKS_PER_WHOLE // 2,
    "whole": TICKS_PER_WHOLE,
}

# Symbolic (non-fraction) time signatures map directly to their tick capacity.
# Common time (C) is 4/4; cut time (C/) is 2/2 -- both total 64 ticks here.
SYMBOLIC_TIME_SIGNATURE_TICKS = {
    "C": 4 * BASE_TICKS["quarter"],
    "C/": 4 * BASE_TICKS["quarter"],
}

_FERMATA_SUFFIX = "_fermata"


def _strip_fermata(suffix):
    if suffix.endswith(_FERMATA_SUFFIX):
        return suffix[: -len(_FERMATA_SUFFIX)]
    return suffix


def split_note_or_rest(token):
    """Return (pitch_or_None, duration_suffix) for a note-/rest- token."""
    if token.startswith("note-"):
        pitch, suffix = token[len("note-"):].split("_", 1)
        return pitch, suffix
    if token.startswith("rest-"):
        return None, token[len("rest-"):]
    raise ValueError(f"not a note/rest token: {token!r}")


def duration_ticks(duration_suffix):
    """Tick cost of a duration suffix, e.g. 'quarter', 'quarter..', 'eighth_fermata'."""
    suffix = _strip_fermata(duration_suffix)
    dots = 0
    while suffix.endswith("."):
        dots += 1
        suffix = suffix[:-1]
    value = BASE_TICKS[suffix]
    add = value
    for _ in range(dots):
        add /= 2
        value += add
    return int(round(value))


def token_tick_cost(token):
    """Tick cost of any token: 0 for structural tokens (barline, tie, clef, ...)."""
    if token.startswith("note-") or token.startswith("rest-"):
        _, suffix = split_note_or_rest(token)
        return duration_ticks(suffix)
    return 0


def measure_ticks(time_signature_token):
    """Ticks-per-measure for a timeSignature-* token, handling symbolic C / C/."""
    body = time_signature_token[len("timeSignature-"):]
    if body in SYMBOLIC_TIME_SIGNATURE_TICKS:
        return SYMBOLIC_TIME_SIGNATURE_TICKS[body]
    beats, unit = body.split("/")
    return int(beats) * (TICKS_PER_WHOLE // int(unit))


def collect_duration_costs(tokens):
    """Unique positive tick costs available from note/rest tokens."""
    costs = set()
    for token in tokens:
        cost = token_tick_cost(token)
        if cost > 0:
            costs.add(cost)
    return costs


def can_fill_ticks(remaining, costs):
    """True if *remaining* ticks can be formed exactly from *costs*."""
    if remaining == 0:
        return True
    if remaining < 0:
        return False

    memo: dict[int, bool] = {}

    def fillable(ticks):
        if ticks in memo:
            return memo[ticks]
        if ticks == 0:
            return True
        for cost in costs:
            if cost <= ticks and fillable(ticks - cost):
                memo[ticks] = True
                return True
        memo[ticks] = False
        return False

    return fillable(remaining)


def duration_group_key(token):
    """Grouping key used by the duration-temperature sampling stage.

    Notes are grouped by duration suffix (fermata stripped), so any pitch at
    the same rhythmic value competes only at the pitch-temperature stage.
    Rests get their own 'rest:<suffix>' bucket, separate from same-duration
    notes, since resting is a rhythmic choice with no pitch dimension -- and
    since there is exactly one rest token per duration, these buckets are
    singletons that resolve immediately once chosen. Structural tokens
    (barline, tie, clef-*, keySignature-*, timeSignature-*, specials) are
    singleton groups keyed by themselves.
    """
    if token.startswith("note-"):
        _, suffix = split_note_or_rest(token)
        return _strip_fermata(suffix)
    if token.startswith("rest-"):
        _, suffix = split_note_or_rest(token)
        return "rest:" + _strip_fermata(suffix)
    return token
