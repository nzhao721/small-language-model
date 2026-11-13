"""Key signature helpers for music-LM generation and rendering."""

from __future__ import annotations

import re

from music_lm.pitch import pitch_limits_for_clef, pitch_to_midi
from music_lm.timing import split_note_or_rest

_KEY_TOKEN_RE = re.compile(r"^keySignature-([A-G](?:#|b)?)(M|m)$")

# Relative major used when conditioning a model that lacks minor key tokens.
_RELATIVE_MAJOR_FOR_MINOR: dict[str, str] = {
    "Am": "keySignature-CM",
    "Bbm": "keySignature-DbM",
    "Bm": "keySignature-DM",
    "Cm": "keySignature-EbM",
    "C#m": "keySignature-EM",
    "Dm": "keySignature-FM",
    "Ebm": "keySignature-GbM",
    "Em": "keySignature-GM",
    "F#m": "keySignature-AM",
    "Fm": "keySignature-AbM",
    "G#m": "keySignature-BM",
    "Gm": "keySignature-BbM",
}

# Twelve key centers in circle-of-fifths order: (major suffix, tonic, minor suffix, tonic).
KEY_CENTERS: list[tuple[str, str, str, str]] = [
    ("CM", "C", "Am", "A"),
    ("DbM", "Db", "Bbm", "Bb"),
    ("DM", "D", "Bm", "B"),
    ("EbM", "Eb", "Cm", "C"),
    ("EM", "E", "C#m", "C#"),
    ("FM", "F", "Dm", "D"),
    ("GbM", "Gb", "Ebm", "Eb"),
    ("GM", "G", "Em", "E"),
    ("AbM", "Ab", "Fm", "F"),
    ("AM", "A", "F#m", "F#"),
    ("BbM", "Bb", "Gm", "G"),
    ("BM", "B", "G#m", "G#"),
]

MAJOR_TO_MINOR: dict[str, str] = {
    f"keySignature-{major_suffix}": f"keySignature-{minor_suffix}"
    for major_suffix, _major_tonic, minor_suffix, _minor_tonic in KEY_CENTERS
}


def minor_key_tokens() -> list[str]:
    return [f"keySignature-{minor_suffix}" for _, _, minor_suffix, _ in KEY_CENTERS]


def parse_key_token(key_token: str) -> tuple[str, str]:
    """Return (tonic pitch name, mode) where mode is 'major' or 'minor'."""
    match = _KEY_TOKEN_RE.match(key_token)
    if not match:
        raise ValueError(f"unsupported key signature token: {key_token!r}")
    suffix, mode_flag = match.group(1), match.group(2)
    mode = "major" if mode_flag == "M" else "minor"
    return suffix, mode


def key_token(suffix: str) -> str:
    return f"keySignature-{suffix}"


def model_key_for(key_token: str) -> str:
    """Key token passed to the model (relative major for minor keys)."""
    _tonic, mode = parse_key_token(key_token)
    if mode == "major":
        return key_token
    minor_suffix = key_token.removeprefix("keySignature-")
    relative_major = _RELATIVE_MAJOR_FOR_MINOR.get(minor_suffix)
    if relative_major is None:
        raise ValueError(f"no relative major mapping for {key_token!r}")
    return relative_major


def pitch_class(pitch: str) -> int:
    return pitch_to_midi(pitch) % 12


def tonic_pitch_for_clef(tonic_root: str, clef_token: str) -> str:
    """Pick an octave so the tonic sits inside the clef's pitch range."""
    low, high = pitch_limits_for_clef(clef_token)
    preferred_octaves = (4, 3, 5, 2, 6) if clef_token == "clef-G2" else (3, 2, 4, 1, 5)

    for octave in preferred_octaves:
        pitch = f"{tonic_root}{octave}"
        midi = pitch_to_midi(pitch)
        if low <= midi <= high:
            return pitch

    raise ValueError(
        f"tonic {tonic_root!r} does not fit {clef_token} range "
        f"({low}-{high} MIDI)"
    )


def first_note_for(
    key_token: str,
    clef_token: str,
    duration: str,
    *,
    token_to_id: dict[str, int] | None = None,
) -> str:
    """Build a first note on the key's tonic with a vocab-backed spelling."""
    tonic_root, _mode = parse_key_token(key_token)
    pitch = tonic_pitch_for_clef(tonic_root, clef_token)
    candidates = [f"note-{pitch}_{duration}"]

    if token_to_id is not None:
        for candidate in candidates:
            if candidate in token_to_id:
                return candidate
        raise ValueError(
            f"no vocab token for tonic first note candidates: {candidates}"
        )

    return candidates[0]


def validate_first_note_is_tonic(tokens: list[str], key_token: str) -> None:
    """Raise if the first sounded note is not the key's tonic."""
    tonic_root, _mode = parse_key_token(key_token)
    expected_class = pitch_class(tonic_root + "4")

    for token in tokens:
        if token in {"<bos>", "<eos>", "<pad>", "<unk>"}:
            continue
        if token.startswith(("clef-", "keySignature-", "timeSignature-")):
            continue
        if not token.startswith("note-"):
            continue

        pitch, _suffix = split_note_or_rest(token)
        if pitch_class(pitch) != expected_class:
            raise ValueError(
                f"first note {token} is not tonic {tonic_root} for {key_token}"
            )
        return

    raise ValueError(f"no note found to validate tonic for {key_token}")


def apply_display_key(tokens: list[str], display_key: str, model_key: str) -> list[str]:
    """Replace the model-conditioning key token with the display key for output."""
    if display_key == model_key:
        return list(tokens)
    return [display_key if token == model_key else token for token in tokens]


def all_key_tokens() -> list[tuple[str, str]]:
    """Return (key_token, label) for every major and minor key."""
    keys: list[tuple[str, str]] = []
    for major_suffix, _major_tonic, minor_suffix, _minor_tonic in KEY_CENTERS:
        keys.append((key_token(major_suffix), major_suffix))
        keys.append((key_token(minor_suffix), minor_suffix))
    return keys
