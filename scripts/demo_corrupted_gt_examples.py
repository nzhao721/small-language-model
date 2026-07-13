"""Corrupt PrIMuS semantic GT and render clean / scanned / handwritten examples."""

from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

import cv2
import mashcima as mc
import numpy as np
from mashcima.annotation_to_image import token_groups_to_canvas
from mashcima.Canvas import Canvas
from mashcima.primus_adapter import convert_primus_annotation_to_mashcima_annotation
from mashcima.vocabulary import (
    is_accidental,
    parse_annotation_into_token_groups,
    repair_annotation,
    to_generic,
)
from PIL import Image

from demo_mashcima_camera_distort import camera_like_distort, seed_for_name
from generate_mashcima_batch import ensure_muscima_cache, image_to_uint8, resize_long_edge
from mashcima_writers import configure_mashcima_writers, symbol_repository_for_key
from primus_semantic_render import (
    KEY_SIG_ALTERATIONS,
    KEY_SIG_TO_MEI,
    render_semantic_to_image,
)

ROOT = Path(__file__).resolve().parents[1]
PRIMUS_DIR = ROOT / "data" / "primus"
PRIMUS_TGZ = ROOT / "data" / "archives" / "primusCalvoRizoAppliedSciences2018.tgz"
OUT_DIR = ROOT / "examples" / "corrupted_gt_demo"

SAMPLES = [
    "000051650-1_1_1",
    "000051662-1_1_1",
    "000051756-1_1_1",
]

NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]
ACCIDENTALS = ["", "#", "b"]
SEMANTIC_DURATIONS = [
    "whole",
    "half",
    "quarter",
    "eighth",
    "sixteenth",
]
DURATION_TO_MASH = {
    "double_whole": "w",
    "whole": "w",
    "half": "h",
    "quarter": "q",
    "eighth": "e",
    "sixteenth": "s",
    "thirty_second": "s",
}
DURATION_TO_MASH_REST = {
    "double_whole": "br",
    "whole": "wr",
    "half": "hr",
    "quarter": "qr",
    "eighth": "er",
    "sixteenth": "sr",
    "thirty_second": "sr",
}
DURATION_ALIASES = {
    "t": "thirty_second",
    "s": "sixteenth",
    "e": "eighth",
    "q": "quarter",
    "h": "half",
    "w": "whole",
}
BARE_DURATION_TO_REST = {
    "w": "wr",
    "h": "hr",
    "q": "qr",
    "e": "er",
    "s": "sr",
    "t": "sr",
}
MASH_REST_TYPES = ["lr", "br", "wr", "hr", "qr", "er", "sr"]
MASH_NOTE_REST_RE = re.compile(
    r"^(?:"
    r"lr|br|wr|hr|qr|er|sr|tr"
    r"|w-?\d+|h-?\d+|q-?\d+|e-?\d+|s-?\d+|t-?\d+"
    r"|e=-?\d+|=e=-?\d+|=e-?\d+"
    r"|s=-?\d+|=s=-?\d+|=s-?\d+"
    r"|t=-?\d+|=t=-?\d+|=t-?\d+"
    r")$"
)
SEMANTIC_NOTE_RE = re.compile(r"^note-([A-G](?:#|b)?)(\d)_(.+)$")
SEMANTIC_REST_RE = re.compile(r"^rest-(.+)$")
HEADER_PREFIXES = ("clef-", "keySignature-", "timeSignature-")

NOTE_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PITCH_OFFSETS = [-12, -7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 12]
PITCH_OFFSET_WEIGHTS = [1, 2, 3, 4, 6, 10, 14, 14, 10, 6, 4, 3, 2, 1]
STAFF_OFFSETS = [-4, -3, -2, -1, 1, 2, 3, 4]
STAFF_OFFSET_WEIGHTS = [1, 2, 5, 9, 9, 5, 2, 1]
DURATION_OFFSETS = [-2, -1, 1, 2]
DURATION_OFFSET_WEIGHTS = [1, 5, 5, 1]
CLUSTER_EXTEND_PROB = 0.72
CLUSTER_BOOST = 3.2


def find_incipt_dir(incipit_id: str) -> Path:
    matches = [p.parent for p in PRIMUS_DIR.rglob(f"{incipit_id}.semantic")]
    if not matches:
        raise FileNotFoundError(f"No PrIMuS folder for {incipit_id}")
    return matches[0]


def split_semantic(tokens: list[str]) -> tuple[list[str], list[str]]:
    header: list[str] = []
    body: list[str] = []
    in_header = True
    for token in tokens:
        if in_header and token.startswith(HEADER_PREFIXES):
            header.append(token)
        else:
            in_header = False
            body.append(token)
    return header, body


def mashcima_pitch(token: str) -> int | None:
    digits = re.search(r"(-?\d+)$", token)
    return int(digits.group(1)) if digits else None


def set_mashcima_pitch(token: str, pitch: int) -> str:
    if mashcima_pitch(token) is None:
        return token
    return re.sub(r"-?\d+$", str(pitch), token)


def normalize_semantic_duration(duration: str) -> str:
    base, dots = split_duration_body(duration)
    base = DURATION_ALIASES.get(base, base)
    return base + "." * dots


def mash_rest_for_duration(duration: str) -> str:
    base, _ = split_duration_body(normalize_semantic_duration(duration))
    return DURATION_TO_MASH_REST.get(base, "qr")


def mash_note_duration_letter(duration: str) -> str:
    base, _ = split_duration_body(normalize_semantic_duration(duration))
    return DURATION_TO_MASH.get(base, "q")


def set_mashcima_duration(token: str, duration: str, rng: random.Random | None = None) -> str:
    mash = mash_note_duration_letter(duration)
    pitch = mashcima_pitch(token)
    if pitch is None:
        return rng.choice(MASH_REST_TYPES) if rng else MASH_REST_TYPES[0]
    if "=" in token:
        if token.startswith("=") and token.count("=") >= 2:
            return f"={mash}={pitch}"
        if token.endswith("="):
            return f"{mash}={pitch}"
        return f"={mash}{pitch}"
    return f"{mash}{pitch}"


def split_duration_body(body: str) -> tuple[str, int]:
    dots = 0
    while body.endswith("."):
        body = body[:-1]
        dots += 1
    if body.endswith("_fermata"):
        body = body[: -len("_fermata")]
    return body, dots


def semantic_note_to_midi(name_acc: str, octave: int) -> int:
    name = name_acc[0]
    acc = name_acc[1:] if len(name_acc) > 1 else ""
    semitone = NOTE_TO_SEMITONE[name]
    if acc == "#":
        semitone += 1
    elif acc == "b":
        semitone -= 1
    return semitone + 12 * (octave + 1)


def midi_to_semantic_note(midi: int) -> tuple[str, int]:
    midi = max(36, min(96, midi))
    octave = midi // 12 - 1
    semitone = midi % 12
    names = [
        ("C", 0),
        ("Db", 1),
        ("D", 2),
        ("Eb", 3),
        ("E", 4),
        ("F", 5),
        ("Gb", 6),
        ("G", 7),
        ("Ab", 8),
        ("A", 9),
        ("Bb", 10),
        ("B", 11),
    ]
    name, _ = min(names, key=lambda item: abs(item[1] - semitone))
    if name.endswith("b"):
        return name[0] + "b", octave
    return name[0], octave


def nearby_duration(base: str, rng: random.Random) -> str:
    if base not in SEMANTIC_DURATIONS:
        return rng.choice(SEMANTIC_DURATIONS)
    idx = SEMANTIC_DURATIONS.index(base)
    offset = rng.choices(DURATION_OFFSETS, weights=DURATION_OFFSET_WEIGHTS, k=1)[0]
    return SEMANTIC_DURATIONS[max(0, min(len(SEMANTIC_DURATIONS) - 1, idx + offset))]


def nearby_semantic_note(from_token: str, rng: random.Random) -> str:
    m = SEMANTIC_NOTE_RE.match(from_token)
    if not m:
        return random_semantic_note(rng)

    name_acc, octave_s, body = m.group(1), int(m.group(2)), m.group(3)
    base_duration, dots = split_duration_body(body)
    duration = nearby_duration(base_duration, rng)
    duration += "." * dots

    midi = semantic_note_to_midi(name_acc, octave_s)
    if rng.random() < 0.82:
        offset = rng.choices(PITCH_OFFSETS, weights=PITCH_OFFSET_WEIGHTS, k=1)[0]
        midi += offset
    else:
        midi += rng.randint(-12, 12)

    new_name, new_octave = midi_to_semantic_note(midi)
    return f"note-{new_name}{new_octave}_{duration}"


def nearby_semantic_rest(from_token: str, rng: random.Random) -> str:
    m = SEMANTIC_REST_RE.match(from_token)
    if not m:
        return random_semantic_rest(rng)
    base_duration, dots = split_duration_body(m.group(1))
    duration = nearby_duration(base_duration, rng)
    return f"rest-{duration}{'.' * dots}"


def random_semantic_note(rng: random.Random) -> str:
    name = rng.choice(NOTE_NAMES) + rng.choice(ACCIDENTALS)
    octave = str(rng.randint(2, 6))
    duration = rng.choice(SEMANTIC_DURATIONS)
    return f"note-{name}{octave}_{duration}"


def random_semantic_rest(rng: random.Random) -> str:
    return f"rest-{rng.choice(SEMANTIC_DURATIONS)}"


def nearby_mashcima_note(from_token: str, rng: random.Random) -> str:
    pitch = mashcima_pitch(from_token)
    if pitch is None:
        return random_mashcima_note_rest(rng)

    offset = rng.choices(STAFF_OFFSETS, weights=STAFF_OFFSET_WEIGHTS, k=1)[0]
    new_pitch = max(-12, min(12, pitch + offset))
    new_token = set_mashcima_pitch(from_token, new_pitch)

    if not is_beamed_mash(from_token) and rng.random() < 0.35:
        mash_kind = re.sub(r"-?\d+$", "", from_token)
        if mash_kind in {"w", "h", "q", "e", "s", "t"}:
            kind_to_idx = {"w": 1, "h": 2, "q": 3, "e": 4, "s": 5, "t": 5}
            idx = kind_to_idx[mash_kind]
            choices = [i for i in (idx - 1, idx + 1) if 1 <= i <= 5]
            if choices:
                idx_to_kind = {1: "w", 2: "h", 3: "q", 4: "e", 5: "s"}
                new_kind = idx_to_kind[rng.choice(choices)]
                new_token = f"{new_kind}{new_pitch}"
    return new_token


def nearby_mashcima_note_rest(from_sem: str, from_mash: str, rng: random.Random) -> str:
    if from_mash == "|" or SEMANTIC_REST_RE.match(from_sem):
        base = from_sem if SEMANTIC_REST_RE.match(from_sem) else "rest-quarter"
        new_sem = nearby_semantic_rest(base, rng)
        _, rest_body = new_sem.split("-", 1)
        base_duration, _ = split_duration_body(rest_body)
        return mash_rest_for_duration(base_duration)
    if SEMANTIC_NOTE_RE.match(from_sem):
        return nearby_mashcima_note(from_mash, rng)
    return random_mashcima_note_rest(rng)


def random_mashcima_note_rest(rng: random.Random) -> str:
    if rng.random() < 0.85:
        pitch = rng.randint(-5, 10)
        mash = rng.choice(["w", "h", "q", "e", "s", "=e=", "e=", "=e"])
        if mash in {"w", "h", "q", "e", "s"}:
            return f"{mash}{pitch}"
        if mash == "=e=":
            return f"=e={pitch}"
        if mash == "e=":
            return f"e={pitch}"
        return f"=e{pitch}"
    return rng.choice(MASH_REST_TYPES)


def is_mash_alignable(token: str) -> bool:
    return token == "|" or bool(MASH_NOTE_REST_RE.match(token))


def is_sem_alignable(token: str) -> bool:
    return token == "barline" or bool(SEMANTIC_NOTE_RE.match(token) or SEMANTIC_REST_RE.match(token))


def extract_aligned_pairs(
    semantic_body: list[str],
    mash_tokens: list[str],
) -> list[tuple[str, str]]:
    mash_alignable = [t for t in mash_tokens if is_mash_alignable(t)]
    sem_alignable = [t for t in semantic_body if is_sem_alignable(t)]
    if len(sem_alignable) != len(mash_alignable):
        raise ValueError(
            f"Alignment mismatch: {len(sem_alignable)} semantic vs {len(mash_alignable)} mashcima"
        )
    return list(zip(sem_alignable, mash_alignable, strict=True))


def rebuild_semantic_body(semantic_body: list[str], pairs: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    pair_iter = iter(pairs)

    for token in semantic_body:
        if token == "barline":
            try:
                sem, _ = next(pair_iter)
                out.append(sem if sem == "barline" else "barline")
            except StopIteration:
                pass
            continue
        if SEMANTIC_NOTE_RE.match(token) or SEMANTIC_REST_RE.match(token):
            try:
                sem, _ = next(pair_iter)
                out.append(sem)
            except StopIteration:
                pass

    for sem, _ in pair_iter:
        out.append(sem)
    return out


def rebuild_mashcima_tokens(mash_tokens: list[str], pairs: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    pair_iter = iter(pairs)

    for token in mash_tokens:
        if not is_mash_alignable(token):
            out.append(token)
            continue
        try:
            _, mash = next(pair_iter)
            out.append(mash if mash == "|" else mash)
        except StopIteration:
            pass

    for _, mash in pair_iter:
        out.append(mash)
    return out


_DIATONIC_INDEX = {name: index for index, name in enumerate("CDEFGAB")}
_CLEF_REFERENCE = {
    "clef-G2": ("G", 4, -2),
    "clef-F4": ("F", 3, 2),
    "clef-F3": ("F", 3, 0),
    "clef-C1": ("C", 4, -4),
    "clef-C2": ("C", 4, -2),
    "clef-C3": ("C", 4, 0),
    "clef-C4": ("C", 4, 2),
}


def _diatonic_position(name: str, octave: int) -> int:
    return octave * 7 + _DIATONIC_INDEX[name]


def semantic_note_to_mash_pitch(token: str, clef_token: str) -> int:
    """Map a semantic note to Mashcima's staff-position coordinate."""
    match = SEMANTIC_NOTE_RE.match(token)
    if match is None:
        raise ValueError(f"not a semantic note: {token}")
    if clef_token not in _CLEF_REFERENCE:
        raise ValueError(f"unsupported clef for Mashcima synchronization: {clef_token}")
    ref_name, ref_octave, ref_pitch = _CLEF_REFERENCE[clef_token]
    name = match.group(1)[0]
    octave = int(match.group(2))
    return ref_pitch + (
        _diatonic_position(name, octave)
        - _diatonic_position(ref_name, ref_octave)
    )


def _mash_note_for_semantic(
    semantic_token: str,
    clef_token: str,
    previous_mash: str,
) -> str:
    match = SEMANTIC_NOTE_RE.match(semantic_token)
    if match is None:
        raise ValueError(f"not a semantic note: {semantic_token}")
    duration, _dots = split_duration_body(match.group(3))
    kind = mash_note_duration_letter(duration)
    pitch = semantic_note_to_mash_pitch(semantic_token, clef_token)
    # Preserve a valid beam position only when the replacement duration remains
    # beamable. Pitch and duration always come from the semantic token.
    if kind in {"e", "s"} and "=" in previous_mash:
        if previous_mash.startswith("=") and previous_mash.count("=") >= 2:
            return f"={kind}={pitch}"
        if previous_mash.endswith("="):
            return f"{kind}={pitch}"
        return f"={kind}{pitch}"
    return f"{kind}{pitch}"


def _key_signature_state(key_token: str | None) -> dict[str, str]:
    if key_token is None:
        return {}
    mei = KEY_SIG_TO_MEI.get(key_token, "0")
    if mei == "0":
        return {}
    accidental = "#" if mei.endswith("s") else "b"
    return {letter: accidental for letter in KEY_SIG_ALTERATIONS.get(mei, ())}


_SEMANTIC_CLEF_TO_MASH = {
    "clef-G2": "clef.G2",
    "clef-F4": "clef.F2",
    "clef-F3": "clef.F3",
    "clef-C1": "clef.C1",
    "clef-C2": "clef.C2",
    "clef-C3": "clef.C3",
    "clef-C4": "clef.C4",
}


def _semantic_time_to_mashcima(time_token: str) -> list[str]:
    if time_token == "timeSignature-C":
        return ["time.C"]
    if time_token in {"timeSignature-C/", "timeSignature-C|"}:
        return ["time.C/"]
    match = re.fullmatch(r"timeSignature-(\d+)/(\d+)", time_token)
    if match is None:
        raise ValueError(f"unsupported time signature token: {time_token}")
    return [f"time.{match.group(1)}", f"time.{match.group(2)}"]


def _semantic_header_to_mashcima(semantic_tokens: list[str]) -> list[str]:
    """Build Mashcima clef / key / time header from PrIMuS semantic tokens."""
    out: list[str] = []
    clef_token = next(t for t in semantic_tokens if t.startswith("clef-"))
    key_token = next((t for t in semantic_tokens if t.startswith("keySignature-")), None)
    time_token = next(t for t in semantic_tokens if t.startswith("timeSignature-"))

    mash_clef = _SEMANTIC_CLEF_TO_MASH.get(clef_token)
    if mash_clef is None:
        raise ValueError(f"unsupported clef for Mashcima render: {clef_token}")
    out.append(mash_clef)

    # Key-signature accidentals: place on conventional staff positions for the clef.
    key_state = _key_signature_state(key_token)
    if key_state:
        # Approximate staff positions for key accidentals (sharp/flat series order).
        sharp_order = ["f", "c", "g", "d", "a", "e", "b"]
        flat_order = ["b", "e", "a", "d", "g", "c", "f"]
        sharp_pos = {"f": 4, "c": 1, "g": 5, "d": 2, "a": 6, "e": 3, "b": 7}
        flat_pos = {"b": 3, "e": 6, "a": 2, "d": 5, "g": 1, "c": 4, "f": 0}
        # Shift relative to treble; bass (F2) is -2 staff steps vs G2 for letter positions.
        bass_shift = -2 if mash_clef.startswith("clef.F") else 0
        if any(v == "#" for v in key_state.values()):
            for letter in sharp_order:
                if key_state.get(letter) == "#":
                    out.append(f"#{sharp_pos[letter] + bass_shift}")
        else:
            for letter in flat_order:
                if key_state.get(letter) == "b":
                    out.append(f"b{flat_pos[letter] + bass_shift}")

    out.extend(_semantic_time_to_mashcima(time_token))
    return out


def _append_semantic_event_as_mashcima(
    out: list[str],
    semantic: str,
    *,
    clef_token: str,
    key_state: dict[str, str],
    measure_state: dict[str, str],
    previous_mash: str = "",
) -> None:
    """Append Mashcima tokens for one semantic note/rest/barline event."""
    if semantic == "barline":
        out.append("|")
        measure_state.clear()
        return

    rest_match = SEMANTIC_REST_RE.match(semantic)
    if rest_match is not None:
        duration, dots = split_duration_body(rest_match.group(1))
        out.append(mash_rest_for_duration(duration))
        if dots:
            out.append("*" * min(dots, 2))
        return

    note_match = SEMANTIC_NOTE_RE.match(semantic)
    if note_match is None:
        raise ValueError(f"unsupported semantic event: {semantic}")
    name_acc = note_match.group(1)
    letter = name_acc[0].lower()
    desired = name_acc[1:] if len(name_acc) > 1 else ""
    current = measure_state.get(letter, key_state.get(letter, ""))
    pitch = semantic_note_to_mash_pitch(semantic, clef_token)
    if desired != current:
        out.append(f"{desired or 'N'}{pitch}")
        measure_state[letter] = desired

    out.append(_mash_note_for_semantic(semantic, clef_token, previous_mash))
    _duration, dots = split_duration_body(note_match.group(3))
    if dots:
        out.append("*" * min(dots, 2))
    if note_match.group(3).endswith("_fermata"):
        out.append("fermata")


def mashcima_annotation_from_semantic(
    semantic_text: str,
    *,
    layout_annotation: str | None = None,
) -> str:
    """Build a Mashcima annotation whose note/rest/barline events match ``semantic_text``.

    Prefer ``layout_annotation`` (original or replayed Mashcima) when its event
    count matches, so beams/header layout are preserved. Otherwise rebuild the
    body from the semantic alone (unbeamed notes) using the layout header when
    available, else a header derived from the semantic.
    """
    semantic_tokens = semantic_text.split()
    clef_token = next(t for t in semantic_tokens if t.startswith("clef-"))
    key_token = next(
        (t for t in semantic_tokens if t.startswith("keySignature-")),
        None,
    )
    _, semantic_body = split_semantic(semantic_tokens)
    events = [token for token in semantic_body if is_sem_alignable(token)]

    if layout_annotation:
        source_tokens = layout_annotation.split()
        source_events = [token for token in source_tokens if is_mash_alignable(token)]
        if len(events) == len(source_events):
            return synchronize_mashcima_to_semantic(layout_annotation, semantic_text)
        header = []
        for token in source_tokens:
            if is_mash_alignable(token):
                break
            header.append(token)
    else:
        header = _semantic_header_to_mashcima(semantic_tokens)

    key_state = _key_signature_state(key_token)
    measure_state: dict[str, str] = {}
    out = list(header)
    for semantic in events:
        _append_semantic_event_as_mashcima(
            out,
            semantic,
            clef_token=clef_token,
            key_state=key_state,
            measure_state=measure_state,
            previous_mash="",
        )
    return " ".join(out)


def synchronize_mashcima_to_semantic(annotation: str, semantic_text: str) -> str:
    """Make every handwritten note/rest glyph agree with its semantic event.

    Corruption historically changed the semantic and Mashcima symbols through
    separate random draws. That occasionally produced one wrong notehead,
    staff position, rest, or duration in an otherwise correct handwritten
    render. This rebuild uses semantic tokens as the source of truth while
    retaining the original handwritten layout, beam context, and header.
    """
    semantic_tokens = semantic_text.split()
    clef_token = next(t for t in semantic_tokens if t.startswith("clef-"))
    key_token = next(
        (t for t in semantic_tokens if t.startswith("keySignature-")),
        None,
    )
    _, semantic_body = split_semantic(semantic_tokens)
    events = [token for token in semantic_body if is_sem_alignable(token)]
    source_tokens = annotation.split()
    source_events = [token for token in source_tokens if is_mash_alignable(token)]
    if len(events) != len(source_events):
        raise ValueError(
            "Cannot synchronize handwritten annotation: "
            f"{len(events)} semantic events vs {len(source_events)} Mashcima events"
        )

    key_state = _key_signature_state(key_token)
    measure_state: dict[str, str] = {}
    event_iter = iter(events)
    out: list[str] = []
    after_time = False

    for source in source_tokens:
        if source.startswith("time."):
            after_time = True
            out.append(source)
            continue
        if not is_mash_alignable(source):
            # Key-signature accidentals occur before the time signature and must
            # remain. Event accidentals/duration dots are regenerated below.
            if after_time and (is_accidental(source) or source in {"*", "**", "fermata"}):
                continue
            out.append(source)
            continue

        semantic = next(event_iter)
        before = len(out)
        _append_semantic_event_as_mashcima(
            out,
            semantic,
            clef_token=clef_token,
            key_state=key_state,
            measure_state=measure_state,
            previous_mash=source,
        )
        # _append already mutated measure_state for barlines.
        _ = before

    return " ".join(out)


def is_beamed_mash(token: str) -> bool:
    return "=" in token and MASH_NOTE_REST_RE.match(token) is not None


def beam_context_indices(pairs: list[tuple[str, str]]) -> set[int]:
    """Indices whose mash token is beamed or adjacent to a beamed note (unsafe to corrupt)."""
    blocked: set[int] = set()
    for i, (_, mash) in enumerate(pairs):
        if "=" in mash:
            blocked.add(i)
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(pairs) and "=" in pairs[j][1]:
                blocked.add(i)
    return blocked


def downgrade_thirty_second_token(token: str) -> str:
    """Map unrenderable thirty-second mashcima tokens to sixteenth equivalents."""
    if token == "tr":
        return "sr"
    if token in BARE_DURATION_TO_REST:
        return BARE_DURATION_TO_REST[token]
    if re.fullmatch(r"t-?\d+", token):
        return re.sub(r"^t", "s", token)
    if re.fullmatch(r"=t=-?\d+", token):
        return re.sub(r"^=t=", "=s=", token)
    if re.fullmatch(r"=t-?\d+", token):
        return re.sub(r"^=t", "=s", token)
    if re.fullmatch(r"t=-?\d+", token):
        return re.sub(r"^t=", "s=", token)
    if to_generic(token) == "t":
        return re.sub(r"t", "s", token, count=1)
    return token


def strip_beam_markers(annotation: str) -> str:
    """Drop beam joiners so partially corrupted beam groups can still render."""
    tokens: list[str] = []
    for token in annotation.split():
        if "=" in token and mashcima_pitch(token) is not None:
            token = token.replace("=", "")
        tokens.append(token)
    return " ".join(tokens)


def coerce_renderable_annotation(annotation: str) -> tuple[str, list[str]]:
    """Normalize to an annotation Mashcima can render without assertion failures."""
    warnings: list[str] = []
    candidate = annotation

    for attempt in range(2):
        try:
            groups, parse_warnings = parse_annotation_into_token_groups(candidate)
            warnings.extend(parse_warnings)
            canvas = Canvas()
            token_groups_to_canvas(canvas, groups)
            canvas.finish_construction()
            given = " ".join(candidate.split())
            generated = " ".join(canvas.get_annotations())
            if given != generated:
                warnings.append("using canvas-canonical annotation for render")
                return generated, warnings
            return given, warnings
        except KeyError as exc:
            raise ValueError(f"unsupported mashcima token: {exc}") from exc
        except AssertionError:
            if attempt == 0:
                candidate, relax_warnings = repair_annotation(strip_beam_markers(candidate))
                warnings.extend(relax_warnings)
                warnings.append("relaxed broken beam groups for render")
                continue
            raise

    return candidate, warnings


def normalize_unrenderable_tokens(annotation: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    out: list[str] = []
    for token in annotation.split():
        mapped = downgrade_thirty_second_token(token)
        if mapped in BARE_DURATION_TO_REST:
            rest = BARE_DURATION_TO_REST[mapped]
            warnings.append(f"mapped {mapped} -> {rest}")
            mapped = rest
        elif mapped != token:
            warnings.append(f"mapped {token} -> {mapped}")
        out.append(mapped)
    return " ".join(out), warnings


def select_corrupt_indices(count: int, fraction: float, rng: random.Random) -> set[int]:
    """Pick indices to corrupt, with consecutive notes more likely after a hit."""
    if count == 0:
        return set()

    target = max(1, int(round(count * fraction)))
    base_p = min(0.5, fraction * 0.85)

    chosen: set[int] = set()
    prev_corrupted = False
    for i in range(count):
        p = base_p
        if prev_corrupted:
            p = min(0.88, base_p * CLUSTER_BOOST)
        if rng.random() < p:
            chosen.add(i)
            prev_corrupted = True
        else:
            prev_corrupted = False

    while len(chosen) < target:
        if chosen:
            seeds = list(chosen)
            rng.shuffle(seeds)
            for seed in seeds:
                neighbors = [i for i in (seed - 1, seed + 1) if 0 <= i < count and i not in chosen]
                if neighbors:
                    chosen.add(rng.choice(neighbors))
                    break
            else:
                if rng.random() < CLUSTER_EXTEND_PROB:
                    seed = rng.choice(tuple(chosen))
                    neighbors = [i for i in (seed - 1, seed + 1) if 0 <= i < count and i not in chosen]
                    if neighbors:
                        chosen.add(rng.choice(neighbors))
                        continue
                chosen.add(rng.randrange(count))
            continue
        chosen.add(rng.randrange(count))

    while len(chosen) > target:
        endpoints = [
            i for i in chosen if ((i - 1) not in chosen) ^ ((i + 1) not in chosen)
        ]
        isolated = [i for i in chosen if (i - 1) not in chosen and (i + 1) not in chosen]
        if endpoints and rng.random() < 0.38:
            chosen.remove(rng.choice(endpoints))
        elif isolated:
            chosen.remove(rng.choice(isolated))
        else:
            chosen.remove(rng.choice(tuple(chosen)))

    return chosen


def apply_corruption(
    sem_token: str,
    mash_token: str,
    rng: random.Random,
    ref_sem: str,
    ref_mash: str,
) -> tuple[str, str, str]:
    """Return (kind, semantic, mashcima) where kind is delete|replace|insert."""
    beamed = is_beamed_mash(mash_token)
    op = rng.choices(["change", "change", "delete", "insert"], weights=[4, 4, 1, 2], k=1)[0]
    if beamed and op in {"delete", "insert"}:
        op = "change"
    if sem_token == "barline":
        op = "change"

    if op == "delete":
        return "delete", "", ""

    if op == "insert":
        anchor_sem = ref_sem if SEMANTIC_NOTE_RE.match(ref_sem) or SEMANTIC_REST_RE.match(ref_sem) else sem_token
        anchor_mash = ref_mash if ref_mash != "|" else mash_token
        new_sem = (
            nearby_semantic_note(anchor_sem, rng)
            if rng.random() < 0.85
            else nearby_semantic_rest(anchor_sem, rng)
        )
        new_mash = nearby_mashcima_note_rest(anchor_sem, anchor_mash, rng)
        return "insert", new_sem, new_mash

    if sem_token == "barline" and mash_token == "|":
        if rng.random() < 0.5:
            neighbor_sem = ref_sem if SEMANTIC_NOTE_RE.match(ref_sem) else random_semantic_note(rng)
            neighbor_mash = ref_mash if mash_token != ref_mash else random_mashcima_note_rest(rng)
            return (
                "replace",
                nearby_semantic_note(neighbor_sem, rng),
                nearby_mashcima_note(neighbor_mash, rng),
            )
        return "replace", sem_token, mash_token

    if SEMANTIC_NOTE_RE.match(sem_token) and (rng.random() < 0.55 or beamed):
        return "replace", nearby_semantic_note(sem_token, rng), nearby_mashcima_note(mash_token, rng)

    if SEMANTIC_NOTE_RE.match(sem_token):
        m = SEMANTIC_NOTE_RE.match(sem_token)
        assert m is not None
        base_duration, dots = split_duration_body(m.group(3))
        new_duration = nearby_duration(base_duration, rng) + "." * dots
        return (
            "replace",
            f"note-{m.group(1)}{m.group(2)}_{new_duration}",
            set_mashcima_duration(mash_token, new_duration.rstrip("."), rng),
        )

    if SEMANTIC_REST_RE.match(sem_token):
        new_sem = nearby_semantic_rest(sem_token, rng)
        _, rest_body = new_sem.split("-", 1)
        base_duration, _ = split_duration_body(rest_body)
        mash_rest = mash_rest_for_duration(base_duration)
        return "replace", new_sem, mash_rest

    return "replace", sem_token, mash_token


def corrupt_pairs(
    pairs: list[tuple[str, str]],
    rng: random.Random,
    fraction: float,
) -> list[tuple[str, str]]:
    if not pairs:
        return pairs

    blocked = beam_context_indices(pairs)
    chosen = select_corrupt_indices(len(pairs), fraction, rng) - blocked
    eligible = [i for i in range(len(pairs)) if i not in blocked]
    # Cannot corrupt more positions than are eligible (non-beamed); capping the
    # target here prevents an infinite loop on beam-heavy incipits where the
    # fraction-based target would otherwise exceed len(eligible).
    target = min(max(1, int(round(len(pairs) * fraction))), len(eligible))
    while len(chosen) < target:
        chosen.add(rng.choice(eligible))
    out = list(pairs)

    for idx in sorted(chosen, reverse=True):
        sem_token, mash_token = out[idx]
        ref_sem, ref_mash = out[max(0, idx - 1)]
        kind, new_sem, new_mash = apply_corruption(sem_token, mash_token, rng, ref_sem, ref_mash)

        if kind == "delete":
            del out[idx]
        elif kind == "insert":
            out.insert(idx + 1, (new_sem, new_mash))
        else:
            out[idx] = (new_sem, new_mash)

    return out


def corrupt_tokens(
    semantic_text: str,
    agnostic_text: str,
    rng: random.Random,
    fraction: float,
) -> tuple[str, str]:
    semantic_tokens = semantic_text.split()
    header, body = split_semantic(semantic_tokens)

    mashcima = convert_primus_annotation_to_mashcima_annotation(agnostic_text)
    if mashcima is None:
        raise ValueError("Agnostic annotation could not be converted to Mashcima")

    mash_tokens = mashcima.split()
    pairs = extract_aligned_pairs(body, mash_tokens)
    corrupted_pairs = corrupt_pairs(pairs, rng, fraction)
    corrupted_body = rebuild_semantic_body(body, corrupted_pairs)
    corrupted_mashcima = " ".join(rebuild_mashcima_tokens(mash_tokens, corrupted_pairs))
    corrupted_semantic = "\t".join(header + corrupted_body) + "\t"
    corrupted_mashcima = synchronize_mashcima_to_semantic(
        corrupted_mashcima,
        corrupted_semantic,
    )
    return corrupted_semantic, corrupted_mashcima


def prepare_mashcima_for_render(annotation: str) -> tuple[str, list[str]]:
    normalized, norm_warnings = normalize_unrenderable_tokens(annotation)
    repaired, warnings = repair_annotation(normalized)
    coerced, coerce_warnings = coerce_renderable_annotation(repaired)
    return coerced, norm_warnings + warnings + coerce_warnings


def _synthesize_mashcima(
    annotation: str,
    mode: str,
    writer_key: str = "",
    min_width: int | None = None,
):
    repo = symbol_repository_for_key(writer_key) if writer_key else None
    if mode == "clean":
        return mc.synthesize_for_beauty(annotation, symbol_repository=repo)
    if mode == "handwritten":
        if min_width:
            # Equivalent to synthesize_for_training, without allocating the
            # old fixed 1200px canvas when the final dataset target is smaller.
            return mc.synthesize(
                annotation,
                min_width=min_width,
                transform_image=True,
                symbol_repository=repo,
            )
        return mc.synthesize_for_training(annotation, symbol_repository=repo)
    raise ValueError(mode)


def render_mashcima(
    annotation: str,
    mode: str,
    *,
    writer_key: str = "",
    apply_appearance: bool = True,
    max_dim: int | None = None,
) -> Image.Image:
    annotation, _ = prepare_mashcima_for_render(annotation)
    try:
        img = _synthesize_mashcima(annotation, mode, writer_key, max_dim)
    except AssertionError:
        relaxed = strip_beam_markers(annotation)
        relaxed, _ = repair_annotation(relaxed)
        annotation, _ = coerce_renderable_annotation(relaxed)
        img = _synthesize_mashcima(annotation, mode, writer_key, max_dim)
    arr = image_to_uint8(img)
    # Appearance is pixel-heavy, so discard pixels above the final training
    # resolution before generating paper grain and ink-pressure fields.
    arr = resize_long_edge(arr, max_dim)
    if apply_appearance:
        from mashcima_appearance import apply_mashcima_appearance

        arr = apply_mashcima_appearance(arr, key=writer_key or annotation[:64])
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write {path}")


def write_index_html(incipit_ids: list[str]) -> None:
    rows = []
    for incipit_id in incipit_ids:
        original_sem = (OUT_DIR / f"{incipit_id}_original.semantic").read_text(encoding="utf-8")
        corrupted_sem = (OUT_DIR / f"{incipit_id}_corrupted.semantic").read_text(encoding="utf-8")
        rows.append(
            f"""
    <section>
      <h2>{incipit_id}</h2>
      <div class="grid">
        <figure>
          <figcaption>Original PrIMuS clean (baseline)</figcaption>
          <img src="{incipit_id}_original_clean.png" alt="original clean">
        </figure>
        <figure>
          <figcaption>Corrupted GT — clean render</figcaption>
          <img src="{incipit_id}_corrupted_clean.png" alt="corrupted clean">
        </figure>
        <figure>
          <figcaption>Corrupted GT — scanned (camera)</figcaption>
          <img src="{incipit_id}_corrupted_scanned.png" alt="corrupted scanned">
        </figure>
        <figure>
          <figcaption>Corrupted GT — handwritten</figcaption>
          <img src="{incipit_id}_corrupted_handwritten.png" alt="corrupted handwritten">
        </figure>
      </div>
      <details>
        <summary>Ground truth tokens</summary>
        <pre><strong>Original semantic</strong>\n{original_sem}</pre>
        <pre><strong>Corrupted semantic</strong>\n{corrupted_sem}</pre>
      </details>
    </section>"""
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Corrupted GT examples</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f6f6f6; }}
    h1 {{ margin-bottom: 0.25rem; }}
    p {{ color: #444; }}
    section {{ background: #fff; border-radius: 8px; padding: 1rem 1.25rem; margin: 1.5rem 0; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; }}
    img {{ width: 100%; height: auto; border: 1px solid #ddd; background: #fff; }}
    pre {{ white-space: pre-wrap; word-break: break-word; font-size: 0.8rem; background: #f0f0f0; padding: 0.75rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Corrupted semantic GT examples</h1>
  <p>~33–50% of note/rest tokens were corrupted in clustered runs. Pitch and duration changes favor small intervals (seconds/thirds) and neighboring note values.</p>
  {''.join(rows)}
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    ensure_muscima_cache(PRIMUS_TGZ, ROOT / "data" / "mashcima-cache" / "MUSCIMA-pp_v1.0")
    configure_mashcima_writers("per-score")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    completed: list[str] = []

    for incipit_id in SAMPLES:
        folder = find_incipt_dir(incipit_id)
        semantic_path = folder / f"{incipit_id}.semantic"
        agnostic_path = folder / f"{incipit_id}.agnostic"
        primus_png = folder / f"{incipit_id}.png"

        semantic_text = semantic_path.read_text(encoding="utf-8")
        agnostic_text = agnostic_path.read_text(encoding="utf-8")

        rng = random.Random(seed_for_name(incipit_id) + 17_001)
        fraction = rng.uniform(0.33, 0.5)

        corrupted_semantic, corrupted_mashcima = corrupt_tokens(
            semantic_text, agnostic_text, rng, fraction
        )

        render_annotation, warnings = prepare_mashcima_for_render(corrupted_mashcima)
        if warnings:
            print(f"{incipit_id}: annotation warnings: {warnings[:3]}")

        (OUT_DIR / f"{incipit_id}_original.semantic").write_text(semantic_text, encoding="utf-8")
        (OUT_DIR / f"{incipit_id}_corrupted.semantic").write_text(corrupted_semantic, encoding="utf-8")
        (OUT_DIR / f"{incipit_id}_corrupted.mashcima").write_text(corrupted_mashcima + "\n", encoding="utf-8")

        shutil.copy2(primus_png, OUT_DIR / f"{incipit_id}_original_clean.png")

        clean = render_semantic_to_image(corrupted_semantic)
        handwritten = render_mashcima(
            render_annotation, "handwritten", writer_key=incipit_id
        )
        scanned = camera_like_distort(clean, seed=seed_for_name(incipit_id) + 99)
        handwritten = camera_like_distort(
            handwritten, seed=seed_for_name(incipit_id) + 199
        )

        save_png(clean, OUT_DIR / f"{incipit_id}_corrupted_clean.png")
        save_png(handwritten, OUT_DIR / f"{incipit_id}_corrupted_handwritten.png")
        save_png(scanned, OUT_DIR / f"{incipit_id}_corrupted_scanned.png")

        meta = {
            "incipit_id": incipit_id,
            "corrupt_fraction": round(fraction, 3),
            "warnings": warnings,
        }
        (OUT_DIR / f"{incipit_id}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        completed.append(incipit_id)
        print(f"{incipit_id}: corrupted ~{fraction:.0%} of note/rest tokens")

    write_index_html(completed)
    print(f"Wrote examples to {OUT_DIR}")
    print(f"Open {OUT_DIR / 'index.html'} for side-by-side comparison")


if __name__ == "__main__":
    main()
