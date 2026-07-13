"""Render PrIMuS semantic tokens to digital notation images (Verovio engraving)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import verovio
from PIL import Image

KEY_SIG_ALTERATIONS: dict[str, frozenset[str]] = {
    "0": frozenset(),
    "1s": frozenset("f"),
    "2s": frozenset("fc"),
    "3s": frozenset("fcg"),
    "4s": frozenset("fcgd"),
    "5s": frozenset("fcgda"),
    "6s": frozenset("fcgdae"),
    "7s": frozenset("fcgdaeb"),
    "8s": frozenset("fcgdaebf"),
    "1f": frozenset("b"),
    "2f": frozenset("be"),
    "3f": frozenset("bea"),
    "4f": frozenset("bead"),
    "5f": frozenset("beadg"),
}

KEY_SIG_TO_MEI: dict[str, str] = {
    "keySignature-CM": "0",
    "keySignature-Am": "0",
    "keySignature-GM": "1s",
    "keySignature-Em": "1s",
    "keySignature-DM": "2s",
    "keySignature-Bm": "2s",
    "keySignature-AM": "3s",
    "keySignature-F#m": "3s",
    "keySignature-EM": "4s",
    "keySignature-C#m": "4s",
    "keySignature-BM": "5s",
    "keySignature-G#m": "5s",
    "keySignature-FM": "1f",
    "keySignature-Dm": "1f",
    "keySignature-BbM": "2f",
    "keySignature-Gm": "2f",
    "keySignature-EbM": "3f",
    "keySignature-Cm": "3f",
    "keySignature-AbM": "4f",
    "keySignature-Fm": "4f",
    "keySignature-DbM": "5f",
    "keySignature-Bbm": "5f",
    "keySignature-C#M": "7s",
    "keySignature-F#M": "6s",
    "keySignature-G#M": "8s",
}

DURATION_TO_MEI: dict[str, str] = {
    "double_whole": "breve",
    "whole": "1",
    "half": "2",
    "quarter": "4",
    "eighth": "8",
    "sixteenth": "16",
    "thirty_second": "32",
}

CLEF_RE = re.compile(r"^clef-([A-G])(\d+)$")
TIME_RE = re.compile(r"^timeSignature-(\d+)/(\d+)$")
NOTE_RE = re.compile(r"^note-([A-G](?:#|b)?)(\d+)_(.+)$")
REST_RE = re.compile(r"^rest-(.+)$")
MULTIREST_RE = re.compile(r"^multirest-(\d+)$")
MASH_NOTE_REST_RE = re.compile(
    r"^(?:"
    r"lr|br|wr|hr|qr|er|sr|tr"
    r"|w-?\d+|h-?\d+|q-?\d+|e-?\d+|s-?\d+|t-?\d+"
    r"|e=-?\d+|=e=-?\d+|=e-?\d+"
    r"|s=-?\d+|=s=-?\d+|=s-?\d+"
    r"|t=-?\d+|=t=-?\d+|=t-?\d+"
    r")$"
)

_VEROVIO_OPTS = {
    "scale": 40,
    "pageWidth": 2400,
    "pageHeight": 300,
    "adjustPageHeight": True,
    "adjustPageWidth": True,
    "spacingStaff": 6,
}

_verovio_toolkit: verovio.toolkit | None = None
_RENDER_PADDING_PX = 16


def _split_duration(body: str) -> tuple[str, int, bool]:
    fermata = False
    if body.endswith("_fermata"):
        body = body[: -len("_fermata")]
        fermata = True
    dots = 0
    while body.endswith("."):
        body = body[:-1]
        dots += 1
    return body, dots, fermata


def _parse_clef(token: str) -> tuple[str, str]:
    m = CLEF_RE.match(token)
    if not m:
        raise ValueError(f"unsupported clef token: {token}")
    return m.group(1), m.group(2)


def _parse_time(token: str) -> tuple[str | None, str | None, bool]:
    if token in {"timeSignature-C", "timeSignature-C/"}:
        if token.endswith("/"):
            return None, None, True
        return "4", "4", False
    m = TIME_RE.match(token)
    if not m:
        raise ValueError(f"unsupported time signature token: {token}")
    return m.group(1), m.group(2), False


def _parse_key(token: str | None) -> str:
    if token is None:
        return "0"
    if token in KEY_SIG_TO_MEI:
        return KEY_SIG_TO_MEI[token]
    raise ValueError(f"unsupported key signature token: {token}")


def _is_header_token(token: str) -> bool:
    return token.startswith(("clef-", "keySignature-", "timeSignature-"))


def _note_pitch(name_acc: str, octave: int) -> tuple[str, str, str | None]:
    letter = name_acc[0].lower()
    acc = name_acc[1:] if len(name_acc) > 1 else ""
    if acc == "#":
        return letter, str(octave), "s"
    if acc == "b":
        return letter, str(octave), "f"
    return letter, str(octave), None


def _resolve_note_accid(
    pname_mei: str,
    oct_mei: str,
    explicit_accid: str | None,
    key_sig: str,
    measure_accidentals: dict[tuple[str, str], str],
) -> str | None:
    """Infer MEI accidental, including naturals that cancel key or measure context."""
    pitch = (pname_mei, oct_mei)
    if explicit_accid:
        measure_accidentals[pitch] = explicit_accid
        return explicit_accid

    key_alters = pname_mei in KEY_SIG_ALTERATIONS.get(key_sig, frozenset())
    measure_alters = pitch in measure_accidentals
    if key_alters or measure_alters:
        measure_accidentals.pop(pitch, None)
        return "n"
    return None


def _mei_note(token: str, note_id: int, *, accid: str | None = None) -> str:
    m = NOTE_RE.match(token)
    if not m:
        raise ValueError(f"unsupported note token: {token}")
    pname, oct_, dur_body = m.group(1), int(m.group(2)), m.group(3)
    base, dots, fermata = _split_duration(dur_body)
    if base not in DURATION_TO_MEI:
        raise ValueError(f"unsupported note duration: {base}")
    pname_mei, oct_mei, _explicit = _note_pitch(pname, oct_)
    if accid is None:
        accid = _explicit
    attrs = [f'xml:id="note-{note_id:08d}"', f'dur="{DURATION_TO_MEI[base]}"', f'oct="{oct_mei}"', f'pname="{pname_mei}"']
    if dots:
        attrs.append(f'dots="{dots}"')
    if fermata:
        attrs.append('fermata="above"')
    inner = ""
    if accid:
        inner = f'<accid xml:id="accid-{note_id:08d}" accid="{accid}" />'
    return f"<note {' '.join(attrs)}>{inner}</note>"


def _mei_rest(token: str, rest_id: int) -> str:
    m = REST_RE.match(token)
    if not m:
        raise ValueError(f"unsupported rest token: {token}")
    base, dots, _fermata = _split_duration(m.group(1))
    if base not in DURATION_TO_MEI:
        raise ValueError(f"unsupported rest duration: {base}")
    attrs = [f'xml:id="rest-{rest_id:08d}"', f'dur="{DURATION_TO_MEI[base]}"']
    if dots:
        attrs.append(f'dots="{dots}"')
    return f"<rest {' '.join(attrs)} />"


def _mei_clef(token: str, clef_id: int) -> str:
    shape, line = _parse_clef(token)
    return (
        f'<clef xml:id="clef-{clef_id:08d}" shape="{shape}" line="{line}" />'
    )


def _is_beamable_note_token(token: str) -> bool:
    m = NOTE_RE.match(token)
    if not m:
        return False
    base, _dots, _fermata = _split_duration(m.group(3))
    return base in {"eighth", "sixteenth", "thirty_second"}


def _is_beamed_mash(mash_token: str) -> bool:
    return "=" in mash_token and MASH_NOTE_REST_RE.match(mash_token) is not None


def _infer_beam_groups_for_measure(measure_tokens: list[str]) -> list[tuple[int, int]]:
    """Beam consecutive eighth/sixteenth/thirty-second notes within a measure."""
    groups: list[tuple[int, int]] = []
    current: list[int] = []
    event_idx = 0

    def close_beam() -> None:
        nonlocal current
        if len(current) >= 2:
            groups.append((current[0], current[-1]))
        current = []

    for token in measure_tokens:
        if token.startswith(("clef-", "keySignature-", "timeSignature-")):
            continue
        if token == "tie":
            continue
        if token == "barline":
            continue
        if MULTIREST_RE.match(token):
            close_beam()
            event_idx += 1
            continue
        if REST_RE.match(token):
            close_beam()
            event_idx += 1
            continue
        if NOTE_RE.match(token):
            if _is_beamable_note_token(token):
                current.append(event_idx)
            else:
                close_beam()
            event_idx += 1
            continue
        raise ValueError(f"unsupported semantic token in measure: {token}")

    close_beam()
    return groups


def _apply_beam_wrappers(events: list[str], groups: list[tuple[int, int]]) -> list[str]:
    """Wrap MEI event index ranges in <beam> elements (groups are inclusive)."""
    if not groups:
        return events
    starts = {start: end for start, end in groups}
    out: list[str] = []
    i = 0
    while i < len(events):
        if i in starts:
            end = starts[i]
            chunk = events[i : end + 1]
            inner = "\n                                    ".join(chunk)
            out.append(
                f"<beam>\n                                    {inner}\n                                    </beam>"
            )
            i = end + 1
        else:
            out.append(events[i])
            i += 1
    return out


def _beam_groups_for_measure(
    measure_tokens: list[str],
    pairs: list[tuple[str, str]],
) -> tuple[list[tuple[int, int]], int]:
    """Return beam (start, end) index groups for one measure and pairs consumed."""
    groups: list[tuple[int, int]] = []
    current: list[int] = []
    event_idx = 0
    pair_idx = 0

    def close_beam() -> None:
        nonlocal current
        if len(current) >= 2:
            groups.append((current[0], current[-1]))
        current = []

    for token in measure_tokens:
        if token.startswith(("clef-", "keySignature-", "timeSignature-")):
            continue
        if token == "tie":
            continue
        if token == "barline":
            continue
        if MULTIREST_RE.match(token):
            close_beam()
            event_idx += 1
            continue
        if NOTE_RE.match(token) or REST_RE.match(token):
            if pair_idx >= len(pairs):
                raise ValueError("mash pair alignment exhausted in measure")
            sem_tok, mash_tok = pairs[pair_idx]
            pair_idx += 1
            if NOTE_RE.match(sem_tok) and _is_beamed_mash(mash_tok):
                current.append(event_idx)
            else:
                close_beam()
            event_idx += 1
            continue
        raise ValueError(f"unsupported semantic token in measure: {token}")

    close_beam()
    return groups, pair_idx


def semantic_to_mei(
    semantic_text: str,
    mash_pairs: list[tuple[str, str]] | None = None,
    *,
    autobeam: bool = True,
) -> str:
    """Convert PrIMuS semantic tab-separated tokens to a minimal MEI score.

    When *mash_pairs* is provided, agnostic/Mashcima beam markers are preferred.
    Otherwise (or when they yield no groups), *autobeam* wraps consecutive
    beamable notes in each measure in explicit ``<beam>`` elements.
    """
    tokens = [t for t in semantic_text.split() if t]
    if not tokens:
        raise ValueError("empty semantic annotation")

    clef_tok = next(t for t in tokens if t.startswith("clef-"))
    key_tok = next((t for t in tokens if t.startswith("keySignature-")), None)
    time_tok = next(t for t in tokens if t.startswith("timeSignature-"))

    clef_shape, clef_line = _parse_clef(clef_tok)
    meter_count, meter_unit, cut_time = _parse_time(time_tok)
    key_sig = _parse_key(key_tok)

    body_start = next((i for i, t in enumerate(tokens) if not _is_header_token(t)), len(tokens))
    body = tokens[body_start:]

    measures: list[list[str]] = [[]]
    measure_token_lists: list[list[str]] = [[]]
    measure_accidentals: dict[tuple[str, str], str] = {}
    note_id = 0
    rest_id = 0
    clef_id = 0
    pair_offset = 0

    for token in body:
        if token == "barline":
            measures.append([])
            measure_token_lists.append([])
            measure_accidentals = {}
            continue
        measure_token_lists[-1].append(token)
        if token.startswith("clef-"):
            clef_id += 1
            measures[-1].append(_mei_clef(token, clef_id))
            continue
        if token.startswith("keySignature-"):
            continue
        if token.startswith("timeSignature-"):
            continue
        if token == "tie":
            continue
        if MULTIREST_RE.match(token):
            count = MULTIREST_RE.match(token).group(1)
            measures[-1].append(f'<mRest xml:id="mrest-{note_id:08d}" num="{count}" />')
            note_id += 1
            continue
        if NOTE_RE.match(token):
            note_id += 1
            m = NOTE_RE.match(token)
            pname, oct_ = m.group(1), int(m.group(2))
            pname_mei, oct_mei, explicit_accid = _note_pitch(pname, oct_)
            accid = _resolve_note_accid(
                pname_mei,
                oct_mei,
                explicit_accid,
                key_sig,
                measure_accidentals,
            )
            measures[-1].append(_mei_note(token, note_id, accid=accid))
            continue
        if REST_RE.match(token):
            rest_id += 1
            measures[-1].append(_mei_rest(token, rest_id))
            continue
        raise ValueError(f"unsupported semantic token: {token}")

    if not measures[-1]:
        measures.pop()
        measure_token_lists.pop()

    if mash_pairs is not None or autobeam:
        for measure_events, measure_tokens in zip(measures, measure_token_lists, strict=True):
            groups: list[tuple[int, int]] = []
            consumed = 0
            if mash_pairs is not None:
                groups, consumed = _beam_groups_for_measure(
                    measure_tokens, mash_pairs[pair_offset:]
                )
                pair_offset += consumed
            if not groups and autobeam:
                groups = _infer_beam_groups_for_measure(measure_tokens)
            wrapped = _apply_beam_wrappers(measure_events, groups)
            measure_events[:] = wrapped

    measure_xml: list[str] = []
    for i, layer_tokens in enumerate(measures, start=1):
        layer_body = "\n                                    ".join(layer_tokens)
        measure_xml.append(
            f"""                        <measure xml:id="measure-{i:08d}" right="single">
                            <staff xml:id="staff-{i:08d}" n="1">
                                <layer xml:id="layer-{i:08d}" n="1">
                                    {layer_body}
                                </layer>
                            </staff>
                        </measure>"""
        )

    measures_block = "\n".join(measure_xml)
    if cut_time:
        score_def = f'<scoreDef key.sig="{key_sig}" meter.sym="cut">'
    else:
        score_def = (
            f'<scoreDef key.sig="{key_sig}" meter.count="{meter_count}" '
            f'meter.unit="{meter_unit}">'
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="4.0.0">
    <music>
        <body>
            <mdiv>
                <score>
                    {score_def}
                        <staffGrp>
                            <staffDef clef.shape="{clef_shape}" clef.line="{clef_line}" n="1" lines="5" />
                        </staffGrp>
                    </scoreDef>
                    <section>
{measures_block}
                    </section>
                </score>
            </mdiv>
        </body>
    </music>
</mei>"""


def _get_verovio() -> verovio.toolkit:
    global _verovio_toolkit
    if _verovio_toolkit is None:
        _verovio_toolkit = verovio.toolkit()
        _verovio_toolkit.setOptions(_VEROVIO_OPTS)
    return _verovio_toolkit


_SCREENSHOT_WORKER = Path(__file__).with_name("_playwright_screenshot.py")


def _screenshot_html(html: str, width: int, height: int) -> bytes:
    """Run Playwright in a fresh subprocess to avoid asyncio-loop conflicts in pool workers."""
    payload = json.dumps({"html": html, "width": width, "height": height})
    proc = subprocess.run(
        [sys.executable, str(_SCREENSHOT_WORKER)],
        input=payload.encode("utf-8"),
        capture_output=True,
        timeout=120,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        if not detail:
            detail = f"screenshot worker exited with code {proc.returncode}"
        raise RuntimeError(detail)
    if not proc.stdout:
        raise RuntimeError("screenshot worker returned empty PNG")
    return proc.stdout


def close_render_pool() -> None:
    """Release per-worker render resources."""
    global _verovio_toolkit
    _verovio_toolkit = None


def render_semantic_to_image(
    semantic_text: str,
    mash_pairs: list[tuple[str, str]] | None = None,
    *,
    autobeam: bool = True,
) -> Image.Image:
    """Render PrIMuS semantic tokens to a white-background digital notation PIL image."""
    mei = semantic_to_mei(
        semantic_text,
        mash_pairs=mash_pairs,
        autobeam=autobeam,
    )
    tk = _get_verovio()
    svg = tk.renderData(mei, {})
    m = re.search(
        r'width="([0-9]+(?:\.[0-9]+)?)px"\s+height="([0-9]+(?:\.[0-9]+)?)px"',
        svg,
    )
    width = int(float(m.group(1))) if m else 800
    height = int(float(m.group(2))) if m else 200
    padding = _RENDER_PADDING_PX
    html = (
        "<!DOCTYPE html><html><body style='margin:0;background:white;"
        "width:max-content;height:max-content;overflow:visible'>"
        f"<div style='padding:{padding}px;display:inline-block'>"
        f"{svg}</div></body></html>"
    )
    png_bytes = _screenshot_html(
        html,
        width + 2 * padding,
        height + 2 * padding,
    )
    return Image.open(BytesIO(png_bytes)).convert("RGB")
