#!/usr/bin/env python3
"""Convert PrIMuS semantic tokens to MusicXML (partwise, monophonic)."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from primus_semantic_render import (
    MULTIREST_RE,
    NOTE_RE,
    REST_RE,
    _is_header_token,
    _metric_beam_groups_for_measure,
    _metric_grid,
    _note_pitch,
    _parse_clef,
    _parse_key,
    _parse_time,
    _resolve_note_accid,
    _split_duration,
    _token_duration_whole,
)

from primus_semantic_render import KEY_SIG_TO_MEI

DIVISIONS = 16
MEI_TO_FIFTHS: dict[str, int] = {
    "0": 0,
    "1s": 1,
    "2s": 2,
    "3s": 3,
    "4s": 4,
    "5s": 5,
    "6s": 6,
    "7s": 7,
    "8s": 8,
    "1f": -1,
    "2f": -2,
    "3f": -3,
    "4f": -4,
    "5f": -5,
}

DURATION_TO_XML_TYPE: dict[str, str] = {
    "double_whole": "breve",
    "whole": "whole",
    "half": "half",
    "quarter": "quarter",
    "eighth": "eighth",
    "sixteenth": "16th",
    "thirty_second": "32nd",
}

MEI_ACCID_TO_XML: dict[str, tuple[int, str]] = {
    "f": (-1, "flat"),
    "s": (1, "sharp"),
    "n": (0, "natural"),
    "ff": (-2, "flat-flat"),
    "x": (2, "double-sharp"),
}


@dataclass
class MeasureEvent:
    kind: str
    token: str
    duration_units: int
    tie_start: bool = False
    tie_stop: bool = False
    beam: str | None = None
    explicit_accid: str | None = None
    display_accid: str | None = None
    fermata: bool = False
    header: dict[str, str] | None = None


def _duration_units_for_token(token: str) -> int:
    return int(_token_duration_whole(token) * 4 * DIVISIONS)


def _duration_type_and_dots(token: str) -> tuple[str, int, bool]:
    if NOTE_RE.match(token):
        body = NOTE_RE.match(token).group(3)
    elif REST_RE.match(token):
        body = REST_RE.match(token).group(1)
    else:
        raise ValueError(f"not a note/rest token: {token}")
    base, dots, fermata = _split_duration(body)
    if base not in DURATION_TO_XML_TYPE:
        raise ValueError(f"unsupported duration: {base}")
    return DURATION_TO_XML_TYPE[base], dots, fermata


def _key_fifths_mode(key_token: str | None) -> tuple[int, str]:
    if key_token is None:
        return 0, "major"
    mei = KEY_SIG_TO_MEI[key_token]
    fifths = MEI_TO_FIFTHS[mei]
    mode = "major" if key_token.endswith("M") else "minor"
    return fifths, mode


def _header_dict(token: str) -> dict[str, str]:
    if token.startswith("clef-"):
        shape, line = _parse_clef(token)
        return {"kind": "clef", "shape": shape, "line": line}
    if token.startswith("keySignature-"):
        fifths, mode = _key_fifths_mode(token)
        return {"kind": "key", "fifths": str(fifths), "mode": mode}
    if token.startswith("timeSignature-"):
        count, unit, cut = _parse_time(token)
        return {
            "kind": "time",
            "count": count or "4",
            "unit": unit or "4",
            "cut": "yes" if cut else "no",
        }
    raise ValueError(f"unsupported header token: {token}")


def _beam_roles_for_measure(
    measure_tokens: list[str],
    meter_count: str | None,
    meter_unit: str | None,
    cut_time: bool,
) -> dict[int, str]:
    groups = _metric_beam_groups_for_measure(
        measure_tokens, meter_count, meter_unit, cut_time
    )
    roles: dict[int, str] = {}
    for start, end in groups:
        for idx in range(start, end + 1):
            if idx == start:
                roles[idx] = "begin"
            elif idx == end:
                roles[idx] = "end"
            else:
                roles[idx] = "continue"
    return roles


def _build_measure_events(
    measure_tokens: list[str],
    key_sig: str,
    meter_count: str | None,
    meter_unit: str | None,
    cut_time: bool,
    *,
    autobeam: bool,
) -> list[MeasureEvent]:
    measure_accidentals: dict[tuple[str, str], str] = {}
    current_key_sig = key_sig
    beam_roles = (
        _beam_roles_for_measure(measure_tokens, meter_count, meter_unit, cut_time)
        if autobeam
        else {}
    )
    events: list[MeasureEvent] = []
    event_idx = 0
    tie_pending = False

    for token in measure_tokens:
        if token.startswith(("clef-", "keySignature-", "timeSignature-")):
            if token.startswith("keySignature-"):
                current_key_sig = _parse_key(token)
            events.append(
                MeasureEvent(
                    kind="header",
                    token=token,
                    duration_units=0,
                    header=_header_dict(token),
                )
            )
            continue
        if token == "tie":
            if not events or events[-1].kind != "note":
                raise ValueError("tie token without preceding note")
            events[-1].tie_start = True
            tie_pending = True
            continue
        if token == "barline":
            continue
        if MULTIREST_RE.match(token):
            count = int(MULTIREST_RE.match(token).group(1))
            events.append(
                MeasureEvent(
                    kind="multirest",
                    token=token,
                    duration_units=0,
                    header={"kind": "multirest", "count": str(count)},
                )
            )
            event_idx += 1
            continue
        if NOTE_RE.match(token):
            m = NOTE_RE.match(token)
            pname, oct_ = m.group(1), int(m.group(2))
            pname_mei, oct_mei, explicit_accid = _note_pitch(pname, oct_)
            accid = _resolve_note_accid(
                pname_mei,
                oct_mei,
                explicit_accid,
                current_key_sig,
                measure_accidentals,
            )
            xml_type, dots, fermata = _duration_type_and_dots(token)
            _ = xml_type, dots
            ev = MeasureEvent(
                kind="note",
                token=token,
                duration_units=_duration_units_for_token(token),
                explicit_accid=explicit_accid,
                display_accid=accid,
                fermata=fermata,
                beam=beam_roles.get(event_idx),
            )
            if tie_pending:
                ev.tie_stop = True
                tie_pending = False
            events.append(ev)
            event_idx += 1
            continue
        if REST_RE.match(token):
            _, _, fermata = _duration_type_and_dots(token)
            events.append(
                MeasureEvent(
                    kind="rest",
                    token=token,
                    duration_units=_duration_units_for_token(token),
                    fermata=fermata,
                )
            )
            event_idx += 1
            continue
        raise ValueError(f"unsupported semantic token in measure: {token}")

    return events


def _append_attributes(
    parent: ET.Element,
    headers: list[dict[str, str]],
    *,
    include_divisions: bool = False,
) -> None:
    if not headers and not include_divisions:
        return
    attrs = ET.SubElement(parent, "attributes")
    if include_divisions:
        ET.SubElement(attrs, "divisions").text = str(DIVISIONS)
    for kind in ("key", "time", "clef"):
        for header in headers:
            if header["kind"] != kind:
                continue
            if kind == "clef":
                clef = ET.SubElement(attrs, "clef")
                ET.SubElement(clef, "sign").text = header["shape"]
                ET.SubElement(clef, "line").text = header["line"]
            elif kind == "key":
                key = ET.SubElement(attrs, "key")
                ET.SubElement(key, "fifths").text = header["fifths"]
                ET.SubElement(key, "mode").text = header["mode"]
            elif kind == "time":
                time_el = ET.SubElement(attrs, "time")
                if header["cut"] == "yes":
                    time_el.set("symbol", "cut")
                ET.SubElement(time_el, "beats").text = header["count"]
                ET.SubElement(time_el, "beat-type").text = header["unit"]


def _pitch_alter(
    explicit_accid: str | None, display_accid: str | None
) -> int | None:
    if explicit_accid is not None:
        return MEI_ACCID_TO_XML[explicit_accid][0]
    if display_accid == "n":
        return 0
    return None


def _append_note(parent: ET.Element, event: MeasureEvent) -> None:
    m = NOTE_RE.match(event.token)
    if not m:
        raise ValueError(f"invalid note token: {event.token}")
    pname, octave = m.group(1)[0].upper(), int(m.group(2))
    xml_type, dots, fermata = _duration_type_and_dots(event.token)

    note = ET.SubElement(parent, "note")
    pitch = ET.SubElement(note, "pitch")
    ET.SubElement(pitch, "step").text = pname
    alter = _pitch_alter(event.explicit_accid, event.display_accid)
    if alter is not None:
        ET.SubElement(pitch, "alter").text = str(alter)
    ET.SubElement(pitch, "octave").text = str(octave)

    ET.SubElement(note, "duration").text = str(event.duration_units)
    if event.tie_stop:
        tie = ET.SubElement(note, "tie")
        tie.set("type", "stop")
    if event.tie_start:
        tie = ET.SubElement(note, "tie")
        tie.set("type", "start")
    ET.SubElement(note, "type").text = xml_type
    for _ in range(dots):
        ET.SubElement(note, "dot")

    if event.display_accid is not None:
        _, accidental_name = MEI_ACCID_TO_XML[event.display_accid]
        accidental = ET.SubElement(note, "accidental")
        accidental.text = accidental_name

    if event.beam:
        beam = ET.SubElement(note, "beam")
        beam.set("number", "1")
        beam.text = event.beam

    notations_needed = event.tie_start or event.tie_stop or event.fermata
    if notations_needed:
        notations = ET.SubElement(note, "notations")
        if event.tie_start:
            tied = ET.SubElement(notations, "tied")
            tied.set("type", "start")
        if event.tie_stop:
            tied = ET.SubElement(notations, "tied")
            tied.set("type", "stop")
        if event.fermata:
            fermata = ET.SubElement(notations, "fermata")
            fermata.set("type", "upright")


def _append_rest(parent: ET.Element, event: MeasureEvent) -> None:
    xml_type, dots, fermata = _duration_type_and_dots(event.token)
    note = ET.SubElement(parent, "note")
    ET.SubElement(note, "rest")
    ET.SubElement(note, "duration").text = str(event.duration_units)
    ET.SubElement(note, "type").text = xml_type
    for _ in range(dots):
        ET.SubElement(note, "dot")
    if event.fermata:
        notations = ET.SubElement(note, "notations")
        fermata = ET.SubElement(notations, "fermata")
        fermata.set("type", "upright")


def semantic_to_musicxml(
    semantic_text: str,
    *,
    autobeam: bool = True,
    title: str = "PrIMuS Semantic",
) -> str:
    """Convert PrIMuS semantic tokens to a MusicXML partwise document."""
    tokens = [t for t in semantic_text.split() if t]
    if not tokens:
        raise ValueError("empty semantic annotation")

    key_tok = next((t for t in tokens if t.startswith("keySignature-")), None)
    time_tok = next(t for t in tokens if t.startswith("timeSignature-"))
    key_sig = _parse_key(key_tok)
    meter_count, meter_unit, cut_time = _parse_time(time_tok)

    body_start = next((i for i, t in enumerate(tokens) if not _is_header_token(t)), len(tokens))
    body = tokens[body_start:]

    measure_token_lists: list[list[str]] = [[]]
    for token in body:
        if token == "barline":
            measure_token_lists.append([])
            continue
        measure_token_lists[-1].append(token)
    if not measure_token_lists[-1]:
        measure_token_lists.pop()

    measure_events = [
        _build_measure_events(
            measure_tokens,
            key_sig,
            meter_count,
            meter_unit,
            cut_time,
            autobeam=autobeam,
        )
        for measure_tokens in measure_token_lists
    ]

    root = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = title
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Melody"

    part = ET.SubElement(root, "part", id="P1")
    measure_number = 0
    initial_headers: list[dict[str, str]] = []
    divisions_written = False
    for token in tokens:
        if _is_header_token(token):
            initial_headers.append(_header_dict(token))
        else:
            break

    for measure_idx, events in enumerate(measure_events, start=1):
        if not events:
            continue
        multirest = next((e for e in events if e.kind == "multirest"), None)
        if multirest is not None:
            count = int(multirest.header["count"])  # type: ignore[index]
            measure_len, _beat = _metric_grid(meter_count, meter_unit, cut_time)
            measure_len_units = int(measure_len * 4 * DIVISIONS)
            for _ in range(count):
                measure_number += 1
                measure_el = ET.SubElement(part, "measure", number=str(measure_number))
                if measure_number == 1:
                    _append_attributes(
                        measure_el,
                        initial_headers,
                        include_divisions=not divisions_written,
                    )
                    divisions_written = True
                note = ET.SubElement(measure_el, "note")
                rest = ET.SubElement(note, "rest")
                rest.set("measure", "yes")
                ET.SubElement(note, "duration").text = str(measure_len_units)
            continue

        measure_number += 1
        measure_el = ET.SubElement(part, "measure", number=str(measure_number))
        pending_headers = list(initial_headers) if measure_number == 1 else []
        initial_headers = []

        for event in events:
            if event.kind == "header":
                pending_headers.append(event.header)  # type: ignore[arg-type]
                continue
            if pending_headers:
                _append_attributes(
                    measure_el,
                    pending_headers,
                    include_divisions=not divisions_written,
                )
                divisions_written = True
                pending_headers = []
            if event.kind == "note":
                _append_note(measure_el, event)
            elif event.kind == "rest":
                _append_rest(measure_el, event)

    xml_body = ET.tostring(root, encoding="unicode")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        f"{xml_body}\n"
    )


def convert_file(
    semantic_path: Path,
    output_path: Path,
    *,
    autobeam: bool = True,
) -> Path:
    semantic_text = semantic_path.read_text(encoding="utf-8").strip()
    musicxml = semantic_to_musicxml(
        semantic_text,
        autobeam=autobeam,
        title=semantic_path.stem,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(musicxml, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Paths to .semantic files (default: sample manual golden-set files)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples/musicxml"),
        help="Output directory for .musicxml files",
    )
    parser.add_argument(
        "--all-manual",
        action="store_true",
        help="Convert every file in golden_set/manual/ground_truth/",
    )
    parser.add_argument(
        "--no-autobeam",
        action="store_true",
        help="Disable metric beat-boundary autobeaming",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    if args.all_manual:
        inputs = sorted((root / "golden_set/manual/ground_truth").glob("*.semantic"))
    elif args.inputs:
        inputs = [Path(p) for p in args.inputs]
    else:
        sample_names = ["gs0", "gs1", "gs5", "gs10", "gs15"]
        inputs = [
            root / "golden_set/manual/ground_truth" / f"{name}.semantic"
            for name in sample_names
        ]

    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    written: list[Path] = []
    for semantic_path in inputs:
        if not semantic_path.is_file():
            raise SystemExit(f"missing semantic file: {semantic_path}")
        out_path = out_dir / f"{semantic_path.stem}.musicxml"
        convert_file(
            semantic_path,
            out_path,
            autobeam=not args.no_autobeam,
        )
        written.append(out_path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
