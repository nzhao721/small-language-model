"""Scan all PrIMuS semantic files and list unique musical tokens."""

from __future__ import annotations

import argparse
import re
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMUS_TGZ = ROOT / "data" / "archives" / "primusCalvoRizoAppliedSciences2018.tgz"
DEFAULT_PRIMUS_DIR = ROOT / "data" / "primus"

DURATION_LABELS = {
    "quadruple_whole": "quadruple whole",
    "double_whole": "double whole",
    "whole": "whole",
    "half": "half",
    "quarter": "quarter",
    "eighth": "eighth",
    "sixteenth": "16th",
    "thirty_second": "32nd",
    "sixty_fourth": "64th",
}

CLEF_LABELS = {
    "G1": "G clef line 1",
    "G2": "treble clef (G clef line 2)",
    "G3": "G clef line 3",
    "G4": "G clef line 4",
    "G5": "G clef line 5",
    "F3": "F clef line 3",
    "F4": "bass clef (F clef line 4)",
    "F5": "F clef line 5",
    "C1": "C clef line 1",
    "C2": "alto clef (C clef line 2)",
    "C3": "tenor clef (C clef line 3)",
    "C4": "C clef line 4",
    "C5": "C clef line 5",
}

KEY_MODE_LABELS = {
    "M": "major",
    "m": "minor",
}


def humanize_token(token: str) -> str:
    if token == "barline":
        return "barline"

    if token.startswith("clef-"):
        code = token.removeprefix("clef-")
        return CLEF_LABELS.get(code, f"clef {code}")

    if token.startswith("keySignature-"):
        key = token.removeprefix("keySignature-")
        if key.endswith("M") and len(key) > 1:
            return f"{key[:-1]} major key signature"
        if key.endswith("m") and len(key) > 1:
            return f"{key[:-1]} minor key signature"
        return f"{key} key signature"

    if token.startswith("timeSignature-"):
        return f"{token.removeprefix('timeSignature-')} time signature"

    if token.startswith("multirest-"):
        bars = token.removeprefix("multirest-")
        return f"{bars}-bar rest"

    if token.startswith("rest-"):
        body = token.removeprefix("rest-")
        fermata = False
        if body.endswith("_fermata"):
            fermata = True
            body = body.removesuffix("_fermata")
        dots = body.count(".")
        body = body.rstrip(".")
        duration = DURATION_LABELS.get(body, body.replace("_", " "))
        label = f"{duration} rest"
        if dots:
            label = f"dotted {label}" if dots == 1 else f"{dots}x-dotted {label}"
        if fermata:
            label += " with fermata"
        return label

    if token.startswith("note-"):
        body = token.removeprefix("note-")
        fermata = False
        if body.endswith("_fermata"):
            fermata = True
            body = body.removesuffix("_fermata")

        if body.startswith("grace_"):
            grace = True
            body = body.removeprefix("grace_")
        else:
            grace = False

        match = re.match(
            r"^([A-G](?:#|b)?)(\d)_(.+?)(\.+)?$",
            body,
        )
        if not match:
            return token

        pitch, octave, duration, dots = match.groups()
        dur_label = DURATION_LABELS.get(duration, duration.replace("_", " "))
        label = f"{pitch}{octave} {dur_label} note"
        if dots:
            label = f"dotted {label}" if len(dots) == 1 else f"{len(dots)}x-dotted {label}"
        if grace:
            label = f"grace {label}"
        if fermata:
            label += " with fermata"
        return label

    return token


def iter_semantic_files(primus_tgz: Path, primus_dir: Path):
    if primus_dir.is_dir() and any(primus_dir.rglob("*.semantic")):
        yield from primus_dir.rglob("*.semantic")
        return

    if not primus_tgz.is_file():
        raise FileNotFoundError(
            f"PrIMuS not found. Run scripts/download_datasets.ps1 first. "
            f"Missing: {primus_tgz}"
        )

    with tarfile.open(primus_tgz, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".semantic"):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            yield member.name, extracted.read().decode("utf-8")


def collect_tokens(primus_tgz: Path, primus_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()

    for item in iter_semantic_files(primus_tgz, primus_dir):
        if isinstance(item, tuple):
            _, text = item
        else:
            text = item.read_text(encoding="utf-8")

        for token in text.split():
            if token:
                counts[token] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primus-tgz", type=Path, default=DEFAULT_PRIMUS_TGZ)
    parser.add_argument("--primus-dir", type=Path, default=DEFAULT_PRIMUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "primus_semantic_tokens.txt",
    )
    args = parser.parse_args()

    counts = collect_tokens(args.primus_tgz, args.primus_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for token in sorted(counts):
        lines.append(f"{humanize_token(token)}\t{token}\t{counts[token]}")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Unique tokens: {len(counts)}")
    print(f"Total token occurrences: {sum(counts.values())}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
