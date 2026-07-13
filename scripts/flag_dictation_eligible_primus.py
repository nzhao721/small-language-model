"""Flag PrIMuS incipits whose semantic tokens are within dictation allow-lists."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMUS_TGZ = ROOT / "data" / "archives" / "primusCalvoRizoAppliedSciences2018.tgz"
DEFAULT_PRIMUS_DIR = ROOT / "data" / "primus"
DEFAULT_FLAGS = ROOT / "data" / "primus_dictation_flags.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "primus_dictation_summary.json"

ALLOWED_CLEFS = frozenset({"clef-C2", "clef-C3", "clef-F4", "clef-G2"})
ALLOWED_TIME_SIGNATURES = frozenset(
    {
        "timeSignature-C",
        "timeSignature-C/",
        "timeSignature-2/2",
        "timeSignature-2/4",
        "timeSignature-3/2",
        "timeSignature-3/4",
        "timeSignature-3/8",
        "timeSignature-4/2",
        "timeSignature-4/4",
        "timeSignature-5/4",
        "timeSignature-5/8",
        "timeSignature-6/2",
        "timeSignature-6/4",
        "timeSignature-6/8",
        "timeSignature-7/4",
        "timeSignature-9/8",
        "timeSignature-12/8",
    }
)
ALLOWED_DURATIONS = frozenset(
    {
        "double_whole",
        "whole",
        "half",
        "quarter",
        "eighth",
        "sixteenth",
        "thirty_second",
    }
)

NOTE_RE = re.compile(r"^note-([A-G](?:#|b)?\d)_(.+)$")
REST_RE = re.compile(r"^rest-(.+)$")


def incipit_id_from_path(path: str | Path) -> str:
    path = Path(str(path).replace("\\", "/"))
    for part in reversed(path.parts):
        if part.endswith(".semantic"):
            return part[: -len(".semantic")]
        if re.fullmatch(r"\d{9}(-\d+_\d+_\d+)?", part):
            return part
    raise ValueError(f"Could not parse incipit id from path: {path}")


def parse_duration_body(body: str) -> tuple[str | None, str | None]:
    """Return (duration_base, error_reason). Dots and fermata are stripped first."""
    if body.endswith("_fermata"):
        body = body[: -len("_fermata")]

    dot_count = 0
    while body.endswith("."):
        body = body[:-1]
        dot_count += 1

    if dot_count > 2:
        return None, "too_many_dots"

    if not body:
        return None, "missing_duration"

    if body not in ALLOWED_DURATIONS:
        return None, f"disallowed_duration:{body}"

    return body, None


def token_allowed(token: str) -> tuple[bool, str | None]:
    if token in {"barline", "tie"}:
        return True, None

    if token.startswith("keySignature-"):
        return True, None

    if token.startswith("clef-"):
        if token in ALLOWED_CLEFS:
            return True, None
        return False, f"disallowed_clef:{token}"

    if token.startswith("timeSignature-"):
        if token in ALLOWED_TIME_SIGNATURES:
            return True, None
        return False, f"disallowed_time_signature:{token}"

    if token.startswith("multirest-"):
        return False, f"multirest:{token}"

    if token.startswith("gracenote-"):
        return False, f"grace_note:{token}"

    note_match = NOTE_RE.match(token)
    if note_match:
        _, body = note_match.groups()
        _, err = parse_duration_body(body)
        if err:
            return False, f"note_{err}"
        return True, None

    rest_match = REST_RE.match(token)
    if rest_match:
        (body,) = rest_match.groups()
        _, err = parse_duration_body(body)
        if err:
            return False, f"rest_{err}"
        return True, None

    return False, f"unknown_token:{token}"


def excerpt_is_eligible(tokens: list[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for token in tokens:
        ok, reason = token_allowed(token)
        if not ok and reason:
            reasons.append(reason)
    return len(reasons) == 0, reasons


def iter_semantic_records(primus_tgz: Path, primus_dir: Path):
    if primus_dir.is_dir() and any(primus_dir.rglob("*.semantic")):
        for path in primus_dir.rglob("*.semantic"):
            text = path.read_text(encoding="utf-8")
            tokens = [t for t in text.split() if t]
            yield str(path), incipit_id_from_path(path), tokens
        return

    if not primus_tgz.is_file():
        raise FileNotFoundError(f"PrIMuS not found: {primus_tgz}")

    with tarfile.open(primus_tgz, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".semantic"):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            text = extracted.read().decode("utf-8")
            tokens = [t for t in text.split() if t]
            yield member.name, incipit_id_from_path(member.name), tokens


def scan_corpus(
    primus_tgz: Path,
    primus_dir: Path,
    flags_path: Path,
    summary_path: Path,
) -> dict:
    flags_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    eligible = 0
    rejection_counts: dict[str, int] = {}

    with flags_path.open("w", encoding="utf-8") as out:
        for semantic_path, incipit_id, tokens in iter_semantic_records(primus_tgz, primus_dir):
            total += 1
            is_eligible, reasons = excerpt_is_eligible(tokens)

            if is_eligible:
                eligible += 1
            else:
                for reason in reasons:
                    key = reason.split(":", 1)[0]
                    rejection_counts[key] = rejection_counts.get(key, 0) + 1

            record = {
                "incipit_id": incipit_id,
                "semantic_path": semantic_path.replace("\\", "/"),
                "dictation_eligible": is_eligible,
                "token_count": len(tokens),
            }
            if not is_eligible:
                record["rejection_reasons"] = sorted(set(reasons))

            out.write(json.dumps(record) + "\n")

    summary = {
        "total_excerpts": total,
        "eligible_excerpts": eligible,
        "ineligible_excerpts": total - eligible,
        "eligible_fraction": round(eligible / total, 6) if total else 0.0,
        "top_rejection_reasons": dict(
            sorted(rejection_counts.items(), key=lambda item: item[1], reverse=True)[:25]
        ),
        "allowed_clefs": sorted(ALLOWED_CLEFS),
        "allowed_time_signatures": sorted(ALLOWED_TIME_SIGNATURES),
        "allowed_durations": sorted(ALLOWED_DURATIONS),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primus-tgz", type=Path, default=DEFAULT_PRIMUS_TGZ)
    parser.add_argument("--primus-dir", type=Path, default=DEFAULT_PRIMUS_DIR)
    parser.add_argument("--flags", type=Path, default=DEFAULT_FLAGS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    summary = scan_corpus(args.primus_tgz, args.primus_dir, args.flags, args.summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
