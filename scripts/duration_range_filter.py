"""Filter datasets to note/rest durations from sixteenth through whole (inclusive)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from purge_ineligible_primus import (  # noqa: E402
    MASHCIMA_DIR,
    MASHCIMA_MANIFEST,
    ROOT,
    purge_dataset_roots,
    purge_mashcima,
    remove_tree,
)
from dataset_layout import primus_ground_truth_dir  # noqa: E402

ALLOWED_DURATIONS = frozenset(
    {"sixteenth", "eighth", "quarter", "half", "whole"},
)
NOTE_RE = re.compile(r"^note-([A-G](?:#|b)?\d)_(.+)$")
REST_RE = re.compile(r"^rest-(.+)$")
INCIPIT_DIR_RE = re.compile(r"^\d{9}(-\d+_\d+_\d+)?$")

DEFAULT_PRIMUS_FLAGS = ROOT / "data" / "duration_range_flags.jsonl"
DEFAULT_PRIMUS_SUMMARY = ROOT / "data" / "duration_range_summary.json"
DEFAULT_CORRUPT_FLAGS = ROOT / "data" / "corrupt" / "duration_range_flags.jsonl"
CORRUPT_DIR = ROOT / "data" / "corrupt"
CORRUPT_MANIFEST = CORRUPT_DIR / "manifest.jsonl"
CORRUPT_JOBS = CORRUPT_DIR / "jobs.jsonl"
CAMERA_DISTORT_MANIFEST = ROOT / "data" / "mashcima" / "camera_distort_manifest.jsonl"


def parse_duration_body(body: str) -> tuple[str | None, str | None]:
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
        return None, body
    return body, None


def semantic_tokens_eligible(tokens: list[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for token in tokens:
        note_match = NOTE_RE.match(token)
        if note_match:
            _, err = parse_duration_body(note_match.group(2))
            if err:
                reasons.append(f"note:{err}")
            continue
        rest_match = REST_RE.match(token)
        if rest_match:
            _, err = parse_duration_body(rest_match.group(1))
            if err:
                reasons.append(f"rest:{err}")
    return len(reasons) == 0, sorted(set(reasons))


def incipit_id_from_semantic_path(path: Path) -> str:
    return path.stem


def scan_primus(primus_dir: Path, flags_path: Path, summary_path: Path) -> dict:
    flags_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    eligible = 0
    rejection_counts: dict[str, int] = {}

    with flags_path.open("w", encoding="utf-8") as out:
        gt_dir = primus_ground_truth_dir(primus_dir)
        if gt_dir.is_dir():
            semantic_paths = sorted(gt_dir.glob("*.semantic"))
        else:
            semantic_paths = sorted(primus_dir.rglob("*.semantic"))
        for semantic_path in semantic_paths:
            total += 1
            tokens = [t for t in semantic_path.read_text(encoding="utf-8").split() if t]
            is_eligible, reasons = semantic_tokens_eligible(tokens)
            incipit_id = incipit_id_from_semantic_path(semantic_path)
            if is_eligible:
                eligible += 1
            else:
                for reason in reasons:
                    key = reason.split(":", 1)[0]
                    rejection_counts[key] = rejection_counts.get(key, 0) + 1
            record = {
                "incipit_id": incipit_id,
                "semantic_path": str(semantic_path.relative_to(ROOT)).replace("\\", "/"),
                "duration_eligible": is_eligible,
            }
            if not is_eligible:
                record["rejection_reasons"] = reasons
            out.write(json.dumps(record) + "\n")

    summary = {
        "total_excerpts": total,
        "eligible_excerpts": eligible,
        "ineligible_excerpts": total - eligible,
        "allowed_durations": sorted(ALLOWED_DURATIONS),
        "top_rejection_reasons": dict(
            sorted(rejection_counts.items(), key=lambda item: item[1], reverse=True)[:25]
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def load_ineligible_incipt_ids(flags_path: Path) -> set[str]:
    ineligible: set[str] = set()
    with flags_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if not record.get("duration_eligible", False):
                ineligible.add(record["incipit_id"])
    return ineligible


def scan_corrupt(corrupt_dir: Path, flags_path: Path) -> dict:
    flags_path.parent.mkdir(parents=True, exist_ok=True)
    last: dict[str, dict] = {}
    if CORRUPT_MANIFEST.is_file():
        with CORRUPT_MANIFEST.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                last[row["corrupt_id"]] = row

    total_ok = 0
    eligible = 0
    with flags_path.open("w", encoding="utf-8") as out:
        for corrupt_id, row in last.items():
            if row.get("status") != "ok":
                continue
            total_ok += 1
            semantic_path = corrupt_dir / "semantic" / f"{corrupt_id}.semantic"
            if not semantic_path.is_file():
                sp = row.get("semantic_path")
                if sp:
                    semantic_path = ROOT / sp.replace("/", "\\")
            if not semantic_path.is_file():
                record = {
                    "corrupt_id": corrupt_id,
                    "incipit_id": row.get("incipit_id"),
                    "format": row.get("format"),
                    "duration_eligible": False,
                    "rejection_reasons": ["missing_semantic"],
                }
                out.write(json.dumps(record) + "\n")
                continue

            tokens = [t for t in semantic_path.read_text(encoding="utf-8").split() if t]
            is_eligible, reasons = semantic_tokens_eligible(tokens)
            if is_eligible:
                eligible += 1
            record = {
                "corrupt_id": corrupt_id,
                "incipit_id": row.get("incipit_id"),
                "format": row.get("format"),
                "duration_eligible": is_eligible,
            }
            if not is_eligible:
                record["rejection_reasons"] = reasons
            out.write(json.dumps(record) + "\n")

    return {
        "ok_jobs_scanned": total_ok,
        "eligible_ok_jobs": eligible,
        "ineligible_ok_jobs": total_ok - eligible,
    }


def load_ineligible_corrupt_ids(flags_path: Path) -> set[str]:
    ineligible: set[str] = set()
    with flags_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if not record.get("duration_eligible", False):
                ineligible.add(record["corrupt_id"])
    return ineligible


def purge_corrupt(ineligible: set[str]) -> dict[str, int]:
    stats = {
        "semantic_removed": 0,
        "images_removed": 0,
        "manifest_rows_removed": 0,
        "jobs_removed": 0,
    }
    last: dict[str, dict] = {}
    if CORRUPT_MANIFEST.is_file():
        with CORRUPT_MANIFEST.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                last[row["corrupt_id"]] = row

    kept_manifest: list[str] = []
    for corrupt_id, row in last.items():
        if corrupt_id in ineligible and row.get("status") == "ok":
            stats["manifest_rows_removed"] += 1
            semantic = CORRUPT_DIR / "semantic" / f"{corrupt_id}.semantic"
            if semantic.is_file():
                semantic.unlink()
                stats["semantic_removed"] += 1
            png_path = row.get("png_path")
            if png_path:
                img = ROOT / png_path.replace("/", "\\")
                if img.is_file():
                    img.unlink()
                    stats["images_removed"] += 1
            for fmt in ("clean", "camera", "handwritten"):
                for ext in (".png", ".jpg", ".jpeg"):
                    alt = CORRUPT_DIR / fmt / f"{corrupt_id}{ext}"
                    if alt.is_file():
                        alt.unlink()
                        stats["images_removed"] += 1
            continue
        kept_manifest.append(json.dumps(row))

    with CORRUPT_MANIFEST.open("w", encoding="utf-8") as f:
        for line in kept_manifest:
            f.write(line + "\n")

    if CORRUPT_JOBS.is_file():
        kept_jobs: list[str] = []
        with CORRUPT_JOBS.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                job = json.loads(line)
                if job["corrupt_id"] in ineligible:
                    stats["jobs_removed"] += 1
                    continue
                kept_jobs.append(line)
        with CORRUPT_JOBS.open("w", encoding="utf-8") as f:
            for line in kept_jobs:
                f.write(line + "\n")

    return stats


def filter_camera_distort_manifest(ineligible_incipt_ids: set[str]) -> int:
    if not CAMERA_DISTORT_MANIFEST.is_file():
        return 0
    removed = 0
    kept: list[str] = []
    with CAMERA_DISTORT_MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("incipit_id") in ineligible_incipt_ids:
                removed += 1
                continue
            kept.append(line)
    with CAMERA_DISTORT_MANIFEST.open("w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
    return removed


def rewrite_primus_flags(flags_path: Path, summary_path: Path) -> int:
    kept: list[dict] = []
    with flags_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("duration_eligible", False):
                kept.append(record)
    with flags_path.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record) + "\n")
    summary = {
        "total_excerpts": len(kept),
        "eligible_excerpts": len(kept),
        "ineligible_excerpts": 0,
        "note": "Duration-filtered corpus; only sixteenth..whole durations remain.",
        "allowed_durations": sorted(ALLOWED_DURATIONS),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primus-dir", type=Path, default=ROOT / "data" / "primus")
    parser.add_argument("--primus-flags", type=Path, default=DEFAULT_PRIMUS_FLAGS)
    parser.add_argument("--primus-summary", type=Path, default=DEFAULT_PRIMUS_SUMMARY)
    parser.add_argument("--corrupt-flags", type=Path, default=DEFAULT_CORRUPT_FLAGS)
    parser.add_argument("--scan", action="store_true", help="Scan and write flag files only")
    parser.add_argument("--purge", action="store_true", help="Purge ineligible excerpts from disk")
    args = parser.parse_args()

    if not args.scan and not args.purge:
        args.scan = True
        args.purge = True

    primus_summary = scan_primus(args.primus_dir, args.primus_flags, args.primus_summary)
    corrupt_summary = scan_corrupt(CORRUPT_DIR, args.corrupt_flags)
    print(json.dumps({"primus": primus_summary, "corrupt": corrupt_summary}, indent=2))

    if not args.purge:
        return

    ineligible_incipt = load_ineligible_incipt_ids(args.primus_flags)
    ineligible_corrupt = load_ineligible_corrupt_ids(args.corrupt_flags)

    dataset_removed = purge_dataset_roots(ineligible_incipt)
    mashcima_png_removed, mashcima_manifest_removed = purge_mashcima(ineligible_incipt)
    camera_manifest_removed = filter_camera_distort_manifest(ineligible_incipt)
    corrupt_stats = purge_corrupt(ineligible_corrupt)
    remaining = rewrite_primus_flags(args.primus_flags, args.primus_summary)

    print(
        json.dumps(
            {
                "primus_ineligible_excerpts": len(ineligible_incipt),
                "primus_dirs_removed": dataset_removed,
                "mashcima_pngs_removed": mashcima_png_removed,
                "mashcima_manifest_rows_removed": mashcima_manifest_removed,
                "camera_distort_manifest_rows_removed": camera_manifest_removed,
                "corrupt_ineligible_jobs": len(ineligible_corrupt),
                "corrupt_purge": corrupt_stats,
                "remaining_primus_excerpts": remaining,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
