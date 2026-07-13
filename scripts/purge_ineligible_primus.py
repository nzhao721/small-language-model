"""Delete PrIMuS-family files for dictation-ineligible incipits."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_layout import (  # noqa: E402
    CAMERA_PRIMUS_DIR,
    MASHCIMA_DIR,
    PRIMUS_DIR,
    camera_distorted_path,
    camera_ground_truth_dir,
    mashcima_ground_truth_dir,
    mashcima_image_path,
    primus_agnostic_path,
    primus_ground_truth_dir,
    primus_png_path,
    primus_semantic_path,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLAGS = ROOT / "data" / "primus_dictation_flags.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "primus_dictation_summary.json"
DATASET_ROOTS = (
    PRIMUS_DIR,
    CAMERA_PRIMUS_DIR,
)
MASHCIMA_MANIFEST = MASHCIMA_DIR / "manifest.jsonl"
INCIPIT_DIR_RE = re.compile(r"^\d{9}(-\d+_\d+_\d+)?$")


def _remove_readonly(func, path, _exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_tree(path: Path, retries: int = 3) -> None:
    for attempt in range(retries):
        try:
            shutil.rmtree(path, onexc=_remove_readonly)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.5)


def load_eligible_ids(flags_path: Path) -> set[str]:
    eligible: set[str] = set()
    with flags_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("dictation_eligible", False):
                eligible.add(record["incipit_id"])
    return eligible


def load_ineligible_ids(flags_path: Path) -> set[str]:
    ineligible: set[str] = set()
    with flags_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if not record.get("dictation_eligible", False):
                ineligible.add(record["incipit_id"])
    return ineligible


def _incipit_ids_in_root(root: Path) -> set[str]:
    ids: set[str] = set()
    if root == PRIMUS_DIR:
        gt = primus_ground_truth_dir(root)
        if gt.is_dir():
            ids.update(p.stem for p in gt.glob("*.semantic"))
        for pkg in root.glob("package_*"):
            if pkg.is_dir():
                for inc_dir in pkg.iterdir():
                    if inc_dir.is_dir() and INCIPIT_DIR_RE.fullmatch(inc_dir.name):
                        ids.add(inc_dir.name)
    elif root == CAMERA_PRIMUS_DIR:
        gt = camera_ground_truth_dir(root)
        if gt.is_dir():
            ids.update(p.stem for p in gt.glob("*.semantic"))
        corpus = root / "Corpus"
        if corpus.is_dir():
            for inc_dir in corpus.iterdir():
                if inc_dir.is_dir() and INCIPIT_DIR_RE.fullmatch(inc_dir.name):
                    ids.add(inc_dir.name)
    else:
        for incipit_dir in root.rglob("*"):
            if incipit_dir.is_dir() and INCIPIT_DIR_RE.fullmatch(incipit_dir.name):
                ids.add(incipit_dir.name)
    return ids


def _purge_incipt_from_root(incipit_id: str, root: Path) -> bool:
    removed_any = False
    if root == PRIMUS_DIR:
        for path in (
            primus_png_path(incipit_id, root),
            primus_semantic_path(incipit_id, root),
            primus_agnostic_path(incipit_id, root),
        ):
            if path.is_file():
                path.unlink()
                removed_any = True
        legacy = root / "package_aa" / incipit_id
        if not legacy.is_dir():
            legacy = root / "package_ab" / incipit_id
        if legacy.is_dir():
            remove_tree(legacy)
            removed_any = True
    elif root == CAMERA_PRIMUS_DIR:
        for path in (
            camera_distorted_path(incipit_id, root),
            camera_ground_truth_dir(root) / f"{incipit_id}.semantic",
        ):
            if path.is_file():
                path.unlink()
                removed_any = True
        legacy = root / "Corpus" / incipit_id
        if legacy.is_dir():
            remove_tree(legacy)
            removed_any = True
    else:
        for incipit_dir in root.rglob(incipit_id):
            if incipit_dir.is_dir() and incipit_dir.name == incipit_id:
                remove_tree(incipit_dir)
                removed_any = True
    return removed_any


def purge_dataset_roots(ineligible: set[str]) -> dict[str, int]:
    removed = {str(root): 0 for root in DATASET_ROOTS}

    for root in DATASET_ROOTS:
        if not root.is_dir():
            continue
        for incipit_id in ineligible:
            if _purge_incipt_from_root(incipit_id, root):
                removed[str(root)] += 1
    return removed


def purge_keep_eligible_only(eligible: set[str]) -> dict[str, int]:
    """Delete incipit artifacts not in the eligible set (for post-flag rewrites)."""
    removed = {str(root): 0 for root in DATASET_ROOTS}

    for root in DATASET_ROOTS:
        if not root.is_dir():
            continue
        for incipit_id in _incipit_ids_in_root(root):
            if incipit_id not in eligible and _purge_incipt_from_root(incipit_id, root):
                removed[str(root)] += 1
    return removed


def purge_mashcima(ineligible: set[str]) -> tuple[int, int]:
    png_removed = 0
    if MASHCIMA_DIR.is_dir():
        for incipit_id in ineligible:
            for path in (
                mashcima_image_path(incipit_id),
                mashcima_ground_truth_dir() / f"{incipit_id}.semantic",
                MASHCIMA_DIR / f"{incipit_id}.png",
                MASHCIMA_DIR / f"{incipit_id}.jpg",
            ):
                if path.is_file():
                    path.unlink()
                    png_removed += 1

    manifest_removed = 0
    if MASHCIMA_MANIFEST.is_file():
        kept_lines: list[str] = []
        with MASHCIMA_MANIFEST.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("incipit_id") in ineligible:
                    manifest_removed += 1
                    continue
                kept_lines.append(line)
        with MASHCIMA_MANIFEST.open("w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")

    return png_removed, manifest_removed


def rewrite_flags(flags_path: Path, summary_path: Path) -> int:
    kept: list[dict] = []
    with flags_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("dictation_eligible", False):
                kept.append(record)

    with flags_path.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record) + "\n")

    summary = {
        "total_excerpts": len(kept),
        "eligible_excerpts": len(kept),
        "ineligible_excerpts": 0,
        "eligible_fraction": 1.0 if kept else 0.0,
        "note": "Corpus purged; only dictation-eligible incipits remain on disk.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flags", type=Path, default=DEFAULT_FLAGS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--keep-eligible-only",
        action="store_true",
        help="Delete any on-disk incipit folders not listed as eligible in flags.",
    )
    args = parser.parse_args()

    if args.keep_eligible_only:
        eligible = load_eligible_ids(args.flags)
        dataset_removed = purge_keep_eligible_only(eligible)
        ineligible_count = sum(dataset_removed.values())
    else:
        ineligible = load_ineligible_ids(args.flags)
        dataset_removed = purge_dataset_roots(ineligible)
        ineligible_count = len(ineligible)
        mashcima_png_removed, manifest_removed = purge_mashcima(ineligible)
        remaining = rewrite_flags(args.flags, args.summary)
        print(
            json.dumps(
                {
                    "ineligible_excerpts_deleted": ineligible_count,
                    "primus_dirs_removed": dataset_removed.get(str(ROOT / "data" / "primus"), 0),
                    "camera_primus_dirs_removed": dataset_removed.get(
                        str(ROOT / "data" / "camera-primus"), 0
                    ),
                    "mashcima_pngs_removed": mashcima_png_removed,
                    "mashcima_manifest_rows_removed": manifest_removed,
                    "remaining_excerpts": remaining,
                },
                indent=2,
            )
        )
        return

    print(
        json.dumps(
            {
                "eligible_excerpts_kept": len(eligible),
                "primus_dirs_removed": dataset_removed.get(str(ROOT / "data" / "primus"), 0),
                "camera_primus_dirs_removed": dataset_removed.get(
                    str(ROOT / "data" / "camera-primus"), 0
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
