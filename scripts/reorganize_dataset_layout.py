"""Reorganize primus, mashcima, and camera-primus into images/ + ground_truth/ layout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_layout import (
    CAMERA_PRIMUS_DIR,
    MASHCIMA_DIR,
    PRIMUS_DIR,
    ROOT,
    camera_distorted_path,
    camera_ground_truth_dir,
    camera_images_dir,
    mashcima_ground_truth_dir,
    mashcima_image_path,
    mashcima_images_dir,
    primus_agnostic_dir,
    primus_ground_truth_dir,
    primus_images_dir,
    primus_semantic_path,
)

KEEP_FILES = {
    "manifest.jsonl",
    "camera_distort_manifest.jsonl",
}


def _move_unique(src: Path, dest: Path, dry_run: bool) -> bool:
    if not src.is_file():
        return False
    if dest.is_file():
        if src.resolve() == dest.resolve():
            return False
        src.unlink()
        return False
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    return True


def reorganize_primus(primus_dir: Path, dry_run: bool) -> dict[str, int]:
    stats = {"png_moved": 0, "semantic_moved": 0, "agnostic_moved": 0, "dirs_removed": 0}
    images = primus_images_dir(primus_dir)
    gt = primus_ground_truth_dir(primus_dir)
    agnostic_dir = primus_agnostic_dir(primus_dir)

    for sem in list(primus_dir.rglob("*.semantic")):
        if "ground_truth" in sem.parts:
            continue
        inc = sem.stem
        png = sem.parent / f"{inc}.png"
        if _move_unique(png, images / png.name, dry_run):
            stats["png_moved"] += 1
        if _move_unique(sem, gt / sem.name, dry_run):
            stats["semantic_moved"] += 1
        ag = sem.parent / f"{inc}.agnostic"
        if _move_unique(ag, agnostic_dir / ag.name, dry_run):
            stats["agnostic_moved"] += 1

    for pkg in list(primus_dir.glob("package_*")):
        if pkg.is_dir():
            if not dry_run:
                shutil.rmtree(pkg)
            stats["dirs_removed"] += 1

    return stats


def reorganize_camera(camera_dir: Path, dry_run: bool) -> dict[str, int]:
    stats = {"jpg_moved": 0, "semantic_moved": 0, "dirs_removed": 0}
    images = camera_images_dir(camera_dir)
    gt = camera_ground_truth_dir(camera_dir)
    corpus = camera_dir / "Corpus"

    if corpus.is_dir():
        for inc_dir in list(corpus.iterdir()):
            if not inc_dir.is_dir():
                continue
            inc = inc_dir.name
            for jpg in inc_dir.glob("*_distorted.jpg"):
                if _move_unique(jpg, images / jpg.name, dry_run):
                    stats["jpg_moved"] += 1
            sem = inc_dir / f"{inc}.semantic"
            dest = gt / sem.name
            if sem.is_file() and not dest.is_file():
                if _move_unique(sem, dest, dry_run):
                    stats["semantic_moved"] += 1
            elif sem.is_file() and not dry_run:
                sem.unlink()
        if not dry_run:
            shutil.rmtree(corpus)
        stats["dirs_removed"] += 1

    return stats


def reorganize_mashcima(mashcima_dir: Path, primus_dir: Path, dry_run: bool) -> dict[str, int]:
    stats = {"jpg_moved": 0, "png_moved": 0, "semantic_copied": 0, "files_removed": 0}
    images = mashcima_images_dir(mashcima_dir)
    gt = mashcima_ground_truth_dir(mashcima_dir)
    manifest_path = mashcima_dir / "manifest.jsonl"

    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for img in list(mashcima_dir.glob(pattern)):
            if img.parent.name == "images":
                continue
            dest = images / img.name
            if _move_unique(img, dest, dry_run):
                if img.suffix.lower() == ".png":
                    stats["png_moved"] += 1
                else:
                    stats["jpg_moved"] += 1

    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                inc = row["incipit_id"]
                src = primus_semantic_path(inc, primus_dir)
                dest = gt / f"{inc}.semantic"
                if src.is_file() and not dest.is_file():
                    if not dry_run:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dest)
                    stats["semantic_copied"] += 1

    for child in list(mashcima_dir.iterdir()):
        if child.is_dir() and child.name not in {"images", "ground_truth"}:
            if not dry_run:
                shutil.rmtree(child)
            stats["files_removed"] += 1
        elif child.is_file() and child.name not in KEEP_FILES:
            if not dry_run:
                child.unlink()
            stats["files_removed"] += 1

    return stats


def update_mashcima_manifest(mashcima_dir: Path, dry_run: bool) -> int:
    manifest_path = mashcima_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        return 0
    updated = 0
    rows: list[str] = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            inc = row.get("incipit_id")
            if inc and row.get("status") == "ok":
                row["output_png"] = (
                    mashcima_image_path(inc, mashcima_dir).relative_to(ROOT).as_posix()
                )
                row["semantic_path"] = (
                    mashcima_ground_truth_dir(mashcima_dir) / f"{inc}.semantic"
                ).relative_to(ROOT).as_posix()
                row["primus_path"] = (
                    primus_ground_truth_dir(PRIMUS_DIR) / f"{inc}.semantic"
                ).relative_to(ROOT).as_posix()
                updated += 1
            rows.append(json.dumps(row))
    if not dry_run:
        manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return updated


def update_corrupt_jobs(jobs_path: Path, dry_run: bool) -> int:
    if not jobs_path.is_file():
        return 0
    primus_root = str(PRIMUS_DIR.resolve())
    updated = 0
    rows: list[str] = []
    with jobs_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("primus_folder") != primus_root:
                row["primus_folder"] = primus_root
                updated += 1
            rows.append(json.dumps(row))
    if not dry_run and updated:
        jobs_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return updated


def update_duration_flags(flags_path: Path, dry_run: bool) -> int:
    if not flags_path.is_file():
        return 0
    out: list[str] = []
    n = 0
    with flags_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sp = row.get("semantic_path", "")
            if "/package_" in sp:
                inc = row.get("incipit_id") or Path(sp).stem
                row["semantic_path"] = (
                    primus_ground_truth_dir(PRIMUS_DIR) / f"{inc}.semantic"
                ).relative_to(ROOT).as_posix()
                n += 1
            out.append(json.dumps(row))
    if not dry_run:
        flags_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return n


def cleanup_primus_strays(primus_dir: Path, dry_run: bool) -> int:
    removed = 0
    keep_dirs = {"images", "ground_truth", "agnostic"}
    for child in list(primus_dir.iterdir()):
        if child.is_dir() and child.name not in keep_dirs:
            if not dry_run:
                shutil.rmtree(child)
            removed += 1
        elif child.is_file():
            if not dry_run:
                child.unlink()
            removed += 1
    return removed


def cleanup_camera_strays(camera_dir: Path, dry_run: bool) -> int:
    removed = 0
    for child in list(camera_dir.iterdir()):
        if child.is_dir() and child.name not in {"images", "ground_truth"}:
            if not dry_run:
                shutil.rmtree(child)
            removed += 1
        elif child.is_file():
            if not dry_run:
                child.unlink()
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Reorganizing primus...", flush=True)
    p_stats = reorganize_primus(PRIMUS_DIR, args.dry_run)
    print(p_stats, flush=True)

    print("Reorganizing camera-primus...", flush=True)
    c_stats = reorganize_camera(CAMERA_PRIMUS_DIR, args.dry_run)
    print(c_stats, flush=True)

    print("Reorganizing mashcima...", flush=True)
    m_stats = reorganize_mashcima(MASHCIMA_DIR, PRIMUS_DIR, args.dry_run)
    print(m_stats, flush=True)

    if not args.dry_run:
        print("Cleaning strays...", flush=True)
        print("primus:", cleanup_primus_strays(PRIMUS_DIR, False))
        print("camera:", cleanup_camera_strays(CAMERA_PRIMUS_DIR, False))
        n = update_mashcima_manifest(MASHCIMA_DIR, False)
        print(f"mashcima manifest rows updated: {n}")
        n2 = update_duration_flags(ROOT / "data" / "duration_range_flags.jsonl", False)
        print(f"duration_range_flags rows updated: {n2}")
        n3 = update_corrupt_jobs(ROOT / "data" / "corrupt" / "jobs.jsonl", False)
        print(f"corrupt jobs rows updated: {n3}")


if __name__ == "__main__":
    main()
