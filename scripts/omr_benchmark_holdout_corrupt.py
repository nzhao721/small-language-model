"""Move corrupt benchmark samples to holdout and purge from training data."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "data" / "omr_benchmark_holdout"
CORRUPT_DIR = ROOT / "data" / "corrupt"
CORRUPT_MANIFEST = CORRUPT_DIR / "manifest.jsonl"
CORRUPT_JOBS = CORRUPT_DIR / "jobs.jsonl"
HOLDOUT_MANIFEST = HOLDOUT / "manifest.jsonl"

FORMAT_SUBDIR = {
    "handwritten_corrupt": "handwritten",
    "clean_corrupt": "clean",
    "camera_distorted_corrupt": "camera",
}


def load_selection(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_holdout_manifest(rows: list[dict]) -> None:
    HOLDOUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with HOLDOUT_MANIFEST.open("a", encoding="utf-8") as f:
        for row in rows:
            entry = {
                "moved_at": ts,
                "kind": "corrupt",
                "group": row["group"],
                "corrupt_id": row["corrupt_id"],
                "incipit_id": row["incipit_id"],
                "from_image": row["image_path"],
                "from_semantic": row["ground_truth_semantic_path"],
                "holdout_image": row.get("holdout_image_path"),
                "holdout_semantic": row.get("holdout_semantic_path"),
            }
            f.write(json.dumps(entry) + "\n")


def move_to_holdout(selection: list[dict]) -> list[dict]:
    updated: list[dict] = []
    for item in selection:
        corrupt_id = item["corrupt_id"]
        fmt = FORMAT_SUBDIR[item["group"]]
        img_src = Path(item["image_path"])
        sem_src = Path(item["ground_truth_semantic_path"])

        img_dst = HOLDOUT / "corrupt" / "images" / fmt / img_src.name
        sem_dst = HOLDOUT / "corrupt" / "ground_truth" / f"{corrupt_id}.semantic"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        sem_dst.parent.mkdir(parents=True, exist_ok=True)

        if img_src.is_file():
            shutil.move(str(img_src), str(img_dst))
        elif not img_dst.is_file():
            raise FileNotFoundError(f"Missing corrupt image: {img_src}")

        if sem_src.is_file():
            shutil.move(str(sem_src), str(sem_dst))
        elif not sem_dst.is_file():
            raise FileNotFoundError(f"Missing corrupt semantic: {sem_src}")

        item = dict(item)
        item["holdout_image_path"] = str(img_dst.resolve())
        item["holdout_semantic_path"] = str(sem_dst.resolve())
        item["image_path"] = item["holdout_image_path"]
        item["ground_truth_semantic_path"] = item["holdout_semantic_path"]
        updated.append(item)
    return updated


def purge_corrupt_ids(corrupt_ids: set[str]) -> dict[str, int]:
    stats = {
        "manifest_rows_removed": 0,
        "jobs_removed": 0,
        "images_removed": 0,
        "semantic_removed": 0,
    }

    last: dict[str, dict] = {}
    if CORRUPT_MANIFEST.is_file():
        with CORRUPT_MANIFEST.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                last[row["corrupt_id"]] = row

    kept_manifest: list[str] = []
    for corrupt_id, row in last.items():
        if corrupt_id in corrupt_ids:
            stats["manifest_rows_removed"] += 1
            for fmt in ("clean", "camera", "handwritten"):
                for ext in (".png", ".jpg", ".jpeg"):
                    alt = CORRUPT_DIR / fmt / f"{corrupt_id}{ext}"
                    if alt.is_file():
                        alt.unlink()
                        stats["images_removed"] += 1
            semantic = CORRUPT_DIR / "semantic" / f"{corrupt_id}.semantic"
            if semantic.is_file():
                semantic.unlink()
                stats["semantic_removed"] += 1
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
                if job["corrupt_id"] in corrupt_ids:
                    stats["jobs_removed"] += 1
                    continue
                kept_jobs.append(line)
        with CORRUPT_JOBS.open("w", encoding="utf-8") as f:
            for line in kept_jobs:
                f.write(line + "\n")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "examples" / "omr_benchmark" / "selection_10_corrupt.json",
    )
    parser.add_argument(
        "--update-selection",
        action="store_true",
        help="Rewrite selection JSON with holdout paths after move.",
    )
    args = parser.parse_args()

    selection = load_selection(args.selection)
    corrupt_ids = {s["corrupt_id"] for s in selection}

    moved = move_to_holdout(selection)
    append_holdout_manifest(moved)
    stats = purge_corrupt_ids(corrupt_ids)

    if args.update_selection:
        args.selection.write_text(json.dumps(moved, indent=2), encoding="utf-8")

    print(f"Moved {len(moved)} corrupt samples to {HOLDOUT / 'corrupt'}")
    print(f"Purge stats: {stats}")
    print(f"corrupt_ids: {sorted(corrupt_ids)}")


if __name__ == "__main__":
    main()
