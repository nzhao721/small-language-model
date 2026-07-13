"""Move OMR benchmark selection into holdout folder and purge training data."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_layout import (  # noqa: E402
    camera_distorted_path,
    mashcima_image_path,
    primus_png_path,
    primus_semantic_path,
)

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "data" / "omr_benchmark_holdout"

GROUP_TO_SUBDIR = {
    "handwritten_original": "handwritten",
    "clean_original": "clean",
    "camera_distorted_original": "camera",
}


def incipit_id_from_entry(entry: dict) -> str:
    stem = Path(entry["filename"]).stem
    if stem.endswith("_distorted"):
        stem = stem[: -len("_distorted")]
    return stem


def filter_manifest(path: Path, drop_ids: set[str], id_key: str = "incipit_id") -> int:
    if not path.exists():
        return 0
    kept: list[str] = []
    removed = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get(id_key) in drop_ids:
                removed += 1
                continue
            kept.append(line.rstrip("\n"))
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def purge_corrupt_jobs(drop_ids: set[str]) -> int:
    jobs_path = ROOT / "data" / "corrupt" / "jobs.jsonl"
    if not jobs_path.exists():
        return 0
    kept: list[str] = []
    removed = 0
    with jobs_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("incipit_id") in drop_ids:
                removed += 1
                continue
            kept.append(line.rstrip("\n"))
    jobs_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def move_holdout(
    selection_path: Path,
    *,
    dry_run: bool = False,
) -> None:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    images_dir = HOLDOUT / "images"
    gt_dir = HOLDOUT / "ground_truth"
    manifest_path = HOLDOUT / "manifest.jsonl"

    for sub in GROUP_TO_SUBDIR.values():
        (images_dir / sub).mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    holdout_ids: set[str] = set()
    manifest_rows: list[dict] = []
    semantics_moved: set[str] = set()

    for entry in selection:
        incipit_id = incipit_id_from_entry(entry)
        holdout_ids.add(incipit_id)
        group = entry["group"]
        subdir = GROUP_TO_SUBDIR[group]
        src_image = Path(entry["image_path"])
        dst_image = images_dir / subdir / entry["filename"]

        manifest_rows.append(
            {
                "incipit_id": incipit_id,
                "group": group,
                "original_image_path": str(src_image.resolve()),
                "holdout_image_path": str(dst_image.resolve()),
                "original_semantic_path": entry.get("ground_truth_semantic_path"),
            }
        )

        if not dry_run:
            if src_image.exists():
                shutil.move(str(src_image), str(dst_image))
            entry["image_path"] = str(dst_image.resolve())

        semantic_src = Path(entry["ground_truth_semantic_path"])
        if incipit_id not in semantics_moved and semantic_src.exists():
            semantic_dst = gt_dir / f"{incipit_id}.semantic"
            if not dry_run:
                shutil.move(str(semantic_src), str(semantic_dst))
            semantics_moved.add(incipit_id)
            entry["ground_truth_semantic_path"] = str(
                (gt_dir / f"{incipit_id}.semantic").resolve()
            )

    # Purge all training artifacts for held-out incipits.
    corrupt_rows_to_purge: list[dict] = []
    corrupt_manifest_path = ROOT / "data" / "corrupt" / "manifest.jsonl"
    if corrupt_manifest_path.exists():
        with corrupt_manifest_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("incipit_id") in holdout_ids:
                    corrupt_rows_to_purge.append(row)

    mashcima_removed = filter_manifest(ROOT / "data" / "mashcima" / "manifest.jsonl", holdout_ids)
    corrupt_removed = filter_manifest(corrupt_manifest_path, holdout_ids)
    jobs_removed = purge_corrupt_jobs(holdout_ids) if not dry_run else 0

    extra_removed: dict[str, int] = {
        "mashcima_jpg": 0,
        "primus_png": 0,
        "camera_distorted": 0,
        "corrupt_images": 0,
        "corrupt_semantic": 0,
    }

    if not dry_run:
        for incipit_id in sorted(holdout_ids):
            mashcima_jpg = mashcima_image_path(incipit_id)
            if mashcima_jpg.exists():
                mashcima_jpg.unlink()
                extra_removed["mashcima_jpg"] += 1

            png = primus_png_path(incipit_id)
            if png.exists():
                png.unlink()
                extra_removed["primus_png"] += 1
            semantic = primus_semantic_path(incipit_id)
            if semantic.exists():
                semantic.unlink()

            cam = camera_distorted_path(incipit_id)
            if cam.exists():
                cam.unlink()
                extra_removed["camera_distorted"] += 1

        # Remove corrupt image/semantic files for dropped corrupt rows.
        for row in corrupt_rows_to_purge:
            cid = row.get("corrupt_id")
            if not cid:
                continue
            for fmt, ext in (
                ("handwritten", ".jpg"),
                ("clean", ".png"),
                ("camera", ".png"),
                ("camera", ".jpg"),
            ):
                p = ROOT / "data" / "corrupt" / fmt / f"{cid}{ext}"
                if p.exists():
                    p.unlink()
                    extra_removed["corrupt_images"] += 1
            sem = ROOT / "data" / "corrupt" / "semantic" / f"{cid}.semantic"
            if sem.exists():
                sem.unlink()
                extra_removed["corrupt_semantic"] += 1

        manifest_path.write_text(
            "\n".join(json.dumps(r) for r in manifest_rows) + "\n",
            encoding="utf-8",
        )
        selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")

        for batch_file in sorted((ROOT / "examples" / "omr_benchmark" / "batches_10").glob("batch_*.json")):
            batch = json.loads(batch_file.read_text(encoding="utf-8"))
            by_filename = {e["filename"]: e for e in selection}
            for i, item in enumerate(batch):
                if item["filename"] in by_filename:
                    batch[i] = by_filename[item["filename"]]
            batch_file.write_text(json.dumps(batch, indent=2), encoding="utf-8")

    print(f"Holdout incipits: {len(holdout_ids)}")
    print(f"  mashcima manifest rows removed: {mashcima_removed}")
    print(f"  corrupt manifest rows removed: {corrupt_removed}")
    print(f"  corrupt jobs removed: {jobs_removed}")
    for k, v in extra_removed.items():
        if v:
            print(f"  extra {k} removed: {v}")
    print(f"Holdout dir: {HOLDOUT}")
    if not dry_run:
        print(f"Updated selection: {selection_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "examples" / "omr_benchmark" / "selection_10.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    move_holdout(args.selection, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
