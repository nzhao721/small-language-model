"""Replace golden_set corrupt clean + handwritten holdout samples from data/corrupt."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "golden_set"
CORRUPT = ROOT / "data" / "corrupt"
CONSOLIDATED_IMG = ROOT / "data" / "consolidated" / "images"
CONSOLIDATED_GT = ROOT / "data" / "consolidated" / "ground_truth"
BENCH = ROOT / "examples" / "omr_benchmark"
SELECTION_PATH = BENCH / "selection_10_corrupt.json"
MANIFEST_PATH = GOLDEN / "manifest.jsonl"

GROUP_BY_FORMAT = {
    "handwritten": "handwritten_corrupt",
    "clean": "clean_corrupt",
    "camera": "camera_distorted_corrupt",
}

FORMAT_SUBDIR = {
    "handwritten_corrupt": "handwritten",
    "clean_corrupt": "clean",
    "camera_distorted_corrupt": "camera",
}

REPLACE_GROUPS = {"handwritten_corrupt", "clean_corrupt"}


def load_corrupt_ok(exclude_ids: set[str]) -> dict[str, list[dict]]:
    last_ok: dict[str, dict] = {}
    with (CORRUPT / "manifest.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            last_ok[row["corrupt_id"]] = row

    by_format: dict[str, list[dict]] = {"handwritten": [], "clean": [], "camera": []}
    for row in last_ok.values():
        cid = row["corrupt_id"]
        if cid in exclude_ids:
            continue
        fmt = row["format"]
        if fmt not in by_format:
            continue
        png_path = ROOT / row["png_path"]
        semantic_path = ROOT / row["semantic_path"]
        if png_path.exists() and semantic_path.exists():
            by_format[fmt].append(
                {
                    "corrupt_id": cid,
                    "incipit_id": row["incipit_id"],
                    "format": fmt,
                    "png_path": png_path,
                    "semantic_path": semantic_path,
                }
            )
    return by_format


def pick_n(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    shuffled = pool[:]
    rng.shuffle(shuffled)
    if len(shuffled) < n:
        raise RuntimeError(f"Only found {len(shuffled)}/{n} valid corrupt samples")
    return shuffled[:n]


def build_entry(fmt: str, row: dict) -> dict:
    group = GROUP_BY_FORMAT[fmt]
    img_path = row["png_path"]
    return {
        "group": group,
        "corrupt_id": row["corrupt_id"],
        "incipit_id": row["incipit_id"],
        "filename": img_path.name,
        "image_path": str(img_path.resolve()),
        "ground_truth_semantic_path": str(row["semantic_path"].resolve()),
    }


def remove_from_golden(old_ids: set[str], groups: set[str]) -> None:
    for entry in json.loads(SELECTION_PATH.read_text(encoding="utf-8")):
        if entry["group"] not in groups:
            continue
        cid = entry["corrupt_id"]
        if cid not in old_ids:
            continue
        fmt = FORMAT_SUBDIR[entry["group"]]
        for p in (
            GOLDEN / "corrupt" / "images" / fmt / entry["filename"],
            GOLDEN / "corrupt" / "ground_truth" / f"{cid}.semantic",
        ):
            if p.is_file():
                p.unlink()


def move_to_golden(entry: dict) -> dict:
    cid = entry["corrupt_id"]
    fmt = FORMAT_SUBDIR[entry["group"]]
    img_src = Path(entry["image_path"])
    sem_src = Path(entry["ground_truth_semantic_path"])
    img_dst = GOLDEN / "corrupt" / "images" / fmt / img_src.name
    sem_dst = GOLDEN / "corrupt" / "ground_truth" / f"{cid}.semantic"
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    sem_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(img_src), str(img_dst))
    shutil.move(str(sem_src), str(sem_dst))
    entry = dict(entry)
    entry["image_path"] = str(img_dst.resolve())
    entry["ground_truth_semantic_path"] = str(sem_dst.resolve())
    entry["holdout_image_path"] = entry["image_path"]
    entry["holdout_semantic_path"] = entry["ground_truth_semantic_path"]
    return entry


def purge_corrupt_ids(corrupt_ids: set[str]) -> dict[str, int]:
    stats = {
        "manifest_rows_removed": 0,
        "jobs_removed": 0,
        "images_removed": 0,
        "semantic_removed": 0,
    }

    last: dict[str, dict] = {}
    manifest_path = CORRUPT / "manifest.jsonl"
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            last[row["corrupt_id"]] = row

    kept_manifest: list[str] = []
    for corrupt_id, row in last.items():
        if corrupt_id in corrupt_ids:
            stats["manifest_rows_removed"] += 1
            for fmt in ("clean", "camera", "handwritten"):
                for ext in (".png", ".jpg", ".jpeg"):
                    alt = CORRUPT / fmt / f"{corrupt_id}{ext}"
                    if alt.is_file():
                        alt.unlink()
                        stats["images_removed"] += 1
            semantic = CORRUPT / "semantic" / f"{corrupt_id}.semantic"
            if semantic.is_file():
                semantic.unlink()
                stats["semantic_removed"] += 1
            continue
        kept_manifest.append(json.dumps(row))

    manifest_path.write_text("\n".join(kept_manifest) + ("\n" if kept_manifest else ""), encoding="utf-8")

    jobs_path = CORRUPT / "jobs.jsonl"
    if jobs_path.is_file():
        kept_jobs: list[str] = []
        with jobs_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                job = json.loads(line)
                if job["corrupt_id"] in corrupt_ids:
                    stats["jobs_removed"] += 1
                    continue
                kept_jobs.append(line)
        jobs_path.write_text("\n".join(kept_jobs) + ("\n" if kept_jobs else ""), encoding="utf-8")

    return stats


def purge_consolidated(corrupt_ids: set[str], formats: dict[str, str]) -> dict[str, int]:
    """Remove consolidated copies. formats maps corrupt_id -> 'clean'|'handwritten'."""
    stats = {"images_removed": 0, "gt_removed": 0}
    for cid, fmt in formats.items():
        prefix = f"corrupt_{fmt}__{cid}"
        for folder, key in ((CONSOLIDATED_IMG, "images_removed"), (CONSOLIDATED_GT, "gt_removed")):
            if not folder.is_dir():
                continue
            for path in folder.glob(f"{prefix}.*"):
                if path.is_file():
                    path.unlink()
                    stats[key] += 1
    return stats


def update_golden_manifest(
    removed_ids: set[str],
    new_entries: list[dict],
) -> None:
    rows: list[dict] = []
    if MANIFEST_PATH.is_file():
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") == "corrupt" and row.get("id") in removed_ids:
                continue
            rows.append(row)

    moved_at = datetime.now(timezone.utc).isoformat()
    for entry in new_entries:
        cid = entry["corrupt_id"]
        rows.append(
            {
                "id": cid,
                "incipit_id": entry["incipit_id"],
                "group": entry["group"],
                "image_path": entry["image_path"],
                "gt_path": entry["ground_truth_semantic_path"],
                "source": "corrupt",
                "moved_at": moved_at,
            }
        )

    MANIFEST_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    golden_corrupt_ids = {
        row["id"]
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row.get("source") == "corrupt"
    }

    old_replace = [e for e in selection if e["group"] in REPLACE_GROUPS]
    old_ids = {e["corrupt_id"] for e in old_replace}
    keep = [e for e in selection if e["group"] not in REPLACE_GROUPS]

    pool = load_corrupt_ok(golden_corrupt_ids)
    rng = random.Random(args.seed)
    new_entries: list[dict] = []
    id_to_fmt: dict[str, str] = {}
    for fmt in ("handwritten", "clean"):
        picked = pick_n(pool[fmt], args.n, rng)
        for row in picked:
            entry = build_entry(fmt, row)
            new_entries.append(entry)
            id_to_fmt[entry["corrupt_id"]] = fmt

    if args.dry_run:
        print(json.dumps({"old_ids": sorted(old_ids), "new_ids": sorted(id_to_fmt)}, indent=2))
        return

    remove_from_golden(old_ids, REPLACE_GROUPS)
    moved = [move_to_golden(e) for e in new_entries]
    purge_stats = purge_corrupt_ids(set(id_to_fmt))
    cons_stats = purge_consolidated(set(id_to_fmt), id_to_fmt)
    update_golden_manifest(old_ids, moved)

    updated_selection = (
        [e for e in moved if e["group"] == "handwritten_corrupt"]
        + [e for e in moved if e["group"] == "clean_corrupt"]
        + keep
    )
    SELECTION_PATH.write_text(json.dumps(updated_selection, indent=2) + "\n", encoding="utf-8")

    print(f"Replaced {len(old_ids)} holdout samples with {len(moved)} new ones")
    print(f"Purge corrupt: {purge_stats}")
    print(f"Purge consolidated: {cons_stats}")
    print("New handwritten:", [e["corrupt_id"] for e in moved if e["group"] == "handwritten_corrupt"])
    print("New clean:", [e["corrupt_id"] for e in moved if e["group"] == "clean_corrupt"])


if __name__ == "__main__":
    main()
