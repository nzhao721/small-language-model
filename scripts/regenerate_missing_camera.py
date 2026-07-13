"""Regenerate corrupt camera jobs whose image files are missing on disk."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_corrupt_batch import (  # noqa: E402
    CORRUPT_DIR,
    DEFAULT_JOBS,
    DEFAULT_MANIFEST,
    DEFAULT_PRIMUS_DIR,
    DEFAULT_PRIMUS_TGZ,
    ROOT,
    _init_pool,
    _regen_worker,
    load_jobs,
    load_last_ok_rows,
    load_manifest_records,
)
from generate_mashcima_batch import DEFAULT_MUSCIMA_DIR, ensure_muscima_cache  # noqa: E402

LOG_PATH = CORRUPT_DIR / "regenerate_missing_camera.log"
FORMATS = ("clean", "camera", "handwritten")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def log(msg: str, log_file) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def find_missing_camera_ids(manifest: Path, corrupt_dir: Path) -> list[str]:
    """corrupt_ids with ok camera manifest row, jpg png_path, no image on disk."""
    records = load_manifest_records(manifest)
    missing: list[str] = []
    for corrupt_id, row in records.items():
        if row.get("format") != "camera" or row.get("status") != "ok":
            continue
        png_path = row.get("png_path", "")
        if not png_path.endswith(".jpg"):
            continue
        png = corrupt_dir / "camera" / f"{corrupt_id}.png"
        jpg = corrupt_dir / "camera" / f"{corrupt_id}.jpg"
        if not png.is_file() and not jpg.is_file():
            missing.append(corrupt_id)
    return sorted(missing)


def count_still_missing(manifest: Path, corrupt_dir: Path) -> int:
    return len(find_missing_camera_ids(manifest, corrupt_dir))


def purge_failed_corrupt(corrupt_ids: set[str]) -> dict[str, int]:
    """Remove failed jobs from jobs.jsonl, manifest, semantic, and images."""
    stats = {
        "jobs_removed": 0,
        "manifest_rows_removed": 0,
        "semantic_removed": 0,
        "images_removed": 0,
    }
    if not corrupt_ids:
        return stats

    if DEFAULT_MANIFEST.is_file():
        kept: list[str] = []
        with DEFAULT_MANIFEST.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row["corrupt_id"] in corrupt_ids:
                    stats["manifest_rows_removed"] += 1
                    continue
                kept.append(line)
        with DEFAULT_MANIFEST.open("w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")

    if DEFAULT_JOBS.is_file():
        kept_jobs: list[str] = []
        with DEFAULT_JOBS.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                job = json.loads(line)
                if job["corrupt_id"] in corrupt_ids:
                    stats["jobs_removed"] += 1
                    continue
                kept_jobs.append(line)
        with DEFAULT_JOBS.open("w", encoding="utf-8") as f:
            for line in kept_jobs:
                f.write(line + "\n")

    for corrupt_id in corrupt_ids:
        semantic = CORRUPT_DIR / "semantic" / f"{corrupt_id}.semantic"
        if semantic.is_file():
            semantic.unlink()
            stats["semantic_removed"] += 1
        for fmt in FORMATS:
            for ext in IMAGE_EXTS:
                img = CORRUPT_DIR / fmt / f"{corrupt_id}{ext}"
                if img.is_file():
                    img.unlink()
                    stats["images_removed"] += 1

    return stats


def run(
    *,
    workers: int = 5,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict:
    CORRUPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    missing_ids = find_missing_camera_ids(DEFAULT_MANIFEST, CORRUPT_DIR)
    if limit is not None:
        missing_ids = missing_ids[:limit]

    jobs_by_id = {j["corrupt_id"]: j for j in load_jobs(DEFAULT_JOBS)}
    last_ok_rows = load_last_ok_rows(DEFAULT_MANIFEST)

    pending: list[tuple[dict, dict]] = []
    skipped_no_job: list[str] = []
    skipped_no_row: list[str] = []
    for corrupt_id in missing_ids:
        job = jobs_by_id.get(corrupt_id)
        if job is None:
            skipped_no_job.append(corrupt_id)
            continue
        row = last_ok_rows.get(corrupt_id)
        if row is None:
            skipped_no_row.append(corrupt_id)
            continue
        pending.append((job, row))

    summary = {
        "identified_missing": len(missing_ids),
        "attempted": 0,
        "ok": 0,
        "failed_purged": 0,
        "skipped_no_job": len(skipped_no_job),
        "skipped_no_row": len(skipped_no_row),
        "remaining_missing": len(missing_ids),
        "purge_stats": {},
        "errors": [],
    }

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log(
            f"Missing camera images: {len(missing_ids)}; "
            f"to regenerate: {len(pending)}; dry_run={dry_run}",
            log_file,
        )
        if skipped_no_job:
            log(f"Skipped {len(skipped_no_job)} ids with no jobs.jsonl entry", log_file)
        if skipped_no_row:
            log(f"Skipped {len(skipped_no_row)} ids with no ok manifest row", log_file)

        if dry_run:
            for corrupt_id in missing_ids[:10]:
                log(f"  would regen: {corrupt_id}", log_file)
            if len(missing_ids) > 10:
                log(f"  ... and {len(missing_ids) - 10} more", log_file)
            summary["remaining_missing"] = len(missing_ids)
            return summary

        if not pending:
            log("Nothing to regenerate.", log_file)
            summary["remaining_missing"] = count_still_missing(DEFAULT_MANIFEST, CORRUPT_DIR)
            return summary

        ensure_muscima_cache(DEFAULT_PRIMUS_TGZ, DEFAULT_MUSCIMA_DIR)
        payloads = [
            (job, row, str(DEFAULT_PRIMUS_DIR.resolve()), str(CORRUPT_DIR.resolve()))
            for job, row in pending
        ]
        initargs = (str(DEFAULT_PRIMUS_TGZ.resolve()), str(DEFAULT_MUSCIMA_DIR.resolve()))
        failed_ids: set[str] = []
        summary["attempted"] = len(pending)

        with DEFAULT_MANIFEST.open("a", encoding="utf-8") as manifest:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_pool,
                initargs=initargs,
            ) as pool_exec:
                futures = {
                    pool_exec.submit(_regen_worker, payload): payload[0]["corrupt_id"]
                    for payload in payloads
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Regen missing camera",
                    unit="job",
                ):
                    corrupt_id = futures[future]
                    record = future.result()
                    manifest.write(json.dumps(record) + "\n")
                    manifest.flush()
                    if record.get("status") == "ok":
                        summary["ok"] += 1
                        log(
                            f"OK {corrupt_id} -> {record.get('png_path')}",
                            log_file,
                        )
                    else:
                        failed_ids.add(corrupt_id)
                        err = record.get("error", "unknown error")
                        summary["errors"].append({"corrupt_id": corrupt_id, "error": err})
                        log(f"FAIL {corrupt_id}: {err}", log_file)

        if failed_ids:
            purge_stats = purge_failed_corrupt(failed_ids)
            summary["failed_purged"] = len(failed_ids)
            summary["purge_stats"] = purge_stats
            log(f"Purged {len(failed_ids)} failed jobs: {purge_stats}", log_file)

        summary["remaining_missing"] = count_still_missing(DEFAULT_MANIFEST, CORRUPT_DIR)
        log(
            f"Done: attempted={summary['attempted']} ok={summary['ok']} "
            f"purged={summary['failed_purged']} remaining_missing={summary['remaining_missing']}",
            log_file,
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(workers=args.workers, dry_run=args.dry_run, limit=args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
