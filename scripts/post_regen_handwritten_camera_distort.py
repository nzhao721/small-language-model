"""After per-score handwritten regeneration: camera-distort + sync consolidated.

For each successfully regenerated handwritten image:
- Mashcima + golden original: apply camera_like_distort in place, then overwrite consolidated copy.
- Corrupt/golden corrupt handwritten: already camera-distorted on regen — sync consolidated only.

Skipped (keeps prior files in originals + consolidated):
- Mashcima/corrupt manifest rows whose latest status is not ok.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from consolidate_dataset_pairs import CONSOLIDATED, consolidate_name  # noqa: E402
from dataset_layout import (  # noqa: E402
    MASHCIMA_DIR,
    ROOT,
)
from demo_mashcima_camera_distort import (  # noqa: E402
    camera_like_distort,
    seed_for_name,
)
from generate_corrupt_batch import (  # noqa: E402
    DEFAULT_MANIFEST as CORRUPT_MANIFEST,
    load_last_ok_rows,
    load_manifest_records,
    load_regenerated_ok_ids,
)
from image_compress import compress_image_file, resolved_image_path  # noqa: E402

from PIL import Image

MASHCIMA_MANIFEST = MASHCIMA_DIR / "manifest.jsonl"
GOLDEN_MANIFEST = ROOT / "golden_set" / "manifest.jsonl"
CORRUPT_HANDWRITTEN_DIR = ROOT / "data" / "corrupt" / "handwritten"
LOG_DIR = ROOT / "data" / "logs"
DEFAULT_PROGRESS_LOG = LOG_DIR / "post_regen_handwritten_camera_distort_progress.jsonl"


class ProgressLogger:
    """Append JSONL progress events (flushed) for tailing during long runs."""

    def __init__(self, path: Path, interval: int) -> None:
        self.path = path
        self.interval = max(1, interval)
        self.phase = ""
        self.total = 0
        self.done = 0
        self._phase_started = 0.0
        path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, event: str, **fields: object) -> None:
        elapsed = time.monotonic() - self._phase_started if self._phase_started else 0.0
        rate = self.done / elapsed if elapsed > 0 and self.done else 0.0
        remaining = max(self.total - self.done, 0)
        eta_s = remaining / rate if rate > 0 else None
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "phase": self.phase,
            "done": self.done,
            "total": self.total,
            "pct": round(100.0 * self.done / self.total, 2) if self.total else 0.0,
            "rate_per_min": round(rate * 60, 1) if rate else 0.0,
            "eta_min": round(eta_s / 60, 1) if eta_s is not None else None,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if event in {"progress", "phase_start", "phase_done", "run_start", "run_done"}:
            eta = f" eta_min={row['eta_min']}" if row["eta_min"] is not None else ""
            print(
                f"[post_regen] {event} {self.phase} {self.done}/{self.total} "
                f"({row['pct']}%) rate={row['rate_per_min']}/min{eta}",
                flush=True,
            )

    def run_start(self, **fields: object) -> None:
        self._emit("run_start", **fields)

    def phase_start(self, phase: str, total: int, **fields: object) -> None:
        self.phase = phase
        self.total = total
        self.done = 0
        self._phase_started = time.monotonic()
        self._emit("phase_start", **fields)

    def tick(self, **fields: object) -> None:
        self.done += 1
        if self.done == 1 or self.done == self.total or self.done % self.interval == 0:
            self._emit("progress", **fields)

    def phase_done(self, stats: dict[str, int]) -> None:
        self._emit("phase_done", stats=stats)

    def run_done(self, results: dict) -> None:
        self._emit("run_done", results=results)


def load_last_mashcima_status(manifest: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not manifest.is_file():
        return rows
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            incipit_id = row.get("incipit_id")
            if incipit_id:
                rows[incipit_id] = row
    return rows


def resolve_image(path_hint: Path | None, stem: str, search_dirs: list[Path]) -> Path | None:
    if path_hint and path_hint.is_file():
        return path_hint
    for d in search_dirs:
        for ext in (".jpg", ".jpeg", ".png"):
            p = d / f"{stem}{ext}"
            if p.is_file():
                return p
    return None


def distort_image_in_place(path: Path) -> None:
    original = Image.open(path).convert("RGB")
    distorted = camera_like_distort(original, seed=seed_for_name(path.stem))
    tmp = path.with_suffix(".camera_tmp" + path.suffix)
    try:
        distorted.save(tmp)
        tmp.replace(path)
        compress_image_file(path)
    finally:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)


def sync_consolidated_image(src: Path, group_key: str, match_stem: str, out_dir: Path) -> Path:
    dest = out_dir / "images" / consolidate_name(group_key, match_stem, src.suffix.lower())
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def remove_stale_png_sibling(image_path: Path) -> None:
    """Drop orphan .png when .jpg is the canonical file (regen/compress leftover)."""
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        png = image_path.with_suffix(".png")
        if png.is_file():
            png.unlink()


def find_consolidated_image(out_dir: Path, group_key: str, stem: str) -> Path | None:
    images = out_dir / "images"
    for ext in (".jpg", ".jpeg", ".png"):
        path = images / consolidate_name(group_key, stem, ext)
        if path.is_file():
            return path
    return None


def process_mashcima(
    out_dir: Path,
    dry_run: bool,
    progress: ProgressLogger | None = None,
    *,
    resume: bool = False,
) -> dict[str, int]:
    stats = {"distorted": 0, "synced": 0, "skipped_error": 0, "skipped_resume": 0, "missing": 0}
    rows = load_last_mashcima_status(MASHCIMA_MANIFEST)
    images_dir = MASHCIMA_DIR / "images"

    eligible = 0
    for incipit_id, row in rows.items():
        if row.get("status") != "ok":
            continue
        hint = Path(row["output_png"]) if row.get("output_png") else None
        if hint and not hint.is_absolute():
            hint = ROOT / hint
        if resolve_image(hint, incipit_id, [images_dir]) is not None:
            eligible += 1
    if progress is not None:
        progress.phase_start("mashcima", eligible)

    for incipit_id, row in rows.items():
        status = row.get("status")
        if status != "ok":
            stats["skipped_error"] += 1
            continue

        hint = Path(row["output_png"]) if row.get("output_png") else None
        if hint and not hint.is_absolute():
            hint = ROOT / hint
        img_path = resolve_image(hint, incipit_id, [images_dir])
        if img_path is None:
            stats["missing"] += 1
            continue

        if resume and find_consolidated_image(out_dir, "mashcima", incipit_id) is not None:
            stats["skipped_resume"] += 1
            if progress is not None:
                progress.tick(incipit_id=incipit_id, skipped=True)
            continue

        if not dry_run:
            distort_image_in_place(img_path)
            final_path = resolved_image_path(img_path)
            remove_stale_png_sibling(final_path)
            sync_consolidated_image(
                final_path, "mashcima", incipit_id, out_dir
            )
        stats["distorted"] += 1
        stats["synced"] += 1
        if progress is not None:
            progress.tick(incipit_id=incipit_id)

    if progress is not None:
        progress.phase_done(stats)
    return stats


def process_corrupt_handwritten(
    out_dir: Path,
    dry_run: bool,
    progress: ProgressLogger | None = None,
    *,
    resume: bool = False,
) -> dict[str, int]:
    stats = {"synced": 0, "skipped_error": 0, "skipped_resume": 0, "missing": 0}
    ok_ids = load_regenerated_ok_ids(CORRUPT_MANIFEST, {"handwritten"})
    all_rows = load_manifest_records(CORRUPT_MANIFEST)

    eligible = 0
    for corrupt_id in ok_ids:
        row = all_rows.get(corrupt_id, {})
        if row.get("status") != "ok":
            continue
        img_path = resolve_image(
            ROOT / row["png_path"] if row.get("png_path") else None,
            corrupt_id,
            [CORRUPT_HANDWRITTEN_DIR],
        )
        if img_path is not None:
            eligible += 1
    if progress is not None:
        progress.phase_start("corrupt_handwritten", eligible)

    for corrupt_id in ok_ids:
        row = all_rows.get(corrupt_id, {})
        if row.get("status") != "ok":
            stats["skipped_error"] += 1
            continue
        img_path = resolve_image(
            ROOT / row["png_path"] if row.get("png_path") else None,
            corrupt_id,
            [CORRUPT_HANDWRITTEN_DIR],
        )
        if img_path is None:
            stats["missing"] += 1
            continue
        if resume and find_consolidated_image(out_dir, "corrupt/handwritten", corrupt_id) is not None:
            stats["skipped_resume"] += 1
            if progress is not None:
                progress.tick(corrupt_id=corrupt_id, skipped=True)
            continue
        if not dry_run:
            sync_consolidated_image(
                resolved_image_path(img_path), "corrupt/handwritten", corrupt_id, out_dir
            )
        stats["synced"] += 1
        if progress is not None:
            progress.tick(corrupt_id=corrupt_id)

    if progress is not None:
        progress.phase_done(stats)
    return stats


def process_golden_handwritten(
    out_dir: Path,
    dry_run: bool,
    progress: ProgressLogger | None = None,
    *,
    resume: bool = False,
    force: bool = False,
) -> dict[str, int]:
    stats = {"distorted": 0, "synced": 0, "skipped": 0, "skipped_resume": 0, "missing": 0}
    if not GOLDEN_MANIFEST.is_file():
        return stats

    rows: list[dict] = []
    with GOLDEN_MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    targets = [
        row
        for row in rows
        if row.get("group") in {"handwritten_original", "handwritten_corrupt"}
        and Path(row["image_path"]).is_file()
    ]
    if progress is not None:
        progress.phase_start("golden_handwritten", len(targets))

    for row in targets:
        group = row.get("group", "")
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            stats["missing"] += 1
            continue

        split = "original" if group == "handwritten_original" else "corrupt"
        group_key = f"golden_{split}/handwritten"
        match_stem = image_path.stem

        if (
            resume
            and not force
            and find_consolidated_image(out_dir, group_key, match_stem) is not None
        ):
            stats["skipped_resume"] += 1
            if progress is not None:
                progress.tick(id=row.get("id"), group=group, skipped=True)
            continue

        if group == "handwritten_original":
            if not dry_run:
                distort_image_in_place(image_path)
                sync_consolidated_image(
                    resolved_image_path(image_path), group_key, match_stem, out_dir
                )
            stats["distorted"] += 1
            stats["synced"] += 1
        else:
            if not dry_run:
                compress_image_file(image_path)
                remove_stale_png_sibling(image_path)
                sync_consolidated_image(
                    resolved_image_path(image_path), group_key, match_stem, out_dir
                )
            stats["synced"] += 1
        if progress is not None:
            progress.tick(id=row.get("id"), group=group)

    if progress is not None:
        progress.phase_done(stats)
    return stats


def wait_for_regeneration(summary: Path, poll_seconds: float) -> bool:
    print(f"Waiting for regeneration summary: {summary}", flush=True)
    while True:
        if summary.is_file():
            text = summary.read_text(encoding="utf-8")
            if text.rstrip().endswith("done"):
                print("Regeneration finished (summary: done).", flush=True)
                return True
            if text.rstrip().endswith("failed"):
                print("Regeneration summary reports failed — processing completed steps only.", flush=True)
                return False
        time.sleep(poll_seconds)


def regeneration_running() -> bool:
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'regenerate_all_handwritten|generate_mashcima_batch|generate_corrupt_batch|regenerate_golden_handwritten' } | Measure-Object | Select-Object -ExpandProperty Count",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return int(out.stdout.strip() or "0") > 0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait-summary",
        type=Path,
        default=None,
        help="Poll until this regenerate_all summary.log ends with done/failed",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=120.0,
        help="Poll interval when --wait-summary is set",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=CONSOLIDATED,
        help="Consolidated dataset root (default: data/consolidated)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=DEFAULT_PROGRESS_LOG,
        help="Append JSONL progress events here (default: data/logs/post_regen_handwritten_camera_distort_progress.jsonl)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=500,
        help="Log every N processed images per phase (default: 500)",
    )
    parser.add_argument(
        "--no-progress-log",
        action="store_true",
        help="Disable incremental progress logging",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in consolidated/ (safe restart)",
    )
    parser.add_argument(
        "--golden-only",
        action="store_true",
        help="Process only golden_set handwritten_original + handwritten_corrupt",
    )
    parser.add_argument(
        "--force-golden",
        action="store_true",
        help="Re-apply golden post-regen even if consolidated copies exist",
    )
    args = parser.parse_args()

    if args.wait_summary:
        while regeneration_running():
            print("Regeneration still running...", flush=True)
            time.sleep(args.poll_seconds)
        wait_for_regeneration(args.wait_summary, args.poll_seconds)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "post_regen_handwritten_camera_distort.json"
    progress = None
    if not args.no_progress_log:
        progress = ProgressLogger(args.progress_log, args.progress_interval)
        progress.run_start(dry_run=args.dry_run, out_dir=str(args.out_dir), resume=args.resume)

    results: dict = {
        "dry_run": args.dry_run,
        "out_dir": str(args.out_dir),
    }
    if args.golden_only:
        results["golden_handwritten"] = process_golden_handwritten(
            args.out_dir,
            args.dry_run,
            progress,
            resume=args.resume,
            force=args.force_golden,
        )
    else:
        results["mashcima"] = process_mashcima(
            args.out_dir, args.dry_run, progress, resume=args.resume
        )
        results["corrupt_handwritten"] = process_corrupt_handwritten(
            args.out_dir, args.dry_run, progress, resume=args.resume
        )
        results["golden_handwritten"] = process_golden_handwritten(
            args.out_dir,
            args.dry_run,
            progress,
            resume=args.resume,
            force=args.force_golden,
        )
    log_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if progress is not None:
        progress.run_done(results)
        print(f"Progress log: {args.progress_log}", flush=True)

    print(json.dumps(results, indent=2), flush=True)
    print(f"Wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
