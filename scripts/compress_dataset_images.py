"""Compress dataset images under data/ to <= target KiB via resize + JPEG."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from image_compress import compress_image_file, resolved_image_path

from dataset_layout import (  # noqa: E402
    CAMERA_PRIMUS_DIR,
    MASHCIMA_DIR,
    PRIMUS_DIR,
    camera_images_dir,
    mashcima_images_dir,
    primus_images_dir,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _dataset_image_root(base: Path) -> Path:
    images = base / "images"
    return images if images.is_dir() else base


def dataset_image_roots() -> list[Path]:
    roots = []
    for base in (MASHCIMA_DIR, PRIMUS_DIR, CAMERA_PRIMUS_DIR):
        roots.append(_dataset_image_root(base))
    roots.extend(
        [
            DATA / "corrupt" / "clean",
            DATA / "corrupt" / "camera",
            DATA / "corrupt" / "handwritten",
        ]
    )
    return roots

SKIP_DIR_NAMES = {"archives", "mashcima-cache", "__pycache__"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

DEFAULT_RESUME_LOG = DATA / "compress_dataset_resume.jsonl"
DEFAULT_CONVERSIONS_LOG = DATA / "compress_dataset_conversions.jsonl"

MANIFEST_UPDATES = [
    (DATA / "mashcima" / "manifest.jsonl", "output_png"),
    (DATA / "corrupt" / "manifest.jsonl", "png_path"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def is_image_file(name: str) -> bool:
    if name.startswith("._"):
        return False
    return Path(name).suffix.lower() in IMAGE_SUFFIXES


def collect_images_from_base(base: Path) -> list[Path]:
    paths: list[Path] = []
    if not base.is_dir():
        return paths
    for root, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            if not is_image_file(name):
                continue
            paths.append(Path(root) / name)
    return paths


def collect_images(selected: list[Path] | None = None) -> list[Path]:
    bases = selected if selected is not None else IMAGE_DIRS
    paths: list[Path] = []
    for base in bases:
        log(f"Scanning {base.relative_to(ROOT)} ...")
        paths.extend(collect_images_from_base(base))
    return sorted(paths)


def load_resume_done(resume_log: Path) -> set[str]:
    if not resume_log.is_file():
        return set()
    done: set[str] = set()
    with resume_log.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("action") in {"skip", "compressed", "converted"}:
                done.add(row["path"])
    return done


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def dir_bucket(path: Path) -> str:
    rel = path.relative_to(DATA)
    return rel.parts[0] if rel.parts else "data"


def measure_paths(paths: list[Path]) -> dict[str, tuple[int, int]]:
    by_dir: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_dir[dir_bucket(path)].append(path)
    return {name: _sum_paths(ps) for name, ps in sorted(by_dir.items())}


def _sum_paths(paths: list[Path]) -> tuple[int, int]:
    total = 0
    for p in paths:
        if p.is_file():
            total += p.stat().st_size
    return len(paths), total


def _worker(args: tuple[str, int, bool]) -> dict:
    path_str, max_kb, dry_run = args
    path = Path(path_str)
    try:
        before = path.stat().st_size
    except OSError as exc:
        return {
            "path": path_str,
            "action": "error",
            "before": 0,
            "after": 0,
            "new_path": path_str,
            "error": str(exc),
        }

    max_bytes = max_kb * 1024
    if before <= max_bytes:
        return {
            "path": path_str,
            "action": "skip",
            "before": before,
            "after": before,
            "new_path": path_str,
        }

    if dry_run:
        return {
            "path": path_str,
            "action": "would_compress",
            "before": before,
            "after": None,
            "new_path": path_str,
        }

    try:
        compress_image_file(path, max_kb=max_kb)
        new_path = resolved_image_path(path)
        after = new_path.stat().st_size if new_path.is_file() else 0
        converted = new_path.resolve() != path.resolve()
        return {
            "path": path_str,
            "action": "converted" if converted else "compressed",
            "before": before,
            "after": after,
            "new_path": str(new_path.relative_to(ROOT)).replace("\\", "/"),
        }
    except Exception as exc:  # noqa: BLE001 — batch worker reports per-file failures
        return {
            "path": path_str,
            "action": "error",
            "before": before,
            "after": before,
            "new_path": path_str,
            "error": str(exc),
        }


def process_batch(
    work: list[tuple[str, int, bool]],
    workers: int,
    dry_run: bool,
    resume_log: Path,
    conversions_log: Path,
    desc: str,
) -> tuple[dict[str, int], list[tuple[str, str]], list[str]]:
    counts: dict[str, int] = defaultdict(int)
    conversions: list[tuple[str, str]] = []
    errors: list[str] = []

    if not work:
        return counts, conversions, errors

    if workers <= 1:
        iterator = (_worker(item) for item in work)
        for result in tqdm(iterator, total=len(work), desc=desc, unit="file"):
            _record_result(result, dry_run, resume_log, conversions_log, counts, conversions, errors)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, item) for item in work]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=desc,
                unit="file",
            ):
                _record_result(
                    future.result(),
                    dry_run,
                    resume_log,
                    conversions_log,
                    counts,
                    conversions,
                    errors,
                )
    return counts, conversions, errors


def _record_result(
    result: dict,
    dry_run: bool,
    resume_log: Path,
    conversions_log: Path,
    counts: dict[str, int],
    conversions: list[tuple[str, str]],
    errors: list[str],
) -> None:
    counts[result["action"]] += 1
    if result["action"] == "error":
        errors.append(f"{result['path']}: {result.get('error', '')}")
    elif not dry_run and result["action"] in {"skip", "compressed", "converted"}:
        append_jsonl(resume_log, result)
        if result["action"] == "converted":
            old_rel = str(Path(result["path"]).relative_to(ROOT)).replace("\\", "/")
            conversions.append((old_rel, result["new_path"]))
            append_jsonl(
                conversions_log,
                {"old_path": old_rel, "new_path": result["new_path"]},
            )


def update_manifests(conversions: list[tuple[str, str]]) -> None:
    mapping = {old: new for old, new in conversions if old != new}
    if not mapping:
        return

    for manifest_path, field in MANIFEST_UPDATES:
        if not manifest_path.is_file():
            continue
        lines: list[str] = []
        updated = 0
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                val = row.get(field)
                if isinstance(val, str) and val in mapping:
                    row[field] = mapping[val]
                    updated += 1
                lines.append(json.dumps(row))
        if updated:
            manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            log(f"Updated {updated} {field!r} entries in {manifest_path.relative_to(ROOT)}")


def purge_macos_junk(dry_run: bool) -> int:
    count = 0
    primus = DATA / "primus"
    if not primus.is_dir():
        return 0
    log("Purging macOS AppleDouble junk under data/primus ...")
    for root, _dirnames, filenames in os.walk(primus):
        for name in filenames:
            if not name.startswith("._"):
                continue
            count += 1
            if not dry_run:
                (Path(root) / name).unlink(missing_ok=True)
    return count


def print_stats(label: str, stats: dict[str, tuple[int, int]]) -> None:
    total_files = sum(n for n, _ in stats.values())
    total_bytes = sum(b for _, b in stats.values())
    log(f"\n{label}:")
    log(f"  total: {total_files:,} files, {total_bytes / 1e9:.2f} GB")
    for name, (count, nbytes) in stats.items():
        log(f"  {name}: {count:,} files, {nbytes / 1e9:.2f} GB")


def resolve_selected_dirs(names: list[str] | None) -> list[Path]:
    if not names:
        return dataset_image_roots()
    lookup = {p.relative_to(DATA).as_posix(): p for p in dataset_image_roots()}
    lookup.update({p.name: p for p in dataset_image_roots()})
    lookup.update(
        {
            "mashcima": _dataset_image_root(MASHCIMA_DIR),
            "primus": _dataset_image_root(PRIMUS_DIR),
            "camera-primus": _dataset_image_root(CAMERA_PRIMUS_DIR),
        }
    )
    selected: list[Path] = []
    for name in names:
        key = name.replace("\\", "/").strip("/")
        if key not in lookup:
            raise SystemExit(f"Unknown dir selector: {name!r}")
        selected.append(lookup[key])
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-kb", type=int, default=100, help="Target max file size in KiB")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write files")
    parser.add_argument("--resume-log", type=Path, default=DEFAULT_RESUME_LOG)
    parser.add_argument("--conversions-log", type=Path, default=DEFAULT_CONVERSIONS_LOG)
    parser.add_argument(
        "--update-manifests",
        action="store_true",
        help="Rewrite manifest image paths after png->jpg conversions",
    )
    parser.add_argument(
        "--purge-macos-junk",
        action="store_true",
        help="Delete data/primus/**/._* AppleDouble files before compression",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        help="Limit to top-level data dirs (e.g. mashcima corrupt/camera)",
    )
    parser.add_argument("--skip-stats", action="store_true", help="Skip before/after size totals")
    parser.add_argument("--no-resume", action="store_true", help="Ignore resume log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_dirs = resolve_selected_dirs(args.dirs)

    if args.purge_macos_junk:
        removed = purge_macos_junk(args.dry_run)
        verb = "Would delete" if args.dry_run else "Deleted"
        log(f"{verb} {removed:,} macOS junk files under data/primus")

    all_images = collect_images(selected_dirs)
    before_stats = measure_paths(all_images) if not args.skip_stats else {}
    if before_stats:
        print_stats("Before", before_stats)

    done = set() if args.no_resume or args.dry_run else load_resume_done(args.resume_log)
    pending = [p for p in all_images if str(p) not in done]

    log(
        f"\nImages: {len(all_images):,} total, {len(pending):,} pending "
        f"(max {args.max_kb} KiB, workers={args.workers}, dry_run={args.dry_run})"
    )

    total_counts: dict[str, int] = defaultdict(int)
    all_conversions: list[tuple[str, str]] = []
    all_errors: list[str] = []

    for base in selected_dirs:
        if not base.is_dir():
            continue
        batch_images = [p for p in pending if p.is_relative_to(base)]
        if not batch_images:
            continue
        work = [(str(p), args.max_kb, args.dry_run) for p in batch_images]
        counts, conversions, errors = process_batch(
            work,
            args.workers,
            args.dry_run,
            args.resume_log,
            args.conversions_log,
            desc=f"Compress {base.relative_to(ROOT)}",
        )
        for action, n in counts.items():
            total_counts[action] += n
        all_conversions.extend(conversions)
        all_errors.extend(errors)

    log("\nActions:")
    for action, n in sorted(total_counts.items()):
        log(f"  {action}: {n:,}")

    if all_errors:
        log(f"\nErrors ({len(all_errors)}):")
        for line in all_errors[:20]:
            log(f"  {line}")
        if len(all_errors) > 20:
            log(f"  ... and {len(all_errors) - 20} more")

    if args.update_manifests and not args.dry_run:
        update_manifests(all_conversions)

    if not args.dry_run and not args.skip_stats:
        after_images = collect_images(selected_dirs)
        after_stats = measure_paths(after_images)
        print_stats("After", after_stats)
        saved = sum(b for _, b in before_stats.values()) - sum(b for _, b in after_stats.values())
        log(f"\nSaved: {saved / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
