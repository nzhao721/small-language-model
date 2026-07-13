"""Compress newly-added, not-yet-compressed images directly under data/consolidated/images.

Unlike compress_dataset_images.py (which walks the per-source dataset dirs), this
targets data/consolidated/images/ directly -- for images that were added straight
into the consolidated folder without going through the source-dataset compression
pass. Uses the exact same compress_image_file() convention as the rest of the
pipeline (PNG always converted to <=max-kb JPEG; other formats left alone if
already under the size cap), so the consolidated store stays consistent before
zipping with build_consolidated_zip.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from image_compress import compress_image_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def collect_targets(images_dir: Path, max_kb: int) -> list[Path]:
    max_bytes = max_kb * 1024
    targets: list[Path] = []
    for p in images_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.suffix.lower() == ".png" or p.stat().st_size > max_bytes:
            targets.append(p)
    return sorted(targets)


def _worker(args: tuple[str, int]) -> tuple[str, str | None]:
    path_str, max_kb = args
    path = Path(path_str)
    try:
        compress_image_file(path, max_kb=max_kb)
        return path_str, None
    except Exception as exc:  # noqa: BLE001 -- batch worker reports per-file failures
        return path_str, str(exc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, default=ROOT / "data" / "consolidated" / "images")
    ap.add_argument("--max-kb", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = collect_targets(args.images_dir, args.max_kb)
    print(f"Found {len(targets):,} images to compress in {args.images_dir}")
    if args.dry_run or not targets:
        return

    errors: list[str] = []
    work = [(str(p), args.max_kb) for p in targets]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, item) for item in work]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Compress", unit="file"):
            path_str, err = future.result()
            if err:
                errors.append(f"{path_str}: {err}")

    print(f"Done. {len(targets) - len(errors):,} compressed, {len(errors)} errors.")
    if errors:
        for line in errors[:20]:
            print(f"  {line}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    remaining_png = sum(1 for p in args.images_dir.iterdir() if p.suffix.lower() == ".png")
    print(f"Remaining .png files: {remaining_png:,}")


if __name__ == "__main__":
    main()
