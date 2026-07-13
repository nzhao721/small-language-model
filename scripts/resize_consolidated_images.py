"""Downscale images under data/consolidated/images/ in place to <= --max-dim (default 768px).

Mirrors the resize logic in build_consolidated_zip.py's _resize_image (long side
capped at 768 to match Florence-2's native input resolution, LANCZOS, never
upscaled), but writes the result back to disk instead of into a zip -- so the
consolidated store itself stays small, not just the zip built from it.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _resize_in_place(path_str: str, max_dim: int) -> tuple[str, bool, str | None]:
    path = Path(path_str)
    try:
        with Image.open(path) as im:
            im.load()
            if max(im.width, im.height) <= max_dim:
                return path_str, False, None
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            scale = max_dim / max(im.width, im.height)
            new_size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
            tmp = path.with_suffix(path.suffix + ".resize_tmp")
            if path.suffix.lower() in (".jpg", ".jpeg"):
                im.save(tmp, format="JPEG", quality=90)
            else:
                im.save(tmp, format="PNG", optimize=True)
            tmp.replace(path)
        return path_str, True, None
    except Exception as exc:  # noqa: BLE001 -- batch worker reports per-file failures
        return path_str, False, str(exc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, default=ROOT / "data" / "consolidated" / "images")
    ap.add_argument("--max-dim", type=int, default=768)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = sorted(
        p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    print(f"Scanning {len(targets):,} images in {args.images_dir} (max-dim={args.max_dim})")
    if args.dry_run:
        return

    resized = 0
    errors: list[str] = []
    work = [(str(p), args.max_dim) for p in targets]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_resize_in_place, *item) for item in work]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Resize", unit="file"):
            path_str, did_resize, err = future.result()
            if err:
                errors.append(f"{path_str}: {err}")
            elif did_resize:
                resized += 1

    print(f"Done. {resized:,} resized, {len(targets) - resized - len(errors):,} already <= {args.max_dim}px, {len(errors)} errors.")
    if errors:
        for line in errors[:20]:
            print(f"  {line}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
