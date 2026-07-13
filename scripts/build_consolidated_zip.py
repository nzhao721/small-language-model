"""Bundle data/consolidated/ into a single zip for Google Drive upload.

The Drive web uploader crashes on the ~273k loose files in data/consolidated/
(136k images + 136k .semantic). A single zip uploads far more reliably, and the
Colab notebooks (train_florence2_base_colab.ipynb / smoke_test_florence2_colab.ipynb,
cell 1b) extract it to local SSD -- never onto the slow Drive mount.

Images are downscaled (long side capped at --max-dim, default 768 to match
Florence-2's native input resolution) before being added to the zip, so the
archive -- and the upload -- is smaller. Images already <= max-dim are left
untouched (never upscaled). Ground-truth .semantic files are copied as-is.

Store mode (no compression) is used on purpose: the re-encoded JPEG/PNG bytes
are already compressed, so deflate would waste CPU for ~no size gain. The
archive lays out paths as `consolidated/images/...` and
`consolidated/ground_truth/...` so extractall to the notebook's
LOCAL_DATA_ROOT (/content/omr_data) yields LOCAL_DATA_ROOT/consolidated/.

Usage:
    python scripts/build_consolidated_zip.py
    python scripts/build_consolidated_zip.py --src data/consolidated --out consolidated.zip
    python scripts/build_consolidated_zip.py --max-dim 768 --workers 8
Then upload the resulting zip to MyDrive/omr/consolidated.zip.
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

SUBDIRS = ("images", "ground_truth")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_MAX_DIM = 768


def _resize_image(path_str: str, max_dim: int) -> bytes:
    path = Path(path_str)
    with Image.open(path) as im:
        im.load()
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        scale = max_dim / max(im.width, im.height)
        if scale < 1.0:
            new_size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        if path.suffix.lower() in (".jpg", ".jpeg"):
            im.save(buf, format="JPEG", quality=90)
        else:
            im.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def build_zip(src: Path, out: Path, max_dim: int, workers: int | None) -> None:
    if not src.is_dir():
        raise SystemExit(f"Source dir not found: {src}")

    image_files: list[tuple[Path, str]] = []
    gt_files: list[tuple[Path, str]] = []
    for sub in SUBDIRS:
        d = src / sub
        if not d.is_dir():
            raise SystemExit(f"Expected subdir missing: {d}")
        bucket = image_files if sub == "images" else gt_files
        for fp in d.iterdir():
            if fp.is_file():
                bucket.append((fp, f"consolidated/{sub}/{fp.name}"))

    total = len(image_files) + len(gt_files)
    print(
        f"Zipping {total:,} files ({len(image_files):,} images downscaled to <= {max_dim}px, "
        f"{len(gt_files):,} ground-truth copied as-is) from {src} -> {out}"
    )
    t0 = time.time()
    written = 0
    # allowZip64 is required: >4 GB payload and >65k entries.
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            paths = [str(fp) for fp, _ in image_files]
            results = pool.map(_resize_image, paths, [max_dim] * len(paths), chunksize=64)
            for (fp, arc), data in zip(image_files, results):
                zf.writestr(arc, data)
                written += 1
                if written % 20000 == 0 or written == total:
                    print(f"  {written:,}/{total:,}  ({time.time() - t0:.0f}s)", flush=True)
        for fp, arc in gt_files:
            zf.write(fp, arc)
            written += 1
            if written % 20000 == 0 or written == total:
                print(f"  {written:,}/{total:,}  ({time.time() - t0:.0f}s)", flush=True)

    size_gb = out.stat().st_size / 1e9
    print(f"DONE: {out}  {size_gb:.2f} GB  in {time.time() - t0:.0f}s")
    print("Next: upload to MyDrive/omr/consolidated.zip, then run notebook cell 1b.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/consolidated"))
    ap.add_argument("--out", type=Path, default=Path("consolidated.zip"))
    ap.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_MAX_DIM,
        help="Max image dimension in pixels; larger images are downscaled (never upscaled).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel resize workers (default: os.cpu_count())",
    )
    args = ap.parse_args()
    build_zip(args.src, args.out, args.max_dim, args.workers)


if __name__ == "__main__":
    main()
