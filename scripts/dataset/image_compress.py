"""Compress dataset images to a target size via resize + JPEG quality search."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
from PIL import Image

MAX_DIMENSIONS = (None, 1600, 1200, 1000, 800, 600, 400, 300)


def _jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _best_jpeg_under_kb(img: Image.Image, max_bytes: int) -> tuple[bytes, int]:
    """Return smallest JPEG at or under max_bytes; binary-search quality, then min quality."""
    lo, hi = 5, 95
    best: bytes | None = None
    best_q = 5

    while lo <= hi:
        mid = (lo + hi) // 2
        data = _jpeg_bytes(img, mid)
        if len(data) <= max_bytes:
            best = data
            best_q = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best is not None:
        return best, best_q

    for q in range(5, 96):
        data = _jpeg_bytes(img, q)
        if len(data) <= max_bytes:
            return data, q

    return _jpeg_bytes(img, 5), 5


def _resize_to_max(img: Image.Image, max_dim: int | None) -> Image.Image:
    if max_dim is None:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def compress_image_file(path: Path | str, max_kb: int = 100) -> None:
    """Resize and JPEG-compress *path* in place to <= *max_kb* KiB.

    PNG inputs are always converted to ``.jpg`` (the PNG is removed), even when
    the PNG is already under *max_kb*, so later camera-distort never runs on
    PNG. Non-PNG files at or under the size limit are left unchanged.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    max_bytes = max_kb * 1024
    suffix = path.suffix.lower()
    if suffix != ".png" and path.stat().st_size <= max_bytes:
        return

    with Image.open(path) as im:
        rgb = im.convert("RGB")

    chosen: bytes | None = None
    for max_dim in MAX_DIMENSIONS:
        resized = _resize_to_max(rgb, max_dim)
        data, _ = _best_jpeg_under_kb(resized, max_bytes)
        if len(data) <= max_bytes:
            chosen = data
            break

    if chosen is None:
        chosen, _ = _best_jpeg_under_kb(_resize_to_max(rgb, 300), max_bytes)

    if suffix == ".png":
        out_path = path.with_suffix(".jpg")
    else:
        out_path = path

    tmp = out_path.with_suffix(out_path.suffix + ".compress_tmp")
    try:
        tmp.write_bytes(chosen)
        tmp.replace(out_path)
        if suffix == ".png" and path.is_file():
            path.unlink()
    finally:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)


def write_jpeg_bgr_under_kb(
    bgr,
    path: Path | str,
    max_kb: int = 100,
) -> Path:
    """Fast path: encode a BGR array to JPEG under *max_kb* without PIL search.

    Used by high-volume Mashcima batch generation. Same size target as
    ``compress_image_file``, fewer quality/dimension probes.
    """
    path = Path(path).with_suffix(".jpg")
    max_bytes = max_kb * 1024
    img = bgr
    h, w = img.shape[:2]
    qualities = (82, 70, 58, 46, 34, 22, 12, 5)
    dims = (None, 1600, 1200, 1000, 800, 600, 400, 300)
    cur = img

    for max_dim in dims:
        cur = img
        if max_dim is not None and max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            cur = cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        for quality in qualities:
            ok, buf = cv2.imencode(
                ".jpg",
                cur,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
            )
            if not ok:
                continue
            data = buf.tobytes()
            if len(data) <= max_bytes:
                tmp = path.with_suffix(path.suffix + ".compress_tmp")
                try:
                    tmp.write_bytes(data)
                    tmp.replace(path)
                finally:
                    if tmp.is_file():
                        tmp.unlink(missing_ok=True)
                return path

    # Last resort: tiny encode
    ok, buf = cv2.imencode(".jpg", cur, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
    if not ok:
        raise RuntimeError(f"JPEG encode failed for {path}")
    tmp = path.with_suffix(path.suffix + ".compress_tmp")
    try:
        tmp.write_bytes(buf.tobytes())
        tmp.replace(path)
    finally:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
    return path

def resolved_image_path(path: Path | str) -> Path:
    """Path after compression (``.png`` may have become ``.jpg``)."""
    path = Path(path)
    if path.is_file():
        return path
    jpg = path.with_suffix(".jpg")
    if jpg.is_file():
        return jpg
    return path
