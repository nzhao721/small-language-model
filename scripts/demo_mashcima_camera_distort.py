"""Apply camera-like distortions to Mashcima PNGs (demo samples or full batch)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from image_compress import compress_image_file

from dataset_layout import MASHCIMA_DIR, mashcima_image_path, mashcima_images_dir  # noqa: E402

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "examples" / "mashcima_camera_demo"
DEFAULT_MANIFEST = MASHCIMA_DIR / "camera_distort_manifest.jsonl"
DEFAULT_INPUT_DIR = mashcima_images_dir()
SAMPLES = [
    "000051650-1_1_1.png",
    "000051662-1_1_1.png",
    "000051756-1_1_1.png",
]


def apply_glare(arr: np.ndarray, rng: random.Random, seed: int) -> np.ndarray:
    """Wash out ink toward gray where glare hits the page."""
    h, w = arr.shape[:2]
    out = arr.astype(np.float32)

    # Soften pure-black ink globally (pen + paper never perfect in photos).
    luminance = out.mean(axis=2)
    ink_mask = np.clip((120 - luminance) / 120, 0, 1)
    ink_lift = rng.uniform(18, 45)
    out += ink_mask[..., None] * ink_lift

    # One or two soft glare hotspots (phone flash / window reflection).
    glare = np.zeros((h, w), dtype=np.float32)
    hotspot_count = rng.randint(1, 3)
    for _ in range(hotspot_count):
        cx = rng.uniform(0.15 * w, 0.85 * w)
        cy = rng.uniform(0.2 * h, 0.8 * h)
        rx = rng.uniform(0.12 * w, 0.35 * w)
        ry = rng.uniform(0.25 * h, 0.9 * h)
        yy, xx = np.mgrid[0:h, 0:w]
        spot = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        glare = np.maximum(glare, spot * rng.uniform(0.35, 0.75))

    # Glare pushes pixels toward white; darkest strokes fade most.
    wash_strength = rng.uniform(0.45, 0.8)
    out += glare[..., None] * (255.0 - out) * wash_strength

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_random_shading(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Uneven page lighting via large-scale soft gradients (not a radial vignette)."""
    h, w = arr.shape[:2]
    shade = np.ones((h, w), dtype=np.float32)

    # Very coarse ambient field — a handful of control points, heavily smoothed.
    nh, nw = rng.randint(2, 4), rng.randint(3, 5)
    field = np.array(
        [[rng.uniform(-1.0, 1.0) for _ in range(nw)] for _ in range(nh)],
        dtype=np.float32,
    )
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    # Extra blur so transitions stay soft across the whole staff.
    blur_sigma = max(h, w) * rng.uniform(0.04, 0.09)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    shade += field * rng.uniform(0.16, 0.30)

    # One or two large soft lighting lobes (window / desk lamp / hand shadow).
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(rng.randint(1, 2)):
        cx = rng.uniform(0.1 * w, 0.9 * w)
        cy = rng.uniform(0.0 * h, 1.0 * h)
        # Spans a large fraction of the page.
        rx = rng.uniform(0.55 * w, 1.3 * w)
        ry = rng.uniform(0.7 * h, 1.6 * h)
        spot = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        amp = rng.uniform(-0.28, 0.20)
        shade += spot * amp

    # Optional gentle linear gradient (page tilted under a light).
    if rng.random() < 0.85:
        angle = rng.uniform(0.0, 2.0 * np.pi)
        nx, ny = np.cos(angle), np.sin(angle)
        # Normalize coords to [-1, 1].
        gx = (xx / max(w - 1, 1) - 0.5) * 2.0
        gy = (yy / max(h - 1, 1) - 0.5) * 2.0
        ramp = gx * nx + gy * ny
        shade += ramp * rng.uniform(0.08, 0.18)

    shade = np.clip(shade, 0.62, 1.25)
    out = arr.astype(np.float32) * shade[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def camera_like_distort(img: Image.Image, seed: int = 42) -> Image.Image:
    """Approximate phone/camera capture artifacts on a clean notation image."""
    rng = random.Random(seed)
    rgb = img.convert("RGB")
    # Perspective transforms operate inside a fixed output canvas. Add a white
    # safety border first so edge notes are not cropped by the simulated camera
    # angle. This is especially important for long, tightly engraved systems.
    pad_x = max(16, round(rgb.width * 0.03))
    pad_y = max(12, round(rgb.height * 0.08))
    rgb = ImageOps.expand(rgb, border=(pad_x, pad_y), fill=(255, 255, 255))
    arr = np.array(rgb)

    # Slight perspective / affine warp
    h, w = arr.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dx = rng.uniform(-0.02, 0.02) * w
    dy = rng.uniform(-0.02, 0.02) * h
    dst = np.float32(
        [
            [dx, dy],
            [w - 1 - dx * 0.5, dy * 0.5],
            [w - 1 - dx, h - 1 - dy],
            [dx * 0.5, h - 1 - dy * 0.5],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    arr = cv2.warpPerspective(
        arr,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 240),
    )

    out = Image.fromarray(arr)

    # Blur + sharpen mix (camera softness)
    if rng.random() < 0.8:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.2)))

    # Brightness / contrast jitter
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.85, 1.08))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.9, 1.15))

    # JPEG compression artifacts
    buf = np.array(out)
    quality = rng.randint(42, 72)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(buf, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        out = Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))

    # Sensor noise
    h, w = out.size[1], out.size[0]
    noise = np.random.default_rng(seed).normal(0, rng.uniform(4, 10), (h, w, 3))
    noisy = np.clip(np.array(out, dtype=np.float32) + noise, 0, 255).astype(np.uint8)

    # Irregular lighting instead of a strong radial vignette.
    shaded = apply_random_shading(noisy, rng)
    return Image.fromarray(apply_glare(shaded, rng, seed))


def seed_for_name(name: str) -> int:
    return abs(hash(name)) % 10_000


def load_done_ids(manifest: Path) -> set[str]:
    if not manifest.is_file():
        return set()
    done: set[str] = set()
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                done.add(row["incipit_id"])
    return done


def append_manifest(manifest: Path, incipit_id: str, status: str, error: str | None = None) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, str] = {"incipit_id": incipit_id, "status": status}
    if error:
        row["error"] = error
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def collect_mashcima_images(input_dir: Path) -> list[Path]:
    """PNG and JPEG mashcima outputs (post-compression may be .jpg only)."""
    search_dir = mashcima_images_dir(input_dir) if input_dir == MASHCIMA_DIR else input_dir
    if not search_dir.is_dir():
        search_dir = input_dir
    by_stem: dict[str, Path] = {}
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for path in search_dir.glob(pattern):
            if path.name.startswith("._"):
                continue
            prev = by_stem.get(path.stem)
            if prev is None or path.suffix.lower() in {".jpg", ".jpeg"}:
                by_stem[path.stem] = path
    return sorted(by_stem.values())


def distort_image_in_place(path: Path) -> str:
    """Return incipit_id on success."""
    incipit_id = path.stem
    original = Image.open(path).convert("RGB")
    distorted = camera_like_distort(original, seed=seed_for_name(incipit_id))
    tmp = path.with_suffix(".camera_tmp" + path.suffix)
    try:
        distorted.save(tmp)
        tmp.replace(path)
        compress_image_file(path)
    finally:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
    return incipit_id


def distort_png_in_place(path: Path) -> str:
    return distort_image_in_place(path)


def _worker(path_str: str) -> tuple[str, str, str | None]:
    path = Path(path_str)
    try:
        return distort_image_in_place(path), "ok", None
    except Exception as exc:  # noqa: BLE001 — batch worker reports per-file failures
        return path.stem, "error", str(exc)


def run_demo() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in SAMPLES:
        incipit_id = Path(name).stem
        src = mashcima_image_path(incipit_id)
        if not src.is_file():
            src = mashcima_images_dir() / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing Mashcima sample: {src}")

        original = Image.open(src).convert("RGB")
        distorted = camera_like_distort(original, seed=seed_for_name(name))

        original.save(OUT_DIR / name.replace(".png", "_original.png"))
        distorted.save(OUT_DIR / name.replace(".png", "_camera.png"))

    print(f"Wrote comparisons to {OUT_DIR}")


def run_batch(
    input_dir: Path,
    manifest: Path,
    workers: int,
    limit: int | None,
) -> None:
    images = collect_mashcima_images(input_dir)
    if limit is not None:
        images = images[:limit]

    done = load_done_ids(manifest)
    pending = [p for p in images if p.stem not in done]
    if not pending:
        print(f"All {len(images)} Mashcima images already camera-distorted.")
        return

    ok_count = 0
    err_count = 0

    if workers <= 1:
        for path in tqdm(pending, desc="Camera-distort Mashcima", unit="file"):
            incipit_id, status, error = _worker(str(path))
            append_manifest(manifest, incipit_id, status, error)
            if status == "ok":
                ok_count += 1
            else:
                err_count += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, str(p)): p for p in pending}
            for future in tqdm(as_completed(futures), total=len(pending), desc="Camera-distort Mashcima", unit="file"):
                incipit_id, status, error = future.result()
                append_manifest(manifest, incipit_id, status, error)
                if status == "ok":
                    ok_count += 1
                else:
                    err_count += 1

    skipped = len(images) - len(pending)
    print(
        f"Done: {ok_count} distorted, {skipped} skipped (already done), "
        f"{err_count} errors — manifest {manifest}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Overwrite all Mashcima PNGs in-place with camera distortions.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory of Mashcima images (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Resume manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel workers for --all (default: CPU count - 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N PNGs (for testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        run_batch(args.input_dir, args.manifest, args.workers, args.limit)
    else:
        run_demo()


if __name__ == "__main__":
    main()
