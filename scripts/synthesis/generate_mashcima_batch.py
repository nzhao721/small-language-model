"""Batch-generate Mashcima handwritten PNGs for all convertible PrIMuS incipits."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

_SCRIPTS = Path(__file__).resolve()
while _SCRIPTS.name != "scripts":
    _SCRIPTS = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from _bootstrap import setup  # noqa: E402

REPO_ROOT = setup(__file__)

from image_compress import write_jpeg_bgr_under_kb

import cv2
import mashcima as mc
import numpy as np
from mashcima.Config import Config, download_file
from mashcima.primus_adapter import (
    _iterate_tgz_primus,
    convert_primus_annotation_to_mashcima_annotation,
)
from mashcima.vocabulary import parse_annotation_into_token_groups
from PIL import Image
from tqdm import tqdm

from demo_mashcima_camera_distort import camera_like_distort, seed_for_name  # noqa: E402
from mashcima_appearance import apply_mashcima_appearance  # noqa: E402
from mashcima_writers import (  # noqa: E402
    DEFAULT_WRITER_MODE,
    WRITER_MODES,
    configure_mashcima_writers,
    symbol_repository_for_key,
    writer_number_for_key,
)

from dataset_layout import (  # noqa: E402
    MASHCIMA_DIR,
    PRIMUS_DIR,
    mashcima_ground_truth_dir,
    mashcima_images_dir,
    primus_agnostic_dir,
    primus_agnostic_path,
    primus_semantic_path,
    read_primus_agnostic,
)

ROOT = REPO_ROOT
DEFAULT_PRIMUS_DIR = PRIMUS_DIR
DEFAULT_PRIMUS_TGZ = ROOT / "data" / "archives" / "primusCalvoRizoAppliedSciences2018.tgz"
DEFAULT_OUTPUT_DIR = MASHCIMA_DIR
DEFAULT_MANIFEST = MASHCIMA_DIR / "manifest.jsonl"
DEFAULT_PROGRESS_LOG = ROOT / "data" / "logs" / "mashcima_regen_progress.txt"
DEFAULT_MUSCIMA_DIR = ROOT / "data" / "mashcima-cache" / "MUSCIMA-pp_v1.0"
MUSCIMA_ZIP_URL = (
    "https://github.com/OMR-Research/muscima-pp/releases/download/v1.0/MUSCIMA-pp_v1.0.zip"
)


def download_muscima_pp(muscima_dir: Path) -> None:
    """Download and extract MUSCIMA++ v1.0 if missing."""
    if muscima_dir.is_dir():
        return

    cache_dir = muscima_dir.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "MUSCIMA-pp_v1.0.zip"

    if not zip_path.is_file() or zip_path.stat().st_size < 1_000_000:
        if zip_path.is_file():
            zip_path.unlink()
        download_file(MUSCIMA_ZIP_URL, str(zip_path))

    extract_root = cache_dir / "_extract"
    if extract_root.is_dir():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    extracted = extract_root / "v1.0"
    if not extracted.is_dir():
        raise FileNotFoundError(f"Expected v1.0 folder in {zip_path}")

    shutil.move(str(extracted), str(muscima_dir))
    shutil.rmtree(extract_root)
    zip_path.unlink(missing_ok=True)


def ensure_muscima_cache(primus_tgz: Path, muscima_dir: Path) -> Config:
    """Load Mashcima config after ensuring MUSCIMA++ is available."""
    download_muscima_pp(muscima_dir)

    cfg = Config.manual_download_config(
        muscima_pp_path=str(muscima_dir),
        primus_path=str(primus_tgz),
    )
    Config.DEFAULT_CONFIG = cfg
    return cfg


def image_to_uint8(img: np.ndarray) -> np.ndarray:
    """Mashcima renders float images in [0, 1]; save as black-on-white uint8 PNGs."""
    if img.dtype == np.uint8:
        arr = img.astype(np.uint8)
    else:
        arr = img.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return 255 - arr


def resize_long_edge(img: np.ndarray, max_dim: int | None) -> np.ndarray:
    """Downscale an image so its long edge is at most *max_dim* pixels."""
    if not max_dim or max_dim <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    return cv2.resize(
        img,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def incipit_id_from_path(path: str) -> str:
    """Extract incipit folder name from a PrIMuS archive member path."""
    parts = Path(path.replace("\\", "/")).parts
    for part in reversed(parts):
        if part.endswith(".agnostic"):
            return part[: -len(".agnostic")]
        if re.fullmatch(r"\d{9}(-\d+_\d+_\d+)?", part):
            return part
    raise ValueError(f"Could not parse incipit id from path: {path}")


def iter_primus_annotations(primus_dir: Path, primus_tgz: Path):
    """Yield (path, agnostic annotation) from purged on-disk PrIMuS, else the archive."""
    if primus_dir.is_dir():
        agnostic_dir = primus_agnostic_dir(primus_dir)
        if agnostic_dir.is_dir():
            agnostic_files = list(agnostic_dir.glob("*.agnostic"))
        else:
            agnostic_files = list(primus_dir.rglob("*.agnostic"))
        if agnostic_files:
            for agnostic in agnostic_files:
                incipit_id = agnostic.stem
                rel = primus_agnostic_path(incipit_id, primus_dir).relative_to(ROOT).as_posix()
                yield rel, agnostic.read_text(encoding="utf-8")
            return

    if not primus_tgz.is_file():
        raise FileNotFoundError(f"PrIMuS not found: {primus_dir} or {primus_tgz}")

    for path, text in _iterate_tgz_primus(str(primus_tgz)):
        yield path, text


def iter_primus_ids(primus_dir: Path, primus_tgz: Path):
    """Yield incipit IDs without reading every agnostic file into the parent."""
    if primus_dir.is_dir():
        agnostic_dir = primus_agnostic_dir(primus_dir)
        if agnostic_dir.is_dir():
            agnostic_files = agnostic_dir.glob("*.agnostic")
        else:
            agnostic_files = primus_dir.rglob("*.agnostic")
        found = False
        for agnostic in agnostic_files:
            found = True
            yield agnostic.stem
        if found:
            return

    if not primus_tgz.is_file():
        raise FileNotFoundError(f"PrIMuS not found: {primus_dir} or {primus_tgz}")
    for path, _ in _iterate_tgz_primus(str(primus_tgz)):
        yield incipit_id_from_path(path)


def synthesize_handwritten(
    annotation: str,
    writer_key: str,
    min_width: int | None = 768,
) -> np.ndarray:
    repo = symbol_repository_for_key(writer_key)
    # This is synthesize_for_training with a smaller initial canvas. The old
    # helper always padded every incipit to at least 1200px, only for the image
    # to be reduced to 768px later.
    return mc.synthesize(
        annotation,
        min_width=min_width or 1200,
        transform_image=True,
        symbol_repository=repo,
    )


def count_primus_annotations(primus_dir: Path, primus_tgz: Path) -> int:
    if primus_dir.is_dir():
        agnostic_dir = primus_agnostic_dir(primus_dir)
        if agnostic_dir.is_dir():
            return sum(1 for _ in agnostic_dir.glob("*.agnostic"))
        return sum(1 for _ in primus_dir.rglob("*.agnostic"))
    return sum(1 for _ in iter_primus_annotations(primus_dir, primus_tgz))


def write_mashcima_progress(
    progress_log: Path,
    counts: dict[str, int],
    total: int,
    *,
    running: bool = True,
) -> None:
    processed = sum(counts.values())
    pct = (100.0 * processed / total) if total else 0.0
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = [
        f"[{ts}] mashcima regen progress",
        f"  processed: {processed}/{total} ({pct:.1f}%)",
        f"  ok={counts['ok']} skipped={counts['skipped']} "
        f"invalid={counts['invalid']} error={counts['error']}",
        f"  status={'running' if running else 'done'}",
    ]
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    progress_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Machine-readable sibling for quick numeric reads.
    json_path = progress_log.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "ts": ts,
                "processed": processed,
                "total": total,
                "pct": round(pct, 2),
                "ok": counts["ok"],
                "skipped": counts["skipped"],
                "invalid": counts["invalid"],
                "error": counts["error"],
                "status": "running" if running else "done",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def copy_primus_semantic(incipit_id: str, primus_dir: Path, output_dir: Path) -> Path:
    src = primus_semantic_path(incipit_id, primus_dir)
    dest = mashcima_ground_truth_dir(output_dir) / f"{incipit_id}.semantic"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dest)
    return dest


_pool_ready = False


def _init_mashcima_pool(primus_tgz_s: str, muscima_dir_s: str, writer_mode_s: str) -> None:
    global _pool_ready
    if _pool_ready:
        return
    ensure_muscima_cache(Path(primus_tgz_s), Path(muscima_dir_s))
    configure_mashcima_writers(writer_mode_s)
    # Each process handles independent images. Letting every OpenCV instance
    # create its own thread team badly oversubscribes an 8-thread machine.
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    _pool_ready = True


def process_one_incipit(
    *,
    incipit_id: str,
    primus_dir: Path,
    output_dir: Path,
    writer_mode: str,
    apply_camera: bool,
    max_dim: int | None = 768,
    primus_tgz: Path | None = None,
    primus_annotation: str | None = None,
) -> dict:
    """Render one Mashcima image (appearance + optional camera)."""
    images_dir = mashcima_images_dir(output_dir)
    out_base = images_dir / f"{incipit_id}.png"
    record: dict = {
        "incipit_id": incipit_id,
        "writer_mode": writer_mode,
        "camera": bool(apply_camera),
        "max_dim": max_dim,
    }

    try:
        if primus_annotation is None:
            primus_annotation = read_primus_agnostic(
                incipit_id, primus_dir, primus_tgz or DEFAULT_PRIMUS_TGZ
            )
        record["primus_path"] = str(
            primus_agnostic_path(incipit_id, primus_dir).relative_to(ROOT).as_posix()
        )
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["reason"] = f"read_agnostic: {exc}"
        return record

    converted = convert_primus_annotation_to_mashcima_annotation(primus_annotation)
    if converted is None:
        record["status"] = "skipped"
        record["reason"] = "conversion_returned_none"
        return record

    _, warnings = parse_annotation_into_token_groups(converted)
    if warnings:
        record["status"] = "invalid"
        record["reason"] = "annotation_warnings"
        record["warnings"] = warnings
        return record

    try:
        img = image_to_uint8(synthesize_handwritten(converted, incipit_id, max_dim))
        # All remaining effects are pixel-heavy. Downscale immediately after
        # synthesis instead of processing full-resolution pixels that the
        # dataset's later resize pass would discard.
        img = resize_long_edge(img, max_dim)
        img = apply_mashcima_appearance(img, key=incipit_id)
        if apply_camera:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil = camera_like_distort(pil, seed=seed_for_name(incipit_id))
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            # Camera padding can grow the canvas by about 6%; keep the saved
            # dataset image capped at the requested final resolution.
            img = resize_long_edge(img, max_dim)
        for stale in (out_base, out_base.with_suffix(".jpg"), out_base.with_suffix(".jpeg")):
            stale.unlink(missing_ok=True)
        out_png = write_jpeg_bgr_under_kb(img, out_base)
        gt_path = copy_primus_semantic(incipit_id, primus_dir, output_dir)
    except Exception as exc:  # noqa: BLE001 - log and continue batch
        record["status"] = "error"
        record["reason"] = str(exc)
        return record

    record["status"] = "ok"
    record["output_png"] = str(out_png.relative_to(ROOT)).replace("\\", "/")
    record["semantic_path"] = str(gt_path.relative_to(ROOT).as_posix())
    return record


def _mashcima_worker(
    payload: tuple[str, str, str, str, bool, int | None, str],
) -> dict:
    (
        incipit_id,
        primus_dir_s,
        output_dir_s,
        writer_mode,
        apply_camera,
        max_dim,
        primus_tgz_s,
    ) = payload
    return process_one_incipit(
        incipit_id=incipit_id,
        primus_dir=Path(primus_dir_s),
        output_dir=Path(output_dir_s),
        writer_mode=writer_mode,
        apply_camera=apply_camera,
        max_dim=max_dim,
        primus_tgz=Path(primus_tgz_s),
    )


def generate_batch(
    primus_dir: Path,
    primus_tgz: Path,
    output_dir: Path,
    manifest_path: Path,
    resume: bool = True,
    limit: int | None = None,
    writer_mode: str = DEFAULT_WRITER_MODE,
    progress_log: Path | None = DEFAULT_PROGRESS_LOG,
    apply_camera: bool = True,
    max_dim: int | None = 768,
    workers: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = mashcima_images_dir(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_muscima_cache(primus_tgz, DEFAULT_MUSCIMA_DIR)
    configure_mashcima_writers(writer_mode)

    done_ids: set[str] = set()
    if resume and manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("status") == "ok":
                    done_ids.add(record["incipit_id"])
    elif not resume and manifest_path.is_file():
        # Fresh full regen: truncate so progress counts stay readable.
        manifest_path.write_text("", encoding="utf-8")

    counts = {"ok": 0, "skipped": 0, "invalid": 0, "error": 0}
    total_available = count_primus_annotations(primus_dir, primus_tgz)

    # IDs only — workers read agnostic from disk (avoids pickling 36k annotations).
    work_ids: list[str] = []
    for incipit_id in iter_primus_ids(primus_dir, primus_tgz):
        if resume and incipit_id in done_ids:
            continue
        work_ids.append(incipit_id)

    # A writer repository is expensive to load. Grouping the deterministic
    # per-score assignments lets each worker reuse one repository for a long
    # run instead of thrashing its small LRU cache on nearly every incipit.
    if writer_mode == "per-score":
        work_ids.sort(key=lambda incipit_id: (writer_number_for_key(incipit_id), incipit_id))
    if limit is not None:
        work_ids = work_ids[:limit]

    if not resume and limit is None:
        total = total_available
    else:
        total = len(work_ids) + (len(done_ids) if resume else 0)

    progress_every = 25
    if progress_log is not None:
        write_mashcima_progress(progress_log, counts, total, running=True)

    primus_dir_s = str(primus_dir.resolve())
    output_dir_s = str(output_dir.resolve())
    primus_tgz_s = str(primus_tgz.resolve())
    payloads = [
        (
            incipit_id,
            primus_dir_s,
            output_dir_s,
            writer_mode,
            apply_camera,
            max_dim,
            primus_tgz_s,
        )
        for incipit_id in work_ids
    ]

    def _consume(record: dict, manifest) -> None:
        status = record.get("status", "error")
        if status in counts:
            counts[status] += 1
        else:
            counts["error"] += 1
        manifest.write(json.dumps(record) + "\n")
        if status == "ok":
            manifest.flush()
        if progress_log and sum(counts.values()) % progress_every == 0:
            write_mashcima_progress(progress_log, counts, total)

    with manifest_path.open("a", encoding="utf-8") as manifest:
        if workers <= 1:
            for payload in tqdm(payloads, desc="Mashcima batch", unit="incipit"):
                record = _mashcima_worker(payload)
                _consume(record, manifest)
        else:
            initargs = (
                primus_tgz_s,
                str(DEFAULT_MUSCIMA_DIR.resolve()),
                writer_mode,
            )
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_mashcima_pool,
                initargs=initargs,
                # Recycle occasionally to return native-image allocations to
                # Windows, but not so often that repositories are reloaded.
                max_tasks_per_child=200,
            ) as pool:
                in_flight: dict = {}
                payload_iter = iter(payloads)
                max_in_flight = max(workers * 2, workers)

                def _submit_more() -> None:
                    while len(in_flight) < max_in_flight:
                        try:
                            payload = next(payload_iter)
                        except StopIteration:
                            return
                        fut = pool.submit(_mashcima_worker, payload)
                        in_flight[fut] = True

                _submit_more()
                pbar = tqdm(
                    total=len(payloads),
                    desc=f"Mashcima batch ({workers} workers)",
                    unit="incipit",
                )
                while in_flight:
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        in_flight.pop(future, None)
                        record = future.result()
                        _consume(record, manifest)
                        pbar.update(1)
                        _submit_more()
                pbar.close()

    if progress_log is not None:
        write_mashcima_progress(progress_log, counts, total, running=False)

    print(
        "Finished: "
        f"ok={counts['ok']} skipped={counts['skipped']} "
        f"invalid={counts['invalid']} error={counts['error']}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primus-dir",
        type=Path,
        default=DEFAULT_PRIMUS_DIR,
        help="Path to extracted purged PrIMuS directory",
    )
    parser.add_argument(
        "--primus-tgz",
        type=Path,
        default=DEFAULT_PRIMUS_TGZ,
        help="Fallback PrIMuS archive if extracted directory is missing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated handwritten PNGs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="JSONL manifest of per-incipit results",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Regenerate even if manifest already records success",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of successful generations (for testing)",
    )
    parser.add_argument(
        "--writers",
        choices=WRITER_MODES,
        default=DEFAULT_WRITER_MODE,
        help=(
            "MUSCIMA++ writer selection: per-score (one of 50 writers per incipit, "
            "default), mixed (all writers' glyphs in one pool), or single (writer 01)"
        ),
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=DEFAULT_PROGRESS_LOG,
        help="Writable text/JSON progress file updated during the batch",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Skip camera_like_distort (appearance only)",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=768,
        help=(
            "Cap the final image long edge and run appearance/camera near that "
            "resolution (default: 768; use 0 to disable)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel worker processes (default: CPU count - 1)",
    )
    args = parser.parse_args()

    generate_batch(
        primus_dir=args.primus_dir,
        primus_tgz=args.primus_tgz,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        resume=not args.no_resume,
        limit=args.limit,
        writer_mode=args.writers,
        progress_log=args.progress_log,
        apply_camera=not args.no_camera,
        max_dim=args.max_dim or None,
        workers=max(1, args.workers),
    )


if __name__ == "__main__":
    main()
