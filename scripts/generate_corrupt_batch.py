"""Batch-generate corrupted GT training images (clean, camera, handwritten)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from demo_corrupted_gt_examples import (
    corrupt_tokens,
    mashcima_annotation_from_semantic,
    prepare_mashcima_for_render,
    render_mashcima,
    save_png,
)
from mashcima.primus_adapter import convert_primus_annotation_to_mashcima_annotation
from demo_mashcima_camera_distort import camera_like_distort
from image_compress import compress_image_file, resolved_image_path, write_jpeg_bgr_under_kb
from primus_semantic_render import close_render_pool, render_semantic_to_image
from generate_mashcima_batch import (
    DEFAULT_MUSCIMA_DIR,
    DEFAULT_PRIMUS_DIR,
    DEFAULT_PRIMUS_TGZ,
    ensure_muscima_cache,
    resize_long_edge,
)
from mashcima_writers import (
    DEFAULT_WRITER_MODE,
    WRITER_MODES,
    configure_mashcima_writers,
    writer_number_for_key,
)
from dataset_layout import (
    read_primus_agnostic,
    read_primus_semantic,
    primus_semantic_path,
    primus_agnostic_path,
)

ROOT = Path(__file__).resolve().parents[1]
CORRUPT_DIR = ROOT / "data" / "corrupt"
DEFAULT_MANIFEST = CORRUPT_DIR / "manifest.jsonl"
DEFAULT_JOBS = CORRUPT_DIR / "jobs.jsonl"
DEFERRED_JOBS = CORRUPT_DIR / "deferred_jobs.jsonl"
MASHCIMA_MANIFEST = ROOT / "data" / "mashcima" / "manifest.jsonl"
DEFAULT_PLAN_SEED = 2026_07_07
FORMAT_COUNTS = {"clean": 11_667, "camera": 11_667, "handwritten": 11_666}


def parse_format_counts(spec: str | None) -> dict[str, int]:
    if not spec:
        return dict(FORMAT_COUNTS)
    counts: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        fmt, _, n_s = part.partition(":")
        counts[fmt.strip()] = int(n_s)
    allowed = set(FORMAT_COUNTS)
    unknown = set(counts) - allowed
    if unknown:
        raise ValueError(f"unknown format(s): {sorted(unknown)}")
    if not counts:
        raise ValueError("empty --format-counts")
    return counts


def golden_incipit_exclusions(primus_ids: set[str]) -> set[str]:
    from remove_golden_from_consolidated import build_exclusion_set

    excl, _ = build_exclusion_set()
    return {key for key in excl if key in primus_ids}


def write_format_progress(
    manifest_path: Path,
    progress_log: Path,
    format_counts: dict[str, int],
    running: int = 0,
) -> None:
    targets = format_counts
    ok_counts = {fmt: 0 for fmt in targets}
    latest: dict[str, dict] = {}
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                latest[row["corrupt_id"]] = row
    attempts_ok = attempts_err = 0
    for row in latest.values():
        if row.get("status") == "ok":
            attempts_ok += 1
            fmt = row.get("format")
            if fmt in ok_counts:
                ok_counts[fmt] += 1
        else:
            attempts_err += 1
    total_ok = sum(ok_counts.values())
    total_target = sum(targets.values())
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = [
        f"[{ts}] corrupt batch progress",
        f"  total ok: {total_ok}/{total_target} ({100.0 * total_ok / total_target:.1f}%)",
    ]
    for fmt in ("handwritten", "camera", "clean"):
        if fmt not in targets:
            continue
        ok = ok_counts[fmt]
        tgt = targets[fmt]
        lines.append(f"  {fmt:12} ok={ok:>6}/{tgt} ({100.0 * ok / tgt:.1f}%)")
    lines.append(f"  attempts: ok={attempts_ok} error={attempts_err} running={running}")
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    progress_log.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_manifest_status(manifest: Path) -> dict[str, str]:
    """Last status per corrupt_id (later manifest lines win)."""
    if not manifest.is_file():
        return {}
    status: dict[str, str] = {}
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            status[row["corrupt_id"]] = row.get("status", "error")
    return status


def load_manifest_records(manifest: Path) -> dict[str, dict]:
    """Last manifest row per corrupt_id."""
    records: dict[str, dict] = {}
    if not manifest.is_file():
        return records
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                records[row["corrupt_id"]] = row
    return records


def load_done_ids(manifest: Path) -> set[str]:
    return {cid for cid, st in load_manifest_status(manifest).items() if st == "ok"}


def load_last_ok_rows(manifest: Path) -> dict[str, dict]:
    """Last ok manifest row per corrupt_id (must include semantic_path)."""
    rows: dict[str, dict] = {}
    if not manifest.is_file():
        return rows
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("semantic_path"):
                rows[row["corrupt_id"]] = row
    return rows


def load_regenerated_ok_ids(manifest: Path, formats: set[str]) -> set[str]:
    """corrupt_ids whose latest manifest row is a successful regeneration for a format."""
    return {
        corrupt_id
        for corrupt_id, row in load_manifest_records(manifest).items()
        if row.get("status") == "ok"
        and row.get("regenerated") is True
        and row.get("format") in formats
    }


def load_error_ids(manifest: Path) -> set[str]:
    return {cid for cid, st in load_manifest_status(manifest).items() if st == "error"}


def load_deferred_ids(path: Path = DEFERRED_JOBS) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["corrupt_id"])
    return ids


def defer_jobs(jobs: list[dict], path: Path = DEFERRED_JOBS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for job in jobs:
            f.write(
                json.dumps(
                    {
                        "corrupt_id": job["corrupt_id"],
                        "incipit_id": job.get("incipit_id"),
                        "format": job.get("format"),
                        "reason": "deferred_skip_pending",
                    }
                )
                + "\n"
            )


def build_primus_index(primus_dir: Path, mashcima_manifest: Path | None = None) -> dict[str, Path]:
    """Build incipit_id -> PrIMuS root map (manifest-fast path, no full-tree rglob)."""
    manifest = mashcima_manifest or MASHCIMA_MANIFEST
    if manifest.is_file():
        index: dict[str, Path] = {}
        with manifest.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                incipit_id = row.get("incipit_id")
                if not incipit_id:
                    continue
                sem = primus_semantic_path(incipit_id, primus_dir)
                ag = primus_agnostic_path(incipit_id, primus_dir)
                if sem.is_file() and ag.is_file():
                    index[incipit_id] = primus_dir
        if index:
            return index

    index = {}
    gt_dir = primus_dir / "ground_truth"
    if gt_dir.is_dir():
        for sem in gt_dir.glob("*.semantic"):
            incipit_id = sem.stem
            if primus_agnostic_path(incipit_id, primus_dir).is_file():
                index[incipit_id] = primus_dir
        if index:
            return index

    for semantic in primus_dir.rglob("*.semantic"):
        if semantic.name.startswith("._"):
            continue
        incipit_id = semantic.stem
        if incipit_id in index:
            continue
        agnostic = semantic.parent / f"{incipit_id}.agnostic"
        if agnostic.is_file():
            index[incipit_id] = primus_dir
    return index


def load_convertible_incipt_ids(
    mashcima_manifest: Path,
    primus_index: dict[str, Path],
) -> list[str]:
    if not mashcima_manifest.is_file():
        raise FileNotFoundError(f"Missing Mashcima manifest: {mashcima_manifest}")

    ok_ids: list[str] = []
    with mashcima_manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            incipit_id = row.get("incipit_id")
            if row.get("status") == "ok" and incipit_id in primus_index:
                ok_ids.append(incipit_id)
    if not ok_ids:
        raise RuntimeError("No convertible incipits with PrIMuS semantic/agnostic on disk")
    return ok_ids


def build_jobs(
    pool: list[str],
    primus_index: dict[str, Path],
    total: int,
    plan_seed: int,
    jobs_path: Path,
    format_counts: dict[str, int] | None = None,
) -> list[dict]:
    counts = format_counts or FORMAT_COUNTS
    expected = sum(counts.values())
    if total != expected:
        raise ValueError(f"--total must be {expected} (got {total})")

    rng = random.Random(plan_seed)
    formats: list[str] = []
    for fmt, count in counts.items():
        formats.extend([fmt] * count)
    rng.shuffle(formats)

    jobs: list[dict] = []
    for fmt in formats:
        incipit_id = rng.choice(pool)
        corrupt_id = uuid.uuid4().hex
        corruption_seed = rng.randint(0, 2_147_483_647)
        jobs.append(
            {
                "corrupt_id": corrupt_id,
                "format": fmt,
                "incipit_id": incipit_id,
                "corruption_seed": corruption_seed,
                "primus_folder": str(primus_index[incipit_id]),
            }
        )

    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    with jobs_path.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")
    return jobs


def load_jobs(jobs_path: Path) -> list[dict]:
    jobs: list[dict] = []
    with jobs_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def resolve_incipt_folder(job: dict, primus_dir: Path) -> Path:
    """Always resolve to the PrIMuS dataset root for layout helpers.

    Legacy jobs stored per-incipit paths under package_aa/package_ab; those
    folders are gone after reorganization.
    """
    _ = job
    return primus_dir


def render_corrupt_image(
    fmt: str,
    corrupted_semantic: str,
    render_annotation: str,
    corruption_seed: int,
    *,
    writer_key: str = "",
    apply_appearance: bool = True,
    max_dim: int | None = None,
) -> "Image.Image":
    """Render one corrupt output image for clean, camera, or handwritten."""
    if fmt == "clean":
        return render_semantic_to_image(corrupted_semantic)
    if fmt == "camera":
        clean = render_semantic_to_image(corrupted_semantic)
        return camera_like_distort(clean, seed=corruption_seed + 99)
    if fmt == "handwritten":
        handwritten = render_mashcima(
            render_annotation,
            "handwritten",
            writer_key=writer_key,
            apply_appearance=apply_appearance,
            max_dim=max_dim,
        )
        return camera_like_distort(handwritten, seed=corruption_seed + 99)
    raise ValueError(f"unknown format: {fmt}")


def timeout_for_format(fmt: str, digital_timeout: float, handwritten_timeout: float) -> float:
    if fmt == "handwritten":
        return handwritten_timeout
    return digital_timeout


def cleanup_job_outputs(corrupt_dir: Path, corrupt_id: str, fmt: str) -> None:
    """Remove partial outputs for a job that did not complete successfully."""
    for path in (
        corrupt_dir / "semantic" / f"{corrupt_id}.semantic",
        corrupt_dir / fmt / f"{corrupt_id}.png",
        corrupt_dir / fmt / f"{corrupt_id}.jpg",
    ):
        if path.is_file():
            path.unlink()


def process_job(job: dict, primus_dir: Path, corrupt_dir: Path) -> dict:
    corrupt_id = job["corrupt_id"]
    fmt = job["format"]
    incipit_id = job["incipit_id"]
    corruption_seed = int(job["corruption_seed"])

    folder = resolve_incipt_folder(job, primus_dir)
    semantic_text = read_primus_semantic(incipit_id, folder)
    agnostic_text = read_primus_agnostic(incipit_id, folder, DEFAULT_PRIMUS_TGZ)

    rng = random.Random(corruption_seed)
    corrupt_fraction = rng.uniform(0.33, 0.5)
    corrupted_semantic, corrupted_mashcima = corrupt_tokens(
        semantic_text, agnostic_text, rng, corrupt_fraction
    )
    render_annotation, warnings = prepare_mashcima_for_render(corrupted_mashcima)

    semantic_out = corrupt_dir / "semantic" / f"{corrupt_id}.semantic"
    png_out = corrupt_dir / fmt / f"{corrupt_id}.png"
    try:
        img = render_corrupt_image(
            fmt,
            corrupted_semantic,
            render_annotation,
            corruption_seed,
            writer_key=corrupt_id,
        )
        save_png(img, png_out)
        compress_image_file(png_out)
        png_out = resolved_image_path(png_out)
        semantic_out.parent.mkdir(parents=True, exist_ok=True)
        semantic_out.write_text(corrupted_semantic, encoding="utf-8")
    except Exception:
        cleanup_job_outputs(corrupt_dir, corrupt_id, fmt)
        raise

    return {
        "corrupt_id": corrupt_id,
        "format": fmt,
        "incipit_id": incipit_id,
        "corruption_seed": corruption_seed,
        "corrupt_fraction": round(corrupt_fraction, 4),
        "status": "ok",
        "semantic_path": str(semantic_out.relative_to(ROOT)).replace("\\", "/"),
        "png_path": str(png_out.relative_to(ROOT)).replace("\\", "/"),
        "warnings": warnings[:5] if warnings else [],
    }


def regenerate_job(
    job: dict,
    manifest_row: dict,
    primus_dir: Path,
    corrupt_dir: Path,
    max_dim: int | None = 768,
) -> dict:
    """Re-render an existing job from its saved semantic (and primus for handwritten)."""
    corrupt_id = job["corrupt_id"]
    fmt = job["format"]
    incipit_id = job["incipit_id"]
    corruption_seed = int(job["corruption_seed"])

    semantic_rel = manifest_row.get("semantic_path")
    semantic_path = (ROOT / semantic_rel) if semantic_rel else (corrupt_dir / "semantic" / f"{corrupt_id}.semantic")
    if not semantic_path.is_file():
        # Always prefer the on-disk corrupt semantic — source of truth for labels.
        semantic_path = corrupt_dir / "semantic" / f"{corrupt_id}.semantic"
    if not semantic_path.is_file():
        raise FileNotFoundError(f"Missing semantic file: {semantic_path}")
    corrupted_semantic = semantic_path.read_text(encoding="utf-8")
    semantic_rel_out = str(semantic_path.relative_to(ROOT)).replace("\\", "/")

    folder = resolve_incipt_folder(job, primus_dir)
    if fmt == "handwritten":
        # Build Mashcima so event pitches/durations match the saved semantic exactly.
        # Prefer replaying corruption for beam layout when sync is possible; otherwise
        # rebuild from the semantic (optionally using the original agnostic header).
        layout_annotation: str | None = None
        try:
            semantic_text = read_primus_semantic(incipit_id, folder)
            agnostic_text = read_primus_agnostic(incipit_id, folder, DEFAULT_PRIMUS_TGZ)
            corrupt_fraction = float(manifest_row.get("corrupt_fraction") or 0.4)
            rng = random.Random(corruption_seed)
            _, replayed_mashcima = corrupt_tokens(
                semantic_text, agnostic_text, rng, corrupt_fraction
            )
            layout_annotation = replayed_mashcima
        except Exception:
            try:
                agnostic_text = read_primus_agnostic(incipit_id, folder, DEFAULT_PRIMUS_TGZ)
                layout_annotation = convert_primus_annotation_to_mashcima_annotation(
                    agnostic_text
                )
            except Exception:
                layout_annotation = None

        corrupted_mashcima = mashcima_annotation_from_semantic(
            corrupted_semantic,
            layout_annotation=layout_annotation,
        )
        render_annotation, warnings = prepare_mashcima_for_render(corrupted_mashcima)
    else:
        render_annotation = ""
        warnings = []

    png_out = corrupt_dir / fmt / f"{corrupt_id}.png"
    img = render_corrupt_image(
        fmt,
        corrupted_semantic,
        render_annotation,
        corruption_seed,
        writer_key=corrupt_id,
        apply_appearance=True,
        max_dim=max_dim,
    )
    # Camera padding grows the canvas slightly. Cap it again and encode once,
    # instead of writing a full PNG and reopening it for iterative compression.
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    bgr = resize_long_edge(bgr, max_dim)
    for stale in (png_out, png_out.with_suffix(".jpg"), png_out.with_suffix(".jpeg")):
        stale.unlink(missing_ok=True)
    png_out = write_jpeg_bgr_under_kb(bgr, png_out)

    return {
        "corrupt_id": corrupt_id,
        "format": fmt,
        "incipit_id": incipit_id,
        "corruption_seed": corruption_seed,
        "corrupt_fraction": manifest_row.get("corrupt_fraction"),
        "status": "ok",
        "semantic_path": semantic_rel_out,
        "png_path": str(png_out.relative_to(ROOT)).replace("\\", "/"),
        "warnings": warnings[:5] if warnings else [],
        "regenerated": True,
        "from_saved_semantic": True,
        "appearance": True,
        "max_dim": max_dim,
    }


def _regen_worker(payload: tuple[dict, dict, str, str, int | None]) -> dict:
    job, manifest_row, primus_dir_s, corrupt_dir_s, max_dim = payload
    primus_dir = Path(primus_dir_s)
    corrupt_dir = Path(corrupt_dir_s)
    try:
        return regenerate_job(
            job,
            manifest_row,
            primus_dir,
            corrupt_dir,
            max_dim=max_dim,
        )
    except Exception as exc:
        # These retry failures are deterministic annotation/render failures.
        # Repeating the identical job three times only triples their cost.
        msg = str(exc).strip() or type(exc).__name__
    return {
        "corrupt_id": job["corrupt_id"],
        "format": job.get("format"),
        "incipit_id": job.get("incipit_id"),
        "status": "error",
        "error": msg,
        "regenerated": True,
    }


def _worker(payload: tuple[dict, str, str]) -> dict:
    job, primus_dir_s, corrupt_dir_s = payload
    primus_dir = Path(primus_dir_s)
    corrupt_dir = Path(corrupt_dir_s)
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return process_job(job, primus_dir, corrupt_dir)
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                continue
    assert last_exc is not None
    cleanup_job_outputs(corrupt_dir, job["corrupt_id"], job.get("format", "clean"))
    msg = str(last_exc).strip() or type(last_exc).__name__
    return {
        "corrupt_id": job["corrupt_id"],
        "format": job.get("format"),
        "incipit_id": job.get("incipit_id"),
        "status": "error",
        "error": msg,
    }


_pool_ready = False


def _init_pool(primus_tgz_s: str, muscima_dir_s: str, writer_mode_s: str) -> None:
    global _pool_ready
    if _pool_ready:
        return
    _configure_playwright_browsers_path()
    ensure_muscima_cache(Path(primus_tgz_s), Path(muscima_dir_s))
    configure_mashcima_writers(writer_mode_s)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    _pool_ready = True


def _kill_pool(pool_exec: ProcessPoolExecutor) -> None:
    """Forcibly kill worker processes so hung jobs don't linger as zombies."""
    procs = list(getattr(pool_exec, "_processes", {}).values())
    for proc in procs:
        try:
            proc.kill()
        except Exception:
            pass
    for proc in procs:
        try:
            proc.join(timeout=5)
        except Exception:
            pass
    try:
        pool_exec.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    close_render_pool()


def write_regen_progress(
    progress_log: Path,
    *,
    formats: set[str],
    target: int,
    ok_count: int,
    err_count: int,
    skipped: int,
    running: bool = True,
) -> None:
    done = ok_count + err_count
    processed = skipped + done
    pct = (100.0 * processed / target) if target else 0.0
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    fmt_label = ",".join(sorted(formats))
    lines = [
        f"[{ts}] corrupt regenerate progress ({fmt_label})",
        f"  processed: {processed}/{target} ({pct:.1f}%)",
        f"  ok={ok_count} errors={err_count} skipped_resume={skipped}",
        f"  status={'running' if running else 'done'}",
    ]
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    progress_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress_log.with_suffix(".json").write_text(
        json.dumps(
            {
                "ts": ts,
                "formats": sorted(formats),
                "processed": processed,
                "total": target,
                "pct": round(pct, 2),
                "ok": ok_count,
                "errors": err_count,
                "skipped_resume": skipped,
                "status": "running" if running else "done",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_regenerate_formats(
    primus_dir: Path,
    primus_tgz: Path,
    corrupt_dir: Path,
    manifest_path: Path,
    jobs_path: Path,
    formats: set[str],
    workers: int,
    limit: int | None,
    resume: bool = True,
    retry_regen_errors: bool = False,
    errors_only: bool = False,
    writer_mode: str = DEFAULT_WRITER_MODE,
    progress_log: Path | None = None,
    max_dim: int | None = 768,
    max_tasks_per_child: int = 200,
) -> None:
    """Re-render existing ok jobs for the given formats from saved semantic files."""
    allowed = set(FORMAT_COUNTS)
    unknown = formats - allowed
    if unknown:
        raise ValueError(f"unknown regenerate format(s): {sorted(unknown)}")

    corrupt_dir.mkdir(parents=True, exist_ok=True)
    ensure_muscima_cache(primus_tgz, DEFAULT_MUSCIMA_DIR)
    configure_mashcima_writers(writer_mode)

    jobs_by_id = {j["corrupt_id"]: j for j in load_jobs(jobs_path)}
    manifest_rows = load_manifest_records(manifest_path)
    last_ok_rows = load_last_ok_rows(manifest_path)
    regenerated = load_regenerated_ok_ids(manifest_path, formats) if resume else set()
    pending_jobs: list[tuple[dict, dict]] = []
    seen_ids: set[str] = set()

    for corrupt_id, row in manifest_rows.items():
        if corrupt_id in regenerated or corrupt_id in seen_ids:
            continue
        fmt = row.get("format")
        if fmt not in formats:
            continue
        job = jobs_by_id.get(corrupt_id)
        if job is None:
            continue

        if row.get("status") == "ok":
            if errors_only:
                continue
            pending_jobs.append((job, row))
            seen_ids.add(corrupt_id)
            continue

        if not retry_regen_errors:
            continue
        if row.get("status") != "error":
            continue

        # Prefer a prior ok row when available; otherwise rebuild from the
        # on-disk semantic (always present for handwritten corrupt jobs).
        ok_row = last_ok_rows.get(corrupt_id)
        semantic_path = corrupt_dir / "semantic" / f"{corrupt_id}.semantic"
        if ok_row and ok_row.get("semantic_path"):
            candidate = ROOT / ok_row["semantic_path"]
            if candidate.is_file():
                semantic_path = candidate
        if not semantic_path.is_file():
            continue

        regen_row = dict(ok_row or row)
        regen_row["semantic_path"] = str(semantic_path.relative_to(ROOT)).replace("\\", "/")
        regen_row.setdefault("corruption_seed", job.get("corruption_seed"))
        regen_row.setdefault("corrupt_fraction", job.get("corrupt_fraction", 0.4))
        regen_row["format"] = fmt
        regen_row["incipit_id"] = job.get("incipit_id", row.get("incipit_id"))
        pending_jobs.append((job, regen_row))
        seen_ids.add(corrupt_id)

    # Keep one expensive writer repository hot in each process for a long
    # contiguous run. Random job order otherwise reloads ~100MB+ almost every
    # image because the per-process repository cache is intentionally small.
    if formats == {"handwritten"} and writer_mode == "per-score":
        pending_jobs.sort(
            key=lambda item: (
                writer_number_for_key(item[0]["corrupt_id"]),
                item[0]["corrupt_id"],
            )
        )
    if limit is not None:
        pending_jobs = pending_jobs[:limit]

    skipped = len(regenerated)
    if not pending_jobs:
        if skipped:
            print(
                f"All {skipped:,} ok job(s) already regenerated for formats: "
                f"{', '.join(sorted(formats))}",
                flush=True,
            )
        else:
            print(f"No ok jobs to regenerate for formats: {', '.join(sorted(formats))}")
        return

    retry_count = sum(
        1
        for job, _ in pending_jobs
        if manifest_rows.get(job["corrupt_id"], {}).get("status") == "error"
    )
    fresh_count = len(pending_jobs) - retry_count
    print(
        f"Regenerating {len(pending_jobs):,} job(s) for formats: {', '.join(sorted(formats))}"
        + (f" ({fresh_count:,} pending ok, {retry_count:,} retry errors)" if retry_count else "")
        + (f" ({skipped:,} skipped, resume)" if skipped else ""),
        flush=True,
    )

    payloads = [
        (job, row, str(primus_dir.resolve()), str(corrupt_dir.resolve()), max_dim)
        for job, row in pending_jobs
    ]
    initargs = (
        str(primus_tgz.resolve()),
        str(DEFAULT_MUSCIMA_DIR.resolve()),
        writer_mode,
    )
    ok_count = 0
    err_count = 0
    total_jobs = len(pending_jobs) + skipped
    progress_every = 20
    if progress_log is not None:
        write_regen_progress(
            progress_log,
            formats=formats,
            target=total_jobs,
            ok_count=0,
            err_count=0,
            skipped=skipped,
        )

    with manifest_path.open("a", encoding="utf-8") as manifest:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_pool,
            initargs=initargs,
            max_tasks_per_child=max_tasks_per_child,
        ) as pool_exec:
            # Bound in-flight work so we don't queue thousands of futures/payloads.
            in_flight: dict = {}
            payload_iter = iter(payloads)
            max_in_flight = max(workers * 2, workers)

            def _submit_more() -> None:
                while len(in_flight) < max_in_flight:
                    try:
                        payload = next(payload_iter)
                    except StopIteration:
                        return
                    fut = pool_exec.submit(_regen_worker, payload)
                    in_flight[fut] = True

            _submit_more()
            pbar = tqdm(
                total=total_jobs,
                initial=skipped,
                desc="Regenerate corrupt formats",
                unit="job",
            )
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future, None)
                    record = future.result()
                    manifest.write(json.dumps(record) + "\n")
                    if record.get("status") == "ok":
                        ok_count += 1
                    else:
                        err_count += 1
                    pbar.update(1)
                    if progress_log is not None and (ok_count + err_count) % progress_every == 0:
                        manifest.flush()
                        write_regen_progress(
                            progress_log,
                            formats=formats,
                            target=total_jobs,
                            ok_count=ok_count,
                            err_count=err_count,
                            skipped=skipped,
                        )
                    _submit_more()
            pbar.close()

    if progress_log is not None:
        write_regen_progress(
            progress_log,
            formats=formats,
            target=total_jobs,
            ok_count=ok_count,
            err_count=err_count,
            skipped=skipped,
            running=False,
        )

    print(
        f"Regenerate done: {ok_count} ok, {err_count} errors - appended to {manifest_path}",
        flush=True,
    )


def run_batch(
    primus_dir: Path,
    primus_tgz: Path,
    corrupt_dir: Path,
    manifest_path: Path,
    jobs_path: Path,
    total: int,
    plan_seed: int,
    workers: int,
    resume: bool,
    regenerate_jobs: bool,
    limit: int | None,
    retry_errors: bool = False,
    skip_pending: int = 0,
    job_timeout: float = 120.0,
    handwritten_timeout: float = 180.0,
    max_tasks_per_child: int = 30,
    writer_mode: str = DEFAULT_WRITER_MODE,
    format_counts: dict[str, int] | None = None,
    exclude_golden: bool = False,
    progress_log: Path | None = None,
) -> None:
    counts = format_counts or FORMAT_COUNTS
    deferred_path = corrupt_dir / "deferred_jobs.jsonl"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    ensure_muscima_cache(primus_tgz, DEFAULT_MUSCIMA_DIR)
    configure_mashcima_writers(writer_mode)

    print("Building PrIMuS index from mashcima manifest...", flush=True)
    primus_index = build_primus_index(primus_dir)
    print(f"PrIMuS index: {len(primus_index):,} incipits", flush=True)
    pool = load_convertible_incipt_ids(MASHCIMA_MANIFEST, primus_index)
    if exclude_golden:
        exclude = golden_incipit_exclusions(set(primus_index))
        pool = [incipit_id for incipit_id in pool if incipit_id not in exclude]
        print(f"Sampling pool: {len(pool):,} incipits (excluded {len(exclude):,} golden)", flush=True)
    if regenerate_jobs or not jobs_path.is_file():
        jobs = build_jobs(pool, primus_index, total, plan_seed, jobs_path, counts)
    else:
        jobs = load_jobs(jobs_path)
        for job in jobs:
            if "primus_folder" not in job and job["incipit_id"] in primus_index:
                job["primus_folder"] = str(primus_index[job["incipit_id"]])
        if len(jobs) != total:
            raise ValueError(f"jobs file has {len(jobs)} entries, expected {total}")

    done = load_done_ids(manifest_path) if resume else set()
    attempted = set(load_manifest_status(manifest_path).keys()) if resume else set()
    if retry_errors:
        error_ids = load_error_ids(manifest_path)
        pending = [j for j in jobs if j["corrupt_id"] in error_ids]
    else:
        never_tried = [
            j for j in jobs if j["corrupt_id"] not in done and j["corrupt_id"] not in attempted
        ]
        retries = [
            j
            for j in jobs
            if j["corrupt_id"] not in done
            and j["corrupt_id"] in attempted
        ]
        pending = never_tried + retries
        if never_tried:
            print(
                f"Pending: {len(never_tried):,} never-attempted, {len(retries):,} retries",
                flush=True,
            )

    deferred_ids = load_deferred_ids(deferred_path)
    if deferred_ids:
        pending = [j for j in pending if j["corrupt_id"] not in deferred_ids]

    if skip_pending > 0 and pending:
        to_defer = pending[:skip_pending]
        defer_jobs(to_defer, deferred_path)
        deferred_ids.update(j["corrupt_id"] for j in to_defer)
        pending = [j for j in pending if j["corrupt_id"] not in deferred_ids]
        print(
            f"Deferred {len(to_defer)} pending job(s) to {deferred_path}; "
            f"first deferred: {to_defer[0]['corrupt_id']}",
            flush=True,
        )

    if limit is not None:
        pending = pending[:limit]

    if not pending:
        if retry_errors:
            print("No corrupt jobs with error status to retry.")
        else:
            print(f"All {len(jobs)} corrupt jobs already complete.")
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ok_count = 0
    err_count = 0

    payloads = [(job, str(primus_dir.resolve()), str(corrupt_dir.resolve())) for job in pending]

    initargs = (
        str(primus_tgz.resolve()),
        str(DEFAULT_MUSCIMA_DIR.resolve()),
        writer_mode,
    )

    def _make_pool() -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_pool,
            initargs=initargs,
            max_tasks_per_child=max_tasks_per_child,
        )

    def _kill_pool_local(pool_exec: ProcessPoolExecutor) -> None:
        _kill_pool(pool_exec)

    def _write_timeout(job: dict, manifest) -> None:
        nonlocal err_count
        cleanup_job_outputs(corrupt_dir, job["corrupt_id"], job.get("format", "clean"))
        limit = timeout_for_format(job.get("format", "clean"), job_timeout, handwritten_timeout)
        record = {
            "corrupt_id": job["corrupt_id"],
            "format": job.get("format"),
            "incipit_id": job.get("incipit_id"),
            "status": "error",
            "error": f"job_timeout>{limit:.0f}s",
        }
        manifest.write(json.dumps(record) + "\n")
        manifest.flush()
        err_count += 1
        print(f"Deferred slow job {job['corrupt_id']} (>{limit:.0f}s)", flush=True)

    progress_interval = 20
    jobs_since_progress = 0

    def _maybe_progress(running: int = 0, *, force: bool = False) -> None:
        nonlocal jobs_since_progress
        if progress_log is None:
            return
        jobs_since_progress += 1
        if force or jobs_since_progress >= progress_interval:
            jobs_since_progress = 0
            write_format_progress(manifest_path, progress_log, counts, running)

    with manifest_path.open("a", encoding="utf-8") as manifest:
        with tqdm(total=len(payloads), desc="Corrupt batch", unit="job") as pbar:
            pending_payloads = list(payloads)
            active: dict = {}
            pool_exec = _make_pool()

            while pending_payloads or active:
                # Fill idle worker slots.
                while pending_payloads and len(active) < workers:
                    payload = pending_payloads.pop(0)
                    try:
                        future = pool_exec.submit(_worker, payload)
                    except BrokenProcessPool:
                        pending_payloads.insert(0, payload)
                        _kill_pool_local(pool_exec)
                        for _f, (p, _s, _lim) in list(active.items()):
                            pending_payloads.append(p)
                        active.clear()
                        pool_exec = _make_pool()
                        break
                    job = payload[0]
                    fmt = job.get("format", "clean")
                    active[future] = (
                        payload,
                        time.monotonic(),
                        timeout_for_format(fmt, job_timeout, handwritten_timeout),
                    )

                if not active:
                    if pending_payloads:
                        continue
                    break

                done, _ = wait(set(active.keys()), timeout=5, return_when=FIRST_COMPLETED)
                broke = False
                for future in done:
                    payload, _started, _limit = active.pop(future)
                    try:
                        record = future.result()
                    except BrokenProcessPool:
                        # Worker died (likely OOM); retry this job on a fresh pool.
                        pending_payloads.append(payload)
                        broke = True
                        continue
                    except Exception as exc:
                        job = payload[0]
                        record = {
                            "corrupt_id": job["corrupt_id"],
                            "format": job.get("format"),
                            "incipit_id": job.get("incipit_id"),
                            "status": "error",
                            "error": str(exc).strip() or type(exc).__name__,
                        }
                    else:
                        manifest.write(json.dumps(record) + "\n")
                        manifest.flush()
                        if record.get("status") == "ok":
                            ok_count += 1
                        else:
                            err_count += 1
                        pbar.update(1)
                        _maybe_progress(len(active))
                        continue
                    manifest.write(json.dumps(record) + "\n")
                    manifest.flush()
                    err_count += 1
                    pbar.update(1)
                    _maybe_progress(len(active))

                if broke:
                    for _f, (p, _s, _lim) in list(active.items()):
                        pending_payloads.append(p)
                    active.clear()
                    _kill_pool_local(pool_exec)
                    pool_exec = _make_pool()
                    continue

                # Detect and kill hung workers (jobs exceeding the timeout).
                now = time.monotonic()
                timed_out = [
                    f
                    for f, (_payload, started, limit) in active.items()
                    if now - started > limit
                ]
                if timed_out:
                    for future in timed_out:
                        payload, _started, _limit = active.pop(future)
                        _write_timeout(payload[0], manifest)
                        pbar.update(1)
                        _maybe_progress(len(active))
                    for _f, (p, _s, _lim) in list(active.items()):
                        pending_payloads.append(p)
                    active.clear()
                    _kill_pool_local(pool_exec)
                    pool_exec = _make_pool()

            _kill_pool_local(pool_exec)

    skipped = len(jobs) - len(pending)
    _maybe_progress(0, force=True)
    print(
        f"Done: {ok_count} generated, {skipped} skipped (resume), {err_count} errors - {manifest_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=sum(FORMAT_COUNTS.values()))
    parser.add_argument("--primus-dir", type=Path, default=DEFAULT_PRIMUS_DIR)
    parser.add_argument("--primus-tgz", type=Path, default=DEFAULT_PRIMUS_TGZ)
    parser.add_argument("--output-dir", type=Path, default=CORRUPT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    parser.add_argument("--plan-seed", type=int, default=DEFAULT_PLAN_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-run only jobs whose latest manifest status is error",
    )
    parser.add_argument(
        "--regenerate-formats",
        type=str,
        default=None,
        help="Comma-separated formats to re-render from saved semantic (e.g. clean,camera,handwritten)",
    )
    parser.add_argument(
        "--retry-regen-errors",
        action="store_true",
        help="Also retry format regeneration jobs whose latest manifest status is error",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="With --regenerate-formats, only retry error jobs (skip ok regenerations)",
    )
    parser.add_argument("--regenerate-jobs", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-pending",
        type=int,
        default=0,
        help="Defer the first N pending jobs to deferred_jobs.jsonl and process the rest first",
    )
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=90.0,
        help="Seconds before a slow job is deferred and marked error (parallel workers only)",
    )
    parser.add_argument(
        "--format-counts",
        type=str,
        default=None,
        help="Per-format job counts, e.g. clean:6000,camera:9000,handwritten:15000",
    )
    parser.add_argument(
        "--exclude-golden",
        action="store_true",
        help="Exclude golden-set primus incipit IDs from sampling",
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=None,
        help="Write per-format ok/error progress to this file after each job",
    )
    parser.add_argument(
        "--handwritten-timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds for handwritten jobs (mashcima render is slower)",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=200,
        help="Recycle worker processes after N jobs to limit memory growth",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=768,
        help=(
            "Cap regenerated image long edge and run appearance/camera near "
            "that resolution (default: 768; use 0 to disable)"
        ),
    )
    parser.add_argument(
        "--writers",
        choices=WRITER_MODES,
        default=DEFAULT_WRITER_MODE,
        help=(
            "MUSCIMA++ writer selection for handwritten renders: per-score "
            "(default), mixed, or single (writer 01)"
        ),
    )
    return parser.parse_args()


def _configure_stdio_utf8() -> None:
    """Ensure redirected logs stay UTF-8 on Windows (avoids UTF-16 mojibake)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass


def _configure_playwright_browsers_path() -> None:
    """Point Playwright at an installed browser when Cursor sandbox path is empty."""
    current = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if current:
        root = Path(current)
        if not any(root.rglob("chrome-headless-shell.exe")):
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)

    if os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        return

    search_roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" / "cursor-sandbox-cache",
    ]
    for root in search_roots:
        if not root.is_dir():
            continue
        for exe in root.rglob("chrome-headless-shell.exe"):
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(exe.parents[2])
            return


def main() -> None:
    _configure_stdio_utf8()
    _configure_playwright_browsers_path()
    args = parse_args()
    format_counts = parse_format_counts(args.format_counts)
    expected_total = sum(format_counts.values())
    if args.total != expected_total:
        raise SystemExit(f"--total must be {expected_total} (got {args.total})")
    if args.regenerate_formats:
        formats = {f.strip() for f in args.regenerate_formats.split(",") if f.strip()}
        regen_progress = args.progress_log or (
            ROOT / "data" / "logs" / "corrupt_handwritten_regen_progress.txt"
            if formats == {"handwritten"}
            else ROOT / "data" / "logs" / "corrupt_regen_progress.txt"
        )
        run_regenerate_formats(
            primus_dir=args.primus_dir,
            primus_tgz=args.primus_tgz,
            corrupt_dir=args.output_dir,
            manifest_path=args.manifest,
            jobs_path=args.jobs,
            formats=formats,
            workers=args.workers,
            limit=args.limit,
            resume=not args.no_resume,
            retry_regen_errors=args.retry_regen_errors,
            errors_only=args.errors_only,
            writer_mode=args.writers,
            progress_log=regen_progress,
            max_dim=args.max_dim or None,
            max_tasks_per_child=max(1, args.max_tasks_per_child),
        )
        return
    run_batch(
        primus_dir=args.primus_dir,
        primus_tgz=args.primus_tgz,
        corrupt_dir=args.output_dir,
        manifest_path=args.manifest,
        jobs_path=args.jobs,
        total=args.total,
        plan_seed=args.plan_seed,
        workers=args.workers,
        resume=not args.no_resume,
        retry_errors=args.retry_errors,
        skip_pending=args.skip_pending,
        job_timeout=args.job_timeout,
        handwritten_timeout=args.handwritten_timeout,
        max_tasks_per_child=args.max_tasks_per_child,
        regenerate_jobs=args.regenerate_jobs,
        limit=args.limit,
        writer_mode=args.writers,
        format_counts=format_counts,
        exclude_golden=args.exclude_golden,
        progress_log=args.progress_log,
    )


if __name__ == "__main__":
    main()
