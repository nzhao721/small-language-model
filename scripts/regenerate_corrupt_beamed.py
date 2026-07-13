"""Regenerate corrupt clean/camera images with explicit MEI beaming.

Reads ground-truth semantic from ``data/corrupt/semantic/``, derives beam groups by
replaying corruption to recover Mashcima beam markers, renders via Verovio, and
writes outputs to ``data/corrupt_beamed/`` (original ``data/corrupt/`` images are
left untouched).

Handwritten corrupt images are not regenerated.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from PIL import Image
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from demo_corrupted_gt_examples import (  # noqa: E402
    corrupt_tokens,
    extract_aligned_pairs,
    is_beamed_mash,
    save_png,
    split_semantic,
)
from demo_mashcima_camera_distort import camera_like_distort  # noqa: E402
from dataset_layout import read_primus_agnostic, read_primus_semantic  # noqa: E402
from generate_corrupt_batch import (  # noqa: E402
    CORRUPT_DIR,
    DEFAULT_JOBS,
    DEFAULT_MANIFEST,
    DEFAULT_PRIMUS_DIR,
    ROOT,
    load_jobs,
    load_manifest_records,
    resolve_incipt_folder,
)
from generate_mashcima_batch import DEFAULT_PRIMUS_TGZ  # noqa: E402
from image_compress import compress_image_file, resolved_image_path  # noqa: E402
from primus_semantic_render import close_render_pool, render_semantic_to_image  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "corrupt_beamed"
LOG_PATH = DEFAULT_OUT / "regenerate.log"
MANIFEST_PATH = DEFAULT_OUT / "manifest.jsonl"
DEFERRED_PATH = DEFAULT_OUT / "deferred_jobs.jsonl"
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _norm_semantic(text: str) -> str:
    return " ".join(text.split())


def log_line(msg: str, log_file) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


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
                done.add(row["corrupt_id"])
    return done


def output_image_path(out_dir: Path, fmt: str, corrupt_id: str) -> Path | None:
    folder = out_dir / fmt
    for ext in IMAGE_EXTS:
        path = folder / f"{corrupt_id}{ext}"
        if path.is_file():
            return path
    return None


def mash_pairs_for_semantic(semantic_gt: str, mashcima: str) -> list[tuple[str, str]]:
    tokens = semantic_gt.split()
    _, body = split_semantic(tokens)
    return extract_aligned_pairs(body, mashcima.split())


def replay_mashcima(
    job: dict,
    row: dict,
    semantic_gt: str,
    primus_dir: Path,
) -> tuple[list[tuple[str, str]] | None, str]:
    """Return (mash_pairs or None, beam_source) for rendering."""
    folder = resolve_incipt_folder(job, primus_dir)
    incipit_id = job["incipit_id"]
    sem_orig = read_primus_semantic(incipit_id, folder)
    agnostic = read_primus_agnostic(incipit_id, folder, DEFAULT_PRIMUS_TGZ)
    seed = int(job["corruption_seed"])
    fraction = float(row.get("corrupt_fraction", 0.4))
    target = _norm_semantic(semantic_gt)

    rng = random.Random(seed)
    rng.uniform(0.33, 0.5)
    replayed, mash = corrupt_tokens(sem_orig, agnostic, rng, fraction)
    if _norm_semantic(replayed) == target:
        return mash_pairs_for_semantic(semantic_gt, mash), "replay_rng"

    rng2 = random.Random(seed)
    replayed2, mash2 = corrupt_tokens(sem_orig, agnostic, rng2, fraction)
    if _norm_semantic(replayed2) == target:
        return mash_pairs_for_semantic(semantic_gt, mash2), "replay_direct"

    return None, "replay_miss"


def render_beamed_job(
    job: dict,
    row: dict,
    corrupt_dir: Path,
    out_dir: Path,
    primus_dir: Path,
) -> dict:
    corrupt_id = job["corrupt_id"]
    fmt = job["format"]
    if fmt not in {"clean", "camera"}:
        raise ValueError(f"unsupported format: {fmt}")

    semantic_path = corrupt_dir / "semantic" / f"{corrupt_id}.semantic"
    if not semantic_path.is_file():
        raise FileNotFoundError(f"missing semantic GT: {semantic_path}")
    semantic_gt = semantic_path.read_text(encoding="utf-8")

    mash_pairs, replay_source = replay_mashcima(job, row, semantic_gt, primus_dir)
    has_agnostic_beams = bool(
        mash_pairs and any(is_beamed_mash(mash) for _, mash in mash_pairs)
    )
    clean = render_semantic_to_image(semantic_gt, mash_pairs=mash_pairs, autobeam=True)
    if has_agnostic_beams:
        beam_source = "agnostic_mash"
    elif mash_pairs is not None:
        beam_source = f"semantic_autobeam+{replay_source}"
    else:
        beam_source = "semantic_autobeam"

    if fmt == "clean":
        img = clean
    else:
        seed = int(job["corruption_seed"])
        img = camera_like_distort(clean, seed=seed + 99)

    out_path = out_dir / fmt / f"{corrupt_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_png(img, out_path)
    compress_image_file(out_path)
    final_path = resolved_image_path(out_path)

    return {
        "corrupt_id": corrupt_id,
        "format": fmt,
        "incipit_id": job.get("incipit_id"),
        "status": "ok",
        "beam_source": beam_source,
        "png_path": str(final_path.relative_to(ROOT)).replace("\\", "/"),
    }


def _worker(payload: tuple[dict, dict, str, str, str]) -> dict:
    job, row, corrupt_dir_s, out_dir_s, primus_dir_s = payload
    try:
        return render_beamed_job(
            job,
            row,
            Path(corrupt_dir_s),
            Path(out_dir_s),
            Path(primus_dir_s),
        )
    except Exception as exc:
        msg = str(exc).strip() or type(exc).__name__
        return {
            "corrupt_id": job["corrupt_id"],
            "format": job.get("format"),
            "incipit_id": job.get("incipit_id"),
            "status": "error",
            "error": msg,
        }
    finally:
        close_render_pool()


def _kill_pool(pool_exec: ProcessPoolExecutor) -> None:
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


def defer_job(job: dict) -> None:
    DEFERRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFERRED_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(job) + "\n")


def collect_pending(
    formats: set[str],
    jobs_by_id: dict[str, dict],
    manifest_rows: dict[str, dict],
    corrupt_dir: Path,
    out_dir: Path,
    done_ids: set[str],
    force: bool = False,
) -> list[tuple[dict, dict]]:
    pending: list[tuple[dict, dict]] = []
    for corrupt_id, row in manifest_rows.items():
        if corrupt_id in done_ids and not force:
            continue
        fmt = row.get("format")
        if fmt not in formats:
            continue
        if row.get("status") != "ok":
            continue
        job = jobs_by_id.get(corrupt_id)
        if job is None:
            continue
        semantic_path = corrupt_dir / "semantic" / f"{corrupt_id}.semantic"
        if not semantic_path.is_file():
            continue
        if not force and output_image_path(out_dir, fmt, corrupt_id) is not None:
            continue
        pending.append((job, row))
    pending.sort(key=lambda item: item[0]["corrupt_id"])
    return pending


def run_regenerate(
    corrupt_dir: Path,
    out_dir: Path,
    primus_dir: Path,
    jobs_path: Path,
    source_manifest: Path,
    manifest_path: Path,
    formats: set[str],
    workers: int,
    job_timeout: float,
    limit: int | None,
    force: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        (out_dir / fmt).mkdir(parents=True, exist_ok=True)

    jobs_by_id = {j["corrupt_id"]: j for j in load_jobs(jobs_path)}
    manifest_rows = load_manifest_records(source_manifest)
    done_ids = load_done_ids(manifest_path)
    pending = collect_pending(formats, jobs_by_id, manifest_rows, corrupt_dir, out_dir, done_ids, force=force)
    if limit is not None:
        pending = pending[:limit]

    total_target = sum(
        1
        for _cid, row in manifest_rows.items()
        if row.get("status") == "ok" and row.get("format") in formats
    )
    skipped = total_target - len(pending)

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_line(
            f"START pending={len(pending)} skipped_existing={skipped} "
            f"total_ok={total_target} formats={','.join(sorted(formats))} workers={workers}",
            log_file,
        )

        if not pending:
            log_line("Nothing to do — all jobs already rendered in output folder.", log_file)
            return

        ok_count = 0
        err_count = 0
        timeout_count = 0
        processed = 0

        payloads = [
            (job, row, str(corrupt_dir.resolve()), str(out_dir.resolve()), str(primus_dir.resolve()))
            for job, row in pending
        ]

        with manifest_path.open("a", encoding="utf-8") as manifest:
            with tqdm(total=len(payloads), desc="Beamed corrupt regen", unit="job") as pbar:
                pending_payloads = list(payloads)
                active: dict = {}
                pool_exec = ProcessPoolExecutor(max_workers=workers)

                def write_progress() -> None:
                    log_line(
                        f"PROGRESS processed={processed} ok={ok_count} err={err_count} "
                        f"timeout={timeout_count} remaining={len(pending_payloads) + len(active)} "
                        f"total_target={total_target}",
                        log_file,
                    )

                while pending_payloads or active:
                    while pending_payloads and len(active) < workers:
                        payload = pending_payloads.pop(0)
                        try:
                            future = pool_exec.submit(_worker, payload)
                        except BrokenProcessPool:
                            pending_payloads.insert(0, payload)
                            _kill_pool(pool_exec)
                            for _f, (p, _s) in list(active.items()):
                                pending_payloads.append(p)
                            active.clear()
                            pool_exec = ProcessPoolExecutor(max_workers=workers)
                            break
                        active[future] = (payload, time.monotonic())

                    if not active:
                        if pending_payloads:
                            continue
                        break

                    done, _ = wait(set(active.keys()), timeout=5, return_when=FIRST_COMPLETED)
                    broke = False
                    for future in done:
                        payload, _started = active.pop(future)
                        try:
                            record = future.result()
                        except BrokenProcessPool:
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

                        manifest.write(json.dumps(record) + "\n")
                        manifest.flush()
                        processed += 1
                        if record.get("status") == "ok":
                            ok_count += 1
                        else:
                            err_count += 1
                        pbar.update(1)
                        if processed % 25 == 0 or processed == len(payloads):
                            write_progress()

                    if broke:
                        for _f, (p, _s) in list(active.items()):
                            pending_payloads.append(p)
                        active.clear()
                        _kill_pool(pool_exec)
                        pool_exec = ProcessPoolExecutor(max_workers=workers)
                        continue

                    now = time.monotonic()
                    timed_out = [
                        f
                        for f, (payload, started) in active.items()
                        if now - started > job_timeout
                    ]
                    if timed_out:
                        for future in timed_out:
                            payload, _started = active.pop(future)
                            job = payload[0]
                            defer_job(job)
                            record = {
                                "corrupt_id": job["corrupt_id"],
                                "format": job.get("format"),
                                "incipit_id": job.get("incipit_id"),
                                "status": "error",
                                "error": f"job_timeout>{job_timeout:.0f}s",
                            }
                            manifest.write(json.dumps(record) + "\n")
                            manifest.flush()
                            processed += 1
                            timeout_count += 1
                            err_count += 1
                            pbar.update(1)
                        for _f, (p, _s) in list(active.items()):
                            pending_payloads.append(p)
                        active.clear()
                        _kill_pool(pool_exec)
                        pool_exec = ProcessPoolExecutor(max_workers=workers)
                        write_progress()

                _kill_pool(pool_exec)

        log_line(
            f"DONE processed={processed} ok={ok_count} err={err_count} timeout={timeout_count} "
            f"output={out_dir}",
            log_file,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrupt-dir", type=Path, default=CORRUPT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--primus-dir", type=Path, default=DEFAULT_PRIMUS_DIR)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--formats",
        default="clean,camera",
        help="comma-separated formats to regenerate (default: clean,camera)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--job-timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-render even when output image already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = {f.strip() for f in args.formats.split(",") if f.strip()}
    unknown = formats - {"clean", "camera"}
    if unknown:
        raise SystemExit(f"unsupported format(s): {sorted(unknown)}")
    run_regenerate(
        corrupt_dir=args.corrupt_dir,
        out_dir=args.out_dir,
        primus_dir=args.primus_dir,
        jobs_path=args.jobs,
        source_manifest=args.source_manifest,
        manifest_path=args.manifest,
        formats=formats,
        workers=args.workers,
        job_timeout=args.job_timeout,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
