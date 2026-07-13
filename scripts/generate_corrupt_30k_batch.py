"""Generate 30k corrupt excerpts into a new folder (primus-only, golden excluded).

Uses the same corruption/render pipeline as generate_corrupt_batch.py but:
- samples incipits from data/primus/ground_truth (golden excluded)
- uses mashcima-manifest-ok incipits only (same convertible pool as generate_corrupt_batch)
- retries on failure until per-format ok targets are met
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_corrupt_batch import (  # noqa: E402
    DEFAULT_MUSCIMA_DIR,
    DEFAULT_PRIMUS_TGZ,
    MASHCIMA_MANIFEST,
    _configure_playwright_browsers_path,
    _configure_stdio_utf8,
    _init_pool,
    _kill_pool,
    build_primus_index,
    ensure_muscima_cache,
    load_convertible_incipt_ids,
    process_job,
)
from generate_mashcima_batch import DEFAULT_PRIMUS_DIR  # noqa: E402
from mashcima_writers import DEFAULT_WRITER_MODE, configure_mashcima_writers  # noqa: E402
from remove_golden_from_consolidated import build_exclusion_set  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "corrupt_30k"
DEFAULT_TARGETS = {"handwritten": 15_000, "camera": 9_000, "clean": 6_000}
PROGRESS_LOG = "progress.log"
PROGRESS_JSON = "progress.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_primus_pool(primus_dir: Path, exclude: set[str]) -> list[str]:
    """PrIMuS incipits with semantic+agnostic that convert cleanly, minus golden."""
    index = build_primus_index(primus_dir, mashcima_manifest=Path("__skip_mashcima__"))
    convertible = set(load_convertible_incipt_ids(MASHCIMA_MANIFEST, index))
    pool = sorted(
        incipit_id
        for incipit_id in convertible
        if incipit_id in index and incipit_id not in exclude
    )
    if not pool:
        raise RuntimeError("No convertible primus incipits left after golden exclusion")
    return pool


def golden_incipit_exclusions(primus_ids: set[str]) -> set[str]:
    """Golden keys that correspond to primus incipit IDs."""
    excl, _ = build_exclusion_set()
    return {key for key in excl if key in primus_ids}


def load_ok_counts_by_format(manifest: Path) -> dict[str, int]:
    counts = {fmt: 0 for fmt in DEFAULT_TARGETS}
    if not manifest.is_file():
        return counts
    seen_ok: set[str] = set()
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            corrupt_id = row.get("corrupt_id")
            if not corrupt_id or corrupt_id in seen_ok:
                continue
            if row.get("status") != "ok":
                continue
            fmt = row.get("format")
            if fmt in counts:
                counts[fmt] += 1
                seen_ok.add(corrupt_id)
    return counts


def load_attempt_stats(manifest: Path) -> tuple[int, int]:
    ok = err = 0
    if not manifest.is_file():
        return ok, err
    latest: dict[str, str] = {}
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            latest[row["corrupt_id"]] = row.get("status", "error")
    for status in latest.values():
        if status == "ok":
            ok += 1
        else:
            err += 1
    return ok, err


def timeout_for_format(fmt: str, digital_timeout: float, handwritten_timeout: float) -> float:
    if fmt == "handwritten":
        return handwritten_timeout
    return digital_timeout


def pick_format(rng: random.Random, remaining: dict[str, int]) -> str:
    need = {fmt: n for fmt, n in remaining.items() if n > 0}
    total = sum(need.values())
    roll = rng.uniform(0, total)
    upto = 0.0
    for fmt, n in need.items():
        upto += n
        if roll <= upto:
            return fmt
    return next(iter(need))


def make_job(rng: random.Random, pool: list[str], fmt: str) -> dict:
    return {
        "corrupt_id": uuid.uuid4().hex,
        "format": fmt,
        "incipit_id": rng.choice(pool),
        "corruption_seed": rng.randint(0, 2_147_483_647),
    }


def _cleanup_failed(corrupt_dir: Path, corrupt_id: str, fmt: str) -> None:
    for path in (
        corrupt_dir / "semantic" / f"{corrupt_id}.semantic",
        corrupt_dir / fmt / f"{corrupt_id}.png",
        corrupt_dir / fmt / f"{corrupt_id}.jpg",
    ):
        if path.is_file():
            path.unlink()


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
    _cleanup_failed(corrupt_dir, job["corrupt_id"], job["format"])
    msg = str(last_exc).strip() or type(last_exc).__name__
    return {
        "corrupt_id": job["corrupt_id"],
        "format": job.get("format"),
        "incipit_id": job.get("incipit_id"),
        "status": "error",
        "error": msg,
    }


def write_progress(
    output_dir: Path,
    ok_counts: dict[str, int],
    targets: dict[str, int],
    attempts_ok: int,
    attempts_err: int,
    running: int,
) -> None:
    total_ok = sum(ok_counts.values())
    total_target = sum(targets.values())
    lines = [
        f"[{utc_now()}] corrupt_30k progress",
        f"  total ok: {total_ok}/{total_target} ({100.0 * total_ok / total_target:.1f}%)",
    ]
    for fmt in ("handwritten", "camera", "clean"):
        ok = ok_counts[fmt]
        tgt = targets[fmt]
        lines.append(
            f"  {fmt:12} ok={ok:>6}/{tgt} ({100.0 * ok / tgt:.1f}%)"
        )
    lines.append(f"  attempts: ok={attempts_ok} error={attempts_err} running={running}")
    text = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PROGRESS_LOG).write_text(text + "\n", encoding="utf-8")
    (output_dir / PROGRESS_JSON).write_text(
        json.dumps(
            {
                "updated_at": utc_now(),
                "ok_counts": ok_counts,
                "targets": targets,
                "total_ok": total_ok,
                "total_target": total_target,
                "attempts_ok": attempts_ok,
                "attempts_err": attempts_err,
                "running": running,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(text, flush=True)


def run_batch(
    *,
    primus_dir: Path,
    primus_tgz: Path,
    output_dir: Path,
    targets: dict[str, int],
    workers: int,
    plan_seed: int,
    job_timeout: float,
    handwritten_timeout: float,
    writer_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    ensure_muscima_cache(primus_tgz, DEFAULT_MUSCIMA_DIR)
    configure_mashcima_writers(writer_mode)

    primus_index = build_primus_index(primus_dir, mashcima_manifest=Path("__skip_mashcima__"))
    exclude = golden_incipit_exclusions(set(primus_index))
    pool = build_primus_pool(primus_dir, exclude)
    print(f"Primus pool: {len(pool):,} incipits (excluded {len(exclude):,} golden)", flush=True)

    rng = random.Random(plan_seed)
    ok_counts = load_ok_counts_by_format(manifest_path)
    attempts_ok, attempts_err = load_attempt_stats(manifest_path)

    remaining = {fmt: max(0, targets[fmt] - ok_counts[fmt]) for fmt in targets}
    in_flight = {fmt: 0 for fmt in targets}
    if sum(remaining.values()) == 0:
        print("All targets already met.", flush=True)
        write_progress(output_dir, ok_counts, targets, attempts_ok, attempts_err, 0)
        return

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
        )

    write_progress(output_dir, ok_counts, targets, attempts_ok, attempts_err, 0)

    with manifest_path.open("a", encoding="utf-8") as manifest:
        pool_exec = _make_pool()
        active: dict = {}

        try:
            while sum(remaining.values()) > 0 or active:
                while sum(remaining.values()) > 0 and len(active) < workers:
                    need = {
                        fmt: remaining[fmt] - in_flight[fmt]
                        for fmt in remaining
                        if remaining[fmt] > in_flight[fmt]
                    }
                    if not need:
                        break
                    fmt = pick_format(rng, need)
                    job = make_job(rng, pool, fmt)
                    in_flight[fmt] += 1
                    payload = (job, str(primus_dir.resolve()), str(output_dir.resolve()))
                    try:
                        future = pool_exec.submit(_worker, payload)
                    except BrokenProcessPool:
                        in_flight[fmt] -= 1
                        _kill_pool(pool_exec)
                        active.clear()
                        pool_exec = _make_pool()
                        break
                    active[future] = (job, time.monotonic(), timeout_for_format(fmt, job_timeout, handwritten_timeout))

                if not active:
                    continue

                done, _ = wait(set(active.keys()), timeout=5, return_when=FIRST_COMPLETED)
                timed_out: list = []

                for future in done:
                    job, started, job_limit = active.pop(future)
                    try:
                        record = future.result()
                    except BrokenProcessPool:
                        if job.get("format") in in_flight and in_flight[job["format"]] > 0:
                            in_flight[job["format"]] -= 1
                        for _f, (pending_job, _started, _limit) in list(active.items()):
                            pfmt = pending_job.get("format")
                            if pfmt in in_flight and in_flight[pfmt] > 0:
                                in_flight[pfmt] -= 1
                        active.clear()
                        _kill_pool(pool_exec)
                        pool_exec = _make_pool()
                        break
                    except Exception as exc:
                        record = {
                            "corrupt_id": job["corrupt_id"],
                            "format": job.get("format"),
                            "incipit_id": job.get("incipit_id"),
                            "status": "error",
                            "error": str(exc).strip() or type(exc).__name__,
                        }

                    manifest.write(json.dumps(record) + "\n")
                    manifest.flush()

                    fmt = record.get("format")
                    if fmt in in_flight and in_flight[fmt] > 0:
                        in_flight[fmt] -= 1

                    if record.get("status") == "ok":
                        attempts_ok += 1
                        fmt = record.get("format")
                        if fmt in remaining and remaining[fmt] > 0:
                            ok_counts[fmt] += 1
                            remaining[fmt] -= 1
                    else:
                        attempts_err += 1

                    write_progress(
                        output_dir,
                        ok_counts,
                        targets,
                        attempts_ok,
                        attempts_err,
                        len(active),
                    )
                    continue

                now = time.monotonic()
                for future, (job, started, job_limit) in list(active.items()):
                    if now - started > job_limit:
                        timed_out.append(future)

                if timed_out:
                    for future in timed_out:
                        job, _, job_limit = active.pop(future)
                        fmt = job.get("format")
                        if fmt in in_flight and in_flight[fmt] > 0:
                            in_flight[fmt] -= 1
                        record = {
                            "corrupt_id": job["corrupt_id"],
                            "format": job.get("format"),
                            "incipit_id": job.get("incipit_id"),
                            "status": "error",
                            "error": f"job_timeout>{job_limit:.0f}s",
                        }
                        manifest.write(json.dumps(record) + "\n")
                        manifest.flush()
                        attempts_err += 1
                        _cleanup_failed(output_dir, job["corrupt_id"], job["format"])
                    for future, (job, _started, _limit) in list(active.items()):
                        fmt = job.get("format")
                        if fmt in in_flight and in_flight[fmt] > 0:
                            in_flight[fmt] -= 1
                    active.clear()
                    _kill_pool(pool_exec)
                    pool_exec = _make_pool()
                    write_progress(
                        output_dir,
                        ok_counts,
                        targets,
                        attempts_ok,
                        attempts_err,
                        len(active),
                    )
        finally:
            _kill_pool(pool_exec)

    write_progress(output_dir, ok_counts, targets, attempts_ok, attempts_err, 0)
    print(
        f"Finished: {sum(ok_counts.values()):,} ok across formats; "
        f"{attempts_err:,} scrapped errors -> {manifest_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primus-dir", type=Path, default=DEFAULT_PRIMUS_DIR)
    parser.add_argument("--primus-tgz", type=Path, default=DEFAULT_PRIMUS_TGZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-seed", type=int, default=2026_07_11)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(6, max(1, (os.cpu_count() or 4) - 1)),
        help="Parallel workers (capped low to reduce Playwright contention)",
    )
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for clean/camera jobs",
    )
    parser.add_argument(
        "--handwritten-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for handwritten jobs (mashcima render is slower)",
    )
    parser.add_argument("--writers", default=DEFAULT_WRITER_MODE)
    return parser.parse_args()


def main() -> None:
    _configure_stdio_utf8()
    _configure_playwright_browsers_path()
    args = parse_args()
    run_batch(
        primus_dir=args.primus_dir,
        primus_tgz=args.primus_tgz,
        output_dir=args.output_dir,
        targets=dict(DEFAULT_TARGETS),
        workers=args.workers,
        plan_seed=args.plan_seed,
        job_timeout=args.job_timeout,
        handwritten_timeout=args.handwritten_timeout,
        writer_mode=args.writers,
    )


if __name__ == "__main__":
    main()
