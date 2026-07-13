"""Log numerical progress for compress + batch jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG = DATA / "progress_monitor.log"

COMPRESS_TARGET = 173_005
CORRUPT_TARGET = 35_000


def _read_compress_target() -> int:
    run_log = DATA / "compress_dataset_run.log"
    if not run_log.is_file():
        return COMPRESS_TARGET
    text = run_log.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "total:" in line.lower() and "files" in line.lower():
            parts = line.replace(",", "").split()
            for i, part in enumerate(parts):
                if part.isdigit() and i > 0 and parts[i - 1].endswith(":"):
                    return int(part)
    return COMPRESS_TARGET


def _mashcima_target() -> int:
    manifest = DATA / "mashcima" / "manifest.jsonl"
    if manifest.is_file():
        ids: set[str] = set()
        with manifest.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("status") == "ok" and row.get("incipit_id"):
                    ids.add(row["incipit_id"])
        if ids:
            return len(ids)
    pngs = list((DATA / "mashcima").glob("*.png"))
    return len(pngs) if pngs else 36_274


def manifest_counts(path: Path, id_key: str) -> tuple[int, int, int]:
    seen: dict[str, str] = {}
    if not path.is_file():
        return 0, 0, 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get(id_key)
            if key:
                seen[key] = row.get("status", "error")
    ok = sum(1 for st in seen.values() if st == "ok")
    err = len(seen) - ok
    return ok, err, len(seen)


def compress_done() -> int:
    resume = DATA / "compress_dataset_resume.jsonl"
    if not resume.is_file():
        return 0
    return sum(1 for line in resume.open(encoding="utf-8") if line.strip())


def running_pids(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    f"Where-Object {{ $_.CommandLine -match '{pattern}' }} | "
                    "Select-Object -ExpandProperty ProcessId"
                ),
            ],
            text=True,
            timeout=45,
        )
        return [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
    except (subprocess.SubprocessError, ValueError, OSError):
        return []


def job_status() -> dict:
    """Return structured progress for all dataset jobs."""
    compress_target = _read_compress_target()
    mashcima_target = _mashcima_target()
    comp_done = compress_done()
    cam_ok, cam_err, cam_total = manifest_counts(
        DATA / "mashcima" / "camera_distort_manifest.jsonl", "incipit_id"
    )
    cor_ok, cor_err, cor_total = manifest_counts(DATA / "corrupt" / "manifest.jsonl", "corrupt_id")
    comp_pids = running_pids("compress_dataset_images")
    cam_pids = running_pids("demo_mashcima_camera_distort")
    cor_pids = running_pids("generate_corrupt_batch")
    return {
        "compress": {
            "done": comp_done,
            "target": compress_target,
            "pct": 100.0 * comp_done / compress_target if compress_target else 0.0,
            "running": bool(comp_pids),
            "pids": comp_pids,
        },
        "camera_distort": {
            "ok": cam_ok,
            "error": cam_err,
            "ids": cam_total,
            "target": mashcima_target,
            "pct": 100.0 * cam_ok / mashcima_target if mashcima_target else 0.0,
            "running": bool(cam_pids),
            "pids": cam_pids,
        },
        "corrupt_batch": {
            "ok": cor_ok,
            "error": cor_err,
            "ids": cor_total,
            "target": CORRUPT_TARGET,
            "pct": 100.0 * cor_ok / CORRUPT_TARGET if CORRUPT_TARGET else 0.0,
            "running": bool(cor_pids),
            "pids": cor_pids,
        },
    }


def all_jobs_complete(threshold: float = 0.99) -> bool:
    """True when no job processes run and ok counts meet targets."""
    s = job_status()
    if s["compress"]["running"] or s["camera_distort"]["running"] or s["corrupt_batch"]["running"]:
        return False
    if s["compress"]["done"] < s["compress"]["target"] * threshold:
        return False
    if s["camera_distort"]["ok"] < s["camera_distort"]["target"] * threshold:
        return False
    if s["corrupt_batch"]["ok"] < s["corrupt_batch"]["target"] * threshold:
        return False
    return True


def format_snapshot() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    s = job_status()
    return "\n".join(
        [
            f"[{ts}] Job progress snapshot",
            (
                f"  compress:        {s['compress']['done']:>7}/{s['compress']['target']} "
                f"({s['compress']['pct']:5.1f}%)  running={s['compress']['running']} "
                f"pids={s['compress']['pids']}"
            ),
            (
                f"  camera_distort:  ok={s['camera_distort']['ok']:>6} "
                f"err={s['camera_distort']['error']:>5} "
                f"ids={s['camera_distort']['ids']:>6} "
                f"({s['camera_distort']['pct']:5.1f}% of {s['camera_distort']['target']})  "
                f"running={s['camera_distort']['running']} pids={s['camera_distort']['pids']}"
            ),
            (
                f"  corrupt_batch:   ok={s['corrupt_batch']['ok']:>6} "
                f"err={s['corrupt_batch']['error']:>5} "
                f"ids={s['corrupt_batch']['ids']:>6} "
                f"({s['corrupt_batch']['pct']:5.1f}% of {s['corrupt_batch']['target']})  "
                f"running={s['corrupt_batch']['running']} pids={s['corrupt_batch']['pids']}"
            ),
            f"  all_jobs_complete={all_jobs_complete()}",
        ]
    )


def emit() -> None:
    text = format_snapshot()
    print(text, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(text + "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor dataset job progress.")
    parser.add_argument(
        "--loop-minutes",
        type=float,
        default=5,
        help="Repeat every N minutes (0 = run once).",
    )
    args = parser.parse_args()

    emit()
    if args.loop_minutes > 0:
        interval = args.loop_minutes * 60
        while True:
            time.sleep(interval)
            emit()


if __name__ == "__main__":
    main()
