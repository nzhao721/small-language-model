"""Poll OMR benchmark rerun outputs; merge and score when all 50 batches are fresh."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
JOBS = ROOT / "examples" / "omr_benchmark" / "rerun_jobs.jsonl"
MARKER = ROOT / "data" / "omr_benchmark_rerun_started.json"
LOG = ROOT / "data" / "omr_benchmark_rerun_progress.log"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_jobs() -> list[dict]:
    return [
        json.loads(line)
        for line in JOBS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rerun_start_ts() -> float:
    if MARKER.is_file():
        data = json.loads(MARKER.read_text(encoding="utf-8"))
        started = data.get("started_at", "")
        if started:
            return datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
    return 0.0


def batch_status(jobs: list[dict], min_mtime: float) -> tuple[int, list[str]]:
    ok = 0
    issues: list[str] = []
    for j in jobs:
        path = ROOT / j["output"]
        if not path.is_file():
            issues.append(f"missing: {j['output']}")
            continue
        if path.stat().st_mtime < min_mtime:
            issues.append(f"stale: {j['output']}")
            continue
        n = len([ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()])
        if n != 10:
            issues.append(f"bad_lines({n}): {j['output']}")
            continue
        ok += 1
    return ok, issues


def run(cmd: list[str]) -> None:
    log("RUN " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    jobs = load_jobs()
    min_mtime = rerun_start_ts()
    log(f"Waiting for {len(jobs)} fresh batch files (mtime >= rerun start)...")

    while True:
        ok, issues = batch_status(jobs, min_mtime)
        log(f"progress: {ok}/{len(jobs)} complete")
        if ok == len(jobs):
            break
        if issues:
            log("pending sample: " + "; ".join(issues[:5]))
        time.sleep(120)

    for pred_dir in (
        "examples/omr_benchmark/predictions_10",
        "examples/omr_benchmark/predictions_10_corrupt",
        "examples/omr_benchmark/predictions_manual",
    ):
        run(
            [
                str(PYTHON),
                str(SCRIPTS / "omr_benchmark_merge_predictions.py"),
                "--predictions-dir",
                pred_dir,
            ]
        )

    run([str(PYTHON), str(SCRIPTS / "omr_benchmark_score.py"), "--all-benchmarks"])
    scores = ROOT / "examples" / "omr_benchmark" / "scores.md"
    log("DONE — scores written to " + str(scores))
    print(scores.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
