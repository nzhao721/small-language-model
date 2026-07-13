"""Poll job progress; when all dataset jobs finish, prepare fresh OMR benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from monitor_job_progress import all_jobs_complete, job_status  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG = DATA / "omr_benchmark_deferred.log"
READY_FLAG = DATA / "omr_benchmark_ready.json"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_script(name: str, *args: str) -> None:
    cmd = [str(PYTHON), str(_SCRIPTS / name), *args]
    log(f"RUN {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


def prepare_fresh_benchmark() -> None:
    run_script("omr_benchmark_reset.py")
    run_script("omr_benchmark_select.py", "--n", "100")
    run_script("omr_benchmark_make_batches.py")
    READY_FLAG.write_text(
        json.dumps(
            {
                "ready_at": datetime.now(timezone.utc).isoformat(),
                "selection": str(ROOT / "examples" / "omr_benchmark" / "selection_100.json"),
                "batches_dir": str(ROOT / "examples" / "omr_benchmark" / "batches"),
                "predictions_dir": str(ROOT / "examples" / "omr_benchmark" / "predictions_100"),
                "models": [
                    "claude-opus-4-8-thinking-high",
                    "gemini-3.1-pro",
                    "composer-2.5",
                    "grok-4.5-xhigh",
                    "gpt-5.6-sol-medium",
                ],
                "note": "Fresh 100-sample OMR benchmark prepared; run transcription subagents.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"READY flag written: {READY_FLAG}")
    print("OMR_BENCHMARK_READY", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-minutes", type=float, default=5.0)
    parser.add_argument("--prepare-now", action="store_true", help="Skip wait; reset and prepare.")
    args = parser.parse_args()

    if args.prepare_now:
        prepare_fresh_benchmark()
        return

    log("Waiting for compress + camera_distort + corrupt_batch to complete...")
    while True:
        s = job_status()
        log(
            "poll: compress={:.1f}% camera={:.1f}% corrupt={:.1f}% running=[{},{},{}]".format(
                s["compress"]["pct"],
                s["camera_distort"]["pct"],
                s["corrupt_batch"]["pct"],
                s["compress"]["running"],
                s["camera_distort"]["running"],
                s["corrupt_batch"]["running"],
            )
        )
        if all_jobs_complete():
            log("All jobs complete — preparing fresh OMR benchmark.")
            prepare_fresh_benchmark()
            return
        time.sleep(args.poll_minutes * 60)


if __name__ == "__main__":
    main()
