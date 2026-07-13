"""Wait for dataset compression to finish, then restart batch generators."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "data" / "batch_restart.log"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
COMPRESS_SCRIPT = ROOT / "scripts" / "compress_dataset_images.py"
BATCH_COMMANDS = [
    [str(VENV_PYTHON), str(ROOT / "scripts" / "demo_mashcima_camera_distort.py"), "--all"],
    [str(VENV_PYTHON), str(ROOT / "scripts" / "generate_corrupt_batch.py"), "--total", "35000"],
]


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{stamp}] {msg}\n"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="")


def is_compress_running() -> bool:
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                [
                    "wmic",
                    "process",
                    "where",
                    "name='python.exe'",
                    "get",
                    "CommandLine",
                ],
                text=True,
                errors="replace",
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
        return "compress_dataset_images.py" in out

    try:
        out = subprocess.check_output(["ps", "aux"], text=True, errors="replace")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return "compress_dataset_images.py" in out


def is_batch_running(script_name: str) -> bool:
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                [
                    "wmic",
                    "process",
                    "where",
                    "name='python.exe'",
                    "get",
                    "CommandLine",
                ],
                text=True,
                errors="replace",
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
        return script_name in out

    try:
        out = subprocess.check_output(["ps", "aux"], text=True, errors="replace")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return script_name in out


def start_batches() -> None:
    script_names = [
        "demo_mashcima_camera_distort.py",
        "generate_corrupt_batch.py",
    ]
    for cmd, script_name in zip(BATCH_COMMANDS, script_names):
        if is_batch_running(script_name):
            log(f"Already running, skipped: {script_name}")
            continue
        log(f"Starting: {' '.join(cmd)}")
        subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )


def wait_for_compress(poll_seconds: float) -> None:
    log("Watcher started; waiting for compress_dataset_images.py to exit")
    while is_compress_running():
        time.sleep(poll_seconds)
    log("Compression process no longer running")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="How often to check whether compression is still running",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Start batches immediately without waiting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not VENV_PYTHON.is_file():
        log(f"ERROR: missing venv python at {VENV_PYTHON}")
        sys.exit(1)

    if args.force or not is_compress_running():
        if args.force:
            log("Force mode: starting batches now")
        else:
            log("No compression process detected; starting batches now")
        start_batches()
        return

    wait_for_compress(args.poll_seconds)
    start_batches()


if __name__ == "__main__":
    main()
