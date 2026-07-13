"""Run full per-score handwritten regeneration with file logging."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LOG_DIR = ROOT / "data" / "logs"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)


def run_step(name: str, args: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"=== {name} ===\n"
        f"started: {datetime.now(timezone.utc).isoformat()}\n"
        f"command: {PYTHON} {' '.join(args)}\n\n"
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        proc = subprocess.Popen(
            [str(PYTHON), "-u", *args],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                sys.stdout.write(line)
            except UnicodeEncodeError:
                sys.stdout.buffer.write(line.encode(sys.stdout.encoding or "utf-8", errors="replace"))
                sys.stdout.buffer.flush()
            log.write(line)
        code = proc.wait()
        log.write(f"\nexit_code: {code}\n")
        log.write(f"finished: {datetime.now(timezone.utc).isoformat()}\n")
    print(f"[{name}] exit {code} -> {log_path}", flush=True)
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-mashcima",
        action="store_true",
        help="Skip mashcima batch (~33k images)",
    )
    parser.add_argument(
        "--skip-corrupt",
        action="store_true",
        help="Skip corrupt handwritten regenerate",
    )
    parser.add_argument(
        "--skip-golden",
        action="store_true",
        help="Skip golden_set handwritten (20 images)",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = LOG_DIR / f"regenerate_handwritten_{stamp}.summary.log"
    steps: list[tuple[str, list[str], Path]] = []

    if not args.skip_mashcima:
        steps.append(
            (
                "mashcima_batch",
                [
                    str(SCRIPTS / "generate_mashcima_batch.py"),
                    "--writers",
                    "per-score",
                    "--no-resume",
                    "--manifest",
                    str(ROOT / "data" / "mashcima" / "manifest.jsonl"),
                ],
                LOG_DIR / f"regenerate_handwritten_{stamp}_01_mashcima.log",
            )
        )
    if not args.skip_corrupt:
        steps.append(
            (
                "corrupt_handwritten",
                [
                    str(SCRIPTS / "generate_corrupt_batch.py"),
                    "--writers",
                    "per-score",
                    "--no-resume",
                    "--regenerate-formats",
                    "handwritten",
                ],
                LOG_DIR / f"regenerate_handwritten_{stamp}_02_corrupt.log",
            )
        )
    if not args.skip_golden:
        steps.append(
            (
                "golden_handwritten",
                [
                    str(SCRIPTS / "regenerate_golden_handwritten.py"),
                    "--writers",
                    "per-score",
                ],
                LOG_DIR / f"regenerate_handwritten_{stamp}_03_golden.log",
            )
        )

    with summary.open("w", encoding="utf-8") as f:
        f.write(f"regenerate_all_handwritten_per_score started {stamp}\n")
        for name, cmd_args, log_path in steps:
            f.write(f"  {name}: {log_path.name}\n")

    failed = False
    for name, cmd_args, log_path in steps:
        code = run_step(name, cmd_args, log_path)
        with summary.open("a", encoding="utf-8") as f:
            f.write(f"{name}: exit {code}\n")
        if code != 0:
            failed = True
            print(f"Stopping after failed step: {name}", flush=True)
            break

    post_regen_args = [
        str(SCRIPTS / "post_regen_handwritten_camera_distort.py"),
    ]
    post_log = LOG_DIR / f"regenerate_handwritten_{stamp}_04_camera_distort.log"
    if not failed:
        post_code = run_step("camera_distort_and_consolidated", post_regen_args, post_log)
        with summary.open("a", encoding="utf-8") as f:
            f.write(f"camera_distort_and_consolidated: exit {post_code}\n")
        if post_code != 0:
            failed = True

    with summary.open("a", encoding="utf-8") as f:
        f.write("done\n" if not failed else "failed\n")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
