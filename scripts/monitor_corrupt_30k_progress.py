"""Tail corrupt_30k progress.log / progress.json on an interval."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "corrupt_30k"


def emit(output_dir: Path) -> None:
    log_path = output_dir / "progress.log"
    json_path = output_dir / "progress.json"
    if log_path.is_file():
        print(log_path.read_text(encoding="utf-8"), end="", flush=True)
    elif json_path.is_file():
        print(json.dumps(json.loads(json_path.read_text(encoding="utf-8")), indent=2), flush=True)
    else:
        print(f"No progress file yet in {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    emit(args.output_dir)
    if args.interval_seconds > 0:
        while True:
            time.sleep(args.interval_seconds)
            emit(args.output_dir)


if __name__ == "__main__":
    main()
