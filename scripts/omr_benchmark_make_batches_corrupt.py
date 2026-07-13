"""Split corrupt OMR benchmark selection into batch files for transcription."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "examples" / "omr_benchmark"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=BENCH_DIR / "selection_10_corrupt.json",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCH_DIR / "batches_10_corrupt",
    )
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch_idx = 0
    for i in range(0, len(selection), args.batch_size):
        batch = selection[i : i + args.batch_size]
        out = args.output_dir / f"batch_{batch_idx:02d}.json"
        out.write_text(json.dumps(batch, indent=2), encoding="utf-8")
        batch_idx += 1

    print(f"Created {batch_idx} batches in {args.output_dir}")


if __name__ == "__main__":
    main()
