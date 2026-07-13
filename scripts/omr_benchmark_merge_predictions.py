"""Merge per-batch OMR prediction jsonl files into one model file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "examples" / "omr_benchmark"

MODELS = [
    "claude-opus-4-8-thinking-high",
    "gemini-3.1-pro",
    "composer-2.5",
    "grok-4.5-xhigh",
    "gpt-5.6-sol-medium",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=BENCH_DIR / "predictions_100",
    )
    args = parser.parse_args()

    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        model_dir = args.predictions_dir / model
        out_path = args.predictions_dir / f"{model}.jsonl"
        lines: list[str] = []
        if model_dir.exists():
            for batch_file in sorted(model_dir.glob("batch_*.jsonl")):
                for line in batch_file.read_text(encoding="utf-8-sig").splitlines():
                    line = line.strip().lstrip("\ufeff")
                    if line:
                        lines.append(line)
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"{model}: {len(lines)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
