"""Wipe OMR benchmark artifacts for a fresh run."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "examples" / "omr_benchmark"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-selection", action="store_true")
    args = parser.parse_args()

    if BENCH.is_dir():
        for name in ("predictions_100", "batches", "images"):
            path = BENCH / name
            if path.exists():
                shutil.rmtree(path)
        for name in ("scores.md", "scores_100.json", "selection_100.json"):
            path = BENCH / name
            if path.exists() and not (args.keep_selection and name == "selection_100.json"):
                path.unlink()
    else:
        BENCH.mkdir(parents=True, exist_ok=True)

    print(f"Reset OMR benchmark dir: {BENCH}")


if __name__ == "__main__":
    main()
