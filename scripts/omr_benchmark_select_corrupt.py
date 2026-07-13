"""Pick random corrupt OMR benchmark samples (10 per group) using manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "examples" / "omr_benchmark"
DEFAULT_SELECTION_PATH = OUT_DIR / "selection_10_corrupt.json"

GROUP_BY_FORMAT = {
    "handwritten": "handwritten_corrupt",
    "clean": "clean_corrupt",
    "camera": "camera_distorted_corrupt",
}


def load_corrupt_ok() -> dict[str, list[dict]]:
    path = ROOT / "data" / "corrupt" / "manifest.jsonl"
    last_ok: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            last_ok[row["corrupt_id"]] = row

    by_format: dict[str, list[dict]] = {"handwritten": [], "clean": [], "camera": []}
    for row in last_ok.values():
        fmt = row["format"]
        if fmt not in by_format:
            continue
        png_path = ROOT / row["png_path"]
        semantic_path = ROOT / row["semantic_path"]
        by_format[fmt].append(
            {
                "corrupt_id": row["corrupt_id"],
                "incipit_id": row["incipit_id"],
                "format": fmt,
                "png_path": png_path,
                "semantic_path": semantic_path,
            }
        )
    return by_format


def pick_n(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    valid = [r for r in pool if r["png_path"].exists() and r["semantic_path"].exists()]
    shuffled = valid[:]
    rng.shuffle(shuffled)
    if len(shuffled) < n:
        raise RuntimeError(f"Only found {len(shuffled)}/{n} valid corrupt samples")
    return shuffled[:n]


def build_selection(n_per_group: int = 10, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    corrupt = load_corrupt_ok()
    selections: list[dict] = []

    for fmt in ("handwritten", "clean", "camera"):
        group = GROUP_BY_FORMAT[fmt]
        picked = pick_n(corrupt[fmt], n_per_group, rng)
        for r in picked:
            selections.append(
                {
                    "group": group,
                    "corrupt_id": r["corrupt_id"],
                    "incipit_id": r["incipit_id"],
                    "filename": r["png_path"].name,
                    "image_path": str(r["png_path"].resolve()),
                    "ground_truth_semantic_path": str(r["semantic_path"].resolve()),
                }
            )
    return selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Samples per corrupt group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_SELECTION_PATH)
    args = parser.parse_args()

    selections = build_selection(args.n, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selections, indent=2), encoding="utf-8")

    from collections import Counter

    counts = Counter(s["group"] for s in selections)
    print(f"Wrote {len(selections)} entries to {args.output}")
    for group, count in sorted(counts.items()):
        print(f"  {group}: {count}")


if __name__ == "__main__":
    main()
