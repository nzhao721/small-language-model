"""Pick random OMR benchmark samples (100 per group) using manifests."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_layout import (  # noqa: E402
    camera_distorted_path,
    primus_png_path,
    primus_semantic_path,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "examples" / "omr_benchmark"
DEFAULT_SELECTION_PATH = OUT_DIR / "selection_100.json"

RNG = random.Random(42)


def load_mashcima_ok() -> list[dict]:
    path = ROOT / "data" / "mashcima" / "manifest.jsonl"
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            incipit_id = row["incipit_id"]
            rows.append(
                {
                    "incipit_id": incipit_id,
                    "mashcima_png": ROOT / row["output_png"],
                    "primus_png": primus_png_path(incipit_id),
                    "primus_semantic": primus_semantic_path(incipit_id),
                    "camera_distorted": camera_distorted_path(incipit_id),
                }
            )
    return rows


def load_corrupt_ok() -> dict[str, list[dict]]:
    path = ROOT / "data" / "corrupt" / "manifest.jsonl"
    by_format: dict[str, list[dict]] = {"handwritten": [], "clean": [], "camera": []}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            fmt = row["format"]
            if fmt not in by_format:
                continue
            by_format[fmt].append(
                {
                    "corrupt_id": row["corrupt_id"],
                    "incipit_id": row["incipit_id"],
                    "png_path": ROOT / row["png_path"],
                    "semantic_path": ROOT / row["semantic_path"],
                }
            )
    return by_format


def pick_n(
    pool: list[dict],
    n: int,
    validate,
) -> list[dict]:
    shuffled = pool[:]
    RNG.shuffle(shuffled)
    picked: list[dict] = []
    for item in shuffled:
        if len(picked) >= n:
            break
        if validate(item):
            picked.append(item)
    if len(picked) < n:
        raise RuntimeError(f"Only found {len(picked)}/{n} valid samples")
    return picked


def build_selection(
    n_per_group: int = 100,
    *,
    original_only: bool = False,
) -> list[dict]:
    mashcima = load_mashcima_ok()
    corrupt = load_corrupt_ok() if not original_only else None

    selections: list[dict] = []

    hw_orig = pick_n(
        mashcima,
        n_per_group,
        lambda r: r["mashcima_png"].exists() and r["primus_semantic"].exists(),
    )
    for r in hw_orig:
        selections.append(
            {
                "group": "handwritten_original",
                "filename": r["mashcima_png"].name,
                "image_path": str(r["mashcima_png"].resolve()),
                "ground_truth_semantic_path": str(r["primus_semantic"].resolve()),
            }
        )

    if not original_only:
        hw_corrupt = pick_n(
            corrupt["handwritten"],
            n_per_group,
            lambda r: r["png_path"].exists() and r["semantic_path"].exists(),
        )
        for r in hw_corrupt:
            selections.append(
                {
                    "group": "handwritten_corrupt",
                    "filename": r["png_path"].name,
                    "image_path": str(r["png_path"].resolve()),
                    "ground_truth_semantic_path": str(r["semantic_path"].resolve()),
                }
            )

    clean_orig = pick_n(
        mashcima,
        n_per_group,
        lambda r: r["primus_png"].exists() and r["primus_semantic"].exists(),
    )
    for r in clean_orig:
        selections.append(
            {
                "group": "clean_original",
                "filename": r["primus_png"].name,
                "image_path": str(r["primus_png"].resolve()),
                "ground_truth_semantic_path": str(r["primus_semantic"].resolve()),
            }
        )

    if not original_only:
        clean_corrupt = pick_n(
            corrupt["clean"],
            n_per_group,
            lambda r: r["png_path"].exists() and r["semantic_path"].exists(),
        )
        for r in clean_corrupt:
            selections.append(
                {
                    "group": "clean_corrupt",
                    "filename": r["png_path"].name,
                    "image_path": str(r["png_path"].resolve()),
                    "ground_truth_semantic_path": str(r["semantic_path"].resolve()),
                }
            )

    cam_orig = pick_n(
        mashcima,
        n_per_group,
        lambda r: r["camera_distorted"].exists() and r["primus_semantic"].exists(),
    )
    for r in cam_orig:
        selections.append(
            {
                "group": "camera_distorted_original",
                "filename": r["camera_distorted"].name,
                "image_path": str(r["camera_distorted"].resolve()),
                "ground_truth_semantic_path": str(r["primus_semantic"].resolve()),
            }
        )

    if not original_only:
        cam_corrupt = pick_n(
            corrupt["camera"],
            n_per_group,
            lambda r: r["png_path"].exists() and r["semantic_path"].exists(),
        )
        for r in cam_corrupt:
            selections.append(
                {
                    "group": "camera_distorted_corrupt",
                    "filename": r["png_path"].name,
                    "image_path": str(r["png_path"].resolve()),
                    "ground_truth_semantic_path": str(r["semantic_path"].resolve()),
                }
            )

    return selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Samples per group")
    parser.add_argument(
        "--original-only",
        action="store_true",
        help="Exclude data/corrupt groups (handwritten/clean/camera corrupt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SELECTION_PATH,
        help="Output selection JSON path",
    )
    args = parser.parse_args()

    selections = build_selection(args.n, original_only=args.original_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selections, indent=2), encoding="utf-8")

    from collections import Counter

    counts = Counter(s["group"] for s in selections)
    print(f"Wrote {len(selections)} entries to {args.output}")
    for group, count in sorted(counts.items()):
        print(f"  {group}: {count}")


if __name__ == "__main__":
    main()
