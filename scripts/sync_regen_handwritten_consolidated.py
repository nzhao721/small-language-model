"""Replace consolidated mashcima + corrupt/handwritten pairs after full regen."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from consolidate_dataset_pairs import (  # noqa: E402
    CONSOLIDATED,
    GT_EXT,
    IMAGE_EXTS,
    consolidate_name,
    iter_images,
    prune_consolidated,
)
from dataset_layout import ROOT  # noqa: E402

MASHCIMA_IMAGES = ROOT / "data" / "mashcima" / "images"
MASHCIMA_GT = ROOT / "data" / "mashcima" / "ground_truth"
CORRUPT_HW = ROOT / "data" / "corrupt" / "handwritten"
CORRUPT_SEM = ROOT / "data" / "corrupt" / "semantic"
REPORT_PATH = CONSOLIDATED / "handwritten_sync_report.json"

GROUPS = (
    ("mashcima", MASHCIMA_IMAGES, MASHCIMA_GT),
    ("corrupt/handwritten", CORRUPT_HW, CORRUPT_SEM),
)


def prefix_for_group(group_key: str) -> str:
    return consolidate_name(group_key, "", "").rstrip("_")


def delete_old_pairs(out_images: Path, out_gt: Path, prefixes: tuple[str, ...], dry_run: bool) -> dict[str, int]:
    stats = {"images_deleted": 0, "gt_deleted": 0}
    for folder, key in ((out_images, "images_deleted"), (out_gt, "gt_deleted")):
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if not path.is_file():
                continue
            if any(path.name.startswith(prefix) for prefix in prefixes):
                if not dry_run:
                    path.unlink()
                stats[key] += 1
    return stats


def copy_pairs(
    group_key: str,
    image_dir: Path,
    gt_dir: Path,
    out_images: Path,
    out_gt: Path,
    dry_run: bool,
) -> dict[str, int]:
    stats = {
        "images_copied": 0,
        "gt_copied": 0,
        "missing_gt": 0,
        "missing_gt_ids": [],
    }
    gt_by_stem = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() == GT_EXT}
    images = list(iter_images(image_dir))

    for img in tqdm(images, desc=group_key, unit="pair"):
        stem = img.stem
        gt = gt_by_stem.get(stem)
        if gt is None:
            stats["missing_gt"] += 1
            if len(stats["missing_gt_ids"]) < 20:
                stats["missing_gt_ids"].append(stem)
            continue
        img_name = consolidate_name(group_key, stem, img.suffix.lower())
        gt_name = consolidate_name(group_key, stem, GT_EXT)
        if not dry_run:
            out_images.mkdir(parents=True, exist_ok=True)
            out_gt.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out_images / img_name)
            shutil.copy2(gt, out_gt / gt_name)
        stats["images_copied"] += 1
        stats["gt_copied"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=CONSOLIDATED)
    args = parser.parse_args()

    out_images = args.out_dir / "images"
    out_gt = args.out_dir / "ground_truth"
    prefixes = tuple(prefix_for_group(key) for key, _, _ in GROUPS)

    report: dict = {
        "dry_run": args.dry_run,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "prefixes_replaced": list(prefixes),
        "delete": delete_old_pairs(out_images, out_gt, prefixes, args.dry_run),
        "copy": {},
    }

    for group_key, image_dir, gt_dir in GROUPS:
        report["copy"][group_key] = copy_pairs(
            group_key, image_dir, gt_dir, out_images, out_gt, args.dry_run
        )

    report["orphans"] = prune_consolidated(args.out_dir, args.dry_run)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    print(json.dumps(report, indent=2))
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
