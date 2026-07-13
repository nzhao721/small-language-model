"""Audit image/semantic pairs, delete orphans, copy pairs into consolidated folders.

Originals are left in place; copies go to ``data/consolidated/images/`` and
``data/consolidated/ground_truth/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_layout import ROOT  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
GT_EXT = ".semantic"
CONSOLIDATED = ROOT / "data" / "consolidated"
REPORT_PATH = CONSOLIDATED / "consolidation_report.json"


@dataclass(frozen=True)
class PairGroup:
    """One image directory paired with one ground-truth directory."""

    key: str
    image_dir: Path
    gt_dir: Path
    image_stem_suffix: str = ""  # strip from image stem before matching GT stem


@dataclass(frozen=True)
class SharedSemanticGroup:
    """GT directory shared across multiple image format directories."""

    key: str
    gt_dir: Path
    image_dirs: tuple[Path, ...]
    image_prefixes: tuple[str, ...]


def iter_images(folder: Path) -> Iterable[Path]:
    if not folder.is_dir():
        return
    by_stem: dict[str, Path] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = path.stem
        existing = by_stem.get(stem)
        if existing is None:
            by_stem[stem] = path
            continue
        # Prefer compressed JPEG when both PNG and JPEG exist for one id.
        rank = {".jpg": 0, ".jpeg": 0, ".png": 1}
        if rank.get(path.suffix.lower(), 2) < rank.get(existing.suffix.lower(), 2):
            by_stem[stem] = path
    yield from by_stem.values()


def delete_duplicate_images(folder: Path, dry_run: bool) -> int:
    """Remove extra image extensions when multiple files share one stem."""
    if not folder.is_dir():
        return 0
    by_stem: dict[str, list[Path]] = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            by_stem.setdefault(path.stem, []).append(path)
    removed = 0
    for paths in by_stem.values():
        if len(paths) < 2:
            continue
        rank = {".jpg": 0, ".jpeg": 0, ".png": 1}
        paths.sort(key=lambda p: rank.get(p.suffix.lower(), 2))
        for extra in paths[1:]:
            if not dry_run:
                extra.unlink()
            removed += 1
    return removed


def iter_semantics(folder: Path) -> Iterable[Path]:
    if not folder.is_dir():
        return
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() == GT_EXT:
            yield path


def image_match_stem(path: Path, suffix: str = "") -> str:
    stem = path.stem
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def consolidate_name(group_key: str, stem: str, ext: str) -> str:
    safe = group_key.replace("/", "_").replace("\\", "_")
    return f"{safe}__{stem}{ext}"


def audit_simple_group(
    group: PairGroup,
    dry_run: bool,
    stats: dict,
) -> list[tuple[Path, Path, str, str]]:
    """Return list of (image_path, gt_path, consolidate_key, match_stem) for valid pairs."""
    pairs: list[tuple[Path, Path, str, str]] = []
    gt_by_stem = {p.stem: p for p in iter_semantics(group.gt_dir)}
    image_stems: set[str] = set()

    for img in iter_images(group.image_dir):
        stem = image_match_stem(img, group.image_stem_suffix)
        image_stems.add(stem)
        gt = gt_by_stem.get(stem)
        if gt is None:
            stats["orphan_images_deleted"] += 1
            stats["orphan_images_by_group"][group.key] = (
                stats["orphan_images_by_group"].get(group.key, 0) + 1
            )
            if not dry_run:
                img.unlink()
            continue
        pairs.append((img, gt, group.key, stem))

    for gt in iter_semantics(group.gt_dir):
        if gt.stem not in image_stems:
            stats["orphan_gt_deleted"] += 1
            stats["orphan_gt_by_group"][group.key] = (
                stats["orphan_gt_by_group"].get(group.key, 0) + 1
            )
            if not dry_run:
                gt.unlink()

    stats["pairs_by_group"][group.key] = len(pairs)
    return pairs


def audit_shared_semantic_group(
    group: SharedSemanticGroup,
    dry_run: bool,
    stats: dict,
) -> list[tuple[Path, Path, str, str]]:
    """Corrupt-style layout: one semantic folder, several image format folders."""
    pairs: list[tuple[Path, Path, str, str]] = []
    gt_by_stem = {p.stem: p for p in iter_semantics(group.gt_dir)}
    stems_with_image: set[str] = set()

    for img_dir, prefix in zip(group.image_dirs, group.image_prefixes):
        sub_key = f"{group.key}/{prefix}"
        for img in iter_images(img_dir):
            stem = img.stem
            gt = gt_by_stem.get(stem)
            if gt is None:
                stats["orphan_images_deleted"] += 1
                stats["orphan_images_by_group"][sub_key] = (
                    stats["orphan_images_by_group"].get(sub_key, 0) + 1
                )
                if not dry_run:
                    img.unlink()
                continue
            stems_with_image.add(stem)
            pairs.append((img, gt, sub_key, stem))

    for gt in iter_semantics(group.gt_dir):
        if gt.stem not in stems_with_image:
            stats["orphan_gt_deleted"] += 1
            stats["orphan_gt_by_group"][group.key] = (
                stats["orphan_gt_by_group"].get(group.key, 0) + 1
            )
            if not dry_run:
                gt.unlink()

    stats["pairs_by_group"][group.key] = len(pairs)
    return pairs


def copy_pairs(
    pairs: list[tuple[Path, Path, str, str]],
    out_images: Path,
    out_gt: Path,
    dry_run: bool,
    stats: dict,
) -> None:
    for img, gt, group_key, match_stem in pairs:
        img_name = consolidate_name(group_key, match_stem, img.suffix.lower())
        gt_name = consolidate_name(group_key, match_stem, GT_EXT)
        dest_img = out_images / img_name
        dest_gt = out_gt / gt_name

        if not dry_run:
            out_images.mkdir(parents=True, exist_ok=True)
            out_gt.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest_img)
            shutil.copy2(gt, dest_gt)
        stats["images_copied"] += 1
        stats["gt_copied"] += 1


def build_groups() -> tuple[list[PairGroup], list[SharedSemanticGroup]]:
    data = ROOT / "data"

    simple: list[PairGroup] = [
        PairGroup("primus", data / "primus" / "images", data / "primus" / "ground_truth"),
        PairGroup(
            "mashcima",
            data / "mashcima" / "images",
            data / "mashcima" / "ground_truth",
        ),
        PairGroup(
            "camera_primus",
            data / "camera-primus" / "images",
            data / "camera-primus" / "ground_truth",
            image_stem_suffix="_distorted",
        ),
    ]

    shared: list[SharedSemanticGroup] = []

    corrupt = data / "corrupt"
    shared.append(
        SharedSemanticGroup(
            "corrupt",
            corrupt / "semantic",
            (corrupt / "clean", corrupt / "camera", corrupt / "handwritten"),
            ("clean", "camera", "handwritten"),
        )
    )

    beamed = data / "corrupt_beamed"
    if beamed.is_dir():
        shared.append(
            SharedSemanticGroup(
                "corrupt_beamed",
                corrupt / "semantic",
                (beamed / "clean", beamed / "camera"),
                ("clean", "camera"),
            )
        )

    return simple, shared


def prune_consolidated(out_dir: Path, dry_run: bool) -> dict:
    """Remove consolidated images/GT that lack a matching pair (same stem)."""
    out_images = out_dir / "images"
    out_gt = out_dir / "ground_truth"
    stats = {
        "duplicate_images_deleted": 0,
        "orphan_images_deleted": 0,
        "orphan_gt_deleted": 0,
        "orphan_images_by_prefix": {},
        "orphan_gt_by_prefix": {},
        "images_before": 0,
        "gt_before": 0,
        "images_after": 0,
        "gt_after": 0,
    }
    if not out_images.is_dir() or not out_gt.is_dir():
        return stats

    stats["images_before"] = sum(1 for _ in out_images.iterdir() if _.is_file())
    stats["gt_before"] = sum(1 for _ in out_gt.iterdir() if _.is_file())

    stats["duplicate_images_deleted"] += delete_duplicate_images(out_images, dry_run)

    gt_stems = {p.stem for p in out_gt.iterdir() if p.is_file()}
    img_stems: set[str] = set()
    for img in out_images.iterdir():
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        img_stems.add(img.stem)
        if img.stem in gt_stems:
            continue
        prefix = img.name.split("__", 1)[0] if "__" in img.name else "unknown"
        stats["orphan_images_by_prefix"][prefix] = (
            stats["orphan_images_by_prefix"].get(prefix, 0) + 1
        )
        stats["orphan_images_deleted"] += 1
        if not dry_run:
            img.unlink()

    for gt in out_gt.iterdir():
        if not gt.is_file() or gt.suffix.lower() != GT_EXT:
            continue
        if gt.stem in img_stems:
            continue
        prefix = gt.name.split("__", 1)[0] if "__" in gt.name else "unknown"
        stats["orphan_gt_by_prefix"][prefix] = (
            stats["orphan_gt_by_prefix"].get(prefix, 0) + 1
        )
        stats["orphan_gt_deleted"] += 1
        if not dry_run:
            gt.unlink()

    stats["images_after"] = stats["images_before"] - stats["duplicate_images_deleted"] - (
        0 if dry_run else stats["orphan_images_deleted"]
    )
    stats["gt_after"] = stats["gt_before"] - (
        0 if dry_run else stats["orphan_gt_deleted"]
    )
    if not dry_run:
        stats["images_after"] = sum(1 for _ in out_images.iterdir() if _.is_file())
        stats["gt_after"] = sum(1 for _ in out_gt.iterdir() if _.is_file())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prune-only",
        action="store_true",
        help="Only prune consolidated/ orphans and duplicate extensions (no full rebuild)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=CONSOLIDATED,
        help="consolidated output root (default: data/consolidated)",
    )
    args = parser.parse_args()

    if args.prune_only:
        stats = {
            "dry_run": args.dry_run,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "mode": "prune_only",
            **prune_consolidated(args.out_dir, args.dry_run),
        }
        stats["finished_at"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(stats, indent=2))
        if not args.dry_run:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        return

    stats: dict = {
        "dry_run": args.dry_run,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duplicate_images_deleted": 0,
        "orphan_images_deleted": 0,
        "orphan_gt_deleted": 0,
        "orphan_images_by_group": {},
        "orphan_gt_by_group": {},
        "pairs_by_group": {},
        "images_copied": 0,
        "gt_copied": 0,
    }

    all_pairs: list[tuple[Path, Path, str, str]] = []
    simple, shared = build_groups()

    for group in simple:
        if group.image_dir.is_dir():
            stats["duplicate_images_deleted"] += delete_duplicate_images(
                group.image_dir, args.dry_run
            )
        if group.image_dir.is_dir() or group.gt_dir.is_dir():
            all_pairs.extend(audit_simple_group(group, args.dry_run, stats))

    for group in shared:
        for img_dir in group.image_dirs:
            if img_dir.is_dir():
                stats["duplicate_images_deleted"] += delete_duplicate_images(
                    img_dir, args.dry_run
                )
        if group.gt_dir.is_dir() or any(d.is_dir() for d in group.image_dirs):
            all_pairs.extend(audit_shared_semantic_group(group, args.dry_run, stats))

    out_images = args.out_dir / "images"
    out_gt = args.out_dir / "ground_truth"
    if not args.dry_run:
        if out_images.exists():
            shutil.rmtree(out_images)
        if out_gt.exists():
            shutil.rmtree(out_gt)

    copy_pairs(all_pairs, out_images, out_gt, args.dry_run, stats)

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats["total_pairs"] = len(all_pairs)
    stats["consolidated_images_dir"] = str(out_images)
    stats["consolidated_gt_dir"] = str(out_gt)

    print(json.dumps(stats, indent=2))
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
