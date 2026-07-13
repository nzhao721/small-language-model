"""Remove golden-set samples from data/consolidated/ (images + ground_truth)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_layout import ROOT  # noqa: E402

CONSOLIDATED = ROOT / "data" / "consolidated"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
GT_EXT = ".semantic"


def incipit_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if stem.endswith("_distorted"):
        stem = stem[: -len("_distorted")]
    return stem


def exclusion_id_for_entry(group: str, filename: str) -> str:
    if group.startswith("manual_") or group.endswith("_corrupt"):
        return Path(filename).stem
    return incipit_id_from_filename(filename)


def build_exclusion_set() -> tuple[set[str], dict[str, int]]:
    excl: set[str] = set()
    sources: dict[str, int] = defaultdict(int)

    manifest = ROOT / "golden_set" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        excl.add(entry["id"])
        sources["manifest_id"] += 1
        if entry.get("incipit_id"):
            excl.add(entry["incipit_id"])
            sources["manifest_incipit_id"] += 1

    for name in ("selection_10.json", "selection_10_corrupt.json", "selection_manual.json"):
        path = ROOT / "examples" / "omr_benchmark" / name
        if not path.is_file():
            continue
        for entry in json.loads(path.read_text(encoding="utf-8")):
            gid = exclusion_id_for_entry(entry["group"], entry["filename"])
            excl.add(gid)
            sources[f"benchmark:{name}"] += 1

    for gt in ROOT.glob("golden_set/**/ground_truth/*.semantic"):
        excl.add(gt.stem)
        sources["golden_gt"] += 1

    return excl, dict(sources)


def parse_consolidated_name(name: str) -> tuple[str, str] | None:
    if "__" not in name:
        return None
    group, rest = name.split("__", 1)
    return group, Path(rest).stem


def stem_matches_exclusion(group: str, stem: str, excl: set[str]) -> bool:
    if group.startswith("golden_"):
        return True
    candidates = {stem}
    if stem.endswith("_distorted"):
        candidates.add(stem[: -len("_distorted")])
    return bool(candidates & excl)


def find_removals(
    excl: set[str],
    img_dir: Path,
    gt_dir: Path,
) -> tuple[list[Path], list[Path], Counter, Counter]:
    img_remove: list[Path] = []
    gt_remove: list[Path] = []
    by_group = Counter()
    by_category = Counter()

    for img in img_dir.iterdir():
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        parsed = parse_consolidated_name(img.name)
        if parsed is None:
            continue
        group, stem = parsed
        if not stem_matches_exclusion(group, stem, excl):
            continue
        gt_path = gt_dir / f"{group}__{stem}{GT_EXT}"
        img_remove.append(img)
        if gt_path.is_file():
            gt_remove.append(gt_path)
        by_group[group] += 1
        if group.startswith("golden_"):
            by_category["golden_prefixed"] += 1
        else:
            by_category["non_golden_overlap"] += 1

    return img_remove, gt_remove, by_group, by_category


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    img_dir = CONSOLIDATED / "images"
    gt_dir = CONSOLIDATED / "ground_truth"
    if not img_dir.is_dir() or not gt_dir.is_dir():
        raise SystemExit(f"Missing consolidated dirs under {CONSOLIDATED}")

    images_before = sum(1 for p in img_dir.iterdir() if p.is_file())
    gt_before = sum(1 for p in gt_dir.iterdir() if p.is_file())

    excl, source_counts = build_exclusion_set()
    img_remove, gt_remove, by_group, by_category = find_removals(excl, img_dir, gt_dir)

    if not args.dry_run:
        for path in img_remove + gt_remove:
            path.unlink()

    images_after = images_before - (0 if args.dry_run else len(img_remove))
    gt_after = gt_before - (0 if args.dry_run else len(gt_remove))
    if not args.dry_run:
        images_after = sum(1 for p in img_dir.iterdir() if p.is_file())
        gt_after = sum(1 for p in gt_dir.iterdir() if p.is_file())

    report = {
        "dry_run": args.dry_run,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "exclusion_set_size": len(excl),
        "exclusion_source_counts": source_counts,
        "images_before": images_before,
        "gt_before": gt_before,
        "images_removed": len(img_remove),
        "gt_removed": len(gt_remove),
        "images_after": images_after,
        "gt_after": gt_after,
        "removed_by_group": dict(sorted(by_group.items())),
        "removed_by_category": dict(by_category),
    }
    print(json.dumps(report, indent=2))

    report_path = CONSOLIDATED / "golden_removal_report.json"
    if not args.dry_run:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
