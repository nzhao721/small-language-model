"""Remove duplicate/orphan files under data/mashcima/images.

- Delete .png when a sibling .jpg exists (compress + regen left stale PNGs).
- Delete images whose stem is not in the latest ok mashcima manifest row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_layout import MASHCIMA_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = MASHCIMA_DIR / "images"
MANIFEST = MASHCIMA_DIR / "manifest.jsonl"


def load_ok_stems(manifest: Path) -> set[str]:
    ok: set[str] = set()
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("incipit_id"):
                ok.add(row["incipit_id"])
    return ok


def cleanup(images_dir: Path, ok_stems: set[str], *, dry_run: bool) -> dict[str, int]:
    stats = {"png_when_jpg": 0, "non_ok_stem": 0, "bytes_freed": 0}

    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem
        reason: str | None = None

        if path.suffix.lower() == ".png":
            jpg = path.with_suffix(".jpg")
            if jpg.is_file():
                reason = "png_when_jpg"

        if stem not in ok_stems:
            reason = "non_ok_stem"

        if reason is None:
            continue

        size = path.stat().st_size
        stats[reason] += 1
        stats["bytes_freed"] += size
        if not dry_run:
            path.unlink(missing_ok=True)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not IMAGES_DIR.is_dir():
        raise SystemExit(f"Missing {IMAGES_DIR}")

    ok_stems = load_ok_stems(MANIFEST)
    before = sum(1 for p in IMAGES_DIR.iterdir() if p.is_file())
    stats = cleanup(IMAGES_DIR, ok_stems, dry_run=args.dry_run)
    after = sum(1 for p in IMAGES_DIR.iterdir() if p.is_file())

    print(f"ok stems in manifest: {len(ok_stems):,}")
    print(f"files before: {before:,}")
    print(f"files after:  {after:,}")
    print(f"removed png_when_jpg: {stats['png_when_jpg']:,}")
    print(f"removed non_ok_stem:  {stats['non_ok_stem']:,}")
    print(f"freed: {stats['bytes_freed'] / 1e9:.2f} GB")
    if args.dry_run:
        print("(dry run — no files deleted)")


if __name__ == "__main__":
    main()
