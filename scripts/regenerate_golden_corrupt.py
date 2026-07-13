"""Regenerate only the 30 corrupt golden-set images for visual review."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from demo_corrupted_gt_examples import save_png  # noqa: E402
from demo_mashcima_camera_distort import camera_like_distort  # noqa: E402
from primus_semantic_render import render_semantic_to_image  # noqa: E402
from regenerate_golden_handwritten import (  # noqa: E402
    DEFAULT_MUSCIMA_DIR,
    DEFAULT_PRIMUS_TGZ,
    ensure_muscima_cache,
    load_replay_cache,
    render_corrupt_handwritten,
)
from mashcima_writers import (  # noqa: E402
    DEFAULT_WRITER_MODE,
    configure_mashcima_writers,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "golden_set"
MANIFEST = GOLDEN / "manifest.jsonl"
REPLAY_CACHE = ROOT / "data" / "logs" / "golden_corrupt_replay.json"
GROUP_FORMAT = {
    "clean_corrupt": "clean",
    "camera_distorted_corrupt": "camera",
    "handwritten_corrupt": "handwritten",
}


def _save_exact(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.convert("RGB").save(path, format="JPEG", quality=92, optimize=True)
    else:
        save_png(image, path)


def _validate_image(path: Path) -> dict:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
        width, height = image.size
    if width < 100 or height < 50:
        raise ValueError(f"implausibly small render {width}x{height}: {path}")
    dark_fraction = float(np.mean(np.min(rgb, axis=2) < 220))
    if dark_fraction < 0.002:
        raise ValueError(f"nearly blank render ({dark_fraction:.5f} dark): {path}")
    return {
        "width": width,
        "height": height,
        "dark_fraction": round(dark_fraction, 5),
    }


def _contact_sheet(rows: list[dict], output: Path) -> None:
    thumb_w, thumb_h = 600, 180
    label_h = 32
    columns = 2
    cell_w, cell_h = thumb_w + 20, thumb_h + label_h + 20
    sheet = Image.new(
        "RGB",
        (columns * cell_w, ((len(rows) + columns - 1) // columns) * cell_h),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        with Image.open(row["image_path"]) as source:
            image = ImageOps.contain(source.convert("RGB"), (thumb_w, thumb_h))
        x = (index % columns) * cell_w + 10
        y = (index // columns) * cell_h + label_h
        sheet.paste(image, (x, y))
        draw.text(
            (x, y - label_h + 7),
            f"{row['group']}  {row['id']}  {row['measures']} measures",
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=90, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--review-dir", type=Path, default=GOLDEN / "corrupt_review_fixed")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--writers", default=DEFAULT_WRITER_MODE)
    parser.add_argument("--max-seed", type=int, default=25_000_000)
    parser.add_argument(
        "--groups",
        default="clean_corrupt,camera_distorted_corrupt,handwritten_corrupt",
        help="Comma-separated corrupt golden groups to regenerate",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = args.backup_dir or GOLDEN / "corrupt_backups" / timestamp
    staging = args.review_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requested_groups = {
        group.strip() for group in args.groups.split(",") if group.strip()
    }
    unknown_groups = requested_groups - set(GROUP_FORMAT)
    if unknown_groups:
        raise ValueError(f"Unknown group(s): {sorted(unknown_groups)}")
    targets = [row for row in rows if row.get("group") in requested_groups]
    expected = 10 * len(requested_groups)
    if len(targets) != expected:
        raise RuntimeError(
            f"Expected {expected} selected corrupt golden samples, found {len(targets)}"
        )

    ensure_muscima_cache(DEFAULT_PRIMUS_TGZ, DEFAULT_MUSCIMA_DIR)
    configure_mashcima_writers(args.writers)
    replay_cache = load_replay_cache(REPLAY_CACHE)
    report_rows: list[dict] = []

    for row in tqdm(targets, desc="Golden corrupt", unit="image"):
        group = row["group"]
        fmt = GROUP_FORMAT[group]
        image_path = Path(row["image_path"])
        semantic_path = Path(row["gt_path"])
        semantic = semantic_path.read_text(encoding="utf-8")
        relative = image_path.relative_to(GOLDEN / "corrupt")
        staged_path = staging / relative
        staged_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "clean":
            image = render_semantic_to_image(semantic)
            _save_exact(image, staged_path)
        elif fmt == "camera":
            seed = zlib.crc32(row["id"].encode("utf-8")) % 2_147_483_647
            image = render_semantic_to_image(semantic)
            image = camera_like_distort(image, seed=seed + 99)
            _save_exact(image, staged_path)
        else:
            cached = replay_cache.get(row["id"], {})
            seed, fraction = render_corrupt_handwritten(
                row["id"],
                row["incipit_id"],
                semantic_path,
                staged_path,
                corruption_seed=cached.get("corruption_seed"),
                corrupt_fraction=cached.get("corrupt_fraction"),
                max_seed=args.max_seed,
            )
            replay_cache[row["id"]] = {
                "corruption_seed": seed,
                "corrupt_fraction": fraction,
            }

        metrics = _validate_image(staged_path)
        report_rows.append(
            {
                "id": row["id"],
                "group": group,
                "image_path": str(image_path),
                "semantic_path": str(semantic_path),
                "measures": semantic.split().count("barline") + 1,
                **metrics,
            }
        )

    # Commit only after every staged image passed validation.
    for row in targets:
        image_path = Path(row["image_path"])
        relative = image_path.relative_to(GOLDEN / "corrupt")
        staged_path = staging / relative
        backup_path = backup_dir / relative
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, backup_path)
        shutil.copy2(staged_path, image_path)

    REPLAY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    REPLAY_CACHE.write_text(json.dumps(replay_cache, indent=2), encoding="utf-8")

    for group in requested_groups:
        group_rows = [row for row in report_rows if row["group"] == group]
        _contact_sheet(group_rows, args.review_dir / f"{group}.jpg")
    if len(report_rows) == 30:
        _contact_sheet(report_rows, args.review_dir / "all_30.jpg")
    else:
        _contact_sheet(report_rows, args.review_dir / f"selected_{len(report_rows)}.jpg")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(report_rows),
        "backup_dir": str(backup_dir),
        "review_dir": str(args.review_dir),
        "benchmark_predictions_stale": True,
        "samples": report_rows,
    }
    (args.review_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    shutil.rmtree(staging)
    print(f"Regenerated {len(report_rows)} golden corrupt images")
    print(f"Backup: {backup_dir}")
    print(f"Review: {args.review_dir}")
    print("Existing corrupt benchmark predictions are now stale; do not re-score yet.")


if __name__ == "__main__":
    main()
