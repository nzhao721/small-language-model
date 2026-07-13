"""Regenerate specific mashcima incipits and restore consolidated pairs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from consolidate_dataset_pairs import CONSOLIDATED, consolidate_name  # noqa: E402
from dataset_layout import (  # noqa: E402
    DEFAULT_PRIMUS_TGZ,
    MASHCIMA_DIR,
    read_primus_agnostic,
)
from generate_corrupt_batch import DEFAULT_PRIMUS_DIR  # noqa: E402
from generate_mashcima_batch import (  # noqa: E402
    DEFAULT_MUSCIMA_DIR,
    ensure_muscima_cache,
    image_to_uint8,
    mashcima_ground_truth_dir,
    mashcima_images_dir,
    synthesize_handwritten,
)
from mashcima_appearance import apply_mashcima_appearance  # noqa: E402
from image_compress import compress_image_file, resolved_image_path  # noqa: E402
from mashcima.primus_adapter import convert_primus_annotation_to_mashcima_annotation  # noqa: E402
from mashcima.vocabulary import parse_annotation_into_token_groups  # noqa: E402
from mashcima_writers import DEFAULT_WRITER_MODE, configure_mashcima_writers  # noqa: E402
from post_regen_handwritten_camera_distort import (  # noqa: E402
    distort_image_in_place,
    remove_stale_png_sibling,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS_FILE = ROOT / "data" / "logs" / "mashcima_corrupt_ids.txt"


def load_ids(path: Path | None, inline: list[str]) -> list[str]:
    if inline:
        return inline
    if path and path.is_file():
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    raise SystemExit("Provide --ids or --ids-file")


def sync_consolidated(
    incipit_id: str,
    image_path: Path,
    semantic_path: Path,
    out_dir: Path,
) -> None:
    img_dest = out_dir / "images" / consolidate_name("mashcima", incipit_id, image_path.suffix.lower())
    gt_dest = out_dir / "ground_truth" / consolidate_name("mashcima", incipit_id, ".semantic")
    img_dest.parent.mkdir(parents=True, exist_ok=True)
    gt_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, img_dest)
    shutil.copy2(semantic_path, gt_dest)


def regenerate_one(incipit_id: str, mashcima_dir: Path) -> Path:
    agnostic = read_primus_agnostic(incipit_id, DEFAULT_PRIMUS_DIR, DEFAULT_PRIMUS_TGZ)
    converted = convert_primus_annotation_to_mashcima_annotation(agnostic)
    if converted is None:
        raise ValueError("agnostic conversion returned None")
    _, warnings = parse_annotation_into_token_groups(converted)
    if warnings:
        raise ValueError(f"annotation warnings: {warnings[:3]}")

    images_dir = mashcima_images_dir(mashcima_dir)
    out_png = images_dir / f"{incipit_id}.png"
    for stale in (out_png, out_png.with_suffix(".jpg"), out_png.with_suffix(".jpeg")):
        stale.unlink(missing_ok=True)

    img = image_to_uint8(synthesize_handwritten(converted, incipit_id))
    img = apply_mashcima_appearance(img, key=incipit_id)
    if not cv2.imwrite(str(out_png), img):
        raise RuntimeError("cv2.imwrite failed")

    compress_image_file(out_png)
    image_path = resolved_image_path(out_png)
    distort_image_in_place(image_path)
    final_path = resolved_image_path(image_path)
    compress_image_file(final_path)
    final_path = resolved_image_path(final_path)
    remove_stale_png_sibling(final_path)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--mashcima-dir", type=Path, default=MASHCIMA_DIR)
    parser.add_argument("--out-dir", type=Path, default=CONSOLIDATED)
    parser.add_argument(
        "--writers",
        default=DEFAULT_WRITER_MODE,
        choices=["per-score", "mixed", "single"],
    )
    args = parser.parse_args()

    ids = load_ids(args.ids_file, args.ids)
    ensure_muscima_cache(DEFAULT_PRIMUS_TGZ, DEFAULT_MUSCIMA_DIR)
    configure_mashcima_writers(args.writers)

    results: list[dict] = []
    for incipit_id in ids:
        row: dict = {"incipit_id": incipit_id}
        try:
            image_path = regenerate_one(incipit_id, args.mashcima_dir)
            semantic_path = mashcima_ground_truth_dir(args.mashcima_dir) / f"{incipit_id}.semantic"
            if not semantic_path.is_file():
                raise FileNotFoundError(f"missing semantic: {semantic_path}")
            sync_consolidated(incipit_id, image_path, semantic_path, args.out_dir)
            row["status"] = "ok"
            row["image_path"] = str(image_path)
            row["size_bytes"] = image_path.stat().st_size
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)
        results.append(row)
        print(json.dumps(row), flush=True)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"Done: {ok}/{len(results)} ok", flush=True)


if __name__ == "__main__":
    main()
