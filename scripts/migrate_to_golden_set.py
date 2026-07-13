"""Move OMR benchmark holdout samples into golden_set/ at repo root."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "data" / "omr_benchmark_holdout"
GOLDEN = ROOT / "golden_set"
BENCHMARK = ROOT / "examples" / "omr_benchmark"

ORIGINAL_GROUP_TO_SUBDIR = {
    "handwritten_original": "handwritten",
    "clean_original": "clean",
    "camera_distorted_original": "camera",
}

CORRUPT_GROUP_TO_SUBDIR = {
    "handwritten_corrupt": "handwritten",
    "clean_corrupt": "clean",
    "camera_distorted_corrupt": "camera",
}


def incipit_id_from_entry(entry: dict) -> str:
    stem = Path(entry["filename"]).stem
    if stem.endswith("_distorted"):
        stem = stem[: -len("_distorted")]
    return stem


def ensure_dirs() -> None:
    for rel in [
        "original/images/handwritten",
        "original/images/clean",
        "original/images/camera",
        "original/ground_truth",
        "corrupt/images/handwritten",
        "corrupt/images/clean",
        "corrupt/images/camera",
        "corrupt/ground_truth",
    ]:
        (GOLDEN / rel).mkdir(parents=True, exist_ok=True)


def move_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def migrate_original(selection: list[dict]) -> list[dict]:
    manifest_rows: list[dict] = []
    for entry in selection:
        group = entry["group"]
        subdir = ORIGINAL_GROUP_TO_SUBDIR[group]
        filename = entry["filename"]
        incipit_id = incipit_id_from_entry(entry)
        gt_name = f"{incipit_id}.semantic"

        src_img = Path(entry["image_path"])
        src_gt = Path(entry["ground_truth_semantic_path"])
        dst_img = GOLDEN / "original" / "images" / subdir / filename
        dst_gt = GOLDEN / "original" / "ground_truth" / gt_name

        move_file(src_img, dst_img)
        move_file(src_gt, dst_gt)

        entry["image_path"] = str(dst_img)
        entry["ground_truth_semantic_path"] = str(dst_gt)

        manifest_rows.append(
            {
                "id": incipit_id,
                "group": group,
                "image_path": str(dst_img),
                "gt_path": str(dst_gt),
                "source": "original",
            }
        )
    return manifest_rows


def migrate_corrupt(selection: list[dict]) -> list[dict]:
    manifest_rows: list[dict] = []
    for entry in selection:
        group = entry["group"]
        subdir = CORRUPT_GROUP_TO_SUBDIR[group]
        corrupt_id = entry["corrupt_id"]
        filename = entry["filename"]
        gt_name = f"{corrupt_id}.semantic"

        src_img = Path(entry["image_path"])
        src_gt = Path(entry["ground_truth_semantic_path"])
        dst_img = GOLDEN / "corrupt" / "images" / subdir / filename
        dst_gt = GOLDEN / "corrupt" / "ground_truth" / gt_name

        move_file(src_img, dst_img)
        move_file(src_gt, dst_gt)

        entry["image_path"] = str(dst_img)
        entry["ground_truth_semantic_path"] = str(dst_gt)
        entry["holdout_image_path"] = str(dst_img)
        entry["holdout_semantic_path"] = str(dst_gt)

        manifest_rows.append(
            {
                "id": corrupt_id,
                "incipit_id": entry["incipit_id"],
                "group": group,
                "image_path": str(dst_img),
                "gt_path": str(dst_gt),
                "source": "corrupt",
            }
        )
    return manifest_rows


def update_json_paths(path: Path, old_prefix: str, new_prefix: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated = text.replace(old_prefix, new_prefix)
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for dirpath in sorted(root.rglob("*"), reverse=True):
        if dirpath.is_dir() and not any(dirpath.iterdir()):
            dirpath.rmdir()
    if root.exists() and not any(root.iterdir()):
        root.rmdir()


def check_stragglers(original_manifest: list[dict], corrupt_selection: list[dict]) -> list[str]:
    found: list[str] = []
    for row in original_manifest:
        for key in ("original_image_path", "original_semantic_path"):
            p = Path(row[key])
            if p.exists():
                found.append(str(p))
    corrupt_ids = {e["corrupt_id"] for e in corrupt_selection}
    corrupt_base = ROOT / "data" / "corrupt"
    for cid in corrupt_ids:
        for sub in ("handwritten", "clean", "camera"):
            for ext in (".jpg", ".png", ".semantic"):
                p = corrupt_base / "images" / sub / f"{cid}{ext}"
                if p.exists():
                    found.append(str(p))
                p = corrupt_base / "ground_truth" / f"{cid}{ext}"
                if p.exists():
                    found.append(str(p))
    return found


def main() -> None:
    ensure_dirs()

    selection_path = BENCHMARK / "selection_10.json"
    corrupt_path = BENCHMARK / "selection_10_corrupt.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    corrupt = json.loads(corrupt_path.read_text(encoding="utf-8"))

    old_manifest = [
        json.loads(line)
        for line in (HOLDOUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    original_rows = migrate_original(selection)
    corrupt_rows = migrate_corrupt(corrupt)

    selection_path.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    corrupt_path.write_text(
        json.dumps(corrupt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for batch_dir in (BENCHMARK / "batches_10", BENCHMARK / "batches_10_corrupt"):
        for batch_file in sorted(batch_dir.glob("batch_*.json")):
            data = json.loads(batch_file.read_text(encoding="utf-8"))
            for entry in data:
                if "image_path" in entry and str(HOLDOUT) in entry["image_path"]:
                    entry["image_path"] = entry["image_path"].replace(
                        str(HOLDOUT), str(GOLDEN)
                    ).replace(
                        "\\images\\", "\\original\\images\\"
                    ).replace(
                        "/images/", "/original/images/"
                    )
                    if "\\corrupt\\images\\" in entry["image_path"] or "/corrupt/images/" in entry["image_path"]:
                        pass  # already has corrupt path from replace
                if "ground_truth_semantic_path" in entry and str(HOLDOUT) in entry["ground_truth_semantic_path"]:
                    entry["ground_truth_semantic_path"] = (
                        entry["ground_truth_semantic_path"]
                        .replace(str(HOLDOUT), str(GOLDEN))
                        .replace("\\ground_truth\\", "\\original\\ground_truth\\")
                        .replace("/ground_truth/", "/original/ground_truth/")
                    )
                    if "\\corrupt\\ground_truth\\" in entry["ground_truth_semantic_path"]:
                        pass
                for key in ("holdout_image_path", "holdout_semantic_path"):
                    if key in entry and str(HOLDOUT) in entry[key]:
                        entry[key] = (
                            entry[key]
                            .replace(str(HOLDOUT), str(GOLDEN))
                            .replace("\\corrupt\\images\\", "\\corrupt\\images\\")
                        )
            batch_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    # Fix batch paths properly by re-reading selection files
    for batch_dir, sel_file in (
        (BENCHMARK / "batches_10", selection_path),
        (BENCHMARK / "batches_10_corrupt", corrupt_path),
    ):
        sel = json.loads(sel_file.read_text(encoding="utf-8"))
        by_filename = {e["filename"]: e for e in sel}
        for batch_file in sorted(batch_dir.glob("batch_*.json")):
            data = json.loads(batch_file.read_text(encoding="utf-8"))
            for entry in data:
                ref = by_filename.get(entry["filename"])
                if ref:
                    entry["image_path"] = ref["image_path"]
                    entry["ground_truth_semantic_path"] = ref["ground_truth_semantic_path"]
                    if "holdout_image_path" in ref:
                        entry["holdout_image_path"] = ref["holdout_image_path"]
                    if "holdout_semantic_path" in ref:
                        entry["holdout_semantic_path"] = ref["holdout_semantic_path"]
            batch_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    moved_at = datetime.now(timezone.utc).isoformat()
    manifest_lines = []
    for row in original_rows + corrupt_rows:
        row["moved_at"] = moved_at
        manifest_lines.append(json.dumps(row, ensure_ascii=False))
    (GOLDEN / "manifest.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    old_prefix = str(HOLDOUT)
    for rel in [
        "examples/omr_benchmark/prompts/opus_batch_handwritten.md",
        "examples/omr_benchmark/prompts/opus_batch_clean.md",
        "examples/omr_benchmark/prompts/opus_batch_camera.md",
        "examples/omr_benchmark/prompts/opus_batch_handwritten_corrupt.md",
        "examples/omr_benchmark/prompts/opus_batch_clean_corrupt.md",
        "examples/omr_benchmark/prompts/opus_batch_camera_corrupt.md",
        "_omr_zoom/zoom.py",
        "_omr_zoom/stafflines.py",
        ".tmp_omr/enhance.py",
    ]:
        p = ROOT / rel
        if p.exists():
            update_json_paths(p, old_prefix, str(GOLDEN))

    remove_empty_dirs(HOLDOUT)

    stragglers = check_stragglers(old_manifest[:30], corrupt)

    # Count golden_set files
    img_count = sum(1 for _ in GOLDEN.rglob("*") if _.suffix.lower() in {".jpg", ".png"})
    gt_count = sum(1 for _ in GOLDEN.rglob("*.semantic"))

    print(json.dumps({
        "images_moved": img_count,
        "ground_truths_moved": gt_count,
        "manifest_entries": len(manifest_lines),
        "stragglers": stragglers,
        "holdout_removed": not HOLDOUT.exists(),
        "golden_set": str(GOLDEN),
    }, indent=2))


if __name__ == "__main__":
    main()
