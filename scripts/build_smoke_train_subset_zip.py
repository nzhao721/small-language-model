"""Build smoke_train_subset.zip matching notebooks/smoke_test_florence2_colab.ipynb pair selection."""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "smoke_test_florence2_colab.ipynb"
IMAGES_DIR = REPO / "data" / "consolidated" / "images"
GT_DIR = REPO / "data" / "consolidated" / "ground_truth"
GOLDEN_DIR = REPO / "golden_set"
OUT_ZIP = REPO / "data" / "smoke_train_subset.zip"
MAX_TRAIN_SAMPLES = 400


def load_notebook_utilities() -> dict:
    ns: dict = {"Path": Path, "json": json, "re": re}
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][5]["source"])
    marker = "manifest = build_smoke_manifest()"
    if marker in src:
        src = src.split(marker, 1)[0]
    exec(src, ns)
    return ns


def main() -> None:
    ns = load_notebook_utilities()
    load_golden_selection = ns["load_golden_selection"]
    build_golden_eval_records = ns["build_golden_eval_records"]
    build_pairs_limited = ns["build_pairs_limited"]

    selection = load_golden_selection(GOLDEN_DIR)
    _eval_golden, exclusion_ids = build_golden_eval_records(GOLDEN_DIR, selection)
    train = build_pairs_limited(IMAGES_DIR, GT_DIR, MAX_TRAIN_SAMPLES, exclusion_ids)

    if len(train) != MAX_TRAIN_SAMPLES:
        print(f"WARNING: expected {MAX_TRAIN_SAMPLES} pairs, got {len(train)}")

    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.is_file():
        OUT_ZIP.unlink()

    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rec in train:
            img = Path(rec["image"])
            gt = Path(rec["gt"])
            zf.write(img, f"consolidated/images/{img.name}")
            zf.write(gt, f"consolidated/ground_truth/{gt.name}")

    size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)

    with zipfile.ZipFile(OUT_ZIP, "r") as zf:
        names = zf.namelist()
        zip_images = sum(1 for n in names if n.startswith("consolidated/images/") and not n.endswith("/"))
        zip_sem = sum(1 for n in names if n.endswith(".semantic"))

    print("=== smoke_train_subset.zip ===")
    print(f"path: {OUT_ZIP}")
    print(f"pairs: {len(train)}")
    print(f"zip images: {zip_images}")
    print(f"zip .semantic: {zip_sem}")
    print(f"size_mb: {size_mb:.2f}")
    print(f"exclusion_ids: {len(exclusion_ids)}")
    print(f"golden selection entries: {len(selection)}")


if __name__ == "__main__":
    main()
