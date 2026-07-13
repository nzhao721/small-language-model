"""Generate upscaled overview + overlapping detail tiles for OMR transcription viewing.

Reads a batch JSON and, for each entry's image_path, writes:
  data/_omr_view/<batch>/<stem>/overview.png   (whole strip, fit to ~1560 wide)
  data/_omr_view/<batch>/<stem>/tile_00.png ... (upscaled overlapping segments)

Does NOT touch any ground-truth semantic files.
"""
import json
import os
import sys
from PIL import Image

BATCH = sys.argv[1] if len(sys.argv) > 1 else "batch_12"
batch_file = f"examples/omr_benchmark/batches/{BATCH}.json"
out_root = f"data/_omr_view/{BATCH}"

TILE_W = 1500       # max tile width in upscaled px
OVERLAP = 320       # horizontal overlap between tiles
TARGET_H = 460      # target upscaled staff height
MAX_F = 4.0
MIN_F = 2.0
OVERVIEW_W = 1560

os.makedirs(out_root, exist_ok=True)

with open(batch_file) as f:
    batch = json.load(f)

manifest = {}
for entry in batch:
    path = entry["image_path"]
    stem = os.path.splitext(os.path.basename(path))[0]
    d = os.path.join(out_root, stem)
    os.makedirs(d, exist_ok=True)

    im = Image.open(path).convert("L")
    w, h = im.size

    # Overview: fit to OVERVIEW_W wide (upscale small ones for legibility)
    of = OVERVIEW_W / w
    ov = im.resize((max(1, int(w * of)), max(1, int(h * of))), Image.LANCZOS)
    ov.save(os.path.join(d, "overview.png"))

    # Detail tiles
    f_ = max(MIN_F, min(MAX_F, TARGET_H / h))
    big = im.resize((int(w * f_), int(h * f_)), Image.LANCZOS)
    bw, bh = big.size

    tiles = []
    if bw <= TILE_W:
        big.save(os.path.join(d, "tile_00.png"))
        tiles.append("tile_00.png")
    else:
        step = TILE_W - OVERLAP
        x = 0
        i = 0
        while x < bw:
            right = min(x + TILE_W, bw)
            crop = big.crop((x, 0, right, bh))
            name = f"tile_{i:02d}.png"
            crop.save(os.path.join(d, name))
            tiles.append(name)
            i += 1
            if right >= bw:
                break
            x += step
    manifest[stem] = {"orig_size": [w, h], "factor": round(f_, 2), "tiles": tiles}

print(json.dumps(manifest, indent=2))
