import json
import os
from PIL import Image, ImageOps

BATCH = r"C:\alpha_ai\small-language-model\examples\omr_benchmark\batches\batch_06.json"
OUT = r"C:\alpha_ai\small-language-model\data\omr_tiles"
os.makedirs(OUT, exist_ok=True)

with open(BATCH, "r") as f:
    batch = json.load(f)

TILE_W = 1500      # max tile width shown at full detail
OVERLAP = 200      # overlap between tiles
TARGET_H = 420     # normalize staff height for readability
OVERVIEW_W = 1500  # overview max width

manifest = []
for i, entry in enumerate(batch):
    path = entry["image_path"]
    fn = entry["filename"]
    stem = os.path.splitext(fn)[0]
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    img = img.convert("RGB")
    w, h = img.size

    # Normalize to target height
    scale = TARGET_H / h
    nw, nh = max(1, int(w * scale)), TARGET_H
    norm = img.resize((nw, nh), Image.LANCZOS)

    # Overview (whole thing, capped width)
    ov_scale = min(1.0, OVERVIEW_W / nw)
    ov = norm.resize((max(1, int(nw * ov_scale)), max(1, int(nh * ov_scale))), Image.LANCZOS)
    ovp = os.path.join(OUT, f"{i:02d}_{stem}_overview.png")
    ov.save(ovp)

    # Tiles
    tiles = []
    if nw <= TILE_W:
        tp = os.path.join(OUT, f"{i:02d}_{stem}_tile0.png")
        norm.save(tp)
        tiles.append(tp)
    else:
        x = 0
        t = 0
        step = TILE_W - OVERLAP
        while x < nw:
            box = (x, 0, min(x + TILE_W, nw), nh)
            crop = norm.crop(box)
            tp = os.path.join(OUT, f"{i:02d}_{stem}_tile{t}.png")
            crop.save(tp)
            tiles.append(tp)
            t += 1
            if x + TILE_W >= nw:
                break
            x += step

    manifest.append({"idx": i, "filename": fn, "normw": nw, "normh": nh,
                     "overview": ovp, "tiles": tiles})
    print(f"{i:02d} {fn} norm={nw}x{nh} tiles={len(tiles)}")

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("DONE")
