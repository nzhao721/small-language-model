"""Crop a wide staff image into overlapping, upscaled horizontal segments.

Used to read camera-distorted PrIMuS incipits more accurately for manual
transcription. Writes segment PNGs to a temp folder and prints their paths.
"""
import sys
import os
from PIL import Image, ImageOps


def process(img_path, out_dir, n_segments=5, overlap=0.18, scale=3):
    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(img_path).convert("L")
    # Autocontrast to counteract camera dimming/shadows.
    im = ImageOps.autocontrast(im, cutoff=1)
    w, h = im.size
    seg_w = int(w / n_segments)
    ov = int(seg_w * overlap)
    paths = []
    for i in range(n_segments):
        left = max(0, i * seg_w - ov)
        right = min(w, (i + 1) * seg_w + ov)
        crop = im.crop((left, 0, right, h))
        cw, ch = crop.size
        crop = crop.resize((cw * scale, ch * scale), Image.LANCZOS)
        out = os.path.join(out_dir, f"seg_{i:02d}.png")
        crop.save(out)
        paths.append((out, left, right))
    return paths, (w, h)


if __name__ == "__main__":
    img_path = sys.argv[1]
    out_dir = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    scale = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    paths, size = process(img_path, out_dir, n_segments=n, scale=scale)
    print(f"source_size={size}")
    for p, l, r in paths:
        print(f"{p}  x=[{l},{r}]")
