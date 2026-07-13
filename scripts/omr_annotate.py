"""Annotate PrIMuS staff strips with a labeled pitch ruler, then tile for viewing.

For each entry in a batch JSON:
  - detect the 5 staff-line y positions
  - upscale
  - draw thin colored guide lines at every diatonic step (line & space) with a
    pitch label (letter+octave for the given clef) in a left margin
  - tile horizontally into overlapping segments

Clef letter+octave labels are the *staff-position* diatonic names (NO key
signature applied). Key-signature accidentals are applied later when writing
tokens. This tool only reads geometry from the PNG; it never reads GT semantics.
"""
import json
import os
import sys
from PIL import Image, ImageDraw, ImageFont

BATCH = sys.argv[1] if len(sys.argv) > 1 else "batch_12"
ONLY = sys.argv[2] if len(sys.argv) > 2 else None  # optional stem filter
CLEF = sys.argv[3] if len(sys.argv) > 3 else "G2"  # G2 or C1 or F4 ...

batch_file = f"examples/omr_benchmark/batches/{BATCH}.json"
out_root = f"data/_omr_view/{BATCH}"

TILE_W = 1500
OVERLAP = 340
TARGET_H = 520
MAX_F = 5.0
MIN_F = 2.5
LEFT_PAD = 66

# diatonic scale letters ascending
LETTERS = ["C", "D", "E", "F", "G", "A", "B"]

def pitch_add(letter, octave, steps):
    """Move `steps` diatonic steps up (positive) from letter/octave."""
    idx = LETTERS.index(letter) + octave * 7
    idx += steps
    return LETTERS[idx % 7], idx // 7

# bottom staff line pitch for each clef
BOTTOM_LINE = {
    "G2": ("E", 4),
    "C1": ("C", 4),
    "C3": ("F", 3),
    "C4": ("D", 3),
    "F4": ("G", 2),
    "F3": ("B", 2),
}

def detect_staff_lines(im_L):
    w, h = im_L.size
    px = im_L.load()
    rowdark = []
    for y in range(h):
        c = 0
        for x in range(0, w, 2):  # subsample columns for speed
            if px[x, y] < 128:
                c += 1
        rowdark.append(c)
    thr = 0.5 * (w / 2)
    bands = []
    y = 0
    while y < h:
        if rowdark[y] > thr:
            y0 = y
            while y < h and rowdark[y] > thr:
                y += 1
            bands.append((y0 + y - 1) / 2.0)
        else:
            y += 1
    # If more than 5 bands, keep the 5 with largest dark counts
    if len(bands) > 5:
        bands_sorted = sorted(bands, key=lambda b: -rowdark[int(b)])[:5]
        bands = sorted(bands_sorted)
    return bands

def main():
    with open(batch_file) as f:
        batch = json.load(f)
    manifest = {}
    for entry in batch:
        path = entry["image_path"]
        stem = os.path.splitext(os.path.basename(path))[0]
        if ONLY and ONLY != stem:
            continue
        im = Image.open(path).convert("L")
        w, h = im.size
        lines = detect_staff_lines(im)
        if len(lines) != 5:
            print(f"WARN {stem}: found {len(lines)} staff lines: {lines}")
            if len(lines) < 2:
                continue
        top = lines[0]
        bot = lines[-1]
        spacing = (bot - top) / (len(lines) - 1)
        step = spacing / 2.0  # half a space per diatonic step

        f_ = max(MIN_F, min(MAX_F, TARGET_H / h))
        big = im.convert("RGB").resize((int(w * f_), int(h * f_)), Image.LANCZOS)
        bw, bh = big.size
        canvas = Image.new("RGB", (bw + LEFT_PAD, bh), "white")
        canvas.paste(big, (LEFT_PAD, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        bl_letter, bl_oct = BOTTOM_LINE.get(CLEF, ("E", 4))
        # bottom staff line index (0) up to some range; draw from -4 (below) to +5 (above) steps beyond staff
        # staff spans 8 steps (bottom line to top line = 8 diatonic steps)
        n_lines = len(lines)
        top_line_steps = (n_lines - 1) * 2  # steps from bottom line to top line
        for s in range(-5, top_line_steps + 6):
            y_orig = bot - s * step
            y = y_orig * f_
            if y < -5 or y > bh + 5:
                continue
            letter, octave = pitch_add(bl_letter, bl_oct, s)
            is_line = (s % 2 == 0)
            color = (0, 160, 0) if is_line else (230, 90, 0)
            draw.line([(LEFT_PAD, y), (bw + LEFT_PAD, y)], fill=color, width=1)
            draw.text((2, y - 10), f"{letter}{octave}", fill=color, font=font)

        d = os.path.join(out_root, stem)
        os.makedirs(d, exist_ok=True)
        canvas.save(os.path.join(d, "annot_overview.png"))

        # tile
        cw, ch = canvas.size
        tiles = []
        if cw <= TILE_W:
            canvas.save(os.path.join(d, "annot_00.png"))
            tiles.append("annot_00.png")
        else:
            stepx = TILE_W - OVERLAP
            x = 0
            i = 0
            while x < cw:
                right = min(x + TILE_W, cw)
                # keep left margin labels visible: paste margin onto each tile
                crop = canvas.crop((x, 0, right, ch))
                if x > 0:
                    marg = canvas.crop((0, 0, LEFT_PAD, ch))
                    crop.paste(marg, (0, 0))
                name = f"annot_{i:02d}.png"
                crop.save(os.path.join(d, name))
                tiles.append(name)
                i += 1
                if right >= cw:
                    break
                x += stepx
        manifest[stem] = {"staff_lines": [round(v, 1) for v in lines],
                          "spacing": round(spacing, 2), "factor": round(f_, 2),
                          "clef": CLEF, "tiles": tiles}
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
