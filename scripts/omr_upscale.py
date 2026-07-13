import json
import os
from PIL import Image

BATCH = r"C:\alpha_ai\small-language-model\examples\omr_benchmark\batches\batch_06.json"
OUT = r"C:\alpha_ai\small-language-model\data\omr_upscaled"
os.makedirs(OUT, exist_ok=True)

with open(BATCH, "r") as f:
    batch = json.load(f)

for i, entry in enumerate(batch):
    path = entry["image_path"]
    fn = entry["filename"]
    stem = os.path.splitext(fn)[0]
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # Target staff height readability: scale so height ~ 320px, cap width
    scale = max(1.0, 320.0 / h)
    nw, nh = int(w * scale), int(h * scale)
    big = img.resize((nw, nh), Image.LANCZOS)
    outp = os.path.join(OUT, f"{i:02d}_{stem}.png")
    big.save(outp)
    print(f"{i:02d} {fn} orig={w}x{h} -> {nw}x{nh} :: {outp}")

print("DONE")
