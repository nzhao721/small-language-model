"""Generate compressed image quality previews for examples/compression_preview/."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from image_compress import compress_image_file, resolved_image_path
from dataset_layout import camera_distorted_path, mashcima_image_path, primus_png_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = ROOT / "examples" / "compression_preview"
MAX_KB = 100

SAMPLES: list[tuple[str, Path]] = [
    ("mashcima", mashcima_image_path("000114312-1_1_1")),
    ("mashcima (2)", mashcima_image_path("000051756-1_1_1")),
    ("primus", primus_png_path("230000054-1_8_2")),
    ("camera-primus", camera_distorted_path("000126577-1_1_1")),
    (
        "corrupt / handwritten",
        ROOT / "data" / "corrupt" / "handwritten" / "0af106efaf3647aebba1c9e53c952968.png",
    ),
    (
        "corrupt / clean",
        ROOT / "data" / "corrupt" / "clean" / "763a9b3df7d5468dbb6d078c90dfd193.png",
    ),
    (
        "corrupt / camera",
        ROOT / "data" / "corrupt" / "camera" / "4502448ef3f24e829a3a8ce37b3b9d46.png",
    ),
]


def slug(label: str) -> str:
    return (
        label.lower()
        .replace(" / ", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def kb(size: int) -> float:
    return round(size / 1024, 1)


def main() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for label, src in SAMPLES:
        if not src.is_file():
            print(f"SKIP missing: {src}")
            continue

        name = slug(label)
        original_name = f"{name}_original{src.suffix.lower()}"
        original_path = PREVIEW_DIR / original_name
        shutil.copy2(src, original_path)
        original_kb = kb(original_path.stat().st_size)

        work = PREVIEW_DIR / f"{name}_work{src.suffix.lower()}"
        shutil.copy2(src, work)
        compress_image_file(work, max_kb=MAX_KB)
        compressed_path = resolved_image_path(work)
        compressed_name = f"{name}_compressed{compressed_path.suffix.lower()}"
        final_compressed = PREVIEW_DIR / compressed_name
        if compressed_path.resolve() != final_compressed.resolve():
            if final_compressed.is_file():
                final_compressed.unlink()
            compressed_path.replace(final_compressed)
        compressed_kb = kb(final_compressed.stat().st_size)

        if work.is_file() and work.resolve() != final_compressed.resolve():
            work.unlink(missing_ok=True)

        entries.append(
            {
                "label": label,
                "original": original_name,
                "compressed": compressed_name,
                "original_kb": original_kb,
                "compressed_kb": compressed_kb,
                "source": str(src.relative_to(ROOT)).replace("\\", "/"),
            }
        )
        print(f"{label}: {original_kb} KB -> {compressed_kb} KB")

    meta_path = PREVIEW_DIR / "preview_meta.json"
    meta_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")

    html = render_index(entries)
    index_path = PREVIEW_DIR / "index.html"
    index_path.write_text(html, encoding="utf-8")
    print(f"\nWrote {len(entries)} previews to {PREVIEW_DIR.relative_to(ROOT)}")


def render_index(entries: list[dict]) -> str:
    cards = []
    for e in entries:
        saved = round(e["original_kb"] - e["compressed_kb"], 1)
        pct = round(100 * (1 - e["compressed_kb"] / e["original_kb"]), 1) if e["original_kb"] else 0
        cards.append(
            f"""
    <article class="card">
      <h2>{e["label"]}</h2>
      <p class="meta">Source: <code>{e["source"]}</code></p>
      <div class="pair">
        <figure>
          <figcaption>Original — {e["original_kb"]} KB</figcaption>
          <a href="{e["original"]}"><img src="{e["original"]}" alt="{e["label"]} original" loading="lazy"></a>
        </figure>
        <figure>
          <figcaption>Compressed (≤{MAX_KB} KB) — {e["compressed_kb"]} KB</figcaption>
          <a href="{e["compressed"]}"><img src="{e["compressed"]}" alt="{e["label"]} compressed" loading="lazy"></a>
        </figure>
      </div>
      <p class="savings">Saved {saved} KB ({pct}%)</p>
    </article>"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dataset image compression preview (≤{MAX_KB} KB)</title>
  <style>
    :root {{
      --bg: #f4f4f5;
      --card: #fff;
      --text: #18181b;
      --muted: #52525b;
      --border: #e4e4e7;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      margin: 0;
      padding: 2rem 1.5rem 3rem;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    header {{ max-width: 1100px; margin: 0 auto 2rem; }}
    h1 {{ margin: 0 0 0.5rem; font-size: 1.75rem; }}
    .lead {{ color: var(--muted); margin: 0; max-width: 70ch; }}
    .status {{
      margin-top: 1rem;
      padding: 0.75rem 1rem;
      background: #fef9c3;
      border: 1px solid #fde047;
      border-radius: 8px;
      font-size: 0.95rem;
    }}
    .grid {{
      max-width: 1100px;
      margin: 0 auto;
      display: grid;
      gap: 1.5rem;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }}
    .card h2 {{ margin: 0 0 0.35rem; font-size: 1.15rem; }}
    .meta {{ margin: 0 0 1rem; font-size: 0.85rem; color: var(--muted); }}
    code {{ font-size: 0.8rem; word-break: break-all; }}
    .pair {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1rem;
    }}
    figure {{ margin: 0; }}
    figcaption {{
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: var(--muted);
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
    }}
    a {{ color: inherit; text-decoration: none; }}
    a:hover img {{ outline: 2px solid var(--accent); }}
    .savings {{
      margin: 0.75rem 0 0;
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <header>
    <h1>Dataset image compression preview</h1>
    <p class="lead">
      Samples compressed with the same pipeline as
      <code>scripts/compress_dataset_images.py</code>
      (resize + JPEG quality search, target ≤{MAX_KB} KB per file).
      PNG sources become JPEG; JPG sources are re-encoded in place.
    </p>
    <p class="status">
      Full dataset compression has <strong>not started</strong> yet — these are non-destructive previews only.
    </p>
  </header>
  <main class="grid">
{"".join(cards)}
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    main()
