"""Build examples/compression_preview/ gallery from dataset samples."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from image_compress import compress_image_file, resolved_image_path
from dataset_layout import (  # noqa: E402
    camera_distorted_path,
    mashcima_image_path,
    primus_png_path,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "examples" / "compression_preview"
MAX_KB = 100

SAMPLES: list[tuple[str, Path]] = [
    ("mashcima_000051650", mashcima_image_path("000051650-1_1_1")),
    ("mashcima_000051662", mashcima_image_path("000051662-1_1_1")),
    ("primus_000051650", primus_png_path("000051650-1_1_1")),
    ("camera_primus_distorted", camera_distorted_path("000051650-1_1_1")),
    ("corrupt_clean", next(iter((DATA / "corrupt" / "clean").glob("*.png")), None)),
    ("corrupt_camera", next(iter((DATA / "corrupt" / "camera").glob("*.png")), None)),
    ("corrupt_handwritten", next(iter((DATA / "corrupt" / "handwritten").glob("*.png")), None)),
]


def kb(path: Path) -> float:
    return path.stat().st_size / 1024


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.iterdir():
        if old.name != ".gitkeep":
            old.unlink()

    entries: list[dict] = []

    for label, src in SAMPLES:
        if src is None or not src.is_file():
            print(f"SKIP {label}: source missing")
            continue

        orig_kb = round(kb(src), 1)
        ext = src.suffix.lower()
        dest = OUT / f"{label}{ext}"
        shutil.copy2(src, dest)

        if orig_kb > MAX_KB:
            compress_image_file(dest, max_kb=MAX_KB)
            dest = resolved_image_path(dest)
            action = "compressed"
        else:
            action = "already_small"

        final_kb = round(kb(dest), 1)
        entries.append(
            {
                "label": label,
                "file": dest.name,
                "original_kb": orig_kb,
                "final_kb": final_kb,
                "action": action,
                "source": str(src.relative_to(ROOT)).replace("\\", "/"),
            }
        )
        print(f"{label}: {orig_kb} KB -> {final_kb} KB ({action})")

    html = _build_html(entries)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"\nGallery: {OUT / 'index.html'}")


def _build_html(entries: list[dict]) -> str:
    sections = []
    for e in entries:
        sections.append(
            f"""    <figure>
      <figcaption>{e['label']}</figcaption>
      <img src="{e['file']}" alt="{e['label']}">
      <p class="meta">{e['final_kb']} KB (was {e['original_kb']} KB) &mdash; {e['action']}</p>
      <p class="src">{e['source']}</p>
    </figure>"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Compression preview (&le;100 KB)</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f6f6f6; }}
    h1 {{ margin-bottom: 0.25rem; }}
    .note {{ color: #444; margin-bottom: 1.5rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.25rem; }}
    figure {{ margin: 0; background: #fff; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    figcaption {{ font-weight: 600; margin-bottom: 0.5rem; }}
    img {{ width: 100%; height: auto; border: 1px solid #ddd; background: #fff; }}
    .meta {{ font-size: 0.9rem; margin: 0.5rem 0 0.25rem; }}
    .src {{ font-size: 0.75rem; color: #666; word-break: break-all; margin: 0; }}
  </style>
</head>
<body>
  <h1>Compression preview</h1>
  <p class="note">Sample images compressed to &le;100 KB using resize + JPEG quality search (same logic as <code>scripts/compress_dataset_images.py</code>). Full dataset compression has <strong>not started yet</strong>.</p>
  <div class="grid">
{chr(10).join(sections)}
  </div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
