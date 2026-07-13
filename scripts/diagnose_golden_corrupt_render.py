"""Report whether golden corrupt renders contain all semantic measures."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from primus_semantic_render import _get_verovio, semantic_to_mei  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "golden_set" / "manifest.jsonl"


def main() -> None:
    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("source") == "corrupt"]

    for row in rows:
        semantic = Path(row["gt_path"]).read_text(encoding="utf-8")
        expected = semantic.split().count("barline") + 1
        mei = semantic_to_mei(semantic)
        tk = _get_verovio()
        tk.loadData(mei)
        pages = tk.getPageCount()
        svg = tk.renderToSVG(1)
        rendered = len(re.findall(r'<g[^>]+class="measure"', svg))
        with Image.open(row["image_path"]) as image:
            size = f"{image.width}x{image.height}"
        print(
            f"{row['group']:27} {row['id']} semantic={expected:2} "
            f"page1={rendered:2} pages={pages} image={size}"
        )


if __name__ == "__main__":
    main()
