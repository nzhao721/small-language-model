"""Generate a few Mashcima samples showcasing paper / staff / ink appearance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_layout import (  # noqa: E402
    DEFAULT_PRIMUS_TGZ,
    MASHCIMA_DIR,
    read_primus_agnostic,
)
from generate_mashcima_batch import (  # noqa: E402
    DEFAULT_MUSCIMA_DIR,
    ensure_muscima_cache,
    image_to_uint8,
    synthesize_handwritten,
)
from mashcima.primus_adapter import convert_primus_annotation_to_mashcima_annotation  # noqa: E402
from mashcima_appearance import apply_mashcima_appearance  # noqa: E402
from mashcima_writers import configure_mashcima_writers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "examples" / "mashcima_appearance_demo"
DEFAULT_IDS = [
    "000051650-1_1_1",
    "000051662-1_1_1",
    "000051756-1_1_1",
    "000051665-1_1_1",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--compare-plain", action="store_true", help="Also write *_plain.jpg")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_muscima_cache(DEFAULT_PRIMUS_TGZ, DEFAULT_MUSCIMA_DIR)
    configure_mashcima_writers("per-score")

    for incipit_id in args.ids:
        agnostic = read_primus_agnostic(incipit_id)
        converted = convert_primus_annotation_to_mashcima_annotation(agnostic)
        if converted is None:
            print(f"skip {incipit_id}: conversion failed", file=sys.stderr)
            continue
        raw = image_to_uint8(synthesize_handwritten(converted, incipit_id))
        styled = apply_mashcima_appearance(raw, key=incipit_id)
        out = args.out_dir / f"{incipit_id}.jpg"
        cv2.imwrite(str(out), styled, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"wrote {out.relative_to(ROOT)}")
        if args.compare_plain:
            plain = args.out_dir / f"{incipit_id}_plain.jpg"
            cv2.imwrite(str(plain), raw, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            print(f"wrote {plain.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
