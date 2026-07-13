"""Invert Mashcima PNGs from white-on-black to black-on-white."""

from __future__ import annotations

from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MASHCIMA_DIR = ROOT / "data" / "mashcima"


def invert_png(path: Path) -> None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    cv2.imwrite(str(path), 255 - img)


def main() -> None:
    pngs = sorted(MASHCIMA_DIR.glob("*.png"))
    for path in tqdm(pngs, desc="Invert Mashcima PNGs", unit="file"):
        invert_png(path)
    print(f"Inverted {len(pngs)} files in {MASHCIMA_DIR}")


if __name__ == "__main__":
    main()
