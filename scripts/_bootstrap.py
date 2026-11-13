"""Register scripts/ category subfolders on sys.path for flat cross-imports."""

from __future__ import annotations

import sys
from pathlib import Path


def setup(anchor: str | Path) -> Path:
    """Add each scripts/<category>/ dir to sys.path; return the repository root."""
    scripts = Path(anchor).resolve()
    while scripts.name != "scripts" and scripts.parent != scripts:
        scripts = scripts.parent
    if scripts.name != "scripts":
        raise RuntimeError(f"could not locate scripts/ from {anchor!r}")

    for sub in sorted(scripts.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_") or sub.name == "__pycache__":
            continue
        path = str(sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    return scripts.parent
