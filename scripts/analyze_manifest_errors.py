"""Summarize error rows in camera_distort and corrupt manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def last_rows(path: Path, id_key: str) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seen[row[id_key]] = row
    return seen


def summarize(name: str, path: Path, id_key: str) -> None:
    rows = last_rows(path, id_key)
    ok = [r for r in rows.values() if r.get("status") == "ok"]
    errs = [r for r in rows.values() if r.get("status") != "ok"]
    print(f"=== {name} ===")
    print(f"ok={len(ok)} err={len(errs)} total_ids={len(rows)}")
    msgs = Counter((e.get("error") or "(no error field)")[:150] for e in errs)
    print("Error types:")
    for msg, count in msgs.most_common(25):
        print(f"  {count:5d}x  {msg}")
    if errs:
        sample = errs[:5]
        print("Sample IDs:")
        for e in sample:
            key = e[id_key]
            err = (e.get("error") or "")[:100]
            print(f"  {key}: {err}")
    print()


def main() -> None:
    summarize(
        "CAMERA_DISTORT",
        ROOT / "data" / "mashcima" / "camera_distort_manifest.jsonl",
        "incipit_id",
    )
    summarize(
        "CORRUPT_BATCH",
        ROOT / "data" / "corrupt" / "manifest.jsonl",
        "corrupt_id",
    )


if __name__ == "__main__":
    main()
