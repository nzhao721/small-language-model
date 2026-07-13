"""Diagnostic: reproduce corrupt-batch render failures against current code."""

from __future__ import annotations

import json
import random
import sys
import traceback
from collections import Counter
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

import mashcima as mc  # noqa: E402
from generate_mashcima_batch import (  # noqa: E402
    DEFAULT_MUSCIMA_DIR,
    DEFAULT_PRIMUS_TGZ,
    ensure_muscima_cache,
)
from mashcima_writers import configure_mashcima_writers  # noqa: E402
from generate_corrupt_batch import (  # noqa: E402
    CORRUPT_DIR,
    DEFAULT_JOBS,
    DEFAULT_MANIFEST,
    ROOT,
    load_jobs,
    process_job,
)


def bucket(rec: dict) -> str:
    msg = (rec.get("error") or "").strip()
    if not msg:
        return "empty"
    if msg.startswith("job_timeout"):
        return "timeout"
    if msg in ("t", "'t'"):
        return "keyT"
    if "0-dimensional" in msg:
        return "numpy0d"
    return "other"


def main() -> None:
    ensure_muscima_cache(DEFAULT_PRIMUS_TGZ, DEFAULT_MUSCIMA_DIR)
    configure_mashcima_writers("per-score")

    last: dict[str, dict] = {}
    for line in DEFAULT_MANIFEST.open(encoding="utf-8"):
        r = json.loads(line)
        last[r["corrupt_id"]] = r

    jobs = {j["corrupt_id"]: j for j in load_jobs(DEFAULT_JOBS)}

    want = {"empty": 5, "keyT": 5, "numpy0d": 3}
    samples: dict[str, list[str]] = {k: [] for k in want}
    for cid, r in last.items():
        if r.get("status") != "error":
            continue
        b = bucket(r)
        if b in samples and len(samples[b]) < want[b] and cid in jobs:
            samples[b].append(cid)

    outcome = Counter()
    for b, cids in samples.items():
        print(f"==== bucket {b} ({len(cids)} samples) ====", flush=True)
        for cid in cids:
            j = jobs[cid]
            label = f"  {cid[:10]} {j['incipit_id']:16} {j['format']:11}"
            try:
                rec = process_job(j, ROOT / "data" / "primus", CORRUPT_DIR)
                print(f"{label} -> {rec.get('status')}", flush=True)
                outcome[f"{b}:{rec.get('status')}"] += 1
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc().splitlines()
                frames = [l for l in tb if l.strip().startswith("File")]
                loc = frames[-1].strip() if frames else ""
                print(f"{label} -> {type(e).__name__}: {str(e)[:60]}", flush=True)
                print(f"       at {loc}", flush=True)
                outcome[f"{b}:EXC:{type(e).__name__}"] += 1

    print("---- outcome summary ----", flush=True)
    for k, c in outcome.most_common():
        print(f"{c:4}  {k}", flush=True)


if __name__ == "__main__":
    main()
