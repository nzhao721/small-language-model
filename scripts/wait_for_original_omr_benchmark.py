"""Poll until original-only 10-sample OMR benchmark artifacts exist."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "examples" / "omr_benchmark" / "selection_10.json"
HOLDOUT = ROOT / "data" / "omr_benchmark_holdout"
PRED_BASE = ROOT / "examples" / "omr_benchmark" / "predictions_10"
MODELS = [
    "claude-opus-4-8-thinking-high",
    "gemini-3.1-pro",
    "composer-2.5",
    "grok-4.5-xhigh",
    "gpt-5.6-sol-medium",
]
READY_FLAG = ROOT / "data" / "omr_benchmark_original_10_ready.json"
LOG = ROOT / "data" / "omr_benchmark_corrupt_wait.log"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_prediction_lines(model_dir: Path) -> int:
    if not model_dir.is_dir():
        return 0
    total = 0
    for path in model_dir.glob("*.jsonl"):
        total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


def status() -> dict:
    sel_count = 0
    if SELECTION.is_file():
        sel_count = len(json.loads(SELECTION.read_text(encoding="utf-8")))
    holdout_files = 0
    if HOLDOUT.is_dir():
        holdout_files = sum(1 for _ in HOLDOUT.rglob("*") if _.is_file())
    model_lines = {m: count_prediction_lines(PRED_BASE / m) for m in MODELS}
    ready = (
        sel_count == 30
        and holdout_files > 0
        and all(count_prediction_lines(PRED_BASE / m) >= 30 for m in MODELS)
    )
    return {
        "ready": ready,
        "selection_count": sel_count,
        "holdout_files": holdout_files,
        "model_lines": model_lines,
    }


def main() -> None:
    poll_seconds = 90
    max_wait_seconds = 2 * 60 * 60
    elapsed = 0
    log("Waiting for original-only OMR benchmark (10x3)...")
    while elapsed <= max_wait_seconds:
        s = status()
        log(
            "poll: selection={}/30 holdout_files={} pred_lines={}".format(
                s["selection_count"],
                s["holdout_files"],
                s["model_lines"],
            )
        )
        if s["ready"]:
            READY_FLAG.write_text(json.dumps(s, indent=2), encoding="utf-8")
            log(f"READY: {READY_FLAG}")
            return
        time.sleep(poll_seconds)
        elapsed += poll_seconds
    s = status()
    READY_FLAG.write_text(json.dumps({"ready": False, **s}, indent=2), encoding="utf-8")
    log(f"TIMEOUT after {max_wait_seconds}s — original benchmark incomplete")
    sys.exit(1)


if __name__ == "__main__":
    main()
