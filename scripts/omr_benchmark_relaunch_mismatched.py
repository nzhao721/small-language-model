"""Print job indices that need relaunch with correct subagent model slug."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "examples" / "omr_benchmark" / "rerun_jobs.jsonl"

SUBAGENT_MODEL = {
    "claude-opus-4-8-thinking-high": "claude-opus-4-8-thinking-max",
    "gemini-3.1-pro": "gemini-3.1-pro",
    "composer-2.5": "composer-2.5",
    "grok-4.5-xhigh": "grok-4.5-xhigh",
    "gpt-5.6-sol-medium": "gpt-5.6-sol-medium",
}


def main() -> None:
    jobs = [
        json.loads(line)
        for line in JOBS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Jobs 0-4 were launched with correct model pairing.
    for i, job in enumerate(jobs):
        if i < 5:
            continue
        out = ROOT / job["output"]
        if out.is_file():
            out.unlink()
        print(
            json.dumps(
                {
                    "index": i,
                    "subagent_model": SUBAGENT_MODEL[job["model"]],
                    "prompt_file": f"data/omr_benchmark_job_prompts/job_{i:02d}.txt",
                }
            )
        )


if __name__ == "__main__":
    main()
