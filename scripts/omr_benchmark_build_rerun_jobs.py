"""Build manifest of 50 OMR benchmark rerun jobs (5 models × 10 batches)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "examples" / "omr_benchmark"
OUT = BENCH / "rerun_jobs.jsonl"

MODELS = [
    "claude-opus-4-8-thinking-high",
    "gemini-3.1-pro",
    "composer-2.5",
    "grok-4.5-xhigh",
    "gpt-5.6-sol-medium",
]

SUITES = [
    ("batches_10", "predictions_10"),
    ("batches_10_corrupt", "predictions_10_corrupt"),
    ("batches_manual", "predictions_manual"),
]

PROMPT_TEMPLATE = """Workspace: C:\\alpha_ai\\small-language-model

Repaired golden-set OMR benchmark rerun. Transcribe 10 {group} sheet music images to PrIMuS semantic format.

Batch JSON: {batch_rel}

Process images **serially** (one at a time, in order). For each image:
1. Read the image file only.
2. Transcribe the monophonic melody to a PrIMuS semantic token string.
3. **NEVER** read ground_truth_semantic_path, any .semantic file, or any golden_set/**/ground_truth/ path.

Images (10):
{image_list}

PrIMuS tokens (space-separated in prediction):
- Header: clef-G2 (or clef-C1, clef-C3, clef-F4, clef-C2), keySignature-CM (or DM, GM, BbM, etc.), timeSignature-C (or 2/4, 3/4, 4/4, C/, etc.)
- Notes: note-{{Pitch}}{{Octave}}_{{duration}} e.g. note-C5_quarter, note-F#4_eighth, note-Bb3_half
- Rests: rest-{{duration}} or rest_{{duration}} e.g. rest-quarter
- Barlines: barline
- Durations: whole, half, quarter, eighth, sixteenth, thirty_second, double_whole (append . for dotted)

Output file: {out_rel}

Write exactly 10 JSON lines (one per image, in order):
{{"filename": "<basename from batch entry>", "prediction": "clef-G2 keySignature-CM ..."}}

Create parent directories if needed. Overwrite the output file. Return the output path and confirm 10 lines written."""


def main() -> None:
    jobs: list[dict] = []
    for batches_name, pred_name in SUITES:
        batches_dir = BENCH / batches_name
        pred_dir = BENCH / pred_name
        for batch_file in sorted(batches_dir.glob("batch_*.json")):
            entries = json.loads(batch_file.read_text(encoding="utf-8"))
            group = entries[0]["group"] if entries else "unknown"
            image_lines = []
            for i, entry in enumerate(entries, 1):
                image_lines.append(f"{i}. {entry['image_path']}")
            batch_rel = batch_file.relative_to(ROOT).as_posix()
            for model in MODELS:
                out_path = pred_dir / model / f"{batch_file.stem}.jsonl"
                out_rel = out_path.relative_to(ROOT).as_posix()
                prompt = PROMPT_TEMPLATE.format(
                    group=group,
                    batch_rel=batch_rel,
                    image_list="\n".join(image_lines),
                    out_rel=out_rel,
                )
                jobs.append(
                    {
                        "id": f"{model}|{batch_rel}",
                        "model": model,
                        "batch_file": batch_rel,
                        "batch_name": batch_file.stem,
                        "group": group,
                        "predictions_dir": pred_dir.relative_to(ROOT).as_posix(),
                        "output": out_rel,
                        "n_images": len(entries),
                        "prompt": prompt,
                    }
                )

    OUT.write_text(
        "\n".join(json.dumps(j, ensure_ascii=False) for j in jobs) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(jobs)} jobs to {OUT}")


if __name__ == "__main__":
    main()
