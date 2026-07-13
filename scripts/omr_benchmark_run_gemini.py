"""Run Gemini vision transcription for OMR benchmark batches."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "examples" / "omr_benchmark"


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value

SYSTEM_PROMPT = """You transcribe monophonic sheet music images to PrIMuS semantic format.

Output ONLY a single space-separated token string (no markdown, no explanation).

PrIMuS tokens:
- Header: clef-G2 (or clef-C1, clef-C3, clef-F4, clef-C2, clef-F3), keySignature-CM (or DM, GM, BbM, AbM, Fm, etc.), timeSignature-C (or 2/4, 3/4, 4/4, C/, etc.)
- Notes: note-{Pitch}{Octave}_{duration} e.g. note-C5_quarter, note-F#4_eighth, note-Bb3_half
- Rests: rest-{duration} or rest_{duration}
- Barlines: barline
- Durations: whole, half, quarter, eighth, sixteenth, thirty_second, double_whole (append . for dotted)
- Use # for sharps and b for flats in pitch names (e.g. F#4, Bb4, Eb5)
"""


def load_image_part(path: Path) -> types.Part:
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "image/png"
    data = path.read_bytes()
    return types.Part.from_bytes(data=data, mime_type=mime)


def transcribe_image(client: genai.Client, model: str, image_path: Path) -> str:
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=SYSTEM_PROMPT),
                    load_image_part(image_path),
                    types.Part.from_text(
                        text="Transcribe this monophonic melody. Return only the PrIMuS token string."
                    ),
                ],
            )
        ],
        config=types.GenerateContentConfig(temperature=0.0),
    )
    text = (response.text or "").strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = " ".join(lines).strip()
    return text


def run_batch(
    client: genai.Client,
    model: str,
    batch_path: Path,
    out_path: Path,
    *,
    delay_s: float = 1.0,
) -> int:
    entries = json.loads(batch_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, entry in enumerate(entries):
        image_path = Path(entry["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        print(f"  [{i + 1}/{len(entries)}] {entry['filename']} ...", flush=True)
        prediction = transcribe_image(client, model, image_path)
        record = {"filename": entry["filename"], "prediction": prediction}
        lines.append(json.dumps(record, ensure_ascii=False))
        if delay_s and i + 1 < len(entries):
            time.sleep(delay_s)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def merge_predictions(model_dir: Path, out_path: Path) -> int:
    lines: list[str] = []
    for batch_file in sorted(model_dir.glob("batch_*.jsonl")):
        for line in batch_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                lines.append(line)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--batches-dir", type=Path, default=BENCH / "batches_manual")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCH / "predictions_manual" / "gemini-3.1-pro",
    )
    parser.add_argument("--batch", help="Run single batch stem e.g. batch_00")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    for env_path in (ROOT / ".env", ROOT.parent / ".env.local", ROOT.parent / ".env"):
        _load_env_file(env_path)
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.merge_only:
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir.parent / f"{args.output_dir.name}.jsonl"
    if args.output_dir.name != "gemini-3.1-pro":
        merged_path = args.output_dir.parent / "gemini-3.1-pro.jsonl"

    if args.merge_only:
        n = merge_predictions(args.output_dir, merged_path)
        print(f"Merged {n} lines -> {merged_path}")
        return

    client = genai.Client(api_key=api_key)
    batch_files = sorted(args.batches_dir.glob("batch_*.json"))
    if args.batch:
        batch_files = [args.batches_dir / f"{args.batch}.json"]

    total = 0
    for batch_path in batch_files:
        out_path = args.output_dir / f"{batch_path.stem}.jsonl"
        print(f"Batch {batch_path.name} -> {out_path}", flush=True)
        n = run_batch(client, args.model, batch_path, out_path, delay_s=args.delay)
        print(f"  wrote {n} lines", flush=True)
        total += n

    n = merge_predictions(args.output_dir, merged_path)
    print(f"Done: {total} batch lines, merged {n} -> {merged_path}")


if __name__ == "__main__":
    main()
