"""Generate Grok OMR benchmark prompt files from Opus templates."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "examples" / "omr_benchmark" / "prompts"

REPLACEMENTS = [
    ("# Opus subagent", "# Grok 4.5 high subagent"),
    ("model: claude-opus-4-8-thinking-high", "model: grok-4.5-xhigh"),
    ("description: Opus OMR", "description: Grok OMR"),
    (
        "predictions_10_corrupt/claude-opus-4-8-thinking-high/",
        "predictions_10_corrupt/grok-4.5-xhigh/",
    ),
    (
        "predictions_manual/claude-opus-4-8-thinking-high/",
        "predictions_manual/grok-4.5-xhigh/",
    ),
    (
        "predictions_10/claude-opus-4-8-thinking-high/",
        "predictions_10/grok-4.5-xhigh/",
    ),
]

OPUS_TO_GROK = {
    "opus_batch_handwritten_corrupt.md": "grok_batch_handwritten_corrupt.md",
    "opus_batch_clean_corrupt.md": "grok_batch_clean_corrupt.md",
    "opus_batch_camera_corrupt.md": "grok_batch_camera_corrupt.md",
    "opus_batch_manual_handwritten_00.md": "grok_batch_manual_handwritten_00.md",
    "opus_batch_manual_handwritten_01.md": "grok_batch_manual_handwritten_01.md",
    "opus_batch_manual_digital_00.md": "grok_batch_manual_digital_00.md",
    "opus_batch_manual_digital_01.md": "grok_batch_manual_digital_01.md",
}

for opus_name, grok_name in OPUS_TO_GROK.items():
    src = PROMPTS / opus_name
    dst = PROMPTS / grok_name
    text = src.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    text = text.replace(
        "Overwrite predictions_10_corrupt/grok-4.5-xhigh/",
        "predictions_10_corrupt/grok-4.5-xhigh/",
    )
    if "missing" not in text.lower() and "Status:" in text:
        text = text.replace(
            "**Status:** Rerun required",
            "**Status:** predictions missing — launch to create",
        )
    dst.write_text(text, encoding="utf-8")
    print(f"wrote {dst.name}")
