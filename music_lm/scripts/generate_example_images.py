"""Generate 4-bar music-LM excerpts and render them as notation images."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parents[1]
_RENDERING = _REPO / "scripts" / "rendering"
_MUSIC_LM_DIR = _SCRIPTS.parent

sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_RENDERING))

from generate_music import (  # noqa: E402
    build_prompt,
    constrained_generate,
    load_checkpoint,
)
from music_lm.keys import (  # noqa: E402
    all_key_tokens,
    first_note_for,
    validate_first_note_is_tonic,
)
from music_lm.pitch import MAX_JUMP_SEMITONES, note_pitch_midi, pitch_limits_for_clef
from music_lm.repetition import is_overly_repetitive
from music_lm.timing import measure_ticks, token_tick_cost  # noqa: E402
from primus_semantic_render import render_semantic_to_image  # noqa: E402

CHECKPOINT = _MUSIC_LM_DIR / "checkpoints_medium" / "best.pt"
VOCAB = _MUSIC_LM_DIR / "vocab.json"
OUT_DIR = _MUSIC_LM_DIR / "examples"

PITCH_TEMPERATURE = 0.9
DURATION_TEMPERATURE = 1.2
MAX_CONSECUTIVE_SAME_PITCH = 3
MAX_CONSECUTIVE_SAME_DURATION = 6
MAX_GENERATION_ATTEMPTS = 32

CONTEXTS = [
    {
        "name": "treble_4-4",
        "clef": "clef-G2",
        "time": "timeSignature-4/4",
        "duration": "quarter",
    },
    {
        "name": "treble_6-8",
        "clef": "clef-G2",
        "time": "timeSignature-6/8",
        "duration": "eighth",
    },
    {
        "name": "bass_4-4",
        "clef": "clef-F4",
        "time": "timeSignature-4/4",
        "duration": "quarter",
    },
    {
        "name": "bass_6-8",
        "clef": "clef-F4",
        "time": "timeSignature-6/8",
        "duration": "eighth",
    },
]


def build_configs(token_to_id: dict[str, int]) -> list[dict]:
    configs: list[dict] = []
    for index, (key_token_value, label) in enumerate(all_key_tokens()):
        ctx = CONTEXTS[index % len(CONTEXTS)]
        configs.append(
            {
                "name": f"{ctx['name']}_{label}",
                "clef": ctx["clef"],
                "time": ctx["time"],
                "key": key_token_value,
                "first_note": first_note_for(
                    key_token_value,
                    ctx["clef"],
                    ctx["duration"],
                    token_to_id=token_to_id,
                ),
            }
        )
    return configs


def tokens_to_semantic_text(tokens: list[str]) -> str:
    """Drop specials and stop at eos for rendering."""
    out: list[str] = []
    for token in tokens:
        if token in {"<bos>", "<eos>", "<pad>", "<unk>"}:
            if token == "<eos>":
                break
            continue
        out.append(token)
    return "\t".join(out)


def validate_pitch_constraints(tokens: list[str], clef_token: str) -> None:
    """Raise if any note violates clef range or max jump size."""
    low, high = pitch_limits_for_clef(clef_token)
    prev_pitch: int | None = None

    for token in tokens:
        if token in {"<bos>", "<eos>", "<pad>", "<unk>"}:
            continue
        if token.startswith(("clef-", "keySignature-", "timeSignature-")):
            continue
        if token == "tie":
            continue
        if token.startswith("rest-"):
            prev_pitch = None
            continue

        midi = note_pitch_midi(token)
        if midi is None:
            continue

        if not (low <= midi <= high):
            raise ValueError(
                f"pitch {token} is outside {clef_token} range "
                f"({low}-{high} MIDI)"
            )
        if prev_pitch is not None and abs(midi - prev_pitch) > MAX_JUMP_SEMITONES:
            raise ValueError(
                f"jump from MIDI {prev_pitch} to {midi} exceeds "
                f"{MAX_JUMP_SEMITONES} semitones"
            )
        prev_pitch = midi


def validate_measure_lengths(tokens: list[str], time_sig_token: str) -> None:
    """Raise if any completed measure does not fill the time signature."""
    ticks_per_measure = measure_ticks(time_sig_token)
    current_ticks = 0
    in_measure = False

    for token in tokens:
        if token in {"<bos>", "<eos>", "<pad>", "<unk>"}:
            continue
        if token.startswith(("clef-", "keySignature-", "timeSignature-")):
            continue
        if token == "tie":
            continue
        if token == "barline":
            if in_measure and current_ticks != ticks_per_measure:
                raise ValueError(
                    f"measure has {current_ticks} ticks, expected {ticks_per_measure}"
                )
            current_ticks = 0
            in_measure = True
            continue

        in_measure = True
        current_ticks += token_tick_cost(token)

    if in_measure and 0 < current_ticks != ticks_per_measure:
        raise ValueError(
            f"final measure has {current_ticks} ticks, expected {ticks_per_measure}"
        )


def clear_output_dir() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    clear_output_dir()
    device = "cpu"

    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    token_to_id = vocab["token_to_id"]
    id_to_token = vocab["id_to_token"]
    configs = build_configs(token_to_id)

    model, ckpt = load_checkpoint(str(CHECKPOINT), device)
    print(
        f"Loaded checkpoint epoch={ckpt.get('epoch')} "
        f"val_loss={ckpt.get('val_loss'):.4f}"
    )

    manifest: list[dict] = []

    for sample_idx, cfg in enumerate(configs, start=1):
        prompt_ids = build_prompt(
            token_to_id,
            cfg["clef"],
            cfg["key"],
            cfg["time"],
            cfg["first_note"],
        )

        tokens: list[str] = []
        seed = 4000 + sample_idx
        for attempt in range(MAX_GENERATION_ATTEMPTS):
            seed = 4000 + sample_idx + attempt * 97
            torch.manual_seed(seed)

            ids = constrained_generate(
                model,
                token_to_id,
                id_to_token,
                prompt_ids,
                cfg["time"],
                bars=4,
                temperature_pitch=PITCH_TEMPERATURE,
                temperature_duration=DURATION_TEMPERATURE,
                top_k=None,
                no_rests=True,
                device=device,
                max_consecutive_same_pitch=MAX_CONSECUTIVE_SAME_PITCH,
                max_consecutive_same_duration=MAX_CONSECUTIVE_SAME_DURATION,
            )
            tokens = [id_to_token[tid] for tid in ids]
            validate_measure_lengths(tokens, cfg["time"])
            validate_pitch_constraints(tokens, cfg["clef"])
            validate_first_note_is_tonic(tokens, cfg["key"])
            if not is_overly_repetitive(
                tokens,
                max_same_pitch_run=MAX_CONSECUTIVE_SAME_PITCH,
                max_same_duration_run=MAX_CONSECUTIVE_SAME_DURATION,
            ):
                break
        else:
            raise RuntimeError(
                f"could not generate non-repetitive excerpt for {cfg['key']} "
                f"after {MAX_GENERATION_ATTEMPTS} attempts"
            )

        semantic_text = tokens_to_semantic_text(tokens)

        label = (
            f"{cfg['name']}_p{PITCH_TEMPERATURE:.1f}_d{DURATION_TEMPERATURE:.1f}"
            f"_s{seed}"
        )
        txt_path = OUT_DIR / f"{label}.semantic"
        img_path = OUT_DIR / f"{label}.png"

        txt_path.write_text(semantic_text, encoding="utf-8")
        image = render_semantic_to_image(semantic_text)
        image.save(img_path)

        entry = {
            "file": img_path.name,
            "clef": cfg["clef"],
            "key": cfg["key"],
            "time": cfg["time"],
            "first_note": cfg["first_note"],
            "pitch_temperature": PITCH_TEMPERATURE,
            "duration_temperature": DURATION_TEMPERATURE,
            "seed": seed,
            "tokens": tokens,
        }
        manifest.append(entry)
        print(f"[{sample_idx:02d}] saved {img_path.name} ({cfg['key']})")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Wrote {len(manifest)} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
