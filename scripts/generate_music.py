"""Generate a short monophonic excerpt from the trained music-LM, conditioned
on a starting clef, time signature, (optional) key signature, and first note.

Two decode modes:

* Free sampling (default): the model predicts every token, including barlines
  and end-of-sequence. Fast, but nothing guarantees measures fill correctly or
  that the excerpt is a particular number of bars.

* Deterministic bar mode (--bars N): barlines and the end of the excerpt are
  placed by an exact rhythmic accountant, not the model. At every step the model
  may only choose among tokens that fit the beats remaining in the current
  measure; a barline is forced the instant a measure fills exactly, and <eos> is
  forced once N barlines have been placed. The model still chooses pitches and
  durations, but it can no longer produce a short/long measure or the wrong
  number of bars.

Examples:
    python scripts/generate_music.py --clef clef-G2 --time timeSignature-3/4 \
        --key keySignature-GM --first-note note-G4_quarter --num-samples 3

    python scripts/generate_music.py --clef clef-G2 --time timeSignature-4/4 \
        --key keySignature-GM --first-note note-G4_quarter --bars 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from music_lm.model import GPT, GPTConfig  # noqa: E402
from music_lm.timing import QUARTER_TICKS, duration_ticks, measure_ticks, validate_vocab  # noqa: E402

DEFAULT_DATA_DIR = ROOT / "data" / "music_lm"


def build_prompt_ids(
    token_to_id: dict[str, int],
    clef: str,
    time_sig: str,
    first_note: str,
    key_sig: str | None,
) -> list[int]:
    tokens = ["<bos>", clef]
    if key_sig:
        tokens.append(key_sig)
    tokens += [time_sig, first_note]

    unknown = [t for t in tokens if t not in token_to_id]
    if unknown:
        raise ValueError(f"Unknown token(s) not in vocab: {unknown}")

    return [token_to_id[t] for t in tokens]


def _sample_from_allowed(
    logits: torch.Tensor,
    allowed_ids: list[int],
    temperature: float,
    top_k: int | None,
    device: str,
) -> int:
    """Sample one token id, considering only the allowed ids. Never returns a
    token outside allowed_ids (all others are masked to -inf)."""
    masked = torch.full_like(logits, float("-inf"))
    idx = torch.tensor(sorted(set(allowed_ids)), device=device, dtype=torch.long)
    masked[idx] = logits[idx]
    masked = masked / max(temperature, 1e-6)

    if top_k is not None:
        k = min(top_k, idx.numel())
        kth = torch.topk(masked, k).values[-1]
        masked[masked < kth] = float("-inf")

    probs = F.softmax(masked, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def constrained_generate(
    model: GPT,
    prompt_ids: list[int],
    *,
    vocab: dict,
    time_sig_token: str,
    target_bars: int,
    pickup_beats: float | None,
    temperature: float,
    top_k: int | None,
    block_size: int,
    device: str,
    no_rests: bool = False,
) -> tuple[list[int], list[str]]:
    """Generate with barlines and end-of-excerpt placed deterministically.

    Returns (token_ids_including_bos_and_eos, warnings).
    """
    token_to_id = vocab["token_to_id"]
    id_to_token = vocab["id_to_token"]

    barline_id = token_to_id["barline"]
    eos_id = token_to_id["<eos>"]
    tie_id = token_to_id.get("tie")  # optional

    # Precompute duration ticks per id (None for non-duration tokens) and the
    # smallest note/rest duration (used as the measure-fill safety threshold).
    ticks_by_id: list[int | None] = [duration_ticks(t) for t in id_to_token]
    smallest_tick = validate_vocab(id_to_token)

    is_note = [t.startswith("note-") for t in id_to_token]

    full_measure = measure_ticks(time_sig_token)
    first_cap = full_measure if not pickup_beats else round(pickup_beats * QUARTER_TICKS)
    if first_cap <= 0:
        raise ValueError("pickup-beats must be positive")

    first_note_ticks = ticks_by_id[prompt_ids[-1]]
    if first_note_ticks is None:
        raise ValueError("--first-note must be a note or rest token")
    if first_note_ticks > first_cap:
        raise ValueError(
            f"first note ({id_to_token[prompt_ids[-1]]} = {first_note_ticks} ticks) is longer than the "
            f"first measure capacity ({first_cap} ticks). Pick a shorter first note or a larger pickup."
        )

    remaining = first_cap - first_note_ticks
    bars_done = 0
    last_type = "note"  # the prompt always ends on the first note

    seq = list(prompt_ids)
    warnings: list[str] = []

    # Hard cap on tokens so a pathological state can never loop forever.
    max_notes_per_bar = full_measure // smallest_tick
    hard_cap = len(prompt_ids) + target_bars * (max_notes_per_bar + 2) + 8

    while bars_done < target_bars:
        if len(seq) >= hard_cap:
            warnings.append(f"hit hard token cap ({hard_cap}); stopping after {bars_done} bars")
            break

        force_barline = False
        allowed: list[int] = []

        if remaining == 0:
            # Measure is exactly full: close it. A tie may precede the barline
            # (tie across the barline) only if the last emitted token was a note.
            if last_type == "tie":
                allowed = [barline_id]
            else:
                allowed = [barline_id]
                if last_type == "note" and tie_id is not None:
                    allowed.append(tie_id)
        else:
            if last_type == "tie":
                # A tie must be followed by a note that fits the remaining beats.
                allowed = [
                    i for i, tk in enumerate(ticks_by_id)
                    if tk is not None and 0 < tk <= remaining and is_note[i]
                ]
            else:
                # Any note or rest whose duration fits the remaining beats.
                allowed = [
                    i for i, tk in enumerate(ticks_by_id)
                    if tk is not None and 0 < tk <= remaining and (not no_rests or is_note[i])
                ]
                if last_type == "note" and tie_id is not None:
                    allowed.append(tie_id)

            if not allowed:
                # No token can fill the remaining beats exactly (remaining is
                # smaller than the smallest available duration). Close the
                # measure anyway and record that it is slightly short.
                force_barline = True
                warnings.append(
                    f"bar {bars_done + 1} left {remaining} ticks unfilled "
                    f"({remaining / QUARTER_TICKS:.3f} beats); forced barline"
                )

        if force_barline:
            allowed = [barline_id]

        idx_cond = torch.tensor([seq[-block_size:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(idx_cond)
        next_id = _sample_from_allowed(logits[0, -1, :], allowed, temperature, top_k, device)
        seq.append(next_id)

        if next_id == barline_id:
            bars_done += 1
            remaining = full_measure
            last_type = "barline"
        elif tie_id is not None and next_id == tie_id:
            last_type = "tie"
        else:
            remaining -= ticks_by_id[next_id]  # type: ignore[operator]
            last_type = "note" if is_note[next_id] else "rest"

    seq.append(eos_id)
    return seq, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DATA_DIR / "checkpoints" / "best.pt")
    parser.add_argument("--clef", type=str, required=True, help="e.g. clef-G2")
    parser.add_argument("--time", type=str, required=True, help="e.g. timeSignature-3/4")
    parser.add_argument("--first-note", type=str, required=True, help="e.g. note-C5_quarter")
    parser.add_argument("--key", type=str, default=None, help="e.g. keySignature-GM (optional)")
    parser.add_argument(
        "--bars",
        type=int,
        default=None,
        help="deterministic bar mode: produce exactly N full measures (barlines + end placed by rule)",
    )
    parser.add_argument(
        "--pickup-beats",
        type=float,
        default=None,
        help="deterministic mode only: make the first measure a partial pickup of this many quarter beats",
    )
    parser.add_argument(
        "--no-rests",
        action="store_true",
        help="deterministic mode only: never emit rest tokens (measures filled with notes/ties only)",
    )
    parser.add_argument("--max-len", type=int, default=None, help="free mode only: defaults to the model's block_size")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=None, help="write the (first) sample to this .semantic file")
    args = parser.parse_args()

    if args.bars is not None and args.bars < 1:
        parser.error("--bars must be >= 1")
    if args.pickup_beats is not None and args.bars is None:
        parser.error("--pickup-beats only applies in deterministic bar mode (--bars)")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    vocab = json.loads((args.data_dir / "vocab.json").read_text(encoding="utf-8"))
    token_to_id, id_to_token = vocab["token_to_id"], vocab["id_to_token"]

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f}) "
          f"params={model.num_params():,}")

    prompt_ids = build_prompt_ids(token_to_id, args.clef, args.time, args.first_note, args.key)
    eos_id = token_to_id["<eos>"]

    first_body_tokens: list[str] | None = None
    for i in range(args.num_samples):
        if args.bars is not None:
            seq, warnings = constrained_generate(
                model,
                prompt_ids,
                vocab=vocab,
                time_sig_token=args.time,
                target_bars=args.bars,
                pickup_beats=args.pickup_beats,
                temperature=args.temperature,
                top_k=args.top_k,
                block_size=config.block_size,
                device=args.device,
                no_rests=args.no_rests,
            )
            ids = seq
            for w in warnings:
                print(f"  [warning] {w}")
        else:
            max_len = args.max_len or config.block_size
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
            out = model.generate(
                idx,
                max_new_tokens=max_len - len(prompt_ids),
                eos_id=eos_id,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            ids = out[0].tolist()

        tokens = [id_to_token[t] for t in ids]
        tokens = tokens[1:]  # strip <bos>
        if "<eos>" in tokens:
            tokens = tokens[: tokens.index("<eos>")]
        print(f"[{i + 1}/{args.num_samples}]", "\t".join(tokens))
        if first_body_tokens is None:
            first_body_tokens = tokens

    if args.out is not None and first_body_tokens is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Match the ground-truth .semantic format: tab-separated, trailing tab, no newline.
        args.out.write_text("\t".join(first_body_tokens) + "\t", encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
