"""Conditioned generation from a trained music-LM checkpoint.

Supports two decoding modes:
  --bars N    deterministic, bar-constrained decoding: at every step only
              tokens that fit the beats remaining in the current measure are
              allowed, a barline is forced when a measure fills exactly, and
              <eos> is forced after N barlines.
  (default)   free-form decoding up to --max-len, bounded only by the model's
              own vocabulary (minus the never-emit specials <pad>/<bos>/<unk>).

Both modes sample through the same dual-temperature mechanism: pitch and
duration/rhythm get independent temperatures instead of one conflated knob.
The allowed token ids at a step are partitioned into duration-groups (see
music_lm.timing.duration_group_key); a duration-group is chosen first using
--duration-temperature (softmax over each group's logsumexp'd marginal
logit -- exact, not approximate), then, if that group has more than one
member (i.e. several pitches at the same rhythmic value), the specific token
is chosen using --pitch-temperature and --top-k. Steps where only one token
is allowed (forced barline, forced eos, forced tie) skip sampling entirely.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from music_lm.model import GPT, GPTConfig
from music_lm.pitch import (
    MAX_JUMP_SEMITONES,
    last_note_pitch_midi,
    note_passes_pitch_constraints,
    note_pitch_midi,
    pitch_limits_for_clef,
)
from music_lm.repetition import (
    consecutive_same_duration_run,
    consecutive_same_pitch_run,
    filter_allowed_avoid_duration,
    filter_allowed_avoid_pitch,
)
from music_lm.timing import (
    can_fill_ticks,
    collect_duration_costs,
    duration_group_key,
    measure_ticks,
    token_tick_cost,
)

_MUSIC_LM_DIR = Path(__file__).resolve().parents[1]


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def sample_dual_temperature(logits, allowed_ids, id_to_token, temperature_pitch,
                             temperature_duration, top_k=None):
    if len(allowed_ids) == 1:
        return allowed_ids[0]

    groups = {}
    for tid in allowed_ids:
        key = duration_group_key(id_to_token[tid])
        groups.setdefault(key, []).append(tid)

    keys = list(groups.keys())
    group_logits = torch.stack([torch.logsumexp(logits[groups[k]], dim=0) for k in keys])
    group_probs = F.softmax(group_logits / temperature_duration, dim=0)
    chosen_key = keys[torch.multinomial(group_probs, 1).item()]
    members = groups[chosen_key]

    if len(members) == 1:
        return members[0]

    member_logits = logits[members] / temperature_pitch
    if top_k is not None and 0 < top_k < len(members):
        top_vals, top_idx = torch.topk(member_logits, top_k)
        probs = F.softmax(top_vals, dim=0)
        pick = torch.multinomial(probs, 1).item()
        return members[top_idx[pick].item()]

    probs = F.softmax(member_logits, dim=0)
    pick = torch.multinomial(probs, 1).item()
    return members[pick]


def measure_state_from_prompt(prompt_ids, id_to_token, ticks_per_measure):
    """Subtract note/rest durations already present in the conditioning prompt."""
    ticks_remaining = ticks_per_measure
    prev_was_note = False
    for tid in prompt_ids:
        token = id_to_token[tid]
        cost = token_tick_cost(token)
        if cost > 0:
            ticks_remaining -= cost
            prev_was_note = token.startswith("note-")
    if ticks_remaining < 0:
        raise ValueError(
            "prompt note/rest durations exceed one measure; "
            f"remaining ticks={ticks_remaining}, measure={ticks_per_measure}"
        )
    return ticks_remaining, prev_was_note


def clef_from_prompt(prompt_ids, id_to_token):
    for tid in prompt_ids:
        token = id_to_token[tid]
        if token.startswith("clef-"):
            return token
    return None


def filter_note_ids_by_pitch(note_rest_ids, id_to_token, clef_token, prev_pitch_midi,
                              max_jump_semitones=MAX_JUMP_SEMITONES):
    filtered = []
    for tid in note_rest_ids:
        token = id_to_token[tid]
        if note_passes_pitch_constraints(
            token, clef_token, prev_pitch_midi, max_jump_semitones=max_jump_semitones
        ):
            filtered.append(tid)
    return filtered


def candidate_note_ids(note_rest_ids, id_to_token, clef_token, prev_pitch_midi,
                       ticks_remaining, duration_costs, max_jump_semitones):
    """Note/rest ids that fit rhythm, pitch, and can still complete the measure."""
    strict = []
    in_range = []
    rhythmic = []

    for tid in note_rest_ids:
        token = id_to_token[tid]
        cost = token_tick_cost(token)
        if not (0 < cost <= ticks_remaining):
            continue
        if not can_fill_ticks(ticks_remaining - cost, duration_costs):
            continue
        rhythmic.append(tid)
        if not note_passes_pitch_constraints(
            token, clef_token, prev_pitch_midi, max_jump_semitones=max_jump_semitones
        ):
            low, high = pitch_limits_for_clef(clef_token)
            midi = note_pitch_midi(token)
            if midi is None or not (low <= midi <= high):
                continue
            in_range.append(tid)
            continue
        strict.append(tid)

    if strict:
        return strict
    if in_range:
        return in_range
    return rhythmic


@torch.no_grad()
def constrained_generate(model, token_to_id, id_to_token, prompt_ids, time_sig_token, bars,
                          temperature_pitch, temperature_duration, top_k, no_rests, device,
                          max_jump_semitones=MAX_JUMP_SEMITONES,
                          max_consecutive_same_pitch=None,
                          max_consecutive_same_duration=None):
    block_size = model.config.block_size
    ids = list(prompt_ids)
    clef_token = clef_from_prompt(prompt_ids, id_to_token)
    if clef_token is None:
        raise ValueError("prompt must include a clef token for pitch-constrained generation")
    ticks_per_measure = measure_ticks(time_sig_token)
    ticks_remaining, prev_was_note = measure_state_from_prompt(
        prompt_ids, id_to_token, ticks_per_measure
    )
    prev_pitch_midi = last_note_pitch_midi([id_to_token[tid] for tid in prompt_ids])
    bars_completed = 0
    barline_id = token_to_id["barline"]
    tie_id = token_to_id.get("tie")
    eos_id = token_to_id["<eos>"]

    vocab_size = len(id_to_token)
    note_rest_ids = [
        i for i in range(vocab_size)
        if id_to_token[i].startswith("note-") or id_to_token[i].startswith("rest-")
    ]
    if no_rests:
        note_rest_ids = [i for i in note_rest_ids if not id_to_token[i].startswith("rest-")]
    note_rest_ids = filter_note_ids_by_pitch(
        note_rest_ids, id_to_token, clef_token, prev_pitch_midi=None,
        max_jump_semitones=max_jump_semitones,
    )
    duration_costs = collect_duration_costs(id_to_token[tid] for tid in note_rest_ids)

    while len(ids) < block_size:
        if bars_completed >= bars:
            allowed = [eos_id]
        elif ticks_remaining == 0:
            allowed = [barline_id]
        else:
            allowed = candidate_note_ids(
                note_rest_ids,
                id_to_token,
                clef_token,
                prev_pitch_midi,
                ticks_remaining,
                duration_costs,
                max_jump_semitones,
            )
            if (
                prev_was_note
                and tie_id is not None
                and allowed
                and id_to_token[ids[-1]] != "tie"
            ):
                allowed.append(tie_id)

            if max_consecutive_same_pitch is not None and allowed:
                run_len, same_midi = consecutive_same_pitch_run(ids, id_to_token)
                if run_len >= max_consecutive_same_pitch and same_midi is not None:
                    revised = filter_allowed_avoid_pitch(
                        allowed, id_to_token, same_midi, tie_id=tie_id
                    )
                    if revised:
                        allowed = revised

            if max_consecutive_same_duration is not None and allowed:
                run_len, same_duration = consecutive_same_duration_run(ids, id_to_token)
                if run_len >= max_consecutive_same_duration and same_duration is not None:
                    revised = filter_allowed_avoid_duration(
                        allowed, id_to_token, same_duration, tie_id=tie_id
                    )
                    if revised:
                        allowed = revised

        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_logits = logits[0, -1]

        next_id = sample_dual_temperature(
            next_logits, allowed, id_to_token, temperature_pitch, temperature_duration, top_k
        )
        ids.append(next_id)

        if next_id == barline_id:
            ticks_remaining = ticks_per_measure
            bars_completed += 1
            prev_was_note = False
        elif next_id == eos_id:
            break
        elif next_id == tie_id:
            pass
        else:
            token = id_to_token[next_id]
            ticks_remaining -= token_tick_cost(token)
            prev_was_note = token.startswith("note-")
            if prev_was_note:
                prev_pitch_midi = note_pitch_midi(token)
            elif token.startswith("rest-"):
                prev_pitch_midi = None

    return ids


@torch.no_grad()
def free_generate(model, token_to_id, id_to_token, prompt_ids, max_len,
                   temperature_pitch, temperature_duration, top_k, no_rests, device):
    block_size = model.config.block_size
    ids = list(prompt_ids)
    eos_id = token_to_id["<eos>"]
    vocab_size = len(id_to_token)

    exclude = {token_to_id["<pad>"], token_to_id["<bos>"], token_to_id["<unk>"]}
    allowed = [i for i in range(vocab_size) if i not in exclude]
    if no_rests:
        allowed = [i for i in allowed if not id_to_token[i].startswith("rest-")]

    while len(ids) < min(block_size, max_len):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_logits = logits[0, -1]
        next_id = sample_dual_temperature(
            next_logits, allowed, id_to_token, temperature_pitch, temperature_duration, top_k
        )
        ids.append(next_id)
        if next_id == eos_id:
            break

    return ids


def build_prompt(token_to_id, clef, key, time_sig, first_note):
    tokens = ["<bos>", clef]
    if key:
        tokens.append(key)
    tokens.append(time_sig)
    if first_note:
        tokens.append(first_note)
    unk_id = token_to_id["<unk>"]
    return [token_to_id.get(t, unk_id) for t in tokens]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", default=str(_MUSIC_LM_DIR / "vocab.json"))
    parser.add_argument("--clef", default="clef-G2")
    parser.add_argument("--key", default=None)
    parser.add_argument("--time", default="timeSignature-4/4")
    parser.add_argument("--first-note", default="note-C4_quarter")
    parser.add_argument("--bars", type=int, default=None,
                         help="Deterministic bar-constrained decoding for this many measures.")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--pitch-temperature", type=float, default=0.9)
    parser.add_argument("--duration-temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-rests", action="store_true")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    token_to_id = vocab["token_to_id"]
    id_to_token = vocab["id_to_token"]

    model, ckpt = load_checkpoint(args.checkpoint, args.device)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f}")

    prompt_ids = build_prompt(token_to_id, args.clef, args.key, args.time, args.first_note)

    for i in range(args.num_samples):
        if args.bars:
            ids = constrained_generate(
                model, token_to_id, id_to_token, prompt_ids, args.time, args.bars,
                args.pitch_temperature, args.duration_temperature, args.top_k, args.no_rests, args.device,
            )
        else:
            ids = free_generate(
                model, token_to_id, id_to_token, prompt_ids, args.max_len,
                args.pitch_temperature, args.duration_temperature, args.top_k, args.no_rests, args.device,
            )
        tokens = [id_to_token[tid] for tid in ids]
        print(f"[{i + 1}/{args.num_samples}]", "\t".join(tokens))


if __name__ == "__main__":
    main()
