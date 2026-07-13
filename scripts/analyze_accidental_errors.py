"""Analyze accidental pitch substitutions in golden eval predictions."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import omr_benchmark_score as obs  # noqa: E402
from omr_benchmark_score import normalize_tokens, token_edit_distance  # noqa: E402
from primus_semantic_render import KEY_SIG_ALTERATIONS, KEY_SIG_TO_MEI  # noqa: E402

# Match prior run-3 note-error analysis: strict tokens, no key/time equivalence.
obs.canonicalize_tokens = lambda toks: list(toks)

NOTE_RE = re.compile(r"^note-([A-G][#b]*)(\d+)_(.+)$")
GROUP_DIRS = {
    "handwritten_original": ROOT / "golden_set" / "original" / "ground_truth",
    "clean_original": ROOT / "golden_set" / "original" / "ground_truth",
    "camera_distorted_original": ROOT / "golden_set" / "original" / "ground_truth",
    "handwritten_corrupt": ROOT / "golden_set" / "corrupt" / "ground_truth",
    "clean_corrupt": ROOT / "golden_set" / "corrupt" / "ground_truth",
    "camera_distorted_corrupt": ROOT / "golden_set" / "corrupt" / "ground_truth",
    "manual_handwritten": ROOT / "golden_set" / "manual" / "ground_truth",
    "manual_digital": ROOT / "golden_set" / "manual" / "ground_truth",
}
LETTER_TO_PITCH = {c: c.lower() for c in "ABCDEFG"}


def strip_trailing_barline(tokens: list[str]) -> list[str]:
    return tokens[:-1] if tokens and tokens[-1] == "barline" else tokens


def align_ops(pred: list[str], gt: list[str]) -> list[tuple[str, str | None, str | None]]:
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    ops: list[tuple[str, str | None, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and gt[i - 1] == pred[j - 1]:
            ops.append(("match", gt[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("replace", gt[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("delete", gt[i - 1], None))
            i -= 1
        else:
            ops.append(("insert", None, pred[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def best_align(pred: list[str], gt: list[str]) -> list[tuple[str, str | None, str | None]]:
    best: tuple[int, list[tuple[str, str | None, str | None]]] | None = None
    for p, g in [
        (pred, gt),
        (strip_trailing_barline(pred), gt),
        (pred, strip_trailing_barline(gt)),
        (strip_trailing_barline(pred), strip_trailing_barline(gt)),
    ]:
        dist = token_edit_distance(p, g)
        ops = align_ops(p, g)
        if best is None or dist < best[0]:
            best = (dist, ops)
    return best[1] if best else []


def parse_note(token: str) -> dict | None:
    m = NOTE_RE.match(token)
    if not m:
        return None
    letter, dur = m.group(1), m.group(3)
    return {
        "letter": letter[0],
        "acc": letter[1:] if len(letter) > 1 else "",
        "octave": int(m.group(2)),
        "dur": dur,
        "token": token,
    }


def load_gt(group: str, filename: str) -> list[str]:
    stem = Path(filename).stem
    if stem.endswith("_distorted"):
        stem = stem[: -len("_distorted")]
    path = GROUP_DIRS[group] / f"{stem}.semantic"
    return normalize_tokens(path.read_text(encoding="utf-8"))


def key_tok_from_tokens(tokens: list[str]) -> str | None:
    return next((t for t in tokens if t.startswith("keySignature-")), None)


def key_mei(key_tok: str | None) -> str:
    if not key_tok:
        return "0"
    return KEY_SIG_TO_MEI.get(key_tok, "0")


def key_implied_acc(key_tok: str | None, letter: str) -> str:
    mei = key_mei(key_tok)
    pitch = LETTER_TO_PITCH[letter]
    alters = KEY_SIG_ALTERATIONS.get(mei, frozenset())
    if pitch not in alters:
        return ""
    return "#" if mei.endswith("s") else "b"


def midi_pitch(letter: str, octave: int, acc: str) -> int:
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    for ch in acc:
        if ch == "#":
            base += 1
        elif ch == "b":
            base -= 1
    return (octave + 1) * 12 + base


def scan_context(gt_tokens: list[str], error_gt_idx: int) -> dict:
    key_tok = key_tok_from_tokens(gt_tokens)
    measure_acc: dict[tuple[str, int], str] = {}
    bar = 0
    prior_notes: list[dict] = []

    for i, tok in enumerate(gt_tokens):
        if tok == "barline":
            bar += 1
            measure_acc = {}
            prior_notes = []
            continue
        if not tok.startswith("note-"):
            continue
        note = parse_note(tok)
        if note is None:
            continue
        if i == error_gt_idx:
            pitch = (note["letter"], note["octave"])
            return {
                "key": key_tok,
                "bar": bar,
                "token_acc": note["acc"],
                "key_implied": key_implied_acc(key_tok, note["letter"]),
                "carried_acc": measure_acc.get(pitch, ""),
                "prior_notes_in_bar": prior_notes.copy(),
                "prior_same": [
                    p
                    for p in prior_notes
                    if p["letter"] == note["letter"] and p["octave"] == note["octave"]
                ],
            }
        pitch = (note["letter"], note["octave"])
        if note["acc"]:
            measure_acc[pitch] = note["acc"]
        elif pitch in measure_acc:
            del measure_acc[pitch]
        prior_notes.append(note)

    raise ValueError(f"index {error_gt_idx} not a note")


def gt_token_index_for_note(gt_tokens: list[str], note_count: int) -> int:
    seen = 0
    for i, tok in enumerate(gt_tokens):
        if tok.startswith("note-"):
            if seen == note_count:
                return i
            seen += 1
    raise ValueError("note index out of range")


def classify_error(gt_note: dict, pred_note: dict, ctx: dict) -> tuple[str, list[str]]:
    gt_acc = gt_note["acc"]
    pred_acc = pred_note["acc"]
    key_impl = ctx["key_implied"]
    carried = ctx["carried_acc"]
    tags: list[str] = []

    gt_sound = gt_acc or carried or key_impl
    pred_sound = pred_acc or key_implied_acc(ctx["key"], pred_note["letter"])
    if (
        gt_sound != pred_sound
        and midi_pitch(gt_note["letter"], gt_note["octave"], gt_sound)
        == midi_pitch(pred_note["letter"], pred_note["octave"], pred_sound)
        and gt_acc != pred_acc
    ):
        tags.append("enharmonic_spelling")

    # Pred added accidental where GT is natural in context
    if not gt_acc and not carried and not key_impl and pred_acc:
        return "wrong_explicit", tags + ["wrong_explicit"]

    # GT explicit matches key default; pred dropped it
    if gt_acc and gt_acc == key_impl and not carried:
        return "key_signature_default", tags + ["key_signature_default"]

    # GT bare token, key implies acc, pred used natural token
    if not gt_acc and key_impl and not carried and not pred_acc:
        return "key_signature_default", tags + ["key_signature_default"]

    # GT inherits from bar carry (prior explicit on same pitch, no key default)
    if not gt_acc and carried and not key_impl:
        return "bar_carry_over", tags + ["bar_carry_over"]

    if not gt_acc and ctx["prior_same"]:
        last = ctx["prior_same"][-1]
        if last["acc"] and not key_impl:
            return "bar_carry_over", tags + ["bar_carry_over"]

    # GT has explicit accidental beyond key / carry context
    if gt_acc:
        return "explicit_accidental_on_note", tags + ["explicit_accidental_on_note"]

    # Pred added acc against key default on bare GT
    if not gt_acc and key_impl and pred_acc and pred_acc != key_impl:
        return "wrong_explicit", tags + ["wrong_explicit"]

    return "other/unclear", tags + ["other/unclear"]


def is_manual_group(group: str) -> bool:
    return group.startswith("manual_")


def main() -> None:
    eval_path = ROOT / "eval_golden_after (3).json"
    if len(sys.argv) > 1:
        eval_path = Path(sys.argv[1])

    data = json.loads(eval_path.read_text(encoding="utf-8-sig"))
    errors: list[dict] = []

    for sample in data["samples"]:
        fname = sample["filename"]
        group = sample["group"]
        pred_tokens = normalize_tokens(sample["prediction"])
        gt_tokens = load_gt(group, fname)

        ops = best_align(pred_tokens, gt_tokens)
        gt_note_i = 0
        for op, gt_tok, pred_tok in ops:
            if op == "match":
                if gt_tok and gt_tok.startswith("note-"):
                    gt_note_i += 1
                continue
            if op not in ("replace",):
                if gt_tok and gt_tok.startswith("note-"):
                    gt_note_i += 1
                continue
            gt_n = parse_note(gt_tok or "")
            pred_n = parse_note(pred_tok or "")
            if not gt_n or not pred_n:
                if gt_tok and gt_tok.startswith("note-"):
                    gt_note_i += 1
                continue
            pitch_changed = (
                gt_n["letter"] != pred_n["letter"]
                or gt_n["acc"] != pred_n["acc"]
                or gt_n["octave"] != pred_n["octave"]
            )
            if (
                pitch_changed
                and gt_n["letter"] == pred_n["letter"]
                and gt_n["octave"] == pred_n["octave"]
                and gt_n["acc"] != pred_n["acc"]
            ):
                gt_idx = gt_token_index_for_note(gt_tokens, gt_note_i)
                ctx = scan_context(gt_tokens, gt_idx)
                primary, tags = classify_error(gt_n, pred_n, ctx)
                errors.append(
                    {
                        "file": fname,
                        "group": group,
                        "manual": is_manual_group(group),
                        "bar": ctx["bar"],
                        "key": ctx["key"] or "(none)",
                        "gt": gt_tok,
                        "pred": pred_tok,
                        "gt_acc": gt_n["acc"] or "(none)",
                        "pred_acc": pred_n["acc"] or "(none)",
                        "key_implied": ctx["key_implied"] or "(none)",
                        "carried": ctx["carried_acc"] or "(none)",
                        "prior_in_bar": [
                            f"{p['letter']}{p['acc']}{p['octave']}" for p in ctx["prior_notes_in_bar"][-8:]
                        ],
                        "primary": primary,
                        "tags": tags,
                    }
                )
            if gt_tok and gt_tok.startswith("note-"):
                gt_note_i += 1

    total = len(errors)
    counts = Counter(e["primary"] for e in errors)
    manual_counts = Counter(e["primary"] for e in errors if e["manual"])
    synth_counts = Counter(e["primary"] for e in errors if not e["manual"])

    print(f"Total accidental substitutions: {total}")
    print("\n=== By category ===")
    cats = [
        "key_signature_default",
        "explicit_accidental_on_note",
        "bar_carry_over",
        "wrong_explicit",
        "enharmonic_spelling",
        "other/unclear",
    ]
    for cat in cats:
        n = counts.get(cat, 0)
        pct = 100.0 * n / total if total else 0
        print(f"  {cat}: {n} ({pct:.1f}%)")

    print("\n=== Manual vs synthetic ===")
    manual_total = sum(1 for e in errors if e["manual"])
    synth_total = total - manual_total
    if total:
        print(f"  manual: {manual_total} ({100 * manual_total / total:.1f}%)")
        print(f"  synthetic: {synth_total} ({100 * synth_total / total:.1f}%)")
    for cat in cats:
        m = manual_counts.get(cat, 0)
        s = synth_counts.get(cat, 0)
        print(f"  {cat}: manual={m}, synthetic={s}")

    print("\n=== Examples ===")
    seen_primary: set[str] = set()
    examples: list[dict] = []
    for e in errors:
        if e["primary"] not in seen_primary:
            examples.append(e)
            seen_primary.add(e["primary"])
    for e in errors:
        if len(examples) >= 8:
            break
        if e not in examples:
            examples.append(e)

    for e in examples[:8]:
        print(f"\n--- {e['file']} ({e['group']}) [{e['primary']}] ---")
        print(f"  key={e['key']}  bar={e['bar']}")
        print(f"  prior in bar: {', '.join(e['prior_in_bar']) or '(start)'}")
        print(f"  GT: {e['gt']}  |  pred: {e['pred']}")
        print(
            f"  token acc GT={e['gt_acc']} pred={e['pred_acc']}  "
            f"key-implied={e['key_implied']} carried={e['carried']}"
        )

    out_path = ROOT / "accidental_error_analysis.json"
    out_path.write_text(
        json.dumps({"total": total, "counts": dict(counts), "errors": errors}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
