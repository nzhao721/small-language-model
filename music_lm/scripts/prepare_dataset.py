"""Tokenize and pack PrIMuS excerpts into fixed-width training rows.

Each sequence is a WHOLE incipit -- <bos> + every token in the file + <eos> --
right-padded with <pad> to --block-size. Sequences are never truncated to a
fixed number of bars: --block-size only needs to be large enough that a
4-measure excerpt fits during generation and that whole incipits fit during
training, and 64 comfortably covers both (empirically, the longest whole
incipit and the longest 4-bar prefix in this corpus are both 58 tokens
including <bos>/<eos>).
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

_MUSIC_LM_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from music_lm.keys import MAJOR_TO_MINOR  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/consolidated/ground_truth")
    parser.add_argument("--pattern", default="primus_*.semantic")
    parser.add_argument("--vocab", default=str(_MUSIC_LM_DIR / "vocab.json"))
    parser.add_argument("--out-dir", default=str(_MUSIC_LM_DIR))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--minor-key-prob",
        type=float,
        default=0.5,
        help="Probability of rewriting a major key token to its relative minor.",
    )
    args = parser.parse_args()

    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    token_to_id = vocab["token_to_id"]
    pad_id = token_to_id["<pad>"]
    bos_id = token_to_id["<bos>"]
    eos_id = token_to_id["<eos>"]

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob(args.pattern))
    print(f"Found {len(files)} files matching {args.pattern!r} in {data_dir}")

    def maybe_rewrite_key(tokens, rng):
        if args.minor_key_prob <= 0:
            return tokens
        out = []
        for token in tokens:
            if token in MAJOR_TO_MINOR and rng.random() < args.minor_key_prob:
                out.append(MAJOR_TO_MINOR[token])
            else:
                out.append(token)
        return out

    rows = []
    dropped = 0
    max_real_len = 0
    minor_rewrites = 0
    for f in files:
        name_hash = int(hashlib.md5(f.name.encode("utf-8")).hexdigest()[:8], 16)
        file_rng = random.Random(args.seed ^ name_hash)
        tokens = f.read_text(encoding="utf-8").strip().split()
        rewritten = maybe_rewrite_key(tokens, file_rng)
        minor_rewrites += sum(
            1 for before, after in zip(tokens, rewritten) if before != after
        )
        tokens = rewritten
        # Direct indexing: vocab was built from this same file set, so every
        # token must already be known -- a KeyError here means vocab.json is
        # stale, which should fail loudly rather than silently map to <unk>.
        ids = [bos_id] + [token_to_id[t] for t in tokens] + [eos_id]
        if len(ids) > args.block_size:
            dropped += 1
            continue
        max_real_len = max(max_real_len, len(ids))
        ids = ids + [pad_id] * (args.block_size - len(ids))
        rows.append(ids)

    print(
        f"Packed {len(rows)} sequences (dropped {dropped} exceeding block_size={args.block_size}); "
        f"rewrote {minor_rewrites} major key tokens to relative minor"
    )
    print(f"Max real (unpadded, bos+tokens+eos) length seen: {max_real_len}")

    rng = random.Random(args.seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_fraction))
    val_idx = set(indices[:n_val])

    train_rows = [rows[i] for i in indices if i not in val_idx]
    val_rows = [rows[i] for i in indices if i in val_idx]

    arr_train = np.array(train_rows, dtype=np.int16)
    arr_val = np.array(val_rows, dtype=np.int16)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", arr_train)
    np.save(out_dir / "val.npy", arr_val)

    meta = {
        "block_size": args.block_size,
        "vocab_size": len(vocab["id_to_token"]),
        "pad_token_id": pad_id,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "unk_token_id": token_to_id["<unk>"],
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "seed": args.seed,
        "data_dir": str(data_dir),
        "pattern": args.pattern,
        "dropped_exceeding_block_size": dropped,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote train.npy {arr_train.shape}, val.npy {arr_val.shape}, meta.json -> {out_dir}")


if __name__ == "__main__":
    main()
