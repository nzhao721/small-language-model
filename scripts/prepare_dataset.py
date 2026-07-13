"""Tokenize PrIMuS semantic files with vocab.json and pack them into fixed-width
arrays for training, with a train/val split.

Each sequence becomes: <bos> + tokens + <eos>, right-padded with <pad> to
--block-size. Unknown tokens (shouldn't occur if vocab.json was built from the
same ground-truth dir) fall back to <unk>.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUND_TRUTH_DIR = ROOT / "data" / "primus" / "ground_truth"
DEFAULT_VOCAB = ROOT / "data" / "music_lm" / "vocab.json"
DEFAULT_OUT_DIR = ROOT / "data" / "music_lm"


def encode_file(path: Path, token_to_id: dict[str, int], bos_id: int, eos_id: int, unk_id: int) -> list[int]:
    tokens = path.read_text(encoding="utf-8").split()
    ids = [token_to_id.get(tok, unk_id) for tok in tokens]
    return [bos_id, *ids, eos_id]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    token_to_id = vocab["token_to_id"]
    pad_id, bos_id, eos_id, unk_id = (token_to_id[t] for t in ("<pad>", "<bos>", "<eos>", "<unk>"))

    paths = sorted(args.ground_truth_dir.glob("*.semantic"))
    if not paths:
        raise FileNotFoundError(f"No .semantic files found in {args.ground_truth_dir}")

    rng = random.Random(args.seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_paths = shuffled[:n_val]
    train_paths = shuffled[n_val:]

    def pack(file_paths: list[Path]) -> np.ndarray:
        arr = np.full((len(file_paths), args.block_size), pad_id, dtype=np.int16)
        dropped = 0
        kept = 0
        for row, path in enumerate(file_paths):
            ids = encode_file(path, token_to_id, bos_id, eos_id, unk_id)
            if len(ids) > args.block_size:
                dropped += 1
                continue
            arr[kept, : len(ids)] = ids
            kept += 1
        if dropped:
            print(f"  dropped {dropped} sequences longer than block_size={args.block_size}")
        return arr[:kept]

    train_arr = pack(train_paths)
    val_arr = pack(val_paths)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "train.npy", train_arr)
    np.save(args.out_dir / "val.npy", val_arr)

    meta = {
        "block_size": args.block_size,
        "vocab_size": len(vocab["id_to_token"]),
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "unk_id": unk_id,
        "n_train": int(train_arr.shape[0]),
        "n_val": int(val_arr.shape[0]),
        "seed": args.seed,
        "val_files": [p.name for p in val_paths],
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"train.npy: {train_arr.shape}")
    print(f"val.npy:   {val_arr.shape}")
    print(f"Wrote {args.out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
