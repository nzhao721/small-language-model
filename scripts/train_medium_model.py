"""Train a *Medium* music-LM, tuned to reduce the repetition / lack-of-movement
seen with the Small model.

Why these settings target repetition
-------------------------------------
The PrIMuS incipits are short (<=58 tokens) and the model's context window
(block_size=64) already spans an entire piece, so the fix for "the melody
doesn't move" is not a longer context -- it is more *capacity to model
higher-order structure* across that context:

* n_layer 4 -> 6   : depth is what lets the model compose long-range melodic
                     dependencies (contour direction, motif development) rather
                     than just locally-safe next-note transitions, which is what
                     produces repeated/oscillating notes.
* n_embd  128 -> 256: a wider residual stream can keep more of the melodic
                     history (recent pitches, contour) distinguishable.
* n_head    4 -> 8 : more attention heads can specialize in parallel (e.g. one
                     tracking beat position, another melodic contour).

A bigger model on an ~873K-token corpus overfits easily, and an overfit model
mimics surface repetition, so we regularize harder than the Small run:

* dropout      0.1 -> 0.2
* weight_decay 0.01 -> 0.05

Checkpoints go to a separate dir so the Small model is left intact. Training is
resumable (scripts/train_music_lm.py --resume) and saves best/last every epoch.

This is a thin wrapper: it just invokes scripts/train_music_lm.py with the
Medium hyperparameters. Everything here can also be run by passing these flags
to that script directly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_music_lm.py"
DEFAULT_OUT_DIR = ROOT / "data" / "music_lm" / "checkpoints_medium"

# Note on --batch-size: this box has ~16 GB RAM and is often memory-pressured,
# and CPU training peaks at the activation memory of one batch. batch-size 64
# keeps the footprint modest (a larger batch was observed to get OOM-killed
# mid-epoch here). Total tokens/epoch is unchanged, so epoch wall-clock is
# roughly the same regardless of batch size.
MEDIUM_HPARAMS = {
    "--n-layer": "6",
    "--n-head": "8",
    "--n-embd": "256",
    "--dropout": "0.2",
    "--weight-decay": "0.05",
    "--lr": "3e-4",
    "--warmup-epochs": "2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="cap training batches per epoch (timing probe only)",
    )
    args = parser.parse_args()

    cmd = [sys.executable, str(TRAIN_SCRIPT)]
    for flag, value in MEDIUM_HPARAMS.items():
        cmd += [flag, value]
    cmd += [
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--out-dir", str(args.out_dir),
        "--seed", str(args.seed),
        "--sample-every", str(args.sample_every),
        "--device", args.device,
    ]
    if args.resume is not None:
        cmd += ["--resume", str(args.resume)]
    if args.max_batches is not None:
        cmd += ["--max-batches", str(args.max_batches)]

    print("Medium config:", " ".join(f"{k} {v}" for k, v in MEDIUM_HPARAMS.items()))
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
