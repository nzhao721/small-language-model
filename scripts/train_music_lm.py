"""Train the micro music-LM on packed PrIMuS semantic-token sequences.

Loads data/music_lm/{train,val}.npy (produced by prepare_dataset.py), trains a
GPT-style decoder-only transformer with next-token prediction, and checkpoints
both the best-val-loss and most recent model every epoch so the exact epoch
count is low-stakes and interrupted runs can be resumed with --resume.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from music_lm.model import GPT, GPTConfig  # noqa: E402

DEFAULT_DATA_DIR = ROOT / "data" / "music_lm"

# Architecture flags a resumed run must match, since they determine the
# shape of the saved state_dict.
ARCH_ARGS = ("n_layer", "n_head", "n_embd", "dropout")


class TokenSeqDataset(Dataset):
    def __init__(self, arr: np.ndarray, pad_id: int):
        self.arr = torch.from_numpy(arr.astype(np.int64))
        self.pad_id = pad_id

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __getitem__(self, idx: int):
        seq = self.arr[idx]
        x = seq[:-1]
        y = seq[1:].clone()
        y[y == self.pad_id] = -1  # ignored by cross_entropy(ignore_index=-1)
        return x, y


def evaluate(model: GPT, loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def sample_prompt(model: GPT, vocab: dict, device: str, block_size: int) -> str:
    token_to_id = vocab["token_to_id"]
    id_to_token = vocab["id_to_token"]
    prompt_tokens = ["<bos>", "clef-G2", "keySignature-CM", "timeSignature-2/4", "note-C5_quarter"]
    ids = [token_to_id[t] for t in prompt_tokens]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=block_size - len(ids), eos_id=token_to_id["<eos>"], temperature=0.9, top_k=20)
    return " ".join(id_to_token[i] for i in out[0].tolist())


def make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR / "checkpoints")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="cap training batches per epoch (for timing/throughput probes only; do not use for real runs)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint to resume from, e.g. data/music_lm/checkpoints/last.pt",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    vocab = json.loads((args.data_dir / "vocab.json").read_text(encoding="utf-8"))
    meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))

    train_arr = np.load(args.data_dir / "train.npy")
    val_arr = np.load(args.data_dir / "val.npy")

    train_ds = TokenSeqDataset(train_arr, meta["pad_id"])
    val_ds = TokenSeqDataset(val_arr, meta["pad_id"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    steps_per_epoch = max(1, math.ceil(len(train_ds) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(steps_per_epoch * args.warmup_epochs))

    start_epoch = 1
    best_val_loss = float("inf")
    history: list[dict] = []

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=args.device)
        saved_config = ckpt["config"]
        for key in ARCH_ARGS:
            cli_val = getattr(args, key)
            if saved_config[key] != cli_val:
                raise ValueError(
                    f"--{key.replace('_', '-')}={cli_val} doesn't match checkpoint's {key}={saved_config[key]}. "
                    "Architecture flags must match the checkpoint being resumed."
                )
        config = GPTConfig(**saved_config)
        model = GPT(config).to(args.device)
        model.load_state_dict(ckpt["model_state"])

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
        scheduler.load_state_dict(ckpt["scheduler_state"])

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        history = ckpt.get("history", [])

        if start_epoch > args.epochs:
            print(
                f"Checkpoint is already at epoch {ckpt['epoch']} >= --epochs {args.epochs}; "
                "nothing to do (pass a larger --epochs to keep training)."
            )
            return
    else:
        config = GPTConfig(
            vocab_size=meta["vocab_size"],
            block_size=meta["block_size"],
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )
        model = GPT(config).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = make_scheduler(optimizer, warmup_steps, total_steps)

    print(f"Model params: {model.num_params():,}")
    print(f"Train sequences: {len(train_ds)}  Val sequences: {len(val_ds)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(path: Path, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": config.__dict__,
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "history": history,
            },
            path,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        n_steps = 0
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            n_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if args.max_batches is not None and n_steps >= args.max_batches:
                break

        train_loss = running_loss / n_steps
        val_loss = evaluate(model, val_loader, args.device)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({dt:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.out_dir / "best.pt", epoch, val_loss)
            print(f"  -> new best (val_loss={val_loss:.4f}), saved best.pt")

        # last.pt is saved every epoch (after best_val_loss is updated) so an
        # interrupted run can always resume with --resume checkpoints/last.pt
        save_checkpoint(args.out_dir / "last.pt", epoch, val_loss)

        if args.sample_every and epoch % args.sample_every == 0:
            print("  sample:", sample_prompt(model, vocab, args.device, meta["block_size"]))

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Done. Best val_loss={best_val_loss:.4f}. Checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
