"""Train the validated ~5M-parameter Medium music-LM on packed PrIMuS sequences.

Medium config (fixed): n_layer=6, n_head=8, n_embd=256, dropout=0.2,
weight_decay=0.05, lr=3e-4, warmup_epochs=2, batch_size=64, epochs=24,
sample_every=2. This is the only model size trained by this script.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from music_lm.model import GPT, GPTConfig

_MUSIC_LM_DIR = Path(__file__).resolve().parents[1]

# Validated Medium hyperparameters — not CLI-configurable.
N_LAYER = 6
N_HEAD = 8
N_EMBD = 256
DROPOUT = 0.2
WEIGHT_DECAY = 0.05
LR = 3e-4
BATCH_SIZE = 64
EPOCHS = 24
WARMUP_EPOCHS = 2
SAMPLE_EVERY = 2
DEFAULT_OUT_DIR = str(_MUSIC_LM_DIR / "checkpoints_medium")


class PackedSequenceDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path).astype(np.int64)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]
        return torch.from_numpy(row[:-1]), torch.from_numpy(row[1:])


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def sample_prompt(model, meta, device, prompt_ids, max_new_tokens=40, temperature=0.9):
    model.eval()
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id = meta["eos_token_id"]
    for _ in range(max_new_tokens):
        if ids.shape[1] >= meta["block_size"]:
            break
        logits, _ = model(ids)
        next_logits = logits[0, -1] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos_id:
            break
    model.train()
    return ids[0].tolist()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        total_batches += 1
    model.train()
    return total_loss / max(1, total_batches)


def main():
    parser = argparse.ArgumentParser(
        description="Train the Medium (~5M) music-LM with its validated hyperparameters."
    )
    parser.add_argument("--train-data", default=str(_MUSIC_LM_DIR / "train.npy"))
    parser.add_argument("--val-data", default=str(_MUSIC_LM_DIR / "val.npy"))
    parser.add_argument("--meta", default=str(_MUSIC_LM_DIR / "meta.json"))
    parser.add_argument("--vocab", default=str(_MUSIC_LM_DIR / "vocab.json"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=os.cpu_count(),
                         help="CPU threads for intra-op parallelism (matmuls). Defaults to all logical cores.")
    args = parser.parse_args()

    if args.device == "cpu" and args.num_threads:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(max(1, args.num_threads // 2))
    print(f"torch threads: intra-op={torch.get_num_threads()} inter-op={torch.get_num_interop_threads()}")
    print(
        f"Medium config: n_layer={N_LAYER} n_head={N_HEAD} n_embd={N_EMBD} "
        f"dropout={DROPOUT} weight_decay={WEIGHT_DECAY} lr={LR} "
        f"batch_size={BATCH_SIZE} epochs={EPOCHS} warmup_epochs={WARMUP_EPOCHS}"
    )

    torch.manual_seed(args.seed)

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    id_to_token = vocab["id_to_token"]

    train_ds = PackedSequenceDataset(args.train_data)
    val_ds = PackedSequenceDataset(args.val_data)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    config = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=meta["block_size"],
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        pad_token_id=meta["pad_token_id"],
    )
    model = GPT(config).to(args.device)
    print(f"Model params: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_tokens = ["<bos>", "clef-G2", "timeSignature-4/4", "note-C4_quarter"]
    prompt_ids = [vocab["token_to_id"].get(t, vocab["token_to_id"]["<unk>"]) for t in prompt_tokens]

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss, total_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_batches += 1

        train_loss = total_loss / max(1, total_batches)
        val_loss = evaluate(model, val_loader, args.device)
        dt = time.time() - t0

        is_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": dt})

        print(
            f"epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"  best={best_val_loss:.4f}  {dt:.1f}s{'  *' if is_best else ''}"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": vars(config),
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "history": history,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if is_best:
            torch.save(ckpt, out_dir / "best.pt")

        if epoch % SAMPLE_EVERY == 0 or epoch == EPOCHS:
            sample_ids = sample_prompt(model, meta, args.device, prompt_ids)
            sample_tokens = [id_to_token[i] for i in sample_ids]
            print("  sample:", " ".join(sample_tokens))

    print(f"Done. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
