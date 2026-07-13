"""Scan PrIMuS semantic files and build a 1:1 token vocabulary for the music LM.

Every whitespace-separated token in the ground-truth `.semantic` files becomes
exactly one vocabulary entry (no BPE / merging). Four special tokens are
prepended for training: <pad>, <bos>, <eos>, <unk>.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUND_TRUTH_DIR = ROOT / "data" / "primus" / "ground_truth"
DEFAULT_OUTPUT = ROOT / "data" / "music_lm" / "vocab.json"

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def scan_tokens(ground_truth_dir: Path) -> tuple[Counter[str], list[int]]:
    counts: Counter[str] = Counter()
    lengths: list[int] = []
    for path in sorted(ground_truth_dir.glob("*.semantic")):
        tokens = path.read_text(encoding="utf-8").split()
        counts.update(tokens)
        lengths.append(len(tokens))
    return counts, lengths


def build_vocab(counts: Counter[str]) -> dict:
    tokens = SPECIAL_TOKENS + sorted(counts)
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    return {"token_to_id": token_to_id, "id_to_token": tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.ground_truth_dir.is_dir():
        raise FileNotFoundError(
            f"Ground truth dir not found: {args.ground_truth_dir}. "
            "Run scripts/download_datasets.ps1 first."
        )

    counts, lengths = scan_tokens(args.ground_truth_dir)
    vocab = build_vocab(counts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    categories: Counter[str] = Counter(tok.split("-")[0] for tok in counts)

    print(f"Files scanned: {len(lengths)}")
    print(f"Unique tokens: {len(counts)} (+{len(SPECIAL_TOKENS)} special = {len(vocab['id_to_token'])} total)")
    print(f"Total token occurrences: {sum(counts.values())}")
    print(f"Sequence length: min={min(lengths)} max={max(lengths)} avg={sum(lengths) / len(lengths):.1f}")
    print("Token categories:", dict(sorted(categories.items())))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
