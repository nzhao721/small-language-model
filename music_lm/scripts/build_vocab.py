"""Build the token vocabulary for the music-LM from original PrIMuS excerpts.

Source: data/consolidated/ground_truth/primus_*.semantic only -- the
consolidated ground_truth folder also holds corrupt_handwritten__*.semantic
and mashcima__*.semantic files generated for the OMR project, which are
excluded by the primus_ prefix filter.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_MUSIC_LM_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from music_lm.keys import minor_key_tokens  # noqa: E402

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def categorize(token):
    if token.startswith("note-"):
        return "note"
    if token.startswith("rest-"):
        return "rest"
    if token.startswith("timeSignature-"):
        return "timeSignature"
    if token.startswith("keySignature-"):
        return "keySignature"
    if token.startswith("clef-"):
        return "clef"
    if token in ("barline", "tie"):
        return token
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/consolidated/ground_truth")
    parser.add_argument("--pattern", default="primus_*.semantic")
    parser.add_argument("--out", default=str(_MUSIC_LM_DIR / "vocab.json"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob(args.pattern))
    print(f"Found {len(files)} files matching {args.pattern!r} in {data_dir}")

    counter = Counter()
    lengths = []
    for f in files:
        tokens = f.read_text(encoding="utf-8").strip().split()
        counter.update(tokens)
        lengths.append(len(tokens))

    content_tokens = sorted(set(counter.keys()) | set(minor_key_tokens()))
    added = sorted(set(minor_key_tokens()) - set(counter.keys()))
    vocab_tokens = SPECIAL_TOKENS + content_tokens
    token_to_id = {tok: i for i, tok in enumerate(vocab_tokens)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"token_to_id": token_to_id, "id_to_token": vocab_tokens}, indent=2),
        encoding="utf-8",
    )

    categories = Counter(categorize(tok) for tok in counter)

    print(f"Vocab size (incl. {len(SPECIAL_TOKENS)} specials): {len(vocab_tokens)}")
    print(f"Unique content tokens in corpus: {len(counter)}")
    if added:
        print(f"Added {len(added)} minor key tokens not present in corpus: {', '.join(added)}")
    print("Category breakdown:", dict(categories))
    print(f"Sequence length (raw tokens/file): min={min(lengths)} max={max(lengths)} "
          f"avg={sum(lengths) / len(lengths):.1f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
