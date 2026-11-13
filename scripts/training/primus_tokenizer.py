"""Closed-vocabulary tokenizer for PrIMuS semantic token strings."""

from __future__ import annotations

import json
import re
from pathlib import Path

from transformers import PreTrainedTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOCAB_PATH = ROOT / "data" / "primus_semantic_tokens.txt"

# Match Florence-2 BartTokenizer special-token IDs where possible so decoder_start_token_id
# stays valid after resize_token_embeddings (user retrains from base weights).
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<OCR>"]


def split_semantic_tokens(text: str) -> list[str]:
    """Split a semantic file body into PrIMuS tokens (whitespace / tab separated)."""
    return [part for part in re.split(r"[\s\t]+", text.strip()) if part]


def load_primus_token_list(vocab_path: Path | str = DEFAULT_VOCAB_PATH) -> list[str]:
    """Load unique PrIMuS token strings from the inventory TSV."""
    path = Path(vocab_path)
    tokens: list[str] = []
    seen: set[str] = set(SPECIAL_TOKENS)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        token = parts[1].strip() if len(parts) >= 2 else parts[0].strip()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return sorted(tokens)


class PrimusTokenizer(PreTrainedTokenizer):
    """One PrIMuS semantic type per token ID; no English BPE."""

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        primus_tokens: list[str] | None = None,
        vocab_file: str | Path | None = None,
        **kwargs,
    ):
        if primus_tokens is None:
            if vocab_file is None:
                vocab_file = DEFAULT_VOCAB_PATH
            primus_tokens = load_primus_token_list(vocab_file)

        self.primus_tokens = list(primus_tokens)
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

        for index, token in enumerate(SPECIAL_TOKENS):
            self._token_to_id[token] = index
            self._id_to_token[index] = token

        offset = len(SPECIAL_TOKENS)
        for index, token in enumerate(self.primus_tokens):
            token_id = offset + index
            if token in self._token_to_id:
                raise ValueError(f"duplicate token in vocab: {token!r}")
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token

        super().__init__(
            bos_token="<s>",
            pad_token="<pad>",
            eos_token="</s>",
            unk_token="<unk>",
            additional_special_tokens=["<OCR>"],
            clean_up_tokenization_spaces=False,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    @classmethod
    def from_vocabulary_file(cls, vocab_path: Path | str = DEFAULT_VOCAB_PATH) -> PrimusTokenizer:
        return cls(vocab_file=vocab_path)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        del token_ids_1
        bos = self.bos_token_id
        eos = self.eos_token_id
        if bos is None or eos is None:
            return token_ids_0
        return [bos] + token_ids_0 + [eos]

    def primus_token_ids(self) -> list[int]:
        """Token IDs for semantic types only (excludes specials)."""
        return [
            token_id
            for token_id in range(self.vocab_size)
            if self._id_to_token[token_id] not in SPECIAL_TOKENS
        ]

    def validate_tokens(self, tokens: list[str]) -> list[str]:
        """Return tokens not present in the closed vocabulary."""
        return [token for token in tokens if token not in self._token_to_id]

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return split_semantic_tokens(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id["<unk>"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        prefix = filename_prefix or ""
        vocab_path = path / f"{prefix}primus_vocab.json"
        payload = {
            "special_tokens": SPECIAL_TOKENS,
            "primus_tokens": self.primus_tokens,
        }
        vocab_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return (str(vocab_path),)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        path = Path(pretrained_model_name_or_path)
        vocab_path = path / "primus_vocab.json"
        if vocab_path.is_file():
            payload = json.loads(vocab_path.read_text(encoding="utf-8"))
            return cls(primus_tokens=payload["primus_tokens"], **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def save_pretrained(self, save_directory: str | Path, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        self.save_vocabulary(str(save_directory))
