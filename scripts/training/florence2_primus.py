"""Wire Florence-2 image encoder to a closed PrIMuS semantic tokenizer."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import torch
from transformers import AutoImageProcessor, LogitsProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType

from primus_tokenizer import DEFAULT_VOCAB_PATH, PrimusTokenizer, split_semantic_tokens


class Florence2PrimusProcessor:
    """Image + PrIMuS-text processor compatible with Florence-2 training/inference."""

    attributes = ["image_processor", "tokenizer"]

    def __init__(self, image_processor, tokenizer: PrimusTokenizer):
        if image_processor is None:
            raise ValueError("image_processor is required")
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        if not isinstance(tokenizer, PrimusTokenizer):
            raise TypeError("tokenizer must be a PrimusTokenizer instance")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("image_processor is missing image_seq_length")

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_seq_length = image_processor.image_seq_length

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: int | None = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs,
    ) -> BatchFeature:
        if images is None:
            raise ValueError("images are required")

        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]

        if text is None:
            return BatchFeature(data={"pixel_values": pixel_values})

        if max_length is not None:
            max_length -= self.image_seq_length

        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
        )
        return BatchFeature(data={**inputs, "pixel_values": pixel_values})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self) -> list[str]:
        return list(
            dict.fromkeys(self.tokenizer.model_input_names + self.image_processor.model_input_names)
        )

    def save_pretrained(self, save_directory: str | Path, **kwargs) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.image_processor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        vocab_path: Path | str | None = None,
        trust_remote_code: bool = True,
    ) -> Florence2PrimusProcessor:
        path = Path(pretrained_model_name_or_path)
        image_processor = AutoImageProcessor.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = PrimusTokenizer.from_pretrained(path)
        if vocab_path is not None:
            tokenizer = build_primus_tokenizer(vocab_path)
        return cls(image_processor, tokenizer)


class PrimusConstrainedLogitsProcessor(LogitsProcessor):
    """Mask generation logits to PrIMuS semantic token IDs (+ EOS)."""

    def __init__(self, allowed_token_ids: list[int], eos_token_id: int):
        self.allowed = set(allowed_token_ids)
        self.allowed.add(eos_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, fill_value=float("-inf"))
        for token_id in self.allowed:
            mask[:, token_id] = scores[:, token_id]
        return mask


def build_primus_tokenizer(vocab_path: Path | str = DEFAULT_VOCAB_PATH) -> PrimusTokenizer:
    return PrimusTokenizer.from_vocabulary_file(vocab_path)


def build_florence2_primus_processor(
    model_id: str,
    *,
    vocab_path: Path | str = DEFAULT_VOCAB_PATH,
    trust_remote_code: bool = True,
) -> Florence2PrimusProcessor:
    image_processor = AutoImageProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = build_primus_tokenizer(vocab_path)
    return Florence2PrimusProcessor(image_processor, tokenizer)


def resize_model_embeddings_for_primus(model, tokenizer: PrimusTokenizer) -> None:
    """Resize decoder embeddings to the PrIMuS closed vocabulary."""
    model.resize_token_embeddings(len(tokenizer))
    text_cfg = getattr(model.config, "text_config", model.config)
    text_cfg.vocab_size = len(tokenizer)
    text_cfg.decoder_start_token_id = tokenizer.convert_tokens_to_ids("</s>")
    text_cfg.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    text_cfg.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    text_cfg.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    if hasattr(model.config, "forced_bos_token_id"):
        model.config.forced_bos_token_id = None


def encode_semantic_labels(
    tokenizer: PrimusTokenizer,
    semantic_texts: list[str],
    *,
    max_length: int,
) -> torch.Tensor:
    """Encode space/tab-separated semantic strings as one ID per PrIMuS token."""
    token_lists = [split_semantic_tokens(text) for text in semantic_texts]
    # padding="max_length" (not True) pads every batch to a fixed `max_length`, so the
    # decoder label tensor has a static shape across batches. This matters under
    # torch.compile: dynamic per-batch padding would change the sequence length each
    # step and trigger a fresh recompilation, erasing most of the compile speedup.
    # Keep `max_length` tight (labels are ~24 tokens, max ~58) so the fixed padding
    # adds negligible decoder compute.
    return tokenizer(
        token_lists,
        is_split_into_words=True,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).input_ids


def make_primus_logits_processor(tokenizer: PrimusTokenizer) -> PrimusConstrainedLogitsProcessor:
    return PrimusConstrainedLogitsProcessor(
        allowed_token_ids=tokenizer.primus_token_ids(),
        eos_token_id=tokenizer.eos_token_id,
    )
