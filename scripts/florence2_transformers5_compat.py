"""Florence-2 compatibility shims for transformers 5.x / Python 3.12."""

from __future__ import annotations

import shutil
from pathlib import Path

_PATCHED_FLAG = "_florence2_tf5_compat_patched"

_REMOTE_CODE_PATCHES: list[tuple[str, str, str]] = [
    (
        "configuration_florence2.py",
        "if self.forced_bos_token_id is None and kwargs.get(",
        "if getattr(self, 'forced_bos_token_id', None) is None and kwargs.get(",
    ),
    (
        "processing_florence2.py",
        "tokenizer.additional_special_tokens + \\",
        "getattr(tokenizer, 'additional_special_tokens', []) + \\",
    ),
    (
        "modeling_florence2.py",
        "        return self.language_model._supports_sdpa",
        "        if not hasattr(self, \"language_model\"):\n            return False\n        return self.language_model._supports_sdpa",
    ),
    (
        "modeling_florence2.py",
        "        return self.language_model._supports_flash_attn_2",
        "        if not hasattr(self, \"language_model\"):\n            return False\n        return self.language_model._supports_flash_attn_2",
    ),
]

_SEARCH_ROOTS = [
    Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules",
    Path.home() / ".cache" / "huggingface" / "hub",
]


def _patch_pretrained_config_defaults() -> None:
    """Restore forced_bos_token_id for transformers 5.x (init + getattr)."""
    import transformers.configuration_utils as config_utils

    if not getattr(config_utils.PretrainedConfig, _PATCHED_FLAG, False):
        _orig_init = config_utils.PretrainedConfig.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            if not hasattr(self, "forced_bos_token_id"):
                self.forced_bos_token_id = kwargs.get("forced_bos_token_id", None)

        config_utils.PretrainedConfig.__init__ = _patched_init  # type: ignore[method-assign]
        setattr(config_utils.PretrainedConfig, _PATCHED_FLAG, True)

    ga_flag = "_florence2_tf5_getattr_patched"
    if getattr(config_utils.PretrainedConfig, ga_flag, False):
        return

    _orig_ga = config_utils.PretrainedConfig.__getattribute__

    def _patched_ga(self, key):
        if key == "forced_bos_token_id":
            try:
                return _orig_ga(self, key)
            except AttributeError:
                return None
        return _orig_ga(self, key)

    config_utils.PretrainedConfig.__getattribute__ = _patched_ga  # type: ignore[method-assign]
    setattr(config_utils.PretrainedConfig, ga_flag, True)


def _patch_tokenizer_defaults() -> None:
    """Expose additional_special_tokens for Florence-2 processor on newer tokenizers."""
    import transformers.tokenization_utils_base as tok_base

    tok_flag = "_florence2_tf5_tok_getattr_patched"
    if getattr(tok_base.PreTrainedTokenizerBase, tok_flag, False):
        return

    if not hasattr(tok_base.PreTrainedTokenizerBase, "__getattr__"):
        return

    _orig_getattr = tok_base.PreTrainedTokenizerBase.__getattr__

    def _patched_getattr(self, name):
        if name == "additional_special_tokens":
            try:
                return _orig_getattr(self, name)
            except AttributeError:
                try:
                    added = _orig_getattr(self, "added_tokens_decoder")
                except AttributeError:
                    return []
                return [
                    tok.content
                    for tok in added.values()
                    if getattr(tok, "special", False)
                ]
        return _orig_getattr(self, name)

    tok_base.PreTrainedTokenizerBase.__getattr__ = _patched_getattr  # type: ignore[method-assign]
    setattr(tok_base.PreTrainedTokenizerBase, tok_flag, True)


def _patch_florence2_remote_code_files() -> bool:
    """Patch cached Florence-2 remote-code files in HF hub and transformers_modules."""
    patched_any = False
    for root in _SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for src in root.rglob("*.py"):
            for filename, old, new in _REMOTE_CODE_PATCHES:
                if src.name != filename:
                    continue
                try:
                    text = src.read_text(encoding="utf-8")
                except OSError:
                    continue
                if old in text and new not in text:
                    src.write_text(text.replace(old, new), encoding="utf-8")
                    patched_any = True

    modules_cache = _SEARCH_ROOTS[0]
    if patched_any and modules_cache.is_dir():
        for d in modules_cache.glob("*lorence*"):
            if d.is_dir():
                try:
                    shutil.rmtree(d)
                except OSError:
                    pass
    return patched_any


def _prefetch_florence2_remote_code(model_id: str) -> None:
    """Download remote-code files so file-level patches can run before import."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return

    try:
        snapshot_download(
            model_id,
            allow_patterns=[
                "configuration_florence2.py",
                "processing_florence2.py",
                "modeling_florence2.py",
            ],
        )
    except Exception:
        pass


def patch_florence2_transformers5_compat(*, model_id: str | None = None) -> None:
    """
    Apply Florence-2 compatibility fixes for transformers 5.x.

    Call before AutoProcessor/AutoModel.from_pretrained(..., trust_remote_code=True).
    Pass model_id (e.g. "microsoft/Florence-2-base") to prefetch and patch remote code
    before the first import on a fresh machine.
    """
    _patch_pretrained_config_defaults()
    _patch_tokenizer_defaults()

    if model_id:
        _prefetch_florence2_remote_code(model_id)

    _patch_florence2_remote_code_files()
