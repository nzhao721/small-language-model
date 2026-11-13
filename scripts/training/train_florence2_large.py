#!/usr/bin/env python3
"""Fine-tune Florence-2-large for OMR (local port of train_florence2_large_a100_colab.ipynb).

Requires: torch, transformers, peft, accelerate, pillow, tqdm, safetensors.
Assumes dependencies are already installed (no pip/restart).

VRAM-tier batching matches the Colab notebook (no hard A100 *name* gate). CUDA + BF16
and at least ~38 GB VRAM are required. Eager attention + Inductor without CUDA Graphs.

Example:
  python scripts/train_florence2_large.py \\
      --data-dir data/consolidated \\
      --golden-root golden_set \\
      --output-dir models/florence2-large-handnew-r64-lr1e4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments

_SCRIPTS = Path(__file__).resolve()
while _SCRIPTS.name != "scripts":
    _SCRIPTS = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from _bootstrap import setup  # noqa: E402

REPO_ROOT = setup(__file__)
SCRIPTS_DIR = Path(__file__).resolve().parent

import florence2_train_utils as U
from florence2_primus import (
    build_florence2_primus_processor,
    encode_semantic_labels,
    make_primus_logits_processor,
    resize_model_embeddings_for_primus,
)

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
    "qkv",
    "lm_head",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "consolidated")
    parser.add_argument("--golden-root", type=Path, default=REPO_ROOT / "golden_set")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "models" / "florence2-large-handnew-r64-lr1e4",
    )
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--vocab-path", type=Path, default=None)
    parser.add_argument("--model-id", default="microsoft/Florence-2-large")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-label-length", type=int, default=96)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-export-merged", action="store_true")
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--no-torch-compile", action="store_true")
    parser.add_argument("--hf-repo-id", default="")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="florence2-omr")
    return parser.parse_args()


def maybe_unzip(zip_path: Path | None, data_dir: Path) -> None:
    images_dir = data_dir / "images"
    if images_dir.is_dir() and any(images_dir.iterdir()):
        print(f"Local data already present: {images_dir}")
        return
    if zip_path is None or not zip_path.is_file():
        return
    parent = data_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {parent} ...")
    started = time.time()
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(parent)
    print(f"Extracted in {(time.time() - started) / 60:.1f} minutes")


def discover_vocab(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            SCRIPTS_DIR / "primus_semantic_tokens.txt",
            REPO_ROOT / "data" / "primus_semantic_tokens.txt",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "PrIMuS vocab not found. Pass --vocab-path or place "
        "data/primus_semantic_tokens.txt in the repo."
    )


def configure_inductor_no_cudagraphs() -> None:
    """Safe Inductor path: compile eager-attn model but prohibit CUDA Graph capture."""
    os.environ.pop("TORCHDYNAMO_DISABLE", None)
    os.environ.pop("ACCELERATE_DYNAMO_BACKEND", None)
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    torch._dynamo.config.disable = False
    if hasattr(inductor_config.triton, "cudagraphs"):
        inductor_config.triton.cudagraphs = False
    if hasattr(inductor_config.triton, "cudagraph_trees"):
        inductor_config.triton.cudagraph_trees = False
    print(
        "TorchDynamo enabled; CUDA Graphs:",
        getattr(inductor_config.triton, "cudagraphs", "unavailable"),
    )


def build_compute_plan() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Florence-2-large training")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support was not detected")

    # Same VRAM tiers as the Colab notebook (no hard A100 name gate).
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 70:
        batch_size, grad_accum = 12, 4
    elif vram_gb >= 38:
        batch_size, grad_accum = 6, 8
    else:
        raise RuntimeError(
            f"Only {vram_gb:.1f} GB VRAM is visible; need at least ~38 GB for this recipe"
        )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    configure_inductor_no_cudagraphs()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    return {
        "device": "cuda",
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "use_bf16": True,
        "use_fp16": False,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "grad_ckpt": False,
        "workers": min(12, os.cpu_count() or 4),
    }


def configure_utils(args: argparse.Namespace, plan: dict) -> None:
    output_dir = args.output_dir
    U.GOLDEN_DIR = args.golden_root
    U.IMAGES_DIR = args.data_dir / "images"
    U.GT_DIR = args.data_dir / "ground_truth"
    U.DRIVE_OUTPUT_DIR = output_dir
    U.MANIFEST_PATH = output_dir / "manifest.json"
    U.EVAL_GOLDEN_BEFORE_PATH = output_dir / "eval_golden_before.json"
    U.EVAL_GOLDEN_AFTER_PATH = output_dir / "eval_golden_after.json"
    U.EVAL_GOLDEN_EPOCH_DIR = output_dir / "eval_golden_epochs"
    U.EVAL_GOLDEN_EPOCH_HISTORY_PATH = output_dir / "eval_golden_epoch_history.json"
    U.EVAL_GOLDEN_EPOCH_WRITEUP_PATH = output_dir / "eval_golden_epochs.md"
    U.MANIFEST_VERSION = 5
    U.GOLDEN_BENCHMARK_COUNT = 100
    U.STYLE_WEIGHTS = {"handwritten": 0.50, "clean": 0.25, "camera": 0.25}
    U.STYLE_WEIGHT_CLEAN_VS_CORRUPT = 0.50
    U.USE_CONSTRAINED_DECODE = True
    U.EVAL_GOLDEN_EVERY_EPOCH = True
    U.DEVICE = plan["device"]
    U.TASK_PROMPT = "<OCR>"
    U.MODEL_ID = args.model_id
    U.SEED = args.seed
    U.encode_semantic_labels = encode_semantic_labels
    U.make_primus_logits_processor = make_primus_logits_processor


def find_latest_checkpoint(checkpoint_dir: Path) -> str | None:
    ckpts = sorted(
        glob.glob(str(checkpoint_dir / "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    return ckpts[-1] if ckpts else None


def main() -> None:
    args = parse_args()
    maybe_unzip(args.zip_path, args.data_dir)
    vocab_path = discover_vocab(args.vocab_path)
    plan = build_compute_plan()
    configure_utils(args, plan)

    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {plan['device']}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")
    print(
        f"Compute plan: batch={plan['batch_size']} x accum={plan['grad_accum']} "
        f"(effective {plan['batch_size'] * plan['grad_accum']}) | "
        f"grad_ckpt={plan['grad_ckpt']} | workers={plan['workers']} | "
        f"precision=bf16 | compile={not args.no_torch_compile}"
    )
    print(f"Images dir exists: {U.IMAGES_DIR.is_dir()}")
    print(f"GT dir exists: {U.GT_DIR.is_dir()}")
    print(f"PRIMUS_VOCAB_PATH={vocab_path}")

    manifest = U.load_or_build_manifest()
    train_ds = U.OMRDataset(manifest["splits"]["train"])
    golden_records = manifest["splits"]["eval_golden"]
    train_sample_weights = U.build_style_sample_weights(
        manifest["splits"]["train"],
        U.STYLE_WEIGHTS,
        U.STYLE_WEIGHT_CLEAN_VS_CORRUPT,
    )
    U.print_style_mix_stats(manifest["splits"]["train"], train_sample_weights)
    print(f"train={len(train_ds):,}  golden_eval={len(golden_records):,}")

    U.patch_florence2_transformers5_compat(model_id=args.model_id)
    print(f"Florence-2 compat applied with model_id={args.model_id!r}")

    processor = build_florence2_primus_processor(
        args.model_id,
        vocab_path=vocab_path,
        trust_remote_code=True,
    )
    print(f"PrIMuS tokenizer vocab_size={processor.tokenizer.vocab_size:,}")

    dtype = torch.bfloat16
    print(f"Loading {args.model_id} ({dtype}, eager attention)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    resize_model_embeddings_for_primus(base_model, processor.tokenizer)
    base_model.config.use_cache = False
    print("Base model loaded with resized PrIMuS embeddings")

    U.processor = processor
    U.golden_records = golden_records

    baseline_results = None
    if args.skip_baseline_eval and U.EVAL_GOLDEN_BEFORE_PATH.is_file():
        baseline_results = json.loads(
            U.EVAL_GOLDEN_BEFORE_PATH.read_text(encoding="utf-8")
        )
        print(f"SKIP_BASELINE_EVAL — loaded cached baseline from {U.EVAL_GOLDEN_BEFORE_PATH}")
    else:
        baseline_results = U.run_golden_eval(
            base_model,
            processor,
            golden_records,
            phase="before_training",
            model_id=args.model_id,
            task_prompt=U.TASK_PROMPT,
            device=plan["device"],
            use_lora=False,
        )
        U.print_golden_eval_summary(baseline_results, "Before training (base)")
        U.EVAL_GOLDEN_BEFORE_PATH.write_text(
            json.dumps(baseline_results, indent=2), encoding="utf-8"
        )
        print(f"Wrote {U.EVAL_GOLDEN_BEFORE_PATH}")
    U.baseline_results = baseline_results

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    if plan["grad_ckpt"]:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing: ON")
    else:
        print("Gradient checkpointing: OFF (A100 fast path)")

    collate_fn = U.make_collate_fn(processor, U.TASK_PROMPT, args.max_label_length)

    report_to = []
    if args.use_wandb:
        report_to.append("wandb")
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    steps_per_epoch = max(
        1,
        (len(train_ds) + plan["batch_size"] * plan["grad_accum"] - 1)
        // (plan["batch_size"] * plan["grad_accum"]),
    )
    print(f"~{steps_per_epoch:,} optimizer steps / epoch")

    resume_path = None
    if not args.no_resume:
        resume_path = find_latest_checkpoint(checkpoint_dir)
        if resume_path:
            print(f"Resuming from {resume_path}")

    # Reset any stale compiled wrapper, then re-enable Inductor with CUDA Graphs disabled.
    configure_inductor_no_cudagraphs()
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=plan["batch_size"],
        gradient_accumulation_steps=plan["grad_accum"],
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        torch_compile=not args.no_torch_compile,
        torch_compile_backend="inductor",
        torch_compile_mode="default",
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=plan["workers"],
        dataloader_pin_memory=True,
        dataloader_persistent_workers=(plan["workers"] > 0),
        dataloader_prefetch_factor=(6 if plan["workers"] > 0 else None),
        report_to=report_to,
        seed=args.seed,
    )

    trainer = U.StyleBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
        callbacks=[U.GoldenEpochEvalCallback()],
        train_sample_weights=train_sample_weights,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_path)
    print(train_result)

    best_dir = output_dir / "best"
    latest_ckpt = find_latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        print(f"Copying latest checkpoint -> {best_dir}")
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(latest_ckpt, best_dir)
        processor.save_pretrained(str(best_dir))
    else:
        trainer.save_model(str(best_dir))
        processor.save_pretrained(str(best_dir))
    print(f"Saved adapter snapshot -> {best_dir}")

    after_results = U.run_golden_eval(
        model,
        processor,
        golden_records,
        phase="after_training",
        model_id=args.model_id,
        task_prompt=U.TASK_PROMPT,
        device=plan["device"],
        use_lora=True,
    )
    U.print_golden_eval_summary(after_results, "After training (LoRA)")
    U.EVAL_GOLDEN_AFTER_PATH.write_text(
        json.dumps(after_results, indent=2), encoding="utf-8"
    )
    print(f"Wrote {U.EVAL_GOLDEN_AFTER_PATH}")
    if baseline_results is not None:
        U.compare_golden_eval_results(baseline_results, after_results)

    if not args.no_export_merged:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(final_dir))
        processor.save_pretrained(str(final_dir))
        print(f"Merged model saved -> {final_dir}")
        if args.hf_repo_id:
            if not args.hf_token:
                print("Set HF_TOKEN / --hf-token to push to the Hub.")
            else:
                merged.push_to_hub(args.hf_repo_id, token=args.hf_token)
                processor.push_to_hub(args.hf_repo_id, token=args.hf_token)
                print(f"Pushed to https://huggingface.co/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
