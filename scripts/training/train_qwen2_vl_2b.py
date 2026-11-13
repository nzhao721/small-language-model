#!/usr/bin/env python3
"""Fine-tune Qwen2-VL-2B for OMR (local port of train_qwen2_vl_2b_a100_colab.ipynb).

Requires: torch, transformers, peft, accelerate, pillow, tqdm, safetensors, packaging.
Assumes dependencies are already installed (no pip/restart).

Example:
  python scripts/train_qwen2_vl_2b.py \\
      --data-dir data/consolidated \\
      --golden-root golden_set \\
      --output-dir models/qwen2-vl-2b-handnew
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve()
while _SCRIPTS.name != "scripts":
    _SCRIPTS = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from _bootstrap import setup  # noqa: E402

REPO_ROOT = setup(__file__)

SYSTEM_PROMPT = "You are an optical music recognition system. Return only PrIMuS semantic tokens."
USER_PROMPT = (
    "Transcribe this monophonic sheet-music image into PrIMuS semantic format. "
    "Return only one space-separated token string with no explanation or markdown."
)
STYLE_WEIGHTS = {"handwritten": 0.50, "clean": 0.25, "camera": 0.25}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MANIFEST_VERSION = 1
GROUP_DIRS = {
    "handwritten_original": ("original", "images", "handwritten"),
    "clean_original": ("original", "images", "clean"),
    "camera_distorted_original": ("original", "images", "camera"),
    "handwritten_corrupt": ("corrupt", "images", "handwritten"),
    "clean_corrupt": ("corrupt", "images", "clean"),
    "camera_distorted_corrupt": ("corrupt", "images", "camera"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "consolidated")
    parser.add_argument("--golden-root", type=Path, default=REPO_ROOT / "golden_set")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "models" / "qwen2-vl-2b-handnew")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Cached training manifest (default: <output-dir>/../qwen2-vl-2b-manifest.json)",
    )
    parser.add_argument("--zip-path", type=Path, default=None, help="Optional consolidated.zip to unpack")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--golden-eval-limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-export-merged", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
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


def build_compute_plan() -> tuple[str, float, int, int, int]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen2-VL-2B training")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"BF16 required for the 2B Colab recipe; not supported on {gpu_name}")
    # Same VRAM tiers as the Colab notebook (no hard A100 name gate).
    if vram_gb >= 70:
        batch_size, grad_accum = 8, 4
    elif vram_gb >= 38:
        batch_size, grad_accum = 4, 8
    else:
        raise RuntimeError(f"Only {vram_gb:.1f} GB VRAM is visible; need at least ~38 GB")
    workers = min(12, os.cpu_count() or 4)
    return gpu_name, vram_gb, batch_size, grad_accum, workers


def canonical_id(stem: str) -> str:
    return stem.split("__", 1)[1] if "__" in stem else stem


def classify_record(stem: str) -> tuple[str, str]:
    prefix = stem.split("__", 1)[0].lower()
    source = "corrupt" if prefix.startswith("corrupt_") else "original"
    if "handwritten" in prefix or prefix == "mashcima":
        style = "handwritten"
    elif "camera" in prefix:
        style = "camera"
    else:
        style = "clean"
    return style, source


def load_golden_ids(golden_root: Path) -> set[str]:
    manifest_path = golden_root / "manifest.jsonl"
    ids: set[str] = set()
    if not manifest_path.is_file():
        print("Golden manifest not found; no ID-based leakage exclusions applied")
        return ids
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "incipit_id"):
            if row.get(key):
                ids.add(str(row[key]))
    print(f"Golden exclusion IDs: {len(ids):,}")
    return ids


def build_manifest(data_dir: Path, golden_root: Path) -> list[dict]:
    from tqdm.auto import tqdm

    images_dir = data_dir / "images"
    gt_dir = data_dir / "ground_truth"
    golden_ids = load_golden_ids(golden_root)
    records: list[dict] = []
    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    missing_gt = 0
    excluded = 0
    for image_path in tqdm(image_paths, desc="Scanning consolidated pairs"):
        gt_path = gt_dir / f"{image_path.stem}.semantic"
        if not gt_path.is_file():
            missing_gt += 1
            continue
        if canonical_id(image_path.stem) in golden_ids:
            excluded += 1
            continue
        target = " ".join(gt_path.read_text(encoding="utf-8").split())
        if not target:
            continue
        style, source = classify_record(image_path.stem)
        records.append(
            {
                "image": str(image_path),
                "text": target,
                "stem": image_path.stem,
                "style": style,
                "source": source,
            }
        )
    print(f"pairs={len(records):,} excluded={excluded:,} missing_gt={missing_gt:,}")
    return records


def load_or_build_manifest(
    data_dir: Path, golden_root: Path, manifest_path: Path, rebuild: bool
) -> list[dict]:
    if not rebuild and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("version") == MANIFEST_VERSION:
            print(f"Loaded cached manifest: {len(payload['records']):,} pairs")
            return payload["records"]
    records = build_manifest(data_dir, golden_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"version": MANIFEST_VERSION, "records": records}),
        encoding="utf-8",
    )
    print(f"Cached manifest -> {manifest_path}")
    return records


def stratified_limit(records: list[dict], limit: int | None, seed: int) -> list[dict]:
    if not limit or limit >= len(records):
        return records
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        buckets[(record["style"], record["source"])].append(record)
    selected: list[dict] = []
    for style, style_fraction in STYLE_WEIGHTS.items():
        available_sources = [source for source in ("original", "corrupt") if buckets[(style, source)]]
        if not available_sources:
            continue
        style_quota = round(limit * style_fraction)
        source_quota = style_quota // len(available_sources)
        for source in available_sources:
            bucket = buckets[(style, source)]
            selected.extend(rng.sample(bucket, min(source_quota, len(bucket))))
    if len(selected) < limit:
        selected_stems = {record["stem"] for record in selected}
        remainder = [record for record in records if record["stem"] not in selected_stems]
        selected.extend(rng.sample(remainder, min(limit - len(selected), len(remainder))))
    rng.shuffle(selected)
    return selected[:limit]


def build_sample_weights(records: list[dict]) -> list[float]:
    counts = Counter((record["style"], record["source"]) for record in records)
    weights: list[float] = []
    for record in records:
        style = record["style"]
        present_sources = [source for source in ("original", "corrupt") if counts[(style, source)] > 0]
        target_mass = STYLE_WEIGHTS[style] / len(present_sources)
        weights.append(target_mass / counts[(style, record["source"])])
    return weights


def latest_checkpoint(checkpoint_dir: Path) -> str | None:
    checkpoints = glob.glob(str(checkpoint_dir / "checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.rsplit("-", 1)[-1]))


def levenshtein(left: list[str], right: list[str]) -> int:
    _relative_minor_to_major = {
        "Am": "CM",
        "Em": "GM",
        "Bm": "DM",
        "F#m": "AM",
        "C#m": "EM",
        "G#m": "BM",
        "D#m": "F#M",
        "A#m": "C#M",
        "Dm": "FM",
        "Gm": "BbM",
        "Cm": "EbM",
        "Fm": "AbM",
        "Bbm": "DbM",
        "Ebm": "GbM",
        "Abm": "CbM",
    }
    _time_signature_canon = {
        "timeSignature-4/4": "timeSignature-C",
        "timeSignature-C": "timeSignature-C",
        "timeSignature-2/2": "timeSignature-C/",
        "timeSignature-C/": "timeSignature-C/",
    }

    def canonicalize(token: str) -> str:
        if token.startswith("keySignature-"):
            key = token[len("keySignature-") :]
            key = _relative_minor_to_major.get(key, key)
            return f"keySignature-{key}"
        return _time_signature_canon.get(token, token)

    left = [canonicalize(t) for t in left]
    right = [canonicalize(t) for t in right]
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, 1):
        current = [i]
        for j, right_token in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def resolve_golden_image(golden_root: Path, row: dict) -> Path | None:
    group = row["group"]
    if group not in GROUP_DIRS:
        return None
    filename = str(row["image_path"]).replace("\\", "/").rsplit("/", 1)[-1]
    folder = golden_root.joinpath(*GROUP_DIRS[group])
    exact = folder / filename
    if exact.is_file():
        return exact
    stem = Path(filename).stem
    return next(
        (path for path in folder.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_EXTS),
        None,
    )


def load_golden_records(golden_root: Path, limit: int | None) -> list[dict]:
    manifest_path = golden_root / "manifest.jsonl"
    if not manifest_path.is_file():
        print(f"Skipping golden eval: {manifest_path} not found")
        return []
    records = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_path = resolve_golden_image(golden_root, row)
        source = row.get("source", "original")
        gt_path = golden_root / source / "ground_truth" / f"{row['id']}.semantic"
        if image_path and gt_path.is_file():
            records.append(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "image": str(image_path),
                    "target": " ".join(gt_path.read_text(encoding="utf-8").split()),
                }
            )
    if limit and len(records) > limit:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            buckets[record["group"]].append(record)
        limited = []
        while len(limited) < limit and any(buckets.values()):
            for group in sorted(buckets):
                if buckets[group] and len(limited) < limit:
                    limited.append(buckets[group].pop(0))
        records = limited
    return records


def clean_prediction(text: str) -> str:
    text = re.sub(r"```(?:\w+)?", "", text or "").replace("```", "")
    return " ".join(text.split())


def main() -> int:
    args = parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from torch.utils.data import Dataset, WeightedRandomSampler
    from tqdm.auto import tqdm
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Trainer, TrainingArguments

    data_dir = args.data_dir.resolve()
    golden_root = args.golden_root.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    manifest_path = (
        args.manifest_path.resolve()
        if args.manifest_path
        else output_dir.parent / "qwen2-vl-2b-manifest.json"
    )
    min_pixels = 64 * 28 * 28
    max_pixels = 768 * 768

    maybe_unzip(args.zip_path, data_dir)
    images_dir = data_dir / "images"
    gt_dir = data_dir / "ground_truth"
    if not images_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(f"Expected {images_dir} and {gt_dir}")

    gpu_name, vram_gb, batch_size, grad_accum, workers = build_compute_plan()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
    print(
        f"batch={batch_size} x accum={grad_accum} "
        f"(effective {batch_size * grad_accum}) | workers={workers} | bf16 + TF32 + SDPA"
    )
    print(f"Output: {output_dir}")

    all_records = load_or_build_manifest(data_dir, golden_root, manifest_path, args.rebuild_manifest)
    train_records = stratified_limit(all_records, args.max_train_samples, args.seed)
    train_sample_weights = build_sample_weights(train_records)
    print(f"Training records: {len(train_records):,}")
    print("Strata:", dict(sorted(Counter((r["style"], r["source"]) for r in train_records).items())))

    # torchao preflight (same as notebook; informational / fail-fast)
    import importlib.machinery
    import importlib.metadata

    from packaging.version import InvalidVersion, Version

    torchao_min = Version("0.16.0")
    try:
        torchao_version_text = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        torchao_version_text = None
    torchao_importable = importlib.machinery.PathFinder.find_spec("torchao") is not None
    if torchao_version_text is None:
        if torchao_importable:
            raise RuntimeError("torchao files are importable but package metadata is missing")
        print("torchao preflight: absent (expected for BF16 LoRA)")
    else:
        try:
            torchao_version = Version(torchao_version_text)
        except InvalidVersion as exc:
            raise RuntimeError(f"Cannot validate torchao version {torchao_version_text!r}") from exc
        if torchao_version < torchao_min:
            raise RuntimeError(f"Incompatible torchao {torchao_version}; PEFT requires >= {torchao_min}")
        if not torchao_importable:
            raise RuntimeError(f"torchao {torchao_version} has metadata but cannot be imported")
        print(f"torchao preflight: compatible ({torchao_version})")

    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    print(f"Loading {args.model_id} in BF16 with SDPA...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "qkv",
            "fc1",
            "fc2",
            "lm_head",
        ],
    )
    model = get_peft_model(base_model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    vision_lora = [name for name, _ in model.named_parameters() if "visual" in name and "lora_" in name]
    language_lora = [
        name for name, _ in model.named_parameters() if "model.layers" in name and "lora_" in name
    ]
    print(f"Vision LoRA tensors: {len(vision_lora)}")
    print(f"Language LoRA tensors: {len(language_lora)}")
    if not vision_lora:
        raise RuntimeError("No vision LoRA modules matched; inspect Qwen module names before training")

    assistant_prefix_ids = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    if not assistant_prefix_ids:
        raise RuntimeError("Could not encode Qwen assistant prefix")

    class OMRDataset(Dataset):
        def __init__(self, records: list[dict]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict:
            record = self.records[index]
            try:
                with Image.open(record["image"]) as source:
                    image = source.convert("RGB")
                    image.load()
            except (OSError, SyntaxError, ValueError) as exc:
                print(f"Unreadable image {record['image']}: {exc}")
                return self[(index + 1) % len(self.records)]
            return {"image": image, "text": record["text"], "stem": record["stem"]}

    def find_last_subsequence(sequence: list[int], pattern: list[int]) -> int:
        for start in range(len(sequence) - len(pattern), -1, -1):
            if sequence[start : start + len(pattern)] == pattern:
                return start
        return -1

    def make_messages(target: str | None = None) -> list[dict]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]
        if target is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
        return messages

    class QwenOMRCollator:
        def __call__(self, examples: list[dict]) -> dict:
            texts = [
                processor.apply_chat_template(
                    make_messages(example["text"]),
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for example in examples
            ]
            images = [example["image"] for example in examples]
            batch = processor(
                text=texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=args.max_seq_length,
                return_tensors="pt",
            )
            labels = batch["input_ids"].clone()
            for row in range(labels.shape[0]):
                token_ids = batch["input_ids"][row].tolist()
                marker_start = find_last_subsequence(token_ids, assistant_prefix_ids)
                if marker_start < 0:
                    raise RuntimeError("Assistant marker missing after chat-template tokenization")
                target_start = marker_start + len(assistant_prefix_ids)
                labels[row, :target_start] = -100
                labels[row, batch["attention_mask"][row] == 0] = -100
                if not torch.any(labels[row] != -100):
                    raise RuntimeError(
                        f"Target was fully truncated; raise --max-seq-length above {args.max_seq_length}"
                    )
            batch["labels"] = labels
            return dict(batch)

    train_dataset = OMRDataset(train_records)
    data_collator = QwenOMRCollator()
    probe = data_collator([train_dataset[0], train_dataset[1]])
    print({key: tuple(value.shape) for key, value in probe.items() if hasattr(value, "shape")})
    print("Supervised tokens:", int((probe["labels"] != -100).sum()))
    del probe

    class StyleBalancedTrainer(Trainer):
        def __init__(self, *trainer_args, train_sample_weights=None, **kwargs):
            super().__init__(*trainer_args, **kwargs)
            self._train_sample_weights = train_sample_weights

        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            if self._train_sample_weights is not None and dataset is self.train_dataset:
                return WeightedRandomSampler(
                    self._train_sample_weights,
                    num_samples=len(dataset),
                    replacement=True,
                )
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                return super()._get_train_sampler()

    resume_path = None if args.no_resume else latest_checkpoint(checkpoint_dir)
    if resume_path:
        print(f"Resuming from {resume_path}")

    ta_kwargs = dict(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_train_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_drop_last=True,
        dataloader_num_workers=workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=workers > 0,
        dataloader_prefetch_factor=4 if workers > 0 else None,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )
    try:
        import inspect

        if "torch_compile" in inspect.signature(TrainingArguments.__init__).parameters:
            ta_kwargs["torch_compile"] = False
    except Exception:
        pass
    training_args = TrainingArguments(**ta_kwargs)

    trainer = StyleBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        train_sample_weights=train_sample_weights,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_path)
    print(train_result)

    adapter_dir = output_dir / "final-adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter + processor -> {adapter_dir}")

    # End-only golden eval (notebook cell 8)
    def predict_one(image_path: str) -> str:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        prompt_text = processor.apply_chat_template(
            make_messages(),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        generated = output_ids[:, prompt_length:]
        return clean_prediction(
            processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        )

    golden_records = load_golden_records(golden_root, args.golden_eval_limit)
    if golden_records:
        model.eval()
        rows = []
        for record in tqdm(golden_records, desc="Golden eval"):
            prediction = predict_one(record["image"])
            target_tokens = record["target"].split()
            prediction_tokens = prediction.split()
            edits = levenshtein(target_tokens, prediction_tokens)
            accuracy = (1.0 - edits / max(1, len(target_tokens))) * 100.0
            rows.append({**record, "prediction": prediction, "edits": edits, "accuracy_pct": accuracy})
        by_group = defaultdict(list)
        for row in rows:
            by_group[row["group"]].append(row["accuracy_pct"])
        summary = {
            "model_id": args.model_id,
            "n": len(rows),
            "mean_accuracy_pct": sum(row["accuracy_pct"] for row in rows) / len(rows),
            "exact_matches": sum(row["edits"] == 0 for row in rows),
            "by_group": {group: sum(values) / len(values) for group, values in sorted(by_group.items())},
            "samples": rows,
        }
        eval_path = output_dir / "eval_golden_after.json"
        eval_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, indent=2))
        print(f"Saved -> {eval_path}")

    if not args.no_export_merged:
        merged_dir = output_dir / "final-merged"
        print("Merging LoRA into Qwen2-VL-2B base weights...")
        merged_model = trainer.model.merge_and_unload()
        merged_model.config.use_cache = True
        merged_model.save_pretrained(
            str(merged_dir),
            safe_serialization=True,
            max_shard_size="4GB",
        )
        processor.save_pretrained(str(merged_dir))
        print(f"Standalone merged model -> {merged_dir}")
    else:
        print(f"Merged export disabled; adapter is at {adapter_dir}")

    print("Done. Resume checkpoints remain under:", checkpoint_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"TRAINING FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
