"""Run local VLM inference for OMR benchmark batches (zero-shot)."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "examples" / "omr_benchmark"
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from florence2_transformers5_compat import patch_florence2_transformers5_compat

SYSTEM_PROMPT = """You transcribe monophonic sheet music images to PrIMuS semantic format.

Output ONLY a single space-separated token string (no markdown, no explanation).

PrIMuS tokens:
- Header: clef-G2 (or clef-C1, clef-C3, clef-F4, clef-C2, clef-F3), keySignature-CM (or DM, GM, BbM, AbM, Fm, etc.), timeSignature-C (or 2/4, 3/4, 4/4, C/, etc.)
- Notes: note-{Pitch}{Octave}_{duration} e.g. note-C5_quarter, note-F#4_eighth, note-Bb3_half
- Rests: rest-{duration} or rest_{duration}
- Barlines: barline
- Durations: whole, half, quarter, eighth, sixteenth, thirty_second, double_whole (append . for dotted)
- Use # for sharps and b for flats in pitch names (e.g. F#4, Bb4, Eb5)
"""

USER_PROMPT = (
    "Transcribe this monophonic melody to PrIMuS semantic format. "
    "Return only the space-separated PrIMuS token string."
)

VLM_MODELS: dict[str, dict[str, str]] = {
    "florence-2-base": {
        "hf_id": "microsoft/Florence-2-base",
        "family": "florence",
    },
    "donut-base": {
        "hf_id": "naver-clova-ix/donut-base",
        "family": "donut",
    },
    "florence-2-large": {
        "hf_id": "microsoft/Florence-2-large",
        "family": "florence",
    },
    "qwen2-vl-2b": {
        "hf_id": "Qwen/Qwen2-VL-2B-Instruct",
        "family": "qwen2-vl",
    },
    "qwen2-vl-7b": {
        "hf_id": "Qwen/Qwen2-VL-7B-Instruct",
        "family": "qwen2-vl",
        "quantize_4bit": "true",
    },
}

BENCHMARK_SUITES = [
    ("predictions_10", BENCH / "batches_10"),
    ("predictions_10_corrupt", BENCH / "batches_10_corrupt"),
    ("predictions_manual", BENCH / "batches_manual"),
]


def clean_prediction(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = " ".join(lines).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def device_info() -> dict:
    info = {
        "cuda": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if info["cuda"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_gb"] = round(props.total_memory / 1e9, 2)
    return info


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class Florence2Runner:
    def __init__(self, hf_id: str, device: str) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        patch_florence2_transformers5_compat(model_id=hf_id)
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"Loading {hf_id} on {device} ({dtype}) ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        self.model.eval()
        self.device = device
        self.task_prompt = (
            "<VQA>Transcribe this monophonic sheet music to PrIMuS semantic format "
            "(clef-G2, keySignature-CM, timeSignature-4/4, note-C5_quarter, rest_quarter, barline). "
            "Return only the space-separated token string."
        )

    def predict(self, image_path: Path) -> str:
        image = load_image(image_path)
        # Florence-2 expects square 768x768 inputs; processor may skip resize on transformers 5.x.
        target = 768
        scale = min(target / image.width, target / image.height)
        resized = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        square = Image.new("RGB", (target, target), (255, 255, 255))
        square.paste(
            resized,
            ((target - resized.width) // 2, (target - resized.height) // 2),
        )
        inputs = self.processor(
            text=self.task_prompt,
            images=square,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        beam = 1 if self.device == "cpu" else 3
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=beam,
                do_sample=False,
                use_cache=False,
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task="<VQA>",
            image_size=(square.width, square.height),
        )
        if isinstance(parsed, dict):
            text = parsed.get("<VQA>") or parsed.get("answer") or str(parsed)
        else:
            text = str(parsed)
        return clean_prediction(text)

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DonutRunner:
    def __init__(self, hf_id: str, device: str) -> None:
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"Loading {hf_id} on {device} ({dtype}) ...", flush=True)
        self.processor = DonutProcessor.from_pretrained(hf_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            hf_id,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()
        self.device = device
        self.task_prompt = f"<s>{SYSTEM_PROMPT}\n{USER_PROMPT}"

    def predict(self, image_path: Path) -> str:
        image = load_image(image_path)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        decoder_input_ids = self.processor.tokenizer(
            self.task_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            )
        sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<[^>]+>", "", sequence, count=1).strip()
        return clean_prediction(sequence)

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Qwen2VLRunner:
    def __init__(self, hf_id: str, device: str, *, quantize_4bit: bool = False) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        load_kwargs: dict = {"trust_remote_code": True}
        if quantize_4bit and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["device_map"] = "auto"
                print(f"Loading {hf_id} with 4-bit quantization ...", flush=True)
            except Exception as exc:
                print(f"4-bit load unavailable ({exc}); falling back to float16.", flush=True)
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
        elif device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32
            load_kwargs["device_map"] = "cpu"
            load_kwargs["low_cpu_mem_usage"] = True
            print(f"Loading {hf_id} on CPU (float32, low_cpu_mem_usage) ...", flush=True)

        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(hf_id, **load_kwargs)
        self.model.eval()
        self.device = device
        self.prompt = f"{SYSTEM_PROMPT}\n{USER_PROMPT}"

    def predict(self, image_path: Path) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                        "min_pixels": 256 * 28 * 28,
                        "max_pixels": 768 * 28 * 28,
                    },
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if self.device == "cpu":
            inputs = inputs.to("cpu")
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return clean_prediction(output_text)

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_runner(model_slug: str, info: dict) -> Florence2Runner | DonutRunner | Qwen2VLRunner:
    cfg = VLM_MODELS[model_slug]
    family = cfg["family"]
    device = info["device"]
    hf_id = cfg["hf_id"]
    if family == "florence":
        return Florence2Runner(hf_id, device)
    if family == "donut":
        return DonutRunner(hf_id, device)
    if family == "qwen2-vl":
        quantize = cfg.get("quantize_4bit") == "true"
        if device == "cpu" and model_slug == "qwen2-vl-7b":
            raise RuntimeError(
                "qwen2-vl-7b requires GPU or >16GB RAM for CPU inference; skipped on this machine"
            )
        return Qwen2VLRunner(hf_id, device, quantize_4bit=quantize)
    raise ValueError(f"Unknown family for {model_slug}")


def run_batch(
    runner: Florence2Runner | DonutRunner | Qwen2VLRunner,
    batch_path: Path,
    out_path: Path,
) -> int:
    entries = json.loads(batch_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if out_path.is_file():
        lines = [
            ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
    start_idx = len(lines)
    if start_idx:
        print(f"  Resuming from sample {start_idx + 1}/{len(entries)}", flush=True)
    for i, entry in enumerate(entries[start_idx:], start=start_idx):
        image_path = Path(entry["image_path"])
        if not image_path.is_file():
            print(
                f"  [{i + 1}/{len(entries)}] {entry['filename']} MISSING -> empty prediction",
                flush=True,
            )
            record = {"filename": entry["filename"], "prediction": ""}
            lines.append(json.dumps(record, ensure_ascii=False))
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            continue
        t0 = time.time()
        print(f"  [{i + 1}/{len(entries)}] {entry['filename']} ...", flush=True)
        prediction = runner.predict(image_path)
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.1f}s ({len(prediction)} chars)", flush=True)
        record = {"filename": entry["filename"], "prediction": prediction}
        lines.append(json.dumps(record, ensure_ascii=False))
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def merge_predictions(model_dir: Path, out_path: Path) -> int:
    lines: list[str] = []
    for batch_file in sorted(model_dir.glob("batch_*.jsonl")):
        for line in batch_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                lines.append(line)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def run_model(
    model_slug: str,
    *,
    suites: list[tuple[str, Path]] | None = None,
    batch_filter: str | None = None,
    skip_existing: bool = True,
) -> dict:
    info = device_info()
    print(f"Device info: {info}", flush=True)
    suites = suites or BENCHMARK_SUITES
    status: dict = {"model": model_slug, "device": info, "suites": {}, "error": None}
    runner = None
    try:
        runner = create_runner(model_slug, info)
        for pred_name, batches_dir in suites:
            pred_root = BENCH / pred_name
            model_dir = pred_root / model_slug
            model_dir.mkdir(parents=True, exist_ok=True)
            batch_files = sorted(batches_dir.glob("batch_*.json"))
            if batch_filter:
                batch_files = [batches_dir / f"{batch_filter}.json"]
            suite_total = 0
            for batch_path in batch_files:
                out_path = model_dir / f"{batch_path.stem}.jsonl"
                if skip_existing and out_path.is_file():
                    existing = [
                        ln
                        for ln in out_path.read_text(encoding="utf-8").splitlines()
                        if ln.strip()
                    ]
                    expected = len(json.loads(batch_path.read_text(encoding="utf-8")))
                    nonempty = sum(
                        1
                        for ln in existing
                        if json.loads(ln).get("prediction", "").strip()
                    )
                    if len(existing) >= expected and nonempty >= expected:
                        print(
                            f"Skipping {batch_path.name} ({len(existing)} predictions exist)",
                            flush=True,
                        )
                        suite_total += len(existing)
                        continue
                print(f"Batch {batch_path.name} -> {out_path}", flush=True)
                n = run_batch(runner, batch_path, out_path)
                suite_total += n
            merged = merge_predictions(model_dir, pred_root / f"{model_slug}.jsonl")
            status["suites"][pred_name] = {"batch_lines": suite_total, "merged": merged}
            print(f"Suite {pred_name}: merged {merged} -> {pred_root / f'{model_slug}.jsonl'}", flush=True)
    except Exception as exc:
        status["error"] = str(exc)
        print(f"ERROR for {model_slug}: {exc}", file=sys.stderr, flush=True)
    finally:
        if runner is not None:
            runner.close()
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(VLM_MODELS),
        help="Run a single model slug",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all shortlist models sequentially",
    )
    parser.add_argument("--batch", help="Run single batch stem e.g. batch_00")
    parser.add_argument(
        "--predictions-suite",
        choices=[s[0] for s in BENCHMARK_SUITES],
        help="Run only one benchmark suite",
    )
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument(
        "--status-json",
        type=Path,
        default=BENCH / "vlm_run_status.json",
    )
    args = parser.parse_args()

    if not args.model and not args.all_models:
        parser.error("Specify --model or --all-models")

    suites = BENCHMARK_SUITES
    if args.predictions_suite:
        suites = [s for s in BENCHMARK_SUITES if s[0] == args.predictions_suite]

    models = list(VLM_MODELS) if args.all_models else [args.model]
    all_status: list[dict] = []
    for model_slug in models:
        print(f"\n=== {model_slug} ===", flush=True)
        status = run_model(
            model_slug,
            suites=suites,
            batch_filter=args.batch,
            skip_existing=not args.no_skip_existing,
        )
        all_status.append(status)

    args.status_json.write_text(json.dumps(all_status, indent=2), encoding="utf-8")
    print(f"\nWrote status -> {args.status_json}", flush=True)


if __name__ == "__main__":
    main()
