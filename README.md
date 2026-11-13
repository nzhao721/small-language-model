# omr-monophonic

> ## Evaluation canvases (read this first)
>
> The golden-set evaluation results are presented in **two Cursor canvases** checked into this repo under [`canvases/`](canvases/):
>
> 1. **[`canvases/qwen2-2b-model-summary.canvas.tsx`](canvases/qwen2-2b-model-summary.canvas.tsx)** — overall benchmark summary: fine-tuned Qwen2-VL-2B vs Florence models and frontier models (token accuracy and exact match).
> 2. **[`canvases/qwen2-2b-final-error-analysis.canvas.tsx`](canvases/qwen2-2b-final-error-analysis.canvas.tsx)** — remaining-error analysis for checkpoint-9000 (where edits occur, error types, hardest samples).
>
> Open either `.canvas.tsx` file in Cursor to view the interactive reports beside the chat.

---

LoRA adapter for **monophonic optical music recognition (OMR)**. Given a sheet-music image, the model transcribes it into **PrIMuS semantic tokens** as a single space-separated string.

This is a task-specific adapter on top of [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct). Load the base model first, then apply this adapter with PEFT.

- **Hugging Face model:** [`nzhao721/omr-monophonic`](https://huggingface.co/nzhao721/omr-monophonic)
- **Hugging Face dataset:** [`nzhao721/omr-monophonic-consolidated`](https://huggingface.co/datasets/nzhao721/omr-monophonic-consolidated)

## Model details

| Field | Value |
| --- | --- |
| Base model | `Qwen/Qwen2-VL-2B-Instruct` |
| Method | LoRA (PEFT) |
| Checkpoint | `checkpoint-9000` (2.07 epochs) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Precision | BF16 training |
| Output format | PrIMuS semantic tokens |
| PEFT version | 0.19.1 |

**Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `qkv`, `gate_proj`, `up_proj`, `down_proj`, `fc1`, `fc2`, `lm_head`

## Intended use

Use this adapter when you need to convert **monophonic** sheet-music images into PrIMuS semantic notation for:

- music-theory and ear-training workflows
- melodic dictation / composition exercises
- downstream symbolic music processing

This is **not** a general music understanding model and is **not** trained for polyphonic scores.

## Evaluation

Evaluated on a 100-image golden set using token-level edit distance. Equivalent major/minor key signatures and common/cut-time aliases (`4/4` ↔ `C`, `2/2` ↔ `C/`) are canonicalized before scoring. At most one trailing barline on either side is forgiven.

**For the full interactive breakdown, open the two canvases listed at the top of this README.**

### Token accuracy (%)

| Dataset | Score |
| --- | ---: |
| handwritten_original | 98.49 |
| clean_original | 100.0 |
| camera_distorted_original | 98.5 |
| handwritten_corrupt | 98.81 |
| clean_corrupt | 100.0 |
| camera_distorted_corrupt | 99.57 |
| manual_handwritten | 93.16 |
| manual_digital | 98.69 |
| **Average** | **97.91** |

### Exact match (%)

Exact match is the share of samples with zero token edits after normalization.

| Dataset | Score |
| --- | ---: |
| handwritten_original | 80.0 |
| clean_original | 100.0 |
| camera_distorted_original | 90.0 |
| handwritten_corrupt | 80.0 |
| clean_corrupt | 100.0 |
| camera_distorted_corrupt | 90.0 |
| manual_handwritten | 15.0 |
| manual_digital | 65.0 |
| **Average** | **70.0** |

## Usage

Install dependencies:

```bash
pip install transformers accelerate peft qwen-vl-utils pillow torch
```

Load the adapter:

```python
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_ID = "nzhao721/omr-monophonic"

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(model, ADAPTER_ID)
model.eval()

image = Image.open("score.png").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
                    "Transcribe this monophonic sheet-music image into PrIMuS semantic format. "
                    "Return only one space-separated token string with no explanation or markdown."
                ),
            },
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=512)
response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(response)
```

## Training

- **Dataset:** consolidated OMR corpus (~139k image / PrIMuS semantic pairs)
- **Sampling mix:** 50% handwritten, 25% clean digital, 25% camera-distorted; balanced original/corrupt within each style
- **Hardware:** NVIDIA A100 (40 GB or 80 GB)
- **Objective:** assistant-only loss on PrIMuS token strings (image and prompt tokens masked)

## Repo layout

| Path | Purpose |
| --- | --- |
| `canvases/` | Cursor canvases for eval summary and error analysis |
| `notebooks/` | Colab training, eval, HF upload, and demo server notebooks |
| `demo/` | Localhost demo UI + FastAPI (Colab inference + Verovio render) |
| `scripts/` | Scoring, rendering, dataset packaging utilities |
| `omr_benchmark/` | Frontier and fine-tuned evaluation score tables / JSON |
| `datasets/omr-monophonic-consolidated/` | Dataset card for the HF consolidated archive |

Large weight files and the full consolidated image tree are **not** stored in git. Use the Hugging Face model and dataset links above.

## Limitations

- Monophonic excerpts only; polyphonic or orchestral scores are out of scope.
- Weakest slice is manually captured handwritten examples (`manual_handwritten`: 93.16% token accuracy, 15% exact match).
- Output is PrIMuS semantic tokens, not MEI, MusicXML, or LilyPond.
- Performance depends on image quality, cropping, and similarity to training styles.

## License

This adapter is released under **Apache 2.0**. The base model [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) has its own license; follow the base model terms when using this adapter.

## Citation

If you use PrIMuS or this evaluation framing, cite the PrIMuS OMR work:

```bibtex
@article{calvo2018end,
  title={End-to-End Neural Optical Music Recognition of Monophonic Scores},
  author={Calvo-Zaragoza, Jorge and Rizo, David},
  journal={Applied Sciences},
  volume={8},
  number={4},
  pages={606},
  year={2018}
}
```
