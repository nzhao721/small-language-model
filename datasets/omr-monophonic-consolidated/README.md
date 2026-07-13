---
pretty_name: OMR Monophonic Consolidated
license: apache-2.0
task_categories:
- image-to-text
tags:
- optical-music-recognition
- omr
- primus
- sheet-music
- monophonic
size_categories:
- 100K<n<1M
---

# OMR Monophonic Consolidated

Paired monophonic sheet-music images and **PrIMuS semantic** ground-truth strings used to fine-tune [`NathanZ721/omr-monophonic`](https://huggingface.co/NathanZ721/omr-monophonic).

## Contents

Download `consolidated.zip`. After extraction you get:

```text
consolidated/
  images/          # one image per example
  ground_truth/    # matching *.semantic files (same stem)
```

Each image filename is `{group}__{id}.{ext}` with a sibling `{group}__{id}.semantic`.

### Groups (≈139k pairs)

| Group | Approx. count | Description |
| --- | ---: | --- |
| `primus` | 36,951 | Clean digital PrIMuS renders |
| `camera_primus` | 36,951 | Camera-style distorted PrIMuS renders |
| `mashcima` | 36,584 | Handwritten-style Mashcima synthesis |
| `corrupt_clean` | 9,806 | Corrupted clean digital |
| `corrupt_camera` | 9,734 | Corrupted camera-distorted |
| `corrupt_handwritten` | 9,456 | Corrupted handwritten |

Exact counts may vary slightly by export; see the archive itself.

## Ground-truth format

Each `.semantic` file is a single line of space-separated PrIMuS semantic tokens, for example:

```text
clef-G2 keySignature-GM timeSignature-4/4 note-G4_quarter note-A4_quarter note-B4_quarter note-C5_quarter barline
```

## Intended use

Training and evaluating monophonic optical music recognition (OMR) systems that emit PrIMuS semantic tokens.

**Out of scope:** polyphonic / orchestral scores, audio transcription, MusicXML/MEI as the primary target format.

## Related model

- Adapter: [`NathanZ721/omr-monophonic`](https://huggingface.co/NathanZ721/omr-monophonic)
- Base: [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

## Loading in Colab / Python

```python
from huggingface_hub import hf_hub_download
import zipfile
from pathlib import Path

zip_path = hf_hub_download(
    repo_id="NathanZ721/omr-monophonic-consolidated",
    filename="consolidated.zip",
    repo_type="dataset",
)

out = Path("/content/omr_data")
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out)

images = out / "consolidated" / "images"
gt = out / "consolidated" / "ground_truth"
print(images, gt)
```

## Citation

If you use PrIMuS-derived material, please cite:

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
