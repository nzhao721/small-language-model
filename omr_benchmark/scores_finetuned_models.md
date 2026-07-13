# Fine-Tuned OMR Model Benchmark Scores

All results use the same 100-image golden set and token-level Levenshtein scoring. Relative major/minor key signatures, `4/4` ↔ `C`, and `2/2` ↔ `C/` are treated as equivalent; at most one trailing barline on either side is forgiven.

| Model | Checkpoint evaluated |
| --- | --- |
| Florence-2-base | Final 3-epoch adapter (`florence2-base_eval.json`) |
| Florence-2-large | Final 3-epoch adapter (`eval_florence2-large.json`) |
| Qwen2-VL-2B | `checkpoint-9000` / 2.07 epochs (`eval_golden_checkpoint-9000.json`) |

Samples scored: 100.

## Token accuracy %

| Dataset | florence-2-base | florence-2-large | qwen2-vl-2b |
| --- | --- | --- | --- |
| handwritten_original | 92.08 | 89.77 | 98.49 |
| clean_original | 96.87 | 97.65 | 100.0 |
| camera_distorted_original | 95.61 | 95.61 | 98.5 |
| handwritten_corrupt | 93.74 | 93.26 | 98.81 |
| clean_corrupt | 97.82 | 98.19 | 100.0 |
| camera_distorted_corrupt | 96.39 | 95.52 | 99.57 |
| manual_handwritten | 83.71 | 90.65 | 93.16 |
| manual_digital | 96.47 | 98.31 | 98.69 |
| **Average** | 93.29 | 94.79 | 97.91 |

## Exact match %

Exact match % is the share of samples with zero token edits after normalization.

| Dataset | florence-2-base | florence-2-large | qwen2-vl-2b |
| --- | --- | --- | --- |
| handwritten_original | 10.0 | 20.0 | 80.0 |
| clean_original | 70.0 | 60.0 | 100.0 |
| camera_distorted_original | 60.0 | 60.0 | 90.0 |
| handwritten_corrupt | 10.0 | 40.0 | 80.0 |
| clean_corrupt | 80.0 | 80.0 | 100.0 |
| camera_distorted_corrupt | 50.0 | 50.0 | 90.0 |
| manual_handwritten | 5.0 | 10.0 | 15.0 |
| manual_digital | 45.0 | 60.0 | 65.0 |
| **Average** | 38.0 | 45.0 | 70.0 |
