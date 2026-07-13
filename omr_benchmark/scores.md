# OMR Benchmark Scores

Token-level edit distance against ground-truth semantic tokens; each single-token insert, delete, or replace counts as one edit, with alignment chosen to minimize total edits. Relative major/minor key signatures (same accidentals) and common/cut-time aliases (4/4↔C, 2/2↔C/) count as equal. Accuracy % = max(0, (1 − edits ÷ ground-truth token count) × 100).


Samples scored: 100.

## Token accuracy %

| Dataset | opus | gemini-3.1-pro | composer-2.5 | grok-4.5-high | gpt-5.6-sol |
| --- | --- | --- | --- | --- | --- |
| handwritten_original | 24.14 | 11.73 | 24.42 | 27.63 | 30.2 |
| clean_original | 85.17 | 15.01 | 36.5 | 38.67 | 66.07 |
| camera_distorted_original | 42.33 | 6.6 | 24.26 | 22.86 | 29.53 |
| handwritten_corrupt | 26.01 | 19.38 | 32.89 | 23.56 | 21.89 |
| clean_corrupt | 43.99 | 11.71 | 37.15 | 39.53 | 72.41 |
| camera_distorted_corrupt | 31.13 | 0.0 | 33.36 | 31.33 | 34.86 |
| manual_handwritten | 19.1 | 14.62 | 28.87 | 33.55 | 24.32 |
| manual_digital | 95.89 | 27.41 | 33.98 | 65.25 | 90.72 |
| **Average** | 48.28 | 14.85 | 31.43 | 38.12 | 48.5 |

## Exact match %

Exact match % = share of samples with zero token edits after canonicalization.

| Dataset | opus | gemini-3.1-pro | composer-2.5 | grok-4.5-high | gpt-5.6-sol |
| --- | --- | --- | --- | --- | --- |
| handwritten_original | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| clean_original | 40.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| camera_distorted_original | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| handwritten_corrupt | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| clean_corrupt | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| camera_distorted_corrupt | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| manual_handwritten | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| manual_digital | 5.0 | 0.0 | 0.0 | 0.0 | 40.0 |
| **Average** | 5.0 | 0.0 | 0.0 | 0.0 | 8.0 |
