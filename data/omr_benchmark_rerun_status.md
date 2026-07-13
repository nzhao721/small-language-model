# OMR Benchmark Rerun (Repaired Golden Set)

Started: 2026-07-11

## Handwritten-only rerun (2026-07-12)
- **10 subagents** in parallel: 5 models × 2 batches (handwritten_original + handwritten_corrupt, 10 images each)
- **Job manifest**: `data/omr_handwritten_rerun_jobs.json`
- **Outputs**: `predictions_10/*/batch_00.jsonl` and `predictions_10_corrupt/*/batch_00.jsonl` (overwrite)
- **After completion**: merge both suites, then `scripts/omr_benchmark_score.py --all-benchmarks`

## Configuration
- **50 subagents**: 5 frontier models × 10 batches (10 images each, serial within batch)
- **Models**: claude-opus-4-8-thinking-high (via Opus 4.8 Thinking Max subagent), gemini-3.1-pro, composer-2.5, grok-4.5-xhigh, gpt-5.6-sol-medium
- **Suites**: predictions_10 (30), predictions_10_corrupt (30), predictions_manual (40) = 100 images
- **Job manifest**: `examples/omr_benchmark/rerun_jobs.jsonl`
- **Per-job prompts**: `data/omr_benchmark_job_prompts/job_XX.txt`

## Progress monitoring
- Batch completion: poll `data/omr_benchmark_rerun_progress.log` (written by merge/score watcher)
- Or count batch files:
  ```powershell
  Get-ChildItem -Recurse examples\omr_benchmark\predictions_* -Filter batch_*.jsonl |
    Where-Object { $_.Directory.Name -in @('claude-opus-4-8-thinking-high','gemini-3.1-pro','composer-2.5','grok-4.5-xhigh','gpt-5.6-sol-medium') } |
    ForEach-Object { (Get-Content $_.FullName | Measure-Object -Line).Lines } | Measure-Object -Sum
  ```
  Expect 50 files × 10 lines = 500 total lines.

## Final scoring
When all 50 batches complete, watcher runs:
1. `scripts/omr_benchmark_merge_predictions.py` (×3 suites)
2. `scripts/omr_benchmark_score.py --all-benchmarks` → `examples/omr_benchmark/scores.md`
