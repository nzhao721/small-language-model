# Run all feasible VLM shortlist models on the OMR golden set (CPU/GPU auto-detect).
# Usage: .\scripts\run_vlm_benchmark.ps1
# Re-run safe: skips batches that already have full predictions.

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "scripts\omr_benchmark_run_vlm.py"
$Log = Join-Path $Root "examples\omr_benchmark\vlm_benchmark_run.log"

if (-not (Test-Path $Python)) {
    Write-Error "Missing venv python at $Python"
    exit 1
}

$models = @("florence-2-base", "donut-base", "florence-2-large", "qwen2-vl-2b")
foreach ($model in $models) {
    Write-Host "`n========== $model ==========" -ForegroundColor Cyan
    & $Python $Script --model $model 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$model exited with code $LASTEXITCODE; continuing..."
    }
}

Write-Host "`nScoring..." -ForegroundColor Cyan
& $Python (Join-Path $Root "scripts\omr_benchmark_score.py") --vlm-shortlist
Write-Host "Done. See examples\omr_benchmark\scores_vlm_shortlist.md"
