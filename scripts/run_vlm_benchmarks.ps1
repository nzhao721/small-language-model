# Detached VLM benchmark runner (survives Cursor shell teardown)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..

$models = @(
    "florence-2-base",
    "donut-base",
    "florence-2-large",
    "qwen2-vl-2b",
    "qwen2-vl-7b"
)

foreach ($model in $models) {
    $log = "examples\omr_benchmark\vlm_${model}_run.log"
    $stamp = Get-Date -Format o
    Add-Content -Path $log -Value "`n===== START $model $stamp ====="
    & .\.venv\Scripts\python.exe -u scripts\omr_benchmark_run_vlm.py --model $model 2>&1 |
        Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    Add-Content -Path $log -Value "===== END $model exit=$code $(Get-Date -Format o) ====="
    if ($code -ne 0) {
        Write-Host "Model $model failed with exit $code; continuing to next model."
    }
}

& .\.venv\Scripts\python.exe scripts\omr_benchmark_score.py --vlm-shortlist 2>&1 |
    Tee-Object -FilePath examples\omr_benchmark\vlm_score_run.log -Append
