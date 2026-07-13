param(
    [int]$Workers = 0,
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
$Root = "C:\alpha_ai\small-language-model"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location $Root

if ($Workers -le 0) {
    $cores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
    $Workers = [Math]::Max(1, $cores)
}

$Out = Join-Path $Root "data\corrupt_30k"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$stdout = Join-Path $Out "run.log"
$stderr = Join-Path $Out "_err.log"
$pidFile = Join-Path $Out "batch.pid"

Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match "generate_corrupt_(30k_)?batch\.py" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
if ($env:PLAYWRIGHT_BROWSERS_PATH -eq "") {
    Remove-Item Env:PLAYWRIGHT_BROWSERS_PATH -ErrorAction SilentlyContinue
}

if ($Fresh) {
    Remove-Item (Join-Path $Out "manifest.jsonl") -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $Out "deferred_jobs.jsonl") -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $Out "jobs.jsonl") -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $Out "semantic") -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $Out "clean") -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $Out "camera") -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $Out "handwritten") -Recurse -Force -ErrorAction SilentlyContinue
}

$batchArgs = @(
    "-u", "scripts\generate_corrupt_batch.py",
    "--total", "30000",
    "--format-counts", "clean:6000,camera:9000,handwritten:15000",
    "--output-dir", "data\corrupt_30k",
    "--manifest", "data\corrupt_30k\manifest.jsonl",
    "--jobs", "data\corrupt_30k\jobs.jsonl",
    "--plan-seed", "20260711",
    "--exclude-golden",
    "--writers", "per-score",
    "--workers", "$Workers",
    "--job-timeout", "90",
    "--handwritten-timeout", "180",
    "--max-tasks-per-child", "25",
    "--progress-log", "data\corrupt_30k\progress.log"
)

if (-not (Test-Path (Join-Path $Out "jobs.jsonl"))) {
    $batchArgs += "--regenerate-jobs"
}
if ($Fresh) {
    $batchArgs += "--no-resume"
}

$ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
Add-Content -Path $stdout -Value "`n=== start_corrupt_30k $ts workers=$Workers ===" -Encoding utf8

$proc = Start-Process -FilePath $Py -ArgumentList $batchArgs `
    -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr

Set-Content -Path $pidFile -Value $proc.Id -Encoding ascii
Write-Output "python_pid=$($proc.Id)"
Write-Output "workers=$Workers"
Write-Output "progress=$Out\progress.log"
