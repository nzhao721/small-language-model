# Restart dataset jobs from manifest/resume checkpoints (one instance each).
$ErrorActionPreference = "Stop"
$Root = "C:\alpha_ai\small-language-model"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location $Root

$ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
Add-Content -Path "data\batch_restart.log" -Value "=== restart_dataset_jobs $ts ==="

function Stop-MatchingPython {
    param([string]$Pattern)
    Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match $Pattern } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Start-JobScript {
    param([string]$Name, [string[]]$ScriptArgs)
    $out = Join-Path $Root "data\job_$($Name -replace '[^a-zA-Z0-9]','_').log"
    $err = Join-Path $Root "data\job_$($Name -replace '[^a-zA-Z0-9]','_')_err.log"
    $allArgs = @("-u") + $ScriptArgs
    $proc = Start-Process -FilePath $Py -ArgumentList $allArgs `
        -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $out -RedirectStandardError $err
    Add-Content -Path "data\batch_restart.log" -Value "Started $Name PID=$($proc.Id) args=$($ScriptArgs -join ' ')"
    return $proc.Id
}

# Dedupe long-running helpers
Stop-MatchingPython "monitor_job_progress\.py"
Stop-MatchingPython "wait_and_launch_omr_benchmark\.py"

# Compression — resumes via data/compress_dataset_resume.jsonl (skip if already complete)
# Camera distort — resumes via camera_distort_manifest.jsonl (skip if already complete)

# Format regeneration only (not full corrupt batch); resumes via manifest regenerated rows
& (Join-Path $Root "scripts\restart_regen_formats.ps1") | Out-Null

# Progress monitor every 5 minutes (single instance)
Start-JobScript "progress_monitor" @(
    "scripts\monitor_job_progress.py",
    "--loop-minutes", "5"
) | Out-Null

# OMR benchmark already prepared when corrupt batch last hit 100%; skip waiter unless flag removed
if (-not (Test-Path "data\omr_benchmark_ready.json")) {
    Start-JobScript "omr_wait" @(
        "scripts\wait_and_launch_omr_benchmark.py",
        "--poll-minutes", "5"
    ) | Out-Null
} else {
    Add-Content -Path "data\batch_restart.log" -Value "Skipped omr_wait (omr_benchmark_ready.json present)"
}

& $Py scripts\monitor_job_progress.py --loop-minutes 0

Write-Host "Restarted format regen, progress monitor (5m). OMR waiter skipped if benchmark ready."
