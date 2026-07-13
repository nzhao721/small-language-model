param(
    [int]$Workers = 0
)

$ErrorActionPreference = "Stop"
$Root = "C:\alpha_ai\small-language-model"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location $Root

if ($Workers -le 0) {
    $cores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
    $Workers = [Math]::Min(12, [Math]::Max(1, $cores - 2))
}


$stdout = Join-Path $Root "data\corrupt\regenerate_formats.log"
$stderr = Join-Path $Root "data\corrupt\_err.log"
$pidFile = Join-Path $Root "data\corrupt\regen_formats.pid"

function Stop-MatchingPython {
    param([string]$Pattern)
    Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match $Pattern } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

Stop-MatchingPython "generate_corrupt_batch\.py.*--regenerate-formats"
Start-Sleep -Seconds 1

$ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
try {
    Add-Content -Path $stdout -Value "`n=== restart_regen_formats $ts workers=$Workers ===" -Encoding utf8 -ErrorAction Stop
} catch {
    # stdout may be locked if a stale process still holds the redirect; regen will append after start
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
if ($env:PLAYWRIGHT_BROWSERS_PATH -eq "") {
    Remove-Item Env:PLAYWRIGHT_BROWSERS_PATH -ErrorAction SilentlyContinue
}

$batchArgs = @(
    "-u", "scripts\generate_corrupt_batch.py",
    "--regenerate-formats", "clean,camera,handwritten",
    "--retry-regen-errors",
    "--workers", "$Workers"
)

$proc = Start-Process -FilePath $Py -ArgumentList $batchArgs `
    -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr

Set-Content -Path $pidFile -Value $proc.Id -Encoding ascii

$alive = $false
for ($i = 0; $i -lt 30; $i++) {
    if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) { $alive = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $alive) { throw "Regen python PID $($proc.Id) exited within 30s" }

Write-Output "python_pid=$($proc.Id)"
Write-Output "workers=$Workers"

