# One-time setup for OMR dataset tooling (Mashcima + deps).
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Vendor = Join-Path $Root "vendor\Mashcima"

if (-not (Test-Path $Venv)) {
    py -3 -m venv $Venv
}

& (Join-Path $Venv "Scripts\pip.exe") install --upgrade pip
& (Join-Path $Venv "Scripts\pip.exe") install -r (Join-Path $Root "requirements-omr.txt")

if (-not (Test-Path (Join-Path $Vendor ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Vendor) | Out-Null
    git clone --depth 1 https://github.com/OMR-Research/Mashcima.git $Vendor
}

$SetupPy = Join-Path $Vendor "setup.py"
$content = Get-Content $SetupPy -Raw
if ($content -match "'requests>=2\.15\.1'\s*\r?\n\s*# missing") {
    $content = $content -replace "('requests>=2\.15\.1')\s*\r?\n\s*# missing", "`$1,`r`n`r`n        # missing"
    Set-Content -Path $SetupPy -Value $content -NoNewline
}

& (Join-Path $Venv "Scripts\pip.exe") install -e $Vendor

Write-Host "OMR tooling ready. Next:"
Write-Host "  .\scripts\download_datasets.ps1"
Write-Host "  .\.venv\Scripts\python.exe .\scripts\generate_mashcima_batch.py"
