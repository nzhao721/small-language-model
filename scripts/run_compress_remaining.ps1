# Chain dataset compression after mashcima finishes.
$ErrorActionPreference = "Stop"
$Root = "C:\alpha_ai\small-language-model"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "scripts\compress_dataset_images.py"
$Log = Join-Path $Root "data\compress_remaining.log"

Set-Location $Root

"=== waiting for mashcima compression (PID from compress_mashcima) ===" | Tee-Object -FilePath $Log -Append
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'compress_dataset_images\.py.*--dirs mashcima' }) {
    Start-Sleep -Seconds 30
}

$dirs = @(
    "corrupt/clean",
    "corrupt/camera",
    "corrupt/handwritten",
    "camera-primus",
    "primus"
)
foreach ($d in $dirs) {
    "=== compressing $d ===" | Tee-Object -FilePath $Log -Append
    & $Py -u $Script --skip-stats --update-manifests --dirs $d 2>&1 |
        Tee-Object -FilePath $Log -Append
}

"=== all compression passes done ===" | Tee-Object -FilePath $Log -Append
