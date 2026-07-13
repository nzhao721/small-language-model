# Download and extract PrIMuS and Camera-PrIMuS datasets.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ArchiveDir = Join-Path $Root "data\archives"
$PrimusDir = Join-Path $Root "data\primus"
$CameraDir = Join-Path $Root "data\camera-primus"

$PrimusUrl = "https://grfia.dlsi.ua.es/primus/packages/primusCalvoRizoAppliedSciences2018.tgz"
$CameraUrl = "https://grfia.dlsi.ua.es/primus/packages/CameraPrIMuS.tgz"

$PrimusArchive = Join-Path $ArchiveDir "primusCalvoRizoAppliedSciences2018.tgz"
$CameraArchive = Join-Path $ArchiveDir "CameraPrIMuS.tgz"

New-Item -ItemType Directory -Force -Path $ArchiveDir, $PrimusDir, $CameraDir | Out-Null

function Ensure-Download {
    param(
        [string]$Url,
        [string]$OutFile,
        [long]$MinBytes
    )

    if ((Test-Path $OutFile) -and ((Get-Item $OutFile).Length -ge $MinBytes)) {
        Write-Host "Already downloaded: $OutFile"
        return
    }

    if (Test-Path $OutFile) {
        Remove-Item $OutFile -Force
    }

    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $OutFile
    Write-Host "Saved $OutFile ($('{0:N2}' -f ((Get-Item $OutFile).Length / 1MB)) MB)"
}

function Ensure-Extract {
    param(
        [string]$Archive,
        [string]$TargetDir,
        [string]$MarkerRelativePath
    )

    $Marker = Join-Path $TargetDir $MarkerRelativePath
    if (Test-Path $Marker) {
        Write-Host "Already extracted: $TargetDir"
        return
    }

    Write-Host "Extracting $Archive -> $TargetDir"
    tar -xzf $Archive -C $TargetDir
    Write-Host "Extracted $TargetDir"
}

Ensure-Download -Url $PrimusUrl -OutFile $PrimusArchive -MinBytes 200000000
Ensure-Download -Url $CameraUrl -OutFile $CameraArchive -MinBytes 2000000000

Ensure-Extract -Archive $PrimusArchive -TargetDir $PrimusDir -MarkerRelativePath "package_aa"
Ensure-Extract -Archive $CameraArchive -TargetDir $CameraDir -MarkerRelativePath "package_aa"

Write-Host "Dataset download and extraction complete."
