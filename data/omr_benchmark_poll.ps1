Set-Location "C:\alpha_ai\small-language-model"
$jobs = Get-Content "examples/omr_benchmark/rerun_jobs.jsonl" | ForEach-Object { $_ | ConvertFrom-Json }
$logPath = "data/omr_benchmark_rerun_progress.log"
$deadline = (Get-Date).AddHours(3)
function Get-Status {
  $complete = 0
  $issues = @()
  foreach ($i in 0..($jobs.Count-1)) {
    $path = $jobs[$i].output
    if (Test-Path $path) {
      $n = @(Get-Content $path | Where-Object { $_.Trim() -ne '' }).Count
      if ($n -eq 10) { $complete++ } else { $issues += [pscustomobject]@{idx=$i; lines=$n; output=$path} }
    } else { $issues += [pscustomobject]@{idx=$i; lines=-1; output=$path} }
  }
  return @{complete=$complete; total=$jobs.Count; issues=$issues}
}
while ((Get-Date) -lt $deadline) {
  $s = Get-Status
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $line = "[$ts] Complete: $($s.complete)/$($s.total)"
  Write-Output $line
  Add-Content -Path $logPath -Value $line
  if ($s.complete -eq $s.total) {
    Add-Content -Path $logPath -Value "[$ts] All batches ready."
    exit 0
  }
  Start-Sleep -Seconds 120
}
$s = Get-Status
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$ts] TIMEOUT after 3h: $($s.complete)/$($s.total) complete"
$s.issues | ConvertTo-Json -Compress | Add-Content -Path $logPath
exit 1
