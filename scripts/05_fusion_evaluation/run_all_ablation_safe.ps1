$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location -LiteralPath $projectRoot

$py = "D:\anaconda3\envs\yolo-auto-label\python.exe"
$evalScript = Join-Path $scriptDir "evaluate_ablation.py"
$logDir = Join-Path $scriptDir "run_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

$models = @(
  "yolov8_balanced",
  "yolov9_640_balanced",
  "yolov10_balanced",
  "yolo11_balanced",
  "rtdetr_balanced"
)

$radii = @(500, 1000, 1500)
$cvModes = @("stratified", "community_group")

$jobs = @()

foreach ($model in $models) {
  foreach ($radius in $radii) {
    foreach ($cv in $cvModes) {
      $jobs += [PSCustomObject]@{
        Model = $model
        Radius = $radius
        CV = $cv
      }
    }
  }
}

foreach ($radius in @(500, 1500)) {
  foreach ($cv in $cvModes) {
    $jobs += [PSCustomObject]@{
      Model = "yolov9_1280_balanced"
      Radius = $radius
      CV = $cv
    }
  }
}

$masterLog = Join-Path $logDir ("batch_label_ablation_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

"[$(Get-Date)] BATCH START total_jobs=$($jobs.Count)" | Out-File -FilePath $masterLog -Append -Encoding utf8
"[$(Get-Date)] projectRoot=$projectRoot" | Out-File -FilePath $masterLog -Append -Encoding utf8
"[$(Get-Date)] evalScript=$evalScript" | Out-File -FilePath $masterLog -Append -Encoding utf8

foreach ($job in $jobs) {
  $time = Get-Date -Format "yyyyMMdd_HHmmss"
  $logName = "{0}_{1}m_{2}_{3}.log" -f $job.Model, $job.Radius, $job.CV, $time
  $logPath = Join-Path $logDir $logName

  "[$(Get-Date)] START model=$($job.Model), radius=$($job.Radius), cv=$($job.CV), log=$logPath" | Out-File -FilePath $masterLog -Append -Encoding utf8

  & $py -u $evalScript `
    --model-name $job.Model `
    --poi-radius $job.Radius `
    --gt-mode ground_truth `
    --fusion-policy guarded `
    --cv-mode $job.CV `
    *> $logPath

  $exitCode = $LASTEXITCODE

  if ($exitCode -eq 0) {
    "[$(Get-Date)] DONE  model=$($job.Model), radius=$($job.Radius), cv=$($job.CV), exit=$exitCode" | Out-File -FilePath $masterLog -Append -Encoding utf8
  } else {
    "[$(Get-Date)] FAIL  model=$($job.Model), radius=$($job.Radius), cv=$($job.CV), exit=$exitCode, check=$logPath" | Out-File -FilePath $masterLog -Append -Encoding utf8
  }
}

"[$(Get-Date)] ALL FINISHED" | Out-File -FilePath $masterLog -Append -Encoding utf8
Write-Host "All experiments finished. Master log: $masterLog"
