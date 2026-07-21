$ErrorActionPreference = "Stop"

$url = "https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp311-cp311-win_amd64.whl"
$totalBytes = [int64]2510750521
$partCount = 4
$checkpoint = "D:\人工智能数据\complete\visual_label_label\checkpoint"
$partDir = Join-Path $checkpoint "torch_cuda_parts"
$output = Join-Path $checkpoint "torch-2.5.1+cu124-cp311-cp311-win_amd64.whl"
$worker = "D:\人工智能数据\complete\03_Prediction_and_Evaluation\download_cuda_part.ps1"
$pwsh = "C:\Users\RHL\AppData\Local\Microsoft\WindowsApps\pwsh.exe"

New-Item -ItemType Directory -Path $partDir -Force | Out-Null
$chunkSize = [int64][math]::Ceiling($totalBytes / $partCount)
$processes = @()
$expectedSizes = @{}

for ($index = 0; $index -lt $partCount; $index++) {
    $start = [int64]$index * $chunkSize
    $end = [math]::Min($totalBytes - 1, $start + $chunkSize - 1)
    $expected = $end - $start + 1
    $partPath = Join-Path $partDir ("part_{0:D2}.bin" -f $index)
    $expectedSizes[$partPath] = $expected

    if ((Test-Path -LiteralPath $partPath) -and
        ((Get-Item -LiteralPath $partPath).Length -eq $expected)) {
        continue
    }

    if ((Test-Path -LiteralPath $partPath) -and
        ((Get-Item -LiteralPath $partPath).Length -gt $expected)) {
        Remove-Item -LiteralPath $partPath -Force
    }
    $stdout = Join-Path $partDir ("part_{0:D2}.stdout.log" -f $index)
    $stderr = Join-Path $partDir ("part_{0:D2}.stderr.log" -f $index)
    $processes += Start-Process `
        -FilePath $pwsh `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", $worker,
            "-Url", $url,
            "-RangeStart", "$start",
            "-RangeEnd", "$end",
            "-PartPath", $partPath
        ) `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
}

if ($processes.Count -gt 0) {
    $processes | Wait-Process
    $failed = @($processes | Where-Object { $_.ExitCode -ne 0 })
    if ($failed.Count -gt 0) {
        throw "One or more CUDA wheel download parts failed."
    }
}

foreach ($entry in $expectedSizes.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Key)) {
        throw "Missing CUDA wheel part: $($entry.Key)"
    }
    $actual = (Get-Item -LiteralPath $entry.Key).Length
    if ($actual -ne $entry.Value) {
        throw "CUDA wheel part has wrong size: $($entry.Key), expected=$($entry.Value), actual=$actual"
    }
}

$temporaryOutput = "$output.tmp"
Remove-Item -LiteralPath $temporaryOutput -Force -ErrorAction SilentlyContinue
$destination = [System.IO.File]::Open(
    $temporaryOutput,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
try {
    for ($index = 0; $index -lt $partCount; $index++) {
        $partPath = Join-Path $partDir ("part_{0:D2}.bin" -f $index)
        $source = [System.IO.File]::OpenRead($partPath)
        try {
            $source.CopyTo($destination)
        }
        finally {
            $source.Dispose()
        }
    }
}
finally {
    $destination.Dispose()
}

if ((Get-Item -LiteralPath $temporaryOutput).Length -ne $totalBytes) {
    throw "Combined CUDA wheel has incorrect size."
}
Move-Item -LiteralPath $temporaryOutput -Destination $output -Force
Write-Output $output

