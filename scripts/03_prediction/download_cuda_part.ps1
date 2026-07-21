param(
    [Parameter(Mandatory = $true)]
    [string]$Url,
    [Parameter(Mandatory = $true)]
    [int64]$RangeStart,
    [Parameter(Mandatory = $true)]
    [int64]$RangeEnd,
    [Parameter(Mandatory = $true)]
    [string]$PartPath
)

$ErrorActionPreference = "Stop"
$expectedLength = $RangeEnd - $RangeStart + 1
$temporary = "$PartPath.download"

if ((Test-Path -LiteralPath $PartPath) -and
    ((Get-Item -LiteralPath $PartPath).Length -gt $expectedLength)) {
    Remove-Item -LiteralPath $PartPath -Force
}

while ($true) {
    $existing = if (Test-Path -LiteralPath $PartPath) {
        (Get-Item -LiteralPath $PartPath).Length
    }
    else {
        [int64]0
    }
    if ($existing -eq $expectedLength) {
        break
    }

    $nextStart = $RangeStart + $existing
    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    & curl.exe `
        -L `
        --retry 2 `
        --retry-all-errors `
        --retry-delay 2 `
        --range "$nextStart-$RangeEnd" `
        -o $temporary `
        $Url
    $curlExit = $LASTEXITCODE

    if (Test-Path -LiteralPath $temporary) {
        $downloaded = (Get-Item -LiteralPath $temporary).Length
        if ($downloaded -gt 0) {
            $destination = [System.IO.File]::Open(
                $PartPath,
                [System.IO.FileMode]::Append,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::Read
            )
            $source = [System.IO.File]::OpenRead($temporary)
            try {
                $source.CopyTo($destination)
            }
            finally {
                $source.Dispose()
                $destination.Dispose()
            }
        }
        Remove-Item -LiteralPath $temporary -Force
    }

    $current = if (Test-Path -LiteralPath $PartPath) {
        (Get-Item -LiteralPath $PartPath).Length
    }
    else {
        [int64]0
    }
    if ($current -gt $expectedLength) {
        throw "Part exceeded expected length: $PartPath"
    }
    if ($curlExit -ne 0 -and $current -eq $existing) {
        Start-Sleep -Seconds 3
    }
}

Write-Output $PartPath

