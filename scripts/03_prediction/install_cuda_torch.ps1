$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$python = "D:\anaconda3\envs\yolo-auto-label\python.exe"
$checkpoint = "D:\人工智能数据\complete\visual_label_label\checkpoint"
$torchWheel = Join-Path $checkpoint "torch-2.5.1+cu124-cp311-cp311-win_amd64.whl"
$visionWheel = Join-Path $checkpoint "torchvision-0.20.1+cu124-cp311-cp311-win_amd64.whl"
$expectedTorchBytes = [int64]2510750521
$expectedVisionBytes = [int64]6143069

foreach ($item in @(
    @{ Path = $torchWheel; Size = $expectedTorchBytes },
    @{ Path = $visionWheel; Size = $expectedVisionBytes }
)) {
    if (-not (Test-Path -LiteralPath $item.Path)) {
        throw "Missing CUDA wheel: $($item.Path)"
    }
    $actual = (Get-Item -LiteralPath $item.Path).Length
    if ($actual -ne $item.Size) {
        throw "CUDA wheel size mismatch: $($item.Path), expected=$($item.Size), actual=$actual"
    }
}

& $python -c @"
import zipfile
for path in [r'$torchWheel', r'$visionWheel']:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f'Corrupt wheel member: {path}: {bad}')
print('wheel_integrity_ok')
"@
if ($LASTEXITCODE -ne 0) {
    throw "CUDA wheel integrity check failed."
}

& $python -m pip install --no-deps --force-reinstall $torchWheel $visionWheel
if ($LASTEXITCODE -ne 0) {
    throw "CUDA PyTorch installation failed."
}

& $python -c @"
import torch, torchvision
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('cuda_runtime', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
print('cuda_devices', torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit('CUDA installation completed but no CUDA device is available')
print('gpu', torch.cuda.get_device_name(0))
"@
if ($LASTEXITCODE -ne 0) {
    throw "CUDA verification failed."
}

