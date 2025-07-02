# build_xformers.ps1 - compile xformers with CUDA support
# Prerequisites:
# - Windows with CUDA-capable GPU and CUDA Toolkit 12.8
# - Visual Studio Build Tools with "Desktop development with C++"
# - Python 3.10+ with PyTorch installed in an active virtual environment

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $IsWindows) {
    throw "This script must be run on Windows"
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python not found in PATH"
}

# check python version
$pyVer = python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
if ([Version]$pyVer -lt [Version]'3.10') {
    throw "Python 3.10 or newer is required"
}

# verify PyTorch and CUDA
try {
    $torchCuda = python - <<'PY'
import torch
print('1' if torch.cuda.is_available() else '0')
PY
} catch {
    throw "PyTorch is not installed in the current environment"
}
if ($torchCuda -ne '1') {
    throw "CUDA device not available via PyTorch"
}

# compiler tools
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "cl.exe not found. Install Visual Studio Build Tools (Desktop C++)"
}
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc not found. Install CUDA Toolkit 12.8"
}

# Ensure script runs from repository root
Set-Location $PSScriptRoot

Write-Host "Detecting CUDA architecture via PyTorch..."
$arch = python - <<'PY'
import torch
cap = torch.cuda.get_device_capability()
print(f"{cap[0]}.{cap[1]}")
PY
if (-not $arch) {
    throw "Could not determine CUDA architecture"
}
$env:TORCH_CUDA_ARCH_LIST = $arch
Write-Host "TORCH_CUDA_ARCH_LIST set to $arch"

Write-Host "Installing build dependencies..."
python -m pip install --upgrade ninja

if (-not (Test-Path 'xformers')) {
    Write-Host "Cloning xformers 0.0.31..."
    git clone --branch v0.0.31 --recursive https://github.com/facebookresearch/xformers.git
}

Push-Location xformers
Write-Host "Building xformers (this may take a while)..."
python -m pip install -e .
Pop-Location

Write-Host "Verifying installation..."
python -m xformers.info

