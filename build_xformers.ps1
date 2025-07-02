# build_xformers.ps1 - compile xformers with CUDA support
# Prerequisites:
# - Windows with CUDA-capable GPU and CUDA Toolkit 12.8
# - Visual Studio Build Tools with "Desktop development with C++"
# - Python 3.10+ with PyTorch installed in an active virtual environment

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
    throw "This script must be run on Windows"
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python not found in PATH"
}

# check python version
$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([Version]$pyVer -lt [Version]'3.10') {
    throw "Python 3.10 or newer is required"
}

# verify PyTorch and CUDA
try {
    $torchCuda = python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
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

# Ensure CUDA_HOME is defined. Try CUDA_PATH or default install location.
if (-not $env:CUDA_HOME) {
    if ($env:CUDA_PATH) {
        $env:CUDA_HOME = $env:CUDA_PATH
    } elseif (Test-Path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8') {
        $env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
    } else {
        throw 'CUDA_HOME environment variable is not set. Please set it to your CUDA install root.'
    }
}

# Ensure script runs from repository root
Set-Location $PSScriptRoot

Write-Host "Detecting CUDA architecture via PyTorch..."
$arch = python -c "import torch; cap = torch.cuda.get_device_capability(); print(f'{cap[0]}.{cap[1]}')"
if (-not $arch) {
    throw "Could not determine CUDA architecture"
}
$env:TORCH_CUDA_ARCH_LIST = $arch
Write-Host "TORCH_CUDA_ARCH_LIST set to $arch"

Write-Host "Installing build dependencies..."
python -m pip install --upgrade ninja

Write-Host "Building xformers from source (this may take a while)..."
python -m pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@v0.0.31#egg=xformers

Write-Host "Verifying installation..."
python -m xformers.info

