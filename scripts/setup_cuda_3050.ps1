# Setup script for an NVIDIA RTX 3050 development environment on Windows.
# Run from the repository root with PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_cuda_3050.ps1

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.11+ and try again."
}

if (-not (Test-Path .\.venv)) {
    Write-Host "Creating virtual environment..."
    py -3.12 -m venv .venv
}

Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing the RTX 3050 CUDA stack..."
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt

Write-Host "" 
Write-Host "Validation: checking CUDA capability..."
python -m prototype.doctor

Write-Host "" 
Write-Host "You are ready to run the real backend on the NVIDIA RTX 3050."
Write-Host "Use: python -m prototype.demo --device cuda"
