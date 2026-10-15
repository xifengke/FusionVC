# Run from the repository root. No system Python packages or CUDA Toolkit are replaced.
param([string]$Python = 'E:\Python\python.exe')
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.cache\pip'
if (!(Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv '.venv'
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& '.\.venv\Scripts\python.exe' -m pip install 'torch==2.8.0' --index-url 'https://download.pytorch.org/whl/cu128'
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
$dependencyArgs = @('-m', 'pip', 'install', '-e', '.[dev,geneformer,final_test]', 'pdex==0.1.27', '.\vendor\cell-eval-vcc2025')
if (Test-Path -LiteralPath 'requirements\windows-constraints.txt') {
    $dependencyArgs += @('-c', 'requirements\windows-constraints.txt')
}
& '.\.venv\Scripts\python.exe' @dependencyArgs
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.\.venv\Scripts\python.exe' 'scripts\lock_environment.py'
& '.\.venv\Scripts\python.exe' -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed' }
