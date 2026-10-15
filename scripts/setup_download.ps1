param([string]$Python = 'E:\Python\python.exe')
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.cache\pip'
if (!(Test-Path -LiteralPath '.venv-download\Scripts\python.exe')) {
    & $Python -m venv '.venv-download'
    if ($LASTEXITCODE -ne 0) { throw 'Download environment creation failed' }
}
& '.\.venv-download\Scripts\python.exe' -m pip install -r 'docs\download-tools.lock.txt'
if ($LASTEXITCODE -ne 0) { throw 'Download dependency installation failed' }
