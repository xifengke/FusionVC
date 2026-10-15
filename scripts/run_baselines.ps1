# Run from repository root; every stage fails on error.
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
$env:POLARS_MAX_THREADS = '4'
$env:NUMBA_NUM_THREADS = '4'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python -m fusionvc.cli prepare
if ($LASTEXITCODE -ne 0) { throw 'Preprocessing failed' }
foreach ($experiment in @('01_mlp', '02_transformer')) {
    foreach ($seed in @(42, 43, 44)) {
        & $python -m fusionvc.cli train --config "configs/experiments/$experiment.yaml" --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $experiment/$seed" }
    }
}
foreach ($experiment in @('01_mlp', '02_transformer')) {
    foreach ($seed in @(42, 43, 44)) {
        & $python -m fusionvc.cli evaluate --run "outputs/$experiment/seed_$seed"
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $experiment/$seed" }
    }
}
& $python -m fusionvc.cli controls
if ($LASTEXITCODE -ne 0) { throw 'Auxiliary reference evaluation failed' }
& $python -m fusionvc.cli report
if ($LASTEXITCODE -ne 0) { throw 'Report failed' }
