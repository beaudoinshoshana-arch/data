$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "[1/8] Build Stage-1 datasets"
python scripts/build_stage1_datasets.py

Write-Host "[2/8] Validate Stage-1 outputs"
python scripts/validate_stage1_outputs.py

Write-Host "[3/8] Train Stage-2 surrogate model"
python scripts/train_stage2_model.py

Write-Host "[4/8] Build external fusion dataset"
python scripts/build_external_fusion_dataset.py

Write-Host "[5/8] Train Safe-MARL policy"
python scripts/train_safe_marl.py --epochs 120

Write-Host "[6/8] Install frontend dependencies"
Set-Location (Join-Path $Root "dashboard\frontend")
npm install

Write-Host "[7/8] Build frontend"
npm run build

Write-Host "[8/8] Generate competition report"
Set-Location $Root
node scripts/generate_competition_report.cjs

Write-Host "Pipeline complete."
