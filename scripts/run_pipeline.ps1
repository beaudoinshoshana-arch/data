$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "[1/9] Build Stage-1 datasets"
python scripts/build_stage1_datasets.py

Write-Host "[2/9] Validate Stage-1 outputs"
python scripts/validate_stage1_outputs.py

Write-Host "[3/9] Train Stage-2 surrogate model"
python scripts/train_stage2_model.py

Write-Host "[4/9] Build external fusion dataset"
python scripts/build_external_fusion_dataset.py

Write-Host "[5/9] Train Safe-MARL policy"
python scripts/train_safe_marl.py --epochs 120

Write-Host "[6/9] Evaluate decision benefit and robustness"
python scripts/evaluate_decision_benefits.py

Write-Host "[7/9] Install frontend dependencies"
Set-Location (Join-Path $Root "dashboard\frontend")
npm install

Write-Host "[8/9] Build frontend"
npm run build

Write-Host "[9/9] Generate competition report"
Set-Location $Root
node scripts/generate_competition_report.cjs

Write-Host "Pipeline complete."
