---
name: wwtp-safe-marl
description: Use for this wastewater treatment plant competition project: data fusion, Safe-MARL aeration/PAC dosing decisions, safety-shield validation, dashboard/API delivery, and competition report workflow.
---

# WWTP Safe-MARL Workflow

## Core Order

1. Inspect current artifacts before making claims.
2. Build or validate Stage-1 data, Stage-2 surrogate model, external fusion data, then Safe-MARL recommendations.
3. Expose results through FastAPI and React dashboard.
4. Verify with API smoke tests, `pytest`, and frontend build.
5. Record result/risk/improvement/next step in `docs/reflection_log.md`.

## Non-Negotiables

- Keep `observed` rows as the only supervised ground truth.
- Route all RL actions through `SafetyShield`.
- Preserve explainability: reward components, feasibility, fallback reason, and Chinese action explanation.
- Do not commit raw data, PDFs, Office files, model weights, virtual environments, or very large CSV outputs.

## Key Commands

```powershell
python scripts/validate_stage1_outputs.py
python scripts/build_external_fusion_dataset.py
python scripts/train_safe_marl.py --epochs 40
pytest
uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8000
cd dashboard/frontend
npm run build
npm run dev
```
