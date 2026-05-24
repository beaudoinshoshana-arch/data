# Integrated Real-Time Control Paper Reproduction

This workspace now includes a runnable reproduction pipeline for:

`Integrated real-time intelligent control for wastewater treatment plants: Data-driven modeling for enhanced prediction and regulatory strategies`

## Positioning

This is a paper-faithful reimplementation with an `ASM2D-inspired digital twin fallback`.

It reproduces the paper's method chain:

1. mechanistic parameter calibration
2. batch simulation sweeps
3. higher-order / Gaussian-style dynamic feature extraction
4. feature-parameter surface fitting
5. data-driven prediction with `gamma` calibration
6. steady-state optimization
7. online dynamic replay with mock `Modbus` control logs

It does **not** claim exact numerical replication of the paper, because the original ASM2D code, 2017 plant dataset, and online deployment interfaces are not public in this workspace.

## Run

```powershell
python scripts/reproduce_integrated_realtime_control.py
```

## Configs

- `configs/paper_repro_asm2d.default.json`
  - plant design parameters, kinetic defaults/bounds, preprocessing, calibration budgets
- `configs/paper_repro_batch.default.json`
  - anoxic/aerobic sweep ranges and polynomial surface-model structure
- `configs/paper_repro_online.default.json`
  - steady-state and dynamic-control replay budgets, thresholds, reporting

## Main Outputs

All artifacts are written to:

`outputs/paper_repro_integrated_control/`

Key files:

- `summary.json`
- `report.md`
- `batch_profiles.csv`
- `batch_feature_parameters.csv`
- `feature_parameter_functions.json`
- `real_prediction_comparison.csv`
- `steady_state_pareto.csv`
- `online_dynamic_replay.csv`
- `online_modbus_mock_log.csv`
- `surface_models.pkl`

Figures:

- `figures/steady_state_pareto.png`
- `figures/dynamic_replay.png`
- `figures/batch_fit_examples.png`

## Notes

- Real-data prediction uses the existing stage-1 hourly dataset rather than the paper's unavailable 2017 daily dataset.
- Effluent BOD is approximated with an effluent COD-based proxy because the workspace does not contain a direct effluent BOD series.
- Dynamic replay is a closed-loop simulation with a mock `Modbus` interface, not a live plant deployment.
