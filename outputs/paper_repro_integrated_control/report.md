# Paper Reproduction Report

## Positioning

This run implements a paper-faithful reproduction pipeline with an ASM2D-inspired digital twin fallback.
It follows the paper's method chain, but it cannot be an exact numerical replica because the original code, 2017 plant data, and online interfaces are not public.

## Calibration

- Mechanistic proxy calibration loss: 0.5615
- Mechanistic proxy calibration time: 97.67 s
- Calibrated proxy parameters: {"kla_base_factor": 0.7778469711352604, "kla_from_aeration_weight": 0.43821804006211246, "kla_from_do_gain": 7.893011258816533, "nrr_bias": 0.9998383622232823, "nrr_from_no3_gain": 0.21154716753775143, "nrr_from_flow_gain": 0.0793949825239744, "mechanistic_continuous_gain_per_hour": 0.2000247448497095}
- Mechanistic calibration loss: 0.3920
- Mechanistic calibration time: 121.95 s
- Calibrated parameters: {"kH": 1.600000000024664, "muH": 7.799999999958339, "uPAO": 5.199999999999843}
- Gamma calibration: 0.9000 in 3.85 s (level_plus_delta_with_dynamic_focus)

## Batch Feature Reconstruction

- Mean test reconstruction R2: 0.9905
- Mean test reconstruction RMSE: 0.3463

## Real-data Prediction

- Mechanistic weighted RMSE: 0.4540
- Data-driven weighted RMSE: 0.3755

## Optimization

- Steady-state operating-cost saving: 1.80%
- Dynamic operating-cost saving: 4.45%
- Dynamic replay mean execution time: 0.56 s

## Surface Models

- `anoxic_no3:A`: degree 2, R2=0.9977, RMSE=0.0001
- `anoxic_no3:mu`: degree 1, R2=0.2083, RMSE=5.2155
- `anoxic_no3:delta`: degree 2, R2=0.4505, RMSE=0.6155
- `aerobic_bod:A`: degree 1, R2=0.9942, RMSE=0.0008
- `aerobic_bod:mu`: degree 1, R2=0.0601, RMSE=4.8066
- `aerobic_bod:delta`: degree 1, R2=0.0575, RMSE=0.4759
- `aerobic_nh3:A`: degree 1, R2=0.9997, RMSE=0.0001
- `aerobic_nh3:mu`: degree 1, R2=0.9990, RMSE=0.0348
- `aerobic_nh3:delta`: degree 1, R2=0.7898, RMSE=0.0828

## Output Artifacts

- `batch_profiles.csv`
- `batch_feature_parameters.csv`
- `feature_parameter_functions.json`
- `real_prediction_comparison.csv`
- `steady_state_pareto.csv`
- `online_dynamic_replay.csv`
- `summary.json`
