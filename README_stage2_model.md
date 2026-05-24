# Stage-2 Wastewater Model

This stage adds a deployable modeling layer on top of the stage-1 data assets in `outputs/stage1_data/`.

## Goal

Build a practical surrogate model for next-hour effluent quality and then use that model to recommend bounded aeration and chemical-dose adjustments.

## Design

1. Supervised sample source
   Use only `scenario_tag == observed` rows for supervised learning.
   The simulated `load_up` and `rain_dilution` rows are kept for decision recommendation, not for label fitting, because they reuse observed labels instead of having independent ground truth.

2. Prediction target
   Predict the next-hour effluent indicators:
   `COD`, `NH3N`, `TP`, and `TN`.

3. Feature construction
   Combine the current process state, current control settings, current effluent quality, time-of-day signals, short-history lag and rolling features, and proxy dynamic features built from first/second differences and rolling volatility.

4. Model selection
   Train a small candidate set of nonlinear tree ensembles and select the best model using the validation split that already exists in stage-1 outputs.

5. Control recommendation
   For each scenario row in the test split, perform a scenario-aware local grid search around the baseline aeration and PAC dose.
   The optimizer now reads a JSON constraint config for effluent limits, actuator bounds, and penalty weights, then minimizes predicted pollution plus operating-cost, smoothness, and violation penalties.

## Run

```powershell
python scripts/train_stage2_model.py
```

Optional debug run:

```powershell
python scripts/train_stage2_model.py --max-recommendations 60
```

Custom constraint config:

```powershell
python scripts/train_stage2_model.py --constraint-config configs/stage2_constraints.default.json
```

## Outputs

- `outputs/stage2_model/summary.json`
- `outputs/stage2_model/leaderboard.json`
- `outputs/stage2_model/feature_columns.json`
- `outputs/stage2_model/test_predictions.csv`
- `outputs/stage2_model/scenario_recommendations_test.csv`
- `outputs/stage2_model/dynamic_feature_frame.csv`
- `outputs/stage2_model/constraint_config_used.json`
- `outputs/stage2_model/model_card.md`
- `outputs/stage2_model/best_model.pkl`

## Practical Notes

- The script reuses the chronological `train / val / test` split from stage-1 and does not reshuffle samples.
- Recommendation search is intentionally local to avoid unrealistic jumps in actuator settings.
- The default limits and actuator ranges live in `configs/stage2_constraints.default.json`.
- The proxy dynamic features are meant to approximate the paper's dynamic-feature idea under the current hourly data granularity.
