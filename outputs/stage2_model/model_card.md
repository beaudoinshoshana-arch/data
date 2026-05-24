# Stage-2 Wastewater Model Card

## Purpose

This surrogate model predicts next-hour effluent quality and recommends bounded aeration and PAC-dose actions.

## Targets

- `label_next_effluent_cod_mgL`
- `label_next_effluent_nh3n_mgL`
- `label_next_effluent_tp_mgL`
- `label_next_effluent_tn_mgL`

## Feature Strategy

- Total numeric features: 349
- Proxy dynamic features: 152
- Current design keeps the stage-1 observed rows for supervision and uses scenario rows only for recommendation search.

## Constraint Config

```json
{
  "target_limits": {
    "label_next_effluent_cod_mgL": {
      "name": "COD",
      "upper": 50.0,
      "penalty_weight": 1.0
    },
    "label_next_effluent_nh3n_mgL": {
      "name": "NH3-N",
      "upper": 5.0,
      "penalty_weight": 1.2
    },
    "label_next_effluent_tp_mgL": {
      "name": "TP",
      "upper": 0.5,
      "penalty_weight": 1.0
    },
    "label_next_effluent_tn_mgL": {
      "name": "TN",
      "upper": 15.0,
      "penalty_weight": 1.0
    }
  },
  "decision_bounds": {
    "aeration_intensity_pct": {
      "lower": 10.0,
      "upper": 100.0,
      "max_step": 20.0
    },
    "chemical_dose_pac_mgL": {
      "lower": 2.0,
      "upper": 60.0,
      "max_step": 10.0
    }
  },
  "objective_weights": {
    "energy": 0.1,
    "chemical": 0.12,
    "smoothness": 0.06,
    "constraint_violation": 6.0
  },
  "smoothness_scale": {
    "aeration_intensity_pct": 15.0,
    "chemical_dose_kgph": 5.0
  }
}
```

## Validation Summary

```json
{
  "weighted_normalized_mae": 0.1832335386889497,
  "targets": {
    "label_next_effluent_cod_mgL": {
      "mae": 0.6676089922662026,
      "rmse": 0.9564894749630252,
      "r2": 0.3306893715956134,
      "normalized_mae": 0.12776870568187465,
      "limit_upper": 50.0,
      "actual_compliance_rate": 1.0,
      "predicted_compliance_rate": 1.0,
      "predicted_mean_exceedance": 0.0
    },
    "label_next_effluent_nh3n_mgL": {
      "mae": 0.07459415040741564,
      "rmse": 0.1433929182768767,
      "r2": 0.34432030220208065,
      "normalized_mae": 0.25543608275034,
      "limit_upper": 5.0,
      "actual_compliance_rate": 1.0,
      "predicted_compliance_rate": 1.0,
      "predicted_mean_exceedance": 0.0
    },
    "label_next_effluent_tp_mgL": {
      "mae": 0.007469028220079246,
      "rmse": 0.009331638606897844,
      "r2": -4.233666797973449,
      "normalized_mae": 0.1983556988664242,
      "limit_upper": 0.5,
      "actual_compliance_rate": 1.0,
      "predicted_compliance_rate": 1.0,
      "predicted_mean_exceedance": 0.0
    },
    "label_next_effluent_tn_mgL": {
      "mae": 0.21837826256522413,
      "rmse": 0.33217869999680694,
      "r2": 0.8948448757032005,
      "normalized_mae": 0.14808351401271166,
      "limit_upper": 15.0,
      "actual_compliance_rate": 1.0,
      "predicted_compliance_rate": 1.0,
      "predicted_mean_exceedance": 0.0
    }
  },
  "overall_compliance": {
    "actual_rate": 1.0,
    "predicted_rate": 1.0
  }
}
```

## Test Summary

```json
{
  "weighted_normalized_mae": 0.2409998853149044,
  "targets": {
    "label_next_effluent_cod_mgL": {
      "mae": 0.5458908537807622,
      "rmse": 1.4760536946924032,
      "r2": 0.44014887931853475,
      "normalized_mae": 0.10447397898938159,
      "limit_upper": 50.0,
      "actual_compliance_rate": 1.0,
      "predicted_compliance_rate": 1.0,
      "predicted_mean_exceedance": 0.0
    },
    "label_next_effluent_nh3n_mgL": {
      "mae": 0.13426064147371486,
      "rmse": 0.22678872032491595,
      "r2": 0.33273353677375905,
      "normalized_mae": 0.4597547145222823,
      "limit_upper": 5.0,
      "actual_compliance_rate": 1.0,
      "predicted_compliance_rate": 1.0,
      "predicted_mean_exceedance": 0.0
    },
    "label_next_effluent_tp_mgL": {
      "mae": 0.008129025921071792,
      "rmse": 0.0207312117264697,
      "r2": 0.23801987920362955,
      "normalized_mae": 0.21588332111836164,
      "limit_upper": 0.5,
      "actual_compliance_rate": 1.0,
      "predicted_compliance_rate": 1.0,
      "predicted_mean_exceedance": 0.0
    },
    "label_next_effluent_tn_mgL": {
      "mae": 0.22937347287316148,
      "rmse": 0.4688185690679332,
      "r2": 0.7680566829112517,
      "normalized_mae": 0.1555394272550924,
      "limit_upper": 15.0,
      "actual_compliance_rate": 1.0,
      "predicted_compliance_rate": 1.0,
      "predicted_mean_exceedance": 0.0
    }
  },
  "overall_compliance": {
    "actual_rate": 1.0,
    "predicted_rate": 1.0
  }
}
```

## Top Feature Importance

- `effluent_cod_mgL`: 0.3349
- `effluent_total_indicator`: 0.1222
- `effluent_cod_mgL_lag_1h`: 0.0492
- `effluent_cod_mgL_slope_3h`: 0.0368
- `effluent_cod_mgL_diff_3h`: 0.0361
- `effluent_tp_mgL`: 0.0232
- `effluent_tn_mgL`: 0.0213
- `effluent_tn_mgL_rollmean_3h`: 0.0147
- `effluent_cod_mgL_rollmean_3h`: 0.0146
- `effluent_tn_mgL_lag_1h`: 0.0128

## Assumptions

- Only `observed` rows are used for supervised label fitting.
- Recommendation search stays local to the current actuator values.
- Proxy dynamic features are built from hourly lags, rolling moments, and discrete first/second differences.
