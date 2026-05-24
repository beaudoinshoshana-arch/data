from __future__ import annotations

import argparse
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


TARGET_COLUMNS = [
    "label_next_effluent_cod_mgL",
    "label_next_effluent_nh3n_mgL",
    "label_next_effluent_tp_mgL",
    "label_next_effluent_tn_mgL",
]
DECISION_BASE_COLUMNS = [
    "influent_cod_mgL",
    "influent_bod_mgL",
    "influent_nh3n_mgL",
    "influent_tp_mgL",
    "influent_flow_m3h",
    "reactor_do_mgL",
    "sludge_mlss_mgL",
    "aeration_intensity_pct",
    "chemical_dose_pac_mgL",
    "chemical_dose_kgph",
]
CURRENT_EFFLUENT_COLUMNS = [
    "effluent_cod_mgL",
    "effluent_nh3n_mgL",
    "effluent_tp_mgL",
    "effluent_tn_mgL",
]
OPTIONAL_PLANT_COLUMNS = [
    "influent_tn_mgL",
    "reactor_orp_mv",
    "reactor_no3_mgL",
    "internal_nh3_mgL",
    "effluent_flow_m3h",
    "effluent_ph",
    "effluent_temp_c",
]
CONTEXT_LAG_COLUMNS = [
    "influent_cod_mgL",
    "influent_nh3n_mgL",
    "influent_tp_mgL",
    "influent_flow_m3h",
    "reactor_do_mgL",
    "sludge_mlss_mgL",
    "aeration_intensity_pct",
    "chemical_dose_kgph",
    "effluent_cod_mgL",
    "effluent_nh3n_mgL",
    "effluent_tp_mgL",
    "effluent_tn_mgL",
]
TARGET_WEIGHTS = {
    "label_next_effluent_cod_mgL": 0.35,
    "label_next_effluent_nh3n_mgL": 0.30,
    "label_next_effluent_tp_mgL": 0.20,
    "label_next_effluent_tn_mgL": 0.15,
}
SCENARIO_GRID_OFFSETS = {
    "observed": {
        "aeration": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "pac": np.array([-4.0, -2.0, 0.0, 2.0, 4.0]),
    },
    "load_up": {
        "aeration": np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0]),
        "pac": np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
    },
    "rain_dilution": {
        "aeration": np.array([-16.0, -12.0, -8.0, -4.0, 0.0, 4.0]),
        "pac": np.array([-8.0, -6.0, -4.0, -2.0, 0.0, 2.0]),
    },
}
QUALITY_FLAG_PREFIXES = (
    "flag_negative_or_range_",
    "flag_hampel_",
    "flag_flatline_",
)
DYNAMIC_FEATURE_TOKENS = (
    "_diff_",
    "_slope_",
    "_accel_",
    "_rollstd_",
    "_trend_turn_",
    "_volatility_ratio_",
)
DEFAULT_CONSTRAINT_CONFIG = {
    "target_limits": {
        "label_next_effluent_cod_mgL": {"name": "COD", "upper": 50.0, "penalty_weight": 1.0},
        "label_next_effluent_nh3n_mgL": {"name": "NH3-N", "upper": 5.0, "penalty_weight": 1.2},
        "label_next_effluent_tp_mgL": {"name": "TP", "upper": 0.5, "penalty_weight": 1.0},
        "label_next_effluent_tn_mgL": {"name": "TN", "upper": 15.0, "penalty_weight": 1.0},
    },
    "decision_bounds": {
        "aeration_intensity_pct": {"lower": 10.0, "upper": 100.0, "max_step": 20.0},
        "chemical_dose_pac_mgL": {"lower": 2.0, "upper": 60.0, "max_step": 10.0},
    },
    "objective_weights": {
        "energy": 0.10,
        "chemical": 0.12,
        "smoothness": 0.06,
        "constraint_violation": 6.0,
    },
    "smoothness_scale": {
        "aeration_intensity_pct": 15.0,
        "chemical_dose_kgph": 5.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the stage-2 wastewater surrogate model and generate control recommendations."
    )
    parser.add_argument(
        "--stage1-output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "stage1_data",
        help="Directory produced by build_stage1_datasets.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "stage2_model",
        help="Directory for model artifacts and recommendation reports.",
    )
    parser.add_argument(
        "--constraint-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "stage2_constraints.default.json",
        help="Optional JSON file that defines effluent limits, actuator bounds, and objective penalties.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by the candidate models.",
    )
    parser.add_argument(
        "--max-recommendations",
        type=int,
        default=None,
        help="Optional cap on recommendation rows, useful for quick debugging runs.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def deep_merge_dict(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_constraint_config(path: Path | None) -> tuple[dict[str, object], Path | None]:
    config = deepcopy(DEFAULT_CONSTRAINT_CONFIG)
    if path is None:
        return config, None
    if not path.exists():
        raise FileNotFoundError(f"Missing constraint config: {path}")
    override = json.loads(path.read_text(encoding="utf-8"))
    return deep_merge_dict(config, override), path


def load_stage1_tables(stage1_output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decision_raw_path = stage1_output / "decision_dataset" / "decision_dataset_raw.csv"
    labels_path = stage1_output / "decision_dataset" / "decision_labels_observed.csv"
    plant_path = stage1_output / "plant_real_hourly" / "plant_real_hourly.csv"
    for path in [decision_raw_path, labels_path, plant_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    decision_raw = pd.read_csv(decision_raw_path, parse_dates=["timestamp"])
    labels = pd.read_csv(labels_path, parse_dates=["timestamp"])
    plant = pd.read_csv(plant_path, parse_dates=["timestamp"])
    return decision_raw, labels, plant


def select_quality_flag_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith(QUALITY_FLAG_PREFIXES) and "label_next_" not in column
    ]


def build_observed_training_table(
    decision_raw: pd.DataFrame,
    labels: pd.DataFrame,
    plant: pd.DataFrame,
) -> pd.DataFrame:
    observed = decision_raw.loc[
        decision_raw["scenario_tag"] == "observed",
        ["timestamp", "split", "source_flag", "scenario_tag", "is_simulated", *DECISION_BASE_COLUMNS],
    ].copy()
    observed = observed.merge(
        labels,
        on=["timestamp", "split", "source_flag"],
        how="inner",
        validate="one_to_one",
    )
    extra_columns = [column for column in OPTIONAL_PLANT_COLUMNS if column in plant.columns]
    quality_flag_columns = select_quality_flag_columns(plant)
    current_columns = ["timestamp", *CURRENT_EFFLUENT_COLUMNS, *extra_columns, *quality_flag_columns]
    observed = observed.merge(
        plant[current_columns].copy(),
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    observed = observed.sort_values("timestamp").reset_index(drop=True)
    return observed


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = out["timestamp"].dt.hour.astype(float)
    day_of_week = out["timestamp"].dt.dayofweek.astype(float)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    return out


def safe_volatility_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    scale = denominator.abs().replace(0.0, np.nan)
    return numerator.abs() / scale


def build_proxy_dynamic_feature_block(ordered: pd.DataFrame, column: str) -> pd.DataFrame:
    series = ordered[column].astype(float)
    lag_1h = series.shift(1)
    lag_2h = series.shift(2)
    lag_3h = series.shift(3)
    lag_6h = series.shift(6)
    prev_history = series.shift(1)
    diff_1h = series - lag_1h
    diff_3h = series - lag_3h
    rollstd_3h = prev_history.rolling(window=3, min_periods=2).std()
    rollstd_6h = prev_history.rolling(window=6, min_periods=2).std()
    prev_diff = diff_1h.shift(1)
    return pd.DataFrame(
        {
            f"{column}_lag_1h": lag_1h,
            f"{column}_lag_3h": lag_3h,
            f"{column}_lag_6h": lag_6h,
            f"{column}_rollmean_3h": prev_history.rolling(window=3, min_periods=1).mean(),
            f"{column}_rollmean_6h": prev_history.rolling(window=6, min_periods=1).mean(),
            f"{column}_rollstd_3h": rollstd_3h,
            f"{column}_rollstd_6h": rollstd_6h,
            f"{column}_diff_1h": diff_1h,
            f"{column}_diff_3h": diff_3h,
            f"{column}_slope_3h": diff_3h / 3.0,
            f"{column}_accel_1h": series - 2.0 * lag_1h + lag_2h,
            f"{column}_trend_turn_1h": ((diff_1h * prev_diff) < 0).astype(float),
            f"{column}_volatility_ratio_6h": safe_volatility_ratio(diff_1h, rollstd_6h),
        }
    )


def build_temporal_context(observed: pd.DataFrame) -> pd.DataFrame:
    ordered = add_time_features(observed).sort_values("timestamp").reset_index(drop=True)
    extra_columns = [column for column in OPTIONAL_PLANT_COLUMNS if column in ordered.columns]
    quality_flag_columns = select_quality_flag_columns(ordered)
    current_columns = list(
        dict.fromkeys(
            ["timestamp", "hour_sin", "hour_cos", "dow_sin", "dow_cos", *CURRENT_EFFLUENT_COLUMNS, *extra_columns, *quality_flag_columns]
        )
    )
    context_frames = [ordered[current_columns].copy()]

    dynamic_columns = [
        column
        for column in list(dict.fromkeys([*CONTEXT_LAG_COLUMNS, *OPTIONAL_PLANT_COLUMNS]))
        if column in ordered.columns
    ]
    for column in dynamic_columns:
        context_frames.append(build_proxy_dynamic_feature_block(ordered, column))
    return pd.concat(context_frames, axis=1)


def clip_lower(series: pd.Series, minimum: float) -> pd.Series:
    return series.astype(float).clip(lower=minimum)


def add_optional_process_features(out: pd.DataFrame) -> pd.DataFrame:
    if "influent_tn_mgL" in out.columns:
        influent_tn = clip_lower(out["influent_tn_mgL"], 0.01)
        out["influent_tn_load_kgph"] = influent_tn * clip_lower(out["influent_flow_m3h"], 1.0) / 1000.0
        out["cod_to_tn_ratio"] = clip_lower(out["influent_cod_mgL"], 0.1) / influent_tn
        out["nh3_to_tn_ratio"] = clip_lower(out["influent_nh3n_mgL"], 0.1) / influent_tn
    if "reactor_no3_mgL" in out.columns:
        reactor_no3 = clip_lower(out["reactor_no3_mgL"], 0.01)
        out["do_to_no3_ratio"] = clip_lower(out["reactor_do_mgL"], 0.05) / reactor_no3
    if "internal_nh3_mgL" in out.columns:
        internal_nh3 = clip_lower(out["internal_nh3_mgL"], 0.01)
        out["influent_to_internal_nh3_ratio"] = clip_lower(out["influent_nh3n_mgL"], 0.1) / internal_nh3
    if "effluent_flow_m3h" in out.columns:
        out["effluent_to_influent_flow_ratio"] = clip_lower(out["effluent_flow_m3h"], 1.0) / clip_lower(
            out["influent_flow_m3h"], 1.0
        )
    if "effluent_temp_c" in out.columns:
        out["effluent_temp_centered"] = out["effluent_temp_c"].astype(float) - out["effluent_temp_c"].astype(float).median()
    quality_flag_columns = select_quality_flag_columns(out)
    if quality_flag_columns:
        quality_matrix = out[quality_flag_columns].astype(float)
        out["quality_flag_count_current"] = quality_matrix.sum(axis=1)
        out["quality_flag_any_current"] = (quality_matrix.sum(axis=1) > 0).astype(float)
    return out


def add_rowwise_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flow = clip_lower(out["influent_flow_m3h"], 1.0)
    cod = clip_lower(out["influent_cod_mgL"], 0.1)
    nh3 = clip_lower(out["influent_nh3n_mgL"], 0.1)
    tp = clip_lower(out["influent_tp_mgL"], 0.01)
    do = clip_lower(out["reactor_do_mgL"], 0.05)
    sludge = clip_lower(out["sludge_mlss_mgL"], 1.0)
    pac = clip_lower(out["chemical_dose_pac_mgL"], 0.01)
    chem_kgph = clip_lower(out["chemical_dose_kgph"], 0.01)
    eff_cod = clip_lower(out["effluent_cod_mgL"], 0.1)
    eff_nh3 = clip_lower(out["effluent_nh3n_mgL"], 0.01)
    eff_tp = clip_lower(out["effluent_tp_mgL"], 0.001)
    eff_tn = clip_lower(out["effluent_tn_mgL"], 0.01)

    out["influent_cod_load_kgph"] = cod * flow / 1000.0
    out["influent_nh3n_load_kgph"] = nh3 * flow / 1000.0
    out["influent_tp_load_kgph"] = tp * flow / 1000.0
    out["aeration_per_flow"] = out["aeration_intensity_pct"] / flow
    out["chemical_per_flow"] = chem_kgph / flow
    out["chemical_per_tp"] = pac / tp
    out["do_to_cod_ratio"] = do / cod
    out["do_to_nh3_ratio"] = do / nh3
    out["cod_to_tp_ratio"] = cod / tp
    out["nh3_to_tp_ratio"] = nh3 / tp
    out["sludge_to_flow_ratio"] = sludge / flow
    out["bod_to_cod_ratio"] = out["influent_bod_mgL"] / cod
    out["effluent_total_indicator"] = eff_cod + 10.0 * eff_nh3 + 50.0 * eff_tp + 5.0 * eff_tn
    out["effluent_cod_to_nh3_ratio"] = eff_cod / eff_nh3
    out["effluent_tn_to_tp_ratio"] = eff_tn / eff_tp
    out["reactor_pressure_index"] = out["aeration_intensity_pct"] / do
    out["chemical_vs_cod_load"] = chem_kgph / clip_lower(out["influent_cod_load_kgph"], 0.01)
    out = add_optional_process_features(out)
    return out


def assemble_feature_frame(decision_rows: pd.DataFrame, temporal_context: pd.DataFrame) -> pd.DataFrame:
    context_columns = [
        column
        for column in temporal_context.columns
        if column == "timestamp" or column not in decision_rows.columns
    ]
    features = decision_rows.merge(
        temporal_context[context_columns],
        on="timestamp",
        how="left",
        validate="many_to_one",
    )
    features = add_rowwise_features(features)
    return features


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "timestamp",
        "split",
        "source_flag",
        "scenario_tag",
        "is_simulated",
        *TARGET_COLUMNS,
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def select_dynamic_feature_columns(feature_columns: list[str]) -> list[str]:
    return [
        column
        for column in feature_columns
        if any(token in column for token in DYNAMIC_FEATURE_TOKENS)
    ]


def build_model_candidates(random_state: int) -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=14,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=700,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def target_limit(constraint_config: dict[str, object], target: str) -> dict[str, object]:
    return dict(constraint_config.get("target_limits", {}).get(target, {}))


def evaluate_predictions(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    reference_std: pd.Series,
    constraint_config: dict[str, object],
) -> dict[str, object]:
    metrics: dict[str, dict[str, float]] = {}
    weighted_nmae = 0.0
    actual_overall = pd.Series(True, index=y_true.index, dtype=bool)
    predicted_overall = pd.Series(True, index=y_true.index, dtype=bool)
    has_limits = False

    for idx, target in enumerate(TARGET_COLUMNS):
        mae = float(mean_absolute_error(y_true[target], y_pred[:, idx]))
        rmse = float(np.sqrt(mean_squared_error(y_true[target], y_pred[:, idx])))
        r2 = float(r2_score(y_true[target], y_pred[:, idx]))
        scale = float(max(reference_std[target], 1e-6))
        nmae = mae / scale
        metric_row = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "normalized_mae": nmae,
        }
        limit_cfg = target_limit(constraint_config, target)
        upper = limit_cfg.get("upper")
        if upper is not None:
            has_limits = True
            actual_compliant = pd.Series(y_true[target].to_numpy() <= float(upper), index=y_true.index)
            predicted_compliant = pd.Series(y_pred[:, idx] <= float(upper), index=y_true.index)
            metric_row["limit_upper"] = float(upper)
            metric_row["actual_compliance_rate"] = float(actual_compliant.mean())
            metric_row["predicted_compliance_rate"] = float(predicted_compliant.mean())
            metric_row["predicted_mean_exceedance"] = float(np.maximum(y_pred[:, idx] - float(upper), 0.0).mean())
            actual_overall = actual_overall & actual_compliant
            predicted_overall = predicted_overall & predicted_compliant
        metrics[target] = metric_row
        weighted_nmae += TARGET_WEIGHTS[target] * nmae

    result: dict[str, object] = {
        "weighted_normalized_mae": float(weighted_nmae),
        "targets": metrics,
    }
    if has_limits:
        result["overall_compliance"] = {
            "actual_rate": float(actual_overall.mean()),
            "predicted_rate": float(predicted_overall.mean()),
        }
    return result


def feature_importance_summary(model: Pipeline, feature_columns: list[str], top_k: int = 20) -> list[dict[str, float]]:
    estimator = model.named_steps["model"]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return []
    ranking = pd.Series(importances, index=feature_columns).sort_values(ascending=False).head(top_k)
    return [
        {"feature": str(name), "importance": float(value)}
        for name, value in ranking.items()
    ]


def json_ready(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(key): json_ready(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_ready(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def bounded_values(center: float, offsets: np.ndarray, lower: float, upper: float, max_step: float | None) -> np.ndarray:
    lower_bound = lower if max_step is None else max(lower, center - max_step)
    upper_bound = upper if max_step is None else min(upper, center + max_step)
    values = np.clip(center + offsets, lower_bound, upper_bound)
    values = np.round(values.astype(float), 6)
    return np.unique(values)


def build_candidate_rows(row: pd.Series, constraint_config: dict[str, object]) -> pd.DataFrame:
    config = SCENARIO_GRID_OFFSETS.get(row["scenario_tag"], SCENARIO_GRID_OFFSETS["observed"])
    aeration_bounds = dict(constraint_config["decision_bounds"]["aeration_intensity_pct"])
    pac_bounds = dict(constraint_config["decision_bounds"]["chemical_dose_pac_mgL"])
    aeration_values = bounded_values(
        center=float(row["aeration_intensity_pct"]),
        offsets=config["aeration"],
        lower=float(aeration_bounds["lower"]),
        upper=float(aeration_bounds["upper"]),
        max_step=float(aeration_bounds.get("max_step", np.inf)),
    )
    pac_values = bounded_values(
        center=float(row["chemical_dose_pac_mgL"]),
        offsets=config["pac"],
        lower=float(pac_bounds["lower"]),
        upper=float(pac_bounds["upper"]),
        max_step=float(pac_bounds.get("max_step", np.inf)),
    )
    rows: list[dict[str, object]] = []
    baseline_aeration = float(row["aeration_intensity_pct"])
    baseline_pac = float(row["chemical_dose_pac_mgL"])
    flow = float(row["influent_flow_m3h"])
    for aeration in aeration_values:
        for pac in pac_values:
            candidate = row.to_dict()
            candidate["aeration_intensity_pct"] = float(aeration)
            candidate["chemical_dose_pac_mgL"] = float(pac)
            candidate["chemical_dose_kgph"] = float(np.clip(pac * flow / 1000.0, 0.0, 500.0))
            candidate["is_baseline_candidate"] = bool(
                np.isclose(aeration, baseline_aeration) and np.isclose(pac, baseline_pac)
            )
            rows.append(candidate)
    if not any(candidate["is_baseline_candidate"] for candidate in rows):
        candidate = row.to_dict()
        candidate["is_baseline_candidate"] = True
        rows.append(candidate)
    return pd.DataFrame(rows)


def compute_constraint_penalty(
    predictions: pd.DataFrame,
    constraint_config: dict[str, object],
) -> tuple[pd.Series, pd.Series]:
    penalty = pd.Series(0.0, index=predictions.index, dtype=float)
    feasible = pd.Series(True, index=predictions.index, dtype=bool)
    for target in TARGET_COLUMNS:
        limit_cfg = target_limit(constraint_config, target)
        upper = limit_cfg.get("upper")
        if upper is None:
            continue
        exceedance = (predictions[target] - float(upper)).clip(lower=0.0)
        scale = float(max(abs(float(upper)), 1.0))
        target_weight = float(limit_cfg.get("penalty_weight", 1.0))
        penalty = penalty + target_weight * exceedance / scale
        feasible = feasible & exceedance.le(0.0)
    return penalty, feasible


def candidate_objective(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    reference_scale: pd.Series,
    constraint_config: dict[str, object],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    pollution = pd.Series(0.0, index=candidates.index, dtype=float)
    for target in TARGET_COLUMNS:
        scale = float(max(reference_scale[target], 0.1))
        pollution = pollution + TARGET_WEIGHTS[target] * predictions[target] / scale

    baseline_row = candidates.loc[candidates["is_baseline_candidate"]].iloc[0]
    baseline_aeration = float(baseline_row["aeration_intensity_pct"])
    baseline_chem = float(baseline_row["chemical_dose_kgph"])
    objective_weights = dict(constraint_config.get("objective_weights", {}))
    smoothness_scale = dict(constraint_config.get("smoothness_scale", {}))
    energy = float(objective_weights.get("energy", 0.10)) * candidates["aeration_intensity_pct"] / max(
        baseline_aeration, 10.0
    )
    chemical = float(objective_weights.get("chemical", 0.12)) * candidates["chemical_dose_kgph"] / max(
        baseline_chem, 0.5
    )
    smoothness = float(objective_weights.get("smoothness", 0.06)) * (
        (candidates["aeration_intensity_pct"] - baseline_aeration).abs()
        / float(smoothness_scale.get("aeration_intensity_pct", 15.0))
        + (candidates["chemical_dose_kgph"] - baseline_chem).abs()
        / float(smoothness_scale.get("chemical_dose_kgph", 5.0))
    )
    constraint_penalty, feasible = compute_constraint_penalty(predictions, constraint_config)
    objective = pollution + energy + chemical + smoothness + float(
        objective_weights.get("constraint_violation", 6.0)
    ) * constraint_penalty
    return objective, feasible, constraint_penalty


def generate_recommendations(
    decision_raw: pd.DataFrame,
    temporal_context: pd.DataFrame,
    model: Pipeline,
    feature_columns: list[str],
    reference_scale: pd.Series,
    constraint_config: dict[str, object],
    max_rows: int | None = None,
) -> pd.DataFrame:
    scenario_rows = decision_raw.loc[decision_raw["split"] == "test"].copy()
    scenario_rows = scenario_rows.sort_values(["timestamp", "scenario_tag"]).reset_index(drop=True)
    if max_rows is not None:
        scenario_rows = scenario_rows.head(max_rows).copy()

    recommendations: list[dict[str, object]] = []
    for _, row in scenario_rows.iterrows():
        candidate_rows = build_candidate_rows(row, constraint_config)
        candidate_features = assemble_feature_frame(candidate_rows, temporal_context)
        candidate_matrix = candidate_features[feature_columns]
        candidate_pred = model.predict(candidate_matrix)
        pred_df = pd.DataFrame(candidate_pred, columns=TARGET_COLUMNS, index=candidate_rows.index)
        objective, feasible, constraint_penalty = candidate_objective(
            candidate_rows,
            pred_df,
            reference_scale,
            constraint_config,
        )
        candidate_rows["objective"] = objective
        candidate_rows["is_feasible"] = feasible
        candidate_rows["constraint_penalty"] = constraint_penalty

        baseline = candidate_rows.loc[candidate_rows["is_baseline_candidate"]].iloc[0]
        baseline_pred = pred_df.loc[baseline.name]
        feasible_pool = candidate_rows.loc[candidate_rows["is_feasible"]]
        search_pool = feasible_pool if not feasible_pool.empty else candidate_rows
        best_idx = search_pool["objective"].idxmin()
        best_row = candidate_rows.loc[best_idx]
        best_pred = pred_df.loc[best_idx]

        record: dict[str, object] = {
            "timestamp": row["timestamp"],
            "split": row["split"],
            "scenario_tag": row["scenario_tag"],
            "source_flag": row["source_flag"],
            "used_feasible_pool": bool(not feasible_pool.empty),
            "baseline_is_feasible": bool(baseline["is_feasible"]),
            "recommended_is_feasible": bool(best_row["is_feasible"]),
            "baseline_constraint_penalty": float(baseline["constraint_penalty"]),
            "recommended_constraint_penalty": float(best_row["constraint_penalty"]),
            "baseline_aeration_intensity_pct": float(baseline["aeration_intensity_pct"]),
            "recommended_aeration_intensity_pct": float(best_row["aeration_intensity_pct"]),
            "baseline_chemical_dose_pac_mgL": float(baseline["chemical_dose_pac_mgL"]),
            "recommended_chemical_dose_pac_mgL": float(best_row["chemical_dose_pac_mgL"]),
            "baseline_chemical_dose_kgph": float(baseline["chemical_dose_kgph"]),
            "recommended_chemical_dose_kgph": float(best_row["chemical_dose_kgph"]),
            "baseline_objective": float(baseline["objective"]),
            "recommended_objective": float(best_row["objective"]),
        }
        for target in TARGET_COLUMNS:
            pretty_name = target.replace("label_next_", "pred_")
            upper = target_limit(constraint_config, target).get("upper")
            baseline_value = float(baseline_pred[target])
            recommended_value = float(best_pred[target])
            record[f"baseline_{pretty_name}"] = baseline_value
            record[f"recommended_{pretty_name}"] = recommended_value
            record[f"delta_{pretty_name}"] = float(recommended_value - baseline_value)
            if upper is not None:
                record[f"{pretty_name}_upper_limit"] = float(upper)
                record[f"baseline_{pretty_name}_exceedance"] = float(max(baseline_value - float(upper), 0.0))
                record[f"recommended_{pretty_name}_exceedance"] = float(max(recommended_value - float(upper), 0.0))
        recommendations.append(record)
    return pd.DataFrame(recommendations)


def write_model_card(
    path: Path,
    summary: dict[str, object],
    feature_columns: list[str],
    dynamic_feature_columns: list[str],
    constraint_config: dict[str, object],
) -> None:
    top_features = summary.get("top_feature_importance", [])
    validation = summary.get("validation", {})
    test = summary.get("test", {})
    lines = [
        "# Stage-2 Wastewater Model Card",
        "",
        "## Purpose",
        "",
        "This surrogate model predicts next-hour effluent quality and recommends bounded aeration and PAC-dose actions.",
        "",
        "## Targets",
        "",
    ]
    for target in TARGET_COLUMNS:
        lines.append(f"- `{target}`")
    lines.extend(
        [
            "",
            "## Feature Strategy",
            "",
            f"- Total numeric features: {len(feature_columns)}",
            f"- Proxy dynamic features: {len(dynamic_feature_columns)}",
            f"- Current design keeps the stage-1 observed rows for supervision and uses scenario rows only for recommendation search.",
            "",
            "## Constraint Config",
            "",
            "```json",
            json.dumps(constraint_config, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Validation Summary",
            "",
            "```json",
            json.dumps(validation, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Test Summary",
            "",
            "```json",
            json.dumps(test, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Top Feature Importance",
            "",
        ]
    )
    if top_features:
        for item in top_features[:10]:
            lines.append(f"- `{item['feature']}`: {item['importance']:.4f}")
    else:
        lines.append("- Not available for the selected estimator.")
    lines.extend(
        [
            "",
            "## Assumptions",
            "",
            "- Only `observed` rows are used for supervised label fitting.",
            "- Recommendation search stays local to the current actuator values.",
            "- Proxy dynamic features are built from hourly lags, rolling moments, and discrete first/second differences.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output)
    constraint_config, constraint_source = load_constraint_config(args.constraint_config)

    decision_raw, labels, plant = load_stage1_tables(args.stage1_output)
    observed = build_observed_training_table(decision_raw, labels, plant)
    temporal_context = build_temporal_context(observed)
    observed_features = assemble_feature_frame(observed, temporal_context)
    observed_features = observed_features.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    feature_columns = select_feature_columns(observed_features)
    dynamic_feature_columns = select_dynamic_feature_columns(feature_columns)
    quality_feature_columns = [
        column
        for column in feature_columns
        if column.startswith(QUALITY_FLAG_PREFIXES) or column.startswith("quality_flag_")
    ]

    train_mask = observed_features["split"] == "train"
    val_mask = observed_features["split"] == "val"
    test_mask = observed_features["split"] == "test"
    train_val_mask = train_mask | val_mask

    X_train = observed_features.loc[train_mask, feature_columns]
    y_train = observed_features.loc[train_mask, TARGET_COLUMNS]
    X_val = observed_features.loc[val_mask, feature_columns]
    y_val = observed_features.loc[val_mask, TARGET_COLUMNS]
    X_test = observed_features.loc[test_mask, feature_columns]
    y_test = observed_features.loc[test_mask, TARGET_COLUMNS]
    X_train_val = observed_features.loc[train_val_mask, feature_columns]
    y_train_val = observed_features.loc[train_val_mask, TARGET_COLUMNS]

    reference_std = y_train.std(ddof=0).replace(0.0, 1.0)
    train_std = y_train.std(ddof=0)
    reference_scale = train_std.where(train_std > 0.1, y_train.median().abs() + 0.1)

    leaderboard: list[dict[str, object]] = []
    for name, candidate in build_model_candidates(args.random_state).items():
        candidate.fit(X_train, y_train)
        val_pred = candidate.predict(X_val)
        val_metrics = evaluate_predictions(y_val, val_pred, reference_std, constraint_config)
        leaderboard.append(
            {
                "model": name,
                "validation": val_metrics,
            }
        )

    leaderboard = sorted(leaderboard, key=lambda item: item["validation"]["weighted_normalized_mae"])
    best_model_name = str(leaderboard[0]["model"])
    best_model = build_model_candidates(args.random_state)[best_model_name]
    best_model.fit(X_train_val, y_train_val)

    test_pred = best_model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, test_pred, reference_std, constraint_config)
    importance = feature_importance_summary(best_model, feature_columns)

    test_predictions = observed_features.loc[test_mask, ["timestamp", "split", "source_flag"]].copy()
    for idx, target in enumerate(TARGET_COLUMNS):
        short_name = target.replace("label_next_", "")
        test_predictions[f"actual_{short_name}"] = y_test[target].to_numpy()
        test_predictions[f"pred_{short_name}"] = test_pred[:, idx]
        test_predictions[f"abs_err_{short_name}"] = np.abs(
            test_predictions[f"pred_{short_name}"] - test_predictions[f"actual_{short_name}"]
        )
        limit_cfg = target_limit(constraint_config, target)
        upper = limit_cfg.get("upper")
        if upper is not None:
            test_predictions[f"{short_name}_upper_limit"] = float(upper)
            test_predictions[f"actual_{short_name}_is_compliant"] = (
                test_predictions[f"actual_{short_name}"] <= float(upper)
            )
            test_predictions[f"pred_{short_name}_is_compliant"] = (
                test_predictions[f"pred_{short_name}"] <= float(upper)
            )
            test_predictions[f"pred_{short_name}_exceedance"] = np.maximum(
                test_predictions[f"pred_{short_name}"] - float(upper),
                0.0,
            )

    overall_actual_compliance = pd.Series(True, index=test_predictions.index, dtype=bool)
    overall_pred_compliance = pd.Series(True, index=test_predictions.index, dtype=bool)
    for target in TARGET_COLUMNS:
        short_name = target.replace("label_next_", "")
        actual_flag = f"actual_{short_name}_is_compliant"
        pred_flag = f"pred_{short_name}_is_compliant"
        if actual_flag in test_predictions.columns:
            overall_actual_compliance = overall_actual_compliance & test_predictions[actual_flag]
            overall_pred_compliance = overall_pred_compliance & test_predictions[pred_flag]
    if len(test_predictions) > 0 and any(
        key in test_predictions.columns for key in [f"actual_{target.replace('label_next_', '')}_is_compliant" for target in TARGET_COLUMNS]
    ):
        test_predictions["actual_all_targets_compliant"] = overall_actual_compliance
        test_predictions["pred_all_targets_compliant"] = overall_pred_compliance

    recommendations = generate_recommendations(
        decision_raw=decision_raw,
        temporal_context=temporal_context,
        model=best_model,
        feature_columns=feature_columns,
        reference_scale=reference_scale,
        constraint_config=constraint_config,
        max_rows=args.max_recommendations,
    )

    dynamic_feature_frame = observed_features[
        [
            "timestamp",
            "split",
            "source_flag",
            "scenario_tag",
            *dynamic_feature_columns,
        ]
    ].copy()

    model_path = output_dir / "best_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(best_model, handle)

    feature_columns_path = output_dir / "feature_columns.json"
    leaderboard_path = output_dir / "leaderboard.json"
    test_predictions_path = output_dir / "test_predictions.csv"
    recommendations_path = output_dir / "scenario_recommendations_test.csv"
    dynamic_feature_frame_path = output_dir / "dynamic_feature_frame.csv"
    constraint_config_used_path = output_dir / "constraint_config_used.json"
    model_card_path = output_dir / "model_card.md"
    summary_path = output_dir / "summary.json"

    feature_columns_path.write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    leaderboard_path.write_text(
        json.dumps(json_ready(leaderboard), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    constraint_config_used_path.write_text(
        json.dumps(json_ready(constraint_config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    test_predictions.to_csv(test_predictions_path, index=False, encoding="utf-8-sig")
    recommendations.to_csv(recommendations_path, index=False, encoding="utf-8-sig")
    dynamic_feature_frame.to_csv(dynamic_feature_frame_path, index=False, encoding="utf-8-sig")

    summary = {
        "selected_model": best_model_name,
        "row_counts": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
            "recommendation_rows": int(len(recommendations)),
        },
        "feature_count": int(len(feature_columns)),
        "feature_groups": {
            "dynamic_feature_count": int(len(dynamic_feature_columns)),
            "quality_flag_feature_count": int(len(quality_feature_columns)),
            "context_column_count": int(len(temporal_context.columns) - 1),
        },
        "constraint_config_source": str(constraint_source) if constraint_source else "built_in_default",
        "validation": leaderboard[0]["validation"],
        "test": test_metrics,
        "top_feature_importance": importance,
        "model_artifacts": {
            "model": model_path,
            "leaderboard": leaderboard_path,
            "feature_columns": feature_columns_path,
            "test_predictions": test_predictions_path,
            "scenario_recommendations_test": recommendations_path,
            "dynamic_feature_frame": dynamic_feature_frame_path,
            "constraint_config_used": constraint_config_used_path,
            "model_card": model_card_path,
        },
        "assumptions": [
            "Only observed rows are used as supervised samples because simulated scenarios do not have independent ground-truth labels.",
            "Control recommendation is limited to a local scenario-aware search grid so the optimizer stays near feasible operating ranges.",
            "Proxy dynamic features approximate the paper's dynamic idea with hourly lag, rolling, first-difference, and second-difference statistics.",
        ],
    }
    write_model_card(model_card_path, summary, feature_columns, dynamic_feature_columns, constraint_config)
    summary_path.write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
