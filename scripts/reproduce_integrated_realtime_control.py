from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from platypus import GeneticAlgorithm, NSGAII, Problem, Real
from scipy.integrate import cumulative_trapezoid, odeint
from scipy.optimize import curve_fit
from scipy.fft import irfft, rfft
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


REAL_TARGET_COLUMNS = [
    "label_next_effluent_bod_proxy_mgL",
    "label_next_effluent_nh3n_mgL",
    "label_next_effluent_tn_mgL",
]
REAL_TARGET_WEIGHTS = {
    "label_next_effluent_bod_proxy_mgL": 0.35,
    "label_next_effluent_nh3n_mgL": 0.35,
    "label_next_effluent_tn_mgL": 0.30,
}
SURFACE_MODEL_ORDER = [
    ("anoxic_no3", "A"),
    ("anoxic_no3", "mu"),
    ("anoxic_no3", "delta"),
    ("aerobic_bod", "A"),
    ("aerobic_bod", "mu"),
    ("aerobic_bod", "delta"),
    ("aerobic_nh3", "A"),
    ("aerobic_nh3", "mu"),
    ("aerobic_nh3", "delta"),
]


@dataclass
class SurfaceModel:
    name: str
    input_cols: list[str]
    degree: int
    poly: PolynomialFeatures
    reg: LinearRegression
    metrics: dict[str, float]
    formula: str


class ModbusMock:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def write_setpoints(self, timestamp: pd.Timestamp, kla_d1: float, nrr_m3h: float) -> None:
        self.records.append(
            {
                "timestamp": timestamp,
                "kla_d1": float(kla_d1),
                "nrr_m3h": float(nrr_m3h),
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-faithful reimplementation of integrated real-time intelligent control for WWTPs."
    )
    parser.add_argument(
        "--stage1-output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "stage1_data",
        help="Directory produced by build_stage1_datasets.py.",
    )
    parser.add_argument(
        "--asm-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "paper_repro_asm2d.default.json",
        help="Digital twin and calibration config.",
    )
    parser.add_argument(
        "--batch-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "paper_repro_batch.default.json",
        help="Batch sweep and surface-model config.",
    )
    parser.add_argument(
        "--online-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "paper_repro_online.default.json",
        help="Steady-state and online replay config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "paper_repro_integrated_control",
        help="Output directory for the paper reproduction artifacts.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_ready(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(key): json_ready(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_ready(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-9:
        return 0.0
    return float(r2_score(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evenly_spaced_index(length: int, target_size: int) -> np.ndarray:
    if target_size >= length:
        return np.arange(length)
    return np.linspace(0, length - 1, num=target_size, dtype=int)


def gaussian(t: np.ndarray, amplitude: float, mu: float, delta: float) -> np.ndarray:
    safe_delta = max(float(delta), 1e-6)
    return float(amplitude) * np.exp(-0.5 * ((t - float(mu)) / safe_delta) ** 2)


def low_pass_fourier(signal: np.ndarray, keep_ratio: float) -> np.ndarray:
    fft = rfft(signal)
    cutoff = max(3, int(math.ceil(len(fft) * keep_ratio)))
    fft[cutoff:] = 0
    return irfft(fft, n=len(signal))


def unit_removal_curve(time_min: np.ndarray, mu: float, delta: float) -> tuple[np.ndarray, np.ndarray, float]:
    g = gaussian(time_min, 1.0, mu, delta)
    total_area = float(np.trapezoid(g, time_min))
    cumulative_area = cumulative_trapezoid(g, time_min, initial=0.0)
    velocity = -total_area + cumulative_area
    concentration_delta = cumulative_trapezoid(velocity, time_min, initial=0.0)
    removal_unit = float(-concentration_delta[-1])
    return velocity, concentration_delta, removal_unit


def amplitude_from_endpoints(initial_value: float, final_value: float, time_min: np.ndarray, mu: float, delta: float) -> float:
    _, _, removal_unit = unit_removal_curve(time_min, mu, delta)
    desired_removal = max(float(initial_value) - float(final_value), 0.0)
    if removal_unit <= 1e-9:
        return 0.0
    return desired_removal / removal_unit


def solve_profile_odeint(
    initial_value: float,
    amplitude: float,
    mu: float,
    delta: float,
    time_min: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    accel = gaussian(time_min, amplitude, mu, delta)
    total_area = float(np.trapezoid(accel, time_min))

    def ode_system(y: np.ndarray, t: float) -> list[float]:
        accel_t = float(np.interp(t, time_min, accel))
        return [float(y[1]), accel_t]

    y0 = [float(initial_value), -total_area]
    solution = odeint(ode_system, y0, time_min)
    concentration = np.clip(solution[:, 0], a_min=0.0, a_max=None)
    velocity = solution[:, 1]
    return concentration, velocity


def final_from_feature_params(initial_value: np.ndarray, amplitude: np.ndarray, mu: np.ndarray, delta: np.ndarray, time_min: np.ndarray) -> np.ndarray:
    results: list[float] = []
    for init, amp, mean, spread in zip(initial_value, amplitude, mu, delta, strict=True):
        _, concentration_delta, _ = unit_removal_curve(time_min, float(mean), float(spread))
        final_value = float(init) + float(amp) * float(concentration_delta[-1])
        results.append(max(final_value, 0.0))
    return np.asarray(results, dtype=float)


def fit_gaussian_feature_parameters(
    time_min: np.ndarray,
    concentration: np.ndarray,
    keep_ratio: float,
    trim_points: int,
) -> dict[str, float]:
    dt = float(np.median(np.diff(time_min)))
    window_length = min(len(concentration) - (1 - len(concentration) % 2), 31)
    if window_length < 7:
        window_length = 7
    if window_length % 2 == 0:
        window_length -= 1
    smooth = savgol_filter(concentration, window_length=window_length, polyorder=3, mode="interp")
    second = savgol_filter(smooth, window_length=window_length, polyorder=4, deriv=2, delta=dt, mode="interp")
    trimmed_t = time_min[trim_points:]
    trimmed_second = second[trim_points:]
    filtered = np.clip(low_pass_fourier(trimmed_second, keep_ratio), a_min=0.0, a_max=None)

    if filtered.max() <= 1e-9:
        return {
            "A": 0.0,
            "mu": float(trimmed_t[len(trimmed_t) // 2]),
            "delta": 1.0,
            "gaussian_r2": 0.0,
            "gaussian_rmse": 0.0,
        }

    guess_amplitude = float(filtered.max())
    guess_mu = float(trimmed_t[int(np.argmax(filtered))])
    weighted_mean = float(np.average(trimmed_t, weights=np.maximum(filtered, 1e-6)))
    weighted_var = float(np.average((trimmed_t - weighted_mean) ** 2, weights=np.maximum(filtered, 1e-6)))
    guess_delta = max(weighted_var ** 0.5, 1.0)

    try:
        params, _ = curve_fit(
            gaussian,
            trimmed_t,
            filtered,
            p0=[guess_amplitude, guess_mu, guess_delta],
            bounds=([0.0, float(trimmed_t.min()), 0.25], [np.inf, float(trimmed_t.max()), 60.0]),
            maxfev=20000,
        )
    except Exception:
        params = np.asarray([guess_amplitude, guess_mu, guess_delta], dtype=float)
    fitted = gaussian(trimmed_t, *params)
    return {
        "A": float(params[0]),
        "mu": float(params[1]),
        "delta": float(params[2]),
        "gaussian_r2": safe_r2(filtered, fitted),
        "gaussian_rmse": rmse(filtered, fitted),
    }


def anoxic_truth_parameters(
    bod_mgL: float,
    no3_mgL: float,
    nrr_m3h: float,
    kinetic_params: dict[str, float],
    asm_config: dict[str, object],
) -> tuple[float, float, float]:
    design = asm_config["design"]
    cn_ratio = bod_mgL / max(no3_mgL, 0.5)
    k_h = kinetic_params["kH"]
    mu_h = kinetic_params["muH"]
    design_nrr = float(design["nitrate_return_flow_m3h"])
    mu = 34.0 + 2.6 * cn_ratio - 0.75 * no3_mgL + 2.4 * (2.45 - k_h) - 0.0003 * (nrr_m3h - design_nrr)
    delta = 9.5 + 0.45 * cn_ratio + 0.12 * no3_mgL + 0.35 * max(0.0, 6.2 - mu_h)
    score = 0.7 * (cn_ratio - 2.2) + 0.025 * (no3_mgL - 12.0) + 1.2 * (mu_h - 6.0) + 0.8 * (
        nrr_m3h / design_nrr - 1.0
    )
    removal = 0.15 + 0.75 * float(logistic(score))
    final_no3 = max(0.05, no3_mgL * (1.0 - removal))
    return float(np.clip(mu, 12.0, 80.0)), float(np.clip(delta, 4.0, 25.0)), final_no3


def aerobic_bod_truth_parameters(
    bod_mgL: float,
    kla_d1: float,
    kinetic_params: dict[str, float],
) -> tuple[float, float, float]:
    k_h = kinetic_params["kH"]
    mu = 26.0 - 0.12 * (kla_d1 - 90.0) + 0.22 * bod_mgL + 1.0 * (2.6 - k_h)
    delta = 7.5 + 0.025 * bod_mgL + 0.015 * (140.0 - kla_d1)
    score = 0.05 * (kla_d1 - 100.0) + 0.035 * (bod_mgL - 40.0) + 0.9 * (k_h - 2.45)
    removal = 0.55 + 0.35 * float(logistic(score))
    final_bod = max(0.3, bod_mgL * (1.0 - removal))
    return float(np.clip(mu, 8.0, 70.0)), float(np.clip(delta, 3.5, 22.0)), final_bod


def aerobic_nh3_truth_parameters(
    nh3_mgL: float,
    bod_mgL: float,
    kla_d1: float,
    kinetic_params: dict[str, float],
) -> tuple[float, float, float]:
    u_pao = kinetic_params["uPAO"]
    mu = 41.0 - 0.17 * (kla_d1 - 90.0) + 0.48 * nh3_mgL + 0.6 * (4.0 - u_pao)
    delta = 8.0 + 0.018 * nh3_mgL + 0.012 * (140.0 - kla_d1)
    score = 0.065 * (kla_d1 - 100.0) + 0.05 * (nh3_mgL - 14.0) + 1.0 * (u_pao - 3.92) - 0.015 * (
        bod_mgL - 40.0
    )
    removal = 0.45 + 0.45 * float(logistic(score))
    final_nh3 = max(0.05, nh3_mgL * (1.0 - removal))
    return float(np.clip(mu, 10.0, 80.0)), float(np.clip(delta, 4.0, 24.0)), final_nh3


def load_stage1_tables(stage1_output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decision_raw = pd.read_csv(stage1_output / "decision_dataset" / "decision_dataset_raw.csv", parse_dates=["timestamp"])
    labels = pd.read_csv(stage1_output / "decision_dataset" / "decision_labels_observed.csv", parse_dates=["timestamp"])
    plant = pd.read_csv(stage1_output / "plant_real_hourly" / "plant_real_hourly.csv", parse_dates=["timestamp"])
    return decision_raw, labels, plant


def build_repro_observed_base_table(
    decision_raw: pd.DataFrame,
    labels: pd.DataFrame,
    plant: pd.DataFrame,
    asm_config: dict[str, object],
) -> pd.DataFrame:
    observed = decision_raw.loc[
        decision_raw["scenario_tag"] == "observed",
        [
            "timestamp",
            "split",
            "source_flag",
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
        ],
    ].copy()
    observed = observed.merge(labels, on=["timestamp", "split", "source_flag"], how="inner", validate="one_to_one")
    plant_cols = [
        "timestamp",
        "influent_tn_mgL",
        "reactor_orp_mv",
        "reactor_no3_mgL",
        "internal_nh3_mgL",
        "effluent_cod_mgL",
        "effluent_nh3n_mgL",
        "effluent_tp_mgL",
        "effluent_tn_mgL",
        "effluent_flow_m3h",
        "effluent_ph",
        "effluent_temp_c",
    ]
    observed = observed.merge(plant[plant_cols], on="timestamp", how="left", validate="one_to_one")

    bod_proxy_factor = float(asm_config["proxies"]["effluent_bod_proxy_factor"])
    observed["effluent_bod_proxy_mgL"] = observed["effluent_cod_mgL"] * bod_proxy_factor
    observed["label_next_effluent_bod_proxy_mgL"] = observed["label_next_effluent_cod_mgL"] * bod_proxy_factor
    observed["aerobic_bod_proxy_mgL"] = (
        float(asm_config["proxies"].get("bod_proxy_intercept", 0.368762))
        + float(asm_config["proxies"].get("bod_proxy_from_effluent_bod", 0.821128)) * observed["effluent_bod_proxy_mgL"]
        + float(asm_config["proxies"].get("bod_proxy_from_influent_bod", 0.00015)) * observed["influent_bod_mgL"]
        + float(asm_config["proxies"].get("bod_proxy_from_influent_cod", 0.000332)) * observed["influent_cod_mgL"]
        + float(asm_config["proxies"].get("bod_proxy_from_chemical_kgph", 0.005232)) * observed["chemical_dose_kgph"]
        + float(asm_config["proxies"].get("bod_proxy_from_do", 0.003403)) * observed["reactor_do_mgL"]
    ).clip(lower=0.05)
    observed["anoxic_no3_proxy_mgL"] = np.maximum.reduce(
        [
            observed["reactor_no3_mgL"].fillna(0.0).to_numpy(),
            (observed["influent_tn_mgL"].fillna(0.0) - observed["influent_nh3n_mgL"].fillna(0.0)).clip(lower=0.05).to_numpy(),
            (observed["effluent_tn_mgL"].fillna(0.0) - observed["effluent_nh3n_mgL"].fillna(0.0)).clip(lower=0.05).to_numpy(),
        ]
    )
    internal_nh3 = observed["internal_nh3_mgL"].fillna(observed["influent_nh3n_mgL"])
    observed["aerobic_nh3_proxy_mgL"] = np.minimum(
        observed["influent_nh3n_mgL"],
        np.maximum(observed["effluent_nh3n_mgL"], internal_nh3),
    )

    required_cols = [
        "influent_bod_mgL",
        "influent_nh3n_mgL",
        "influent_tn_mgL",
        "reactor_do_mgL",
        "effluent_bod_proxy_mgL",
        "effluent_nh3n_mgL",
        "effluent_tn_mgL",
        "label_next_effluent_bod_proxy_mgL",
        "label_next_effluent_nh3n_mgL",
        "label_next_effluent_tn_mgL",
        "aerobic_bod_proxy_mgL",
        "anoxic_no3_proxy_mgL",
        "aerobic_nh3_proxy_mgL",
    ]
    observed = observed.dropna(subset=required_cols).sort_values("timestamp").reset_index(drop=True)
    return observed


def apply_mechanistic_proxy_mapping(observed: pd.DataFrame, asm_config: dict[str, object]) -> pd.DataFrame:
    design = asm_config["design"]
    proxies = asm_config["proxies"]
    design_kla = float(design["kla_base_d1"])
    design_nrr = float(design["nitrate_return_flow_m3h"])
    kla_base_factor = float(proxies.get("kla_base_factor", 0.72))
    kla_min_d1 = float(proxies.get("kla_min_d1", 60.0))
    kla_max_d1 = float(proxies.get("kla_max_d1", 160.0))
    nrr_bias = float(proxies.get("nrr_bias", 1.0))
    do_center = float(observed["reactor_do_mgL"].median())
    no3_center = float(observed["reactor_no3_mgL"].fillna(observed["anoxic_no3_proxy_mgL"]).median())
    flow_center = float(observed["influent_flow_m3h"].median())

    mapped = observed.copy()
    mapped["kla_proxy_d1"] = (
        design_kla * (kla_base_factor + float(proxies["kla_from_aeration_weight"]) * mapped["aeration_intensity_pct"].fillna(50.0) / 100.0)
        + float(proxies["kla_from_do_gain"]) * (mapped["reactor_do_mgL"].fillna(do_center) - do_center)
    ).clip(lower=kla_min_d1, upper=kla_max_d1)
    mapped["nrr_proxy_m3h"] = design_nrr * (
        nrr_bias
        + float(proxies["nrr_from_no3_gain"])
        * (mapped["reactor_no3_mgL"].fillna(no3_center) - no3_center)
        / max(no3_center, 0.2)
        + float(proxies["nrr_from_flow_gain"])
        * (mapped["influent_flow_m3h"] - flow_center)
        / max(flow_center, 1.0)
    )
    mapped["nrr_proxy_m3h"] = mapped["nrr_proxy_m3h"].clip(
        lower=design_nrr * float(proxies["nrr_min_factor"]),
        upper=design_nrr * float(proxies["nrr_max_factor"]),
    )
    return mapped


def build_repro_observed_table(
    decision_raw: pd.DataFrame,
    labels: pd.DataFrame,
    plant: pd.DataFrame,
    asm_config: dict[str, object],
) -> pd.DataFrame:
    observed = build_repro_observed_base_table(decision_raw, labels, plant, asm_config)
    return apply_mechanistic_proxy_mapping(observed, asm_config)


def select_calibration_subset(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.copy()
    idx = evenly_spaced_index(len(df), sample_size)
    return df.iloc[idx].reset_index(drop=True)


def mechanistic_batch_outputs(df: pd.DataFrame, kinetic_params: dict[str, float], asm_config: dict[str, object]) -> pd.DataFrame:
    anoxic_results = [
        anoxic_truth_parameters(
            bod_mgL=float(row.influent_bod_mgL),
            no3_mgL=float(row.anoxic_no3_proxy_mgL),
            nrr_m3h=float(row.nrr_proxy_m3h),
            kinetic_params=kinetic_params,
            asm_config=asm_config,
        )
        for row in df.itertuples(index=False)
    ]
    aerobic_bod_results = [
        aerobic_bod_truth_parameters(
            bod_mgL=float(row.aerobic_bod_proxy_mgL),
            kla_d1=float(row.kla_proxy_d1),
            kinetic_params=kinetic_params,
        )
        for row in df.itertuples(index=False)
    ]
    aerobic_nh3_results = [
        aerobic_nh3_truth_parameters(
            nh3_mgL=float(row.aerobic_nh3_proxy_mgL),
            bod_mgL=float(row.influent_bod_mgL),
            kla_d1=float(row.kla_proxy_d1),
            kinetic_params=kinetic_params,
        )
        for row in df.itertuples(index=False)
    ]

    output = pd.DataFrame(index=df.index)
    output["anoxic_mu"] = [item[0] for item in anoxic_results]
    output["anoxic_delta"] = [item[1] for item in anoxic_results]
    output["batch_final_no3_mgL"] = [item[2] for item in anoxic_results]
    output["aerobic_bod_mu"] = [item[0] for item in aerobic_bod_results]
    output["aerobic_bod_delta"] = [item[1] for item in aerobic_bod_results]
    output["batch_final_bod_proxy_mgL"] = [item[2] for item in aerobic_bod_results]
    output["aerobic_nh3_mu"] = [item[0] for item in aerobic_nh3_results]
    output["aerobic_nh3_delta"] = [item[1] for item in aerobic_nh3_results]
    output["batch_final_nh3_mgL"] = [item[2] for item in aerobic_nh3_results]
    return output


def mechanistic_predict_real(
    df: pd.DataFrame,
    kinetic_params: dict[str, float],
    asm_config: dict[str, object],
    horizon_min: float = 60.0,
) -> pd.DataFrame:
    mech = mechanistic_batch_outputs(df, kinetic_params, asm_config)
    gain_per_hour = float(asm_config["proxies"]["mechanistic_continuous_gain_per_hour"])
    blend = gain_per_hour * horizon_min / 60.0
    residual_current = np.maximum(
        0.05,
        df["effluent_tn_mgL"].to_numpy()
        - df["effluent_nh3n_mgL"].to_numpy()
        - 0.3 * df["reactor_no3_mgL"].fillna(df["anoxic_no3_proxy_mgL"]).to_numpy(),
    )
    tn_batch = mech["batch_final_nh3_mgL"].to_numpy() + mech["batch_final_no3_mgL"].to_numpy() + residual_current * np.exp(
        -0.15 * (df["nrr_proxy_m3h"].to_numpy() / float(asm_config["design"]["nitrate_return_flow_m3h"]) - 1.0)
    )

    pred = pd.DataFrame(index=df.index)
    pred["pred_bod_proxy_mgL"] = df["effluent_bod_proxy_mgL"] + blend * (mech["batch_final_bod_proxy_mgL"] - df["effluent_bod_proxy_mgL"])
    pred["pred_nh3_mgL"] = df["effluent_nh3n_mgL"] + blend * (mech["batch_final_nh3_mgL"] - df["effluent_nh3n_mgL"])
    pred["pred_tn_mgL"] = df["effluent_tn_mgL"] + blend * (tn_batch - df["effluent_tn_mgL"])
    return pred.clip(lower=0.0)


def weighted_loss(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    loss = 0.0
    for target in REAL_TARGET_COLUMNS:
        short = target.replace("label_next_effluent_", "").replace("_proxy", "")
        pred_col = {
            "bod_mgL": "pred_bod_proxy_mgL",
            "nh3n_mgL": "pred_nh3_mgL",
            "tn_mgL": "pred_tn_mgL",
        }
        pred_key = pred_col[short]
        scale = max(float(y_true[target].std(ddof=0)), 1e-6)
        loss += REAL_TARGET_WEIGHTS[target] * mean_absolute_error(y_true[target], y_pred[pred_key]) / scale
    return float(loss)


def target_prediction_context(target: str) -> tuple[str, str]:
    short = target.replace("label_next_effluent_", "").replace("_proxy", "")
    pred_key = {
        "bod_mgL": "pred_bod_proxy_mgL",
        "nh3n_mgL": "pred_nh3_mgL",
        "tn_mgL": "pred_tn_mgL",
    }[short]
    current_key = {
        "bod_mgL": "effluent_bod_proxy_mgL",
        "nh3n_mgL": "effluent_nh3n_mgL",
        "tn_mgL": "effluent_tn_mgL",
    }[short]
    return pred_key, current_key


def weighted_delta_loss(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    loss = 0.0
    for target in REAL_TARGET_COLUMNS:
        pred_key, current_key = target_prediction_context(target)
        actual_delta = y_true[target] - y_true[current_key]
        pred_delta = y_pred[pred_key] - y_true[current_key]
        scale = max(float(actual_delta.std(ddof=0)), 1e-3)
        loss += REAL_TARGET_WEIGHTS[target] * mean_absolute_error(actual_delta, pred_delta) / scale
    return float(loss)


def gamma_change_score(df: pd.DataFrame) -> pd.Series:
    score = np.zeros(len(df), dtype=float)
    for target in REAL_TARGET_COLUMNS:
        _, current_key = target_prediction_context(target)
        actual_delta = (df[target] - df[current_key]).abs().to_numpy()
        scale = max(float(np.std(actual_delta, ddof=0)), 1e-3)
        score += REAL_TARGET_WEIGHTS[target] * actual_delta / scale
    return pd.Series(score, index=df.index)


def select_gamma_calibration_subset(
    df: pd.DataFrame,
    sample_size: int,
    asm_config: dict[str, object],
) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.copy()
    calibration = asm_config["calibration"]
    focus_share = float(calibration.get("gamma_focus_share", 0.7))
    focus_quantile = float(calibration.get("gamma_dynamic_focus_quantile", 0.7))
    dynamic_score = gamma_change_score(df)
    threshold = float(dynamic_score.quantile(focus_quantile))
    dynamic_idx = list(dynamic_score[dynamic_score >= threshold].sort_values(ascending=False).index)
    focus_size = min(int(round(sample_size * focus_share)), len(dynamic_idx))
    selected = dynamic_idx[:focus_size]
    if len(selected) < sample_size:
        remainder = df.drop(index=selected)
        extra_count = min(sample_size - len(selected), len(remainder))
        extra_idx = remainder.iloc[evenly_spaced_index(len(remainder), extra_count)].index.tolist()
        selected.extend(extra_idx)
    selected = sorted(set(selected))
    return df.loc[selected].reset_index(drop=True)


def gamma_objective_loss(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    gamma: float,
    asm_config: dict[str, object],
) -> float:
    calibration = asm_config["calibration"]
    level_weight = float(calibration.get("gamma_level_loss_weight", 0.4))
    delta_weight = float(calibration.get("gamma_delta_loss_weight", 0.6))
    prior = float(calibration.get("gamma_prior", 0.35))
    prior_weight = float(calibration.get("gamma_prior_weight", 0.03))
    return (
        level_weight * weighted_loss(y_true, y_pred)
        + delta_weight * weighted_delta_loss(y_true, y_pred)
        + prior_weight * abs(float(gamma) - prior)
    )


def estimate_gamma_target_scale_factors(
    calibration_df: pd.DataFrame,
    surface_models: dict[str, SurfaceModel],
    asm_config: dict[str, object],
) -> dict[str, float]:
    calibration = asm_config["calibration"]
    previous_factors = calibration.pop("gamma_target_scale_factors", None)
    try:
        raw_pred = data_driven_predict_real(calibration_df, surface_models, 1.0, asm_config)
    finally:
        if previous_factors is not None:
            calibration["gamma_target_scale_factors"] = previous_factors

    min_scale = float(calibration.get("gamma_target_scale_min", 0.02))
    max_scale = float(calibration.get("gamma_target_scale_max", 0.35))
    factors: dict[str, float] = {}
    for target in REAL_TARGET_COLUMNS:
        pred_key, current_key = target_prediction_context(target)
        actual_delta = (calibration_df[target] - calibration_df[current_key]).abs().to_numpy()
        raw_delta = (raw_pred[pred_key] - calibration_df[current_key]).abs().to_numpy()
        actual_q = max(float(np.quantile(actual_delta, 0.75)), 1e-6)
        raw_q = max(float(np.quantile(raw_delta, 0.75)), 1e-6)
        factors[pred_key] = float(np.clip(actual_q / raw_q, min_scale, max_scale))
    return factors


def estimate_gamma_bridge_coefficients(
    calibration_df: pd.DataFrame,
    surface_models: dict[str, SurfaceModel],
    asm_config: dict[str, object],
) -> dict[str, dict[str, float]]:
    calibration = asm_config["calibration"]
    previous_coeffs = calibration.pop("gamma_bridge_coefficients", None)
    previous_factors = calibration.pop("gamma_target_scale_factors", None)
    try:
        raw_pred = data_driven_predict_real(calibration_df, surface_models, 1.0, asm_config)
    finally:
        if previous_coeffs is not None:
            calibration["gamma_bridge_coefficients"] = previous_coeffs
        if previous_factors is not None:
            calibration["gamma_target_scale_factors"] = previous_factors

    max_abs_slope = float(calibration.get("gamma_bridge_max_abs_slope", 0.12))
    coeffs: dict[str, dict[str, float]] = {}
    for target in REAL_TARGET_COLUMNS:
        pred_key, current_key = target_prediction_context(target)
        actual_delta = (calibration_df[target] - calibration_df[current_key]).to_numpy()
        raw_delta = (raw_pred[pred_key] - calibration_df[current_key]).to_numpy()
        raw_centered = raw_delta - raw_delta.mean()
        actual_centered = actual_delta - actual_delta.mean()
        denom = max(float(np.dot(raw_centered, raw_centered)), 1e-9)
        slope = float(np.dot(actual_centered, raw_centered) / denom)
        slope = float(np.clip(slope, -max_abs_slope, max_abs_slope))
        intercept = float(actual_delta.mean() - slope * raw_delta.mean())
        coeffs[pred_key] = {"slope": slope, "intercept": intercept}
    return coeffs


def mechanistic_proxy_params_from_config(asm_config: dict[str, object]) -> dict[str, float]:
    proxies = asm_config["proxies"]
    return {
        "kla_base_factor": float(proxies.get("kla_base_factor", 0.72)),
        "kla_from_aeration_weight": float(proxies["kla_from_aeration_weight"]),
        "kla_from_do_gain": float(proxies["kla_from_do_gain"]),
        "nrr_bias": float(proxies.get("nrr_bias", 1.0)),
        "nrr_from_no3_gain": float(proxies["nrr_from_no3_gain"]),
        "nrr_from_flow_gain": float(proxies["nrr_from_flow_gain"]),
        "mechanistic_continuous_gain_per_hour": float(proxies["mechanistic_continuous_gain_per_hour"]),
    }


def with_mechanistic_proxy_params(
    asm_config: dict[str, object],
    proxy_params: dict[str, float],
) -> dict[str, object]:
    updated = copy.deepcopy(asm_config)
    proxies = updated["proxies"]
    for key, value in proxy_params.items():
        proxies[key] = float(value)
    return updated


def calibrate_mechanistic_proxy_parameters(
    base_observed_df: pd.DataFrame,
    asm_config: dict[str, object],
    kinetic_params: dict[str, float],
) -> dict[str, object]:
    calibration = asm_config["calibration"]
    sample_size = int(calibration.get("mechanistic_proxy_train_sample_size", calibration["train_sample_size"]))
    population = int(calibration.get("mechanistic_proxy_population_size", calibration["population_size"]))
    iterations = int(calibration.get("mechanistic_proxy_iterations", calibration["iterations"]))
    bounds = calibration["mechanistic_proxy_bounds"]
    prior_weight = float(calibration.get("mechanistic_proxy_prior_weight", 0.0))
    energy_anchor_weight = float(calibration.get("mechanistic_proxy_energy_anchor_weight", 0.0))
    base_train = base_observed_df.loc[base_observed_df["split"] == "train"].reset_index(drop=True)
    initial_params = mechanistic_proxy_params_from_config(asm_config)
    param_names = list(initial_params.keys())
    reference_train = apply_mechanistic_proxy_mapping(base_train, asm_config)
    ref_kla_mean = float(reference_train["kla_proxy_d1"].mean())
    ref_nrr_mean = float(reference_train["nrr_proxy_m3h"].mean())

    problem = Problem(len(param_names), 1)
    problem.types[:] = [Real(*bounds[name]) for name in param_names]

    def objective(vars_: list[float]) -> list[float]:
        proxy_params = {name: float(value) for name, value in zip(param_names, vars_, strict=True)}
        candidate_config = with_mechanistic_proxy_params(asm_config, proxy_params)
        candidate_train = apply_mechanistic_proxy_mapping(base_train, candidate_config)
        calibration_df = select_calibration_subset(candidate_train, sample_size)
        pred = mechanistic_predict_real(calibration_df, kinetic_params, candidate_config)
        loss = weighted_loss(calibration_df, pred)
        if prior_weight > 0.0:
            penalty = 0.0
            for name in param_names:
                lower, upper = bounds[name]
                scale = max(float(upper) - float(lower), 1e-6)
                penalty += abs(proxy_params[name] - initial_params[name]) / scale
            loss += prior_weight * penalty / len(param_names)
        if energy_anchor_weight > 0.0:
            anchor_penalty = (
                abs(float(candidate_train["kla_proxy_d1"].mean()) - ref_kla_mean) / max(ref_kla_mean, 1e-6)
                + abs(float(candidate_train["nrr_proxy_m3h"].mean()) - ref_nrr_mean) / max(ref_nrr_mean, 1e-6)
            )
            loss += energy_anchor_weight * anchor_penalty
        return [float(loss)]

    problem.function = objective
    algorithm = GeneticAlgorithm(problem, population_size=population)
    start = time.perf_counter()
    algorithm.run(population * iterations)
    elapsed = time.perf_counter() - start
    best = min(algorithm.result, key=lambda s: s.objectives[0])
    params = {name: float(value) for name, value in zip(param_names, best.variables, strict=True)}
    return {
        "params": params,
        "loss": float(best.objectives[0]),
        "elapsed_sec": float(elapsed),
        "sample_size": int(min(sample_size, len(base_train))),
    }


def calibrate_mechanistic_parameters(
    train_df: pd.DataFrame,
    asm_config: dict[str, object],
) -> dict[str, object]:
    sample_size = int(asm_config["calibration"]["train_sample_size"])
    calibration_df = select_calibration_subset(train_df, sample_size)
    bounds = asm_config["kinetic_bounds"]
    population = int(asm_config["calibration"]["population_size"])
    iterations = int(asm_config["calibration"]["iterations"])

    problem = Problem(3, 1)
    problem.types[:] = [
        Real(*bounds["kH"]),
        Real(*bounds["muH"]),
        Real(*bounds["uPAO"]),
    ]

    def objective(vars_: list[float]) -> list[float]:
        params = {"kH": float(vars_[0]), "muH": float(vars_[1]), "uPAO": float(vars_[2])}
        pred = mechanistic_predict_real(calibration_df, params, asm_config)
        loss = weighted_loss(calibration_df, pred)
        return [loss]

    problem.function = objective
    algorithm = GeneticAlgorithm(problem, population_size=population)
    start = time.perf_counter()
    algorithm.run(population * iterations)
    elapsed = time.perf_counter() - start
    best = min(algorithm.result, key=lambda s: s.objectives[0])
    params = {
        "kH": float(best.variables[0]),
        "muH": float(best.variables[1]),
        "uPAO": float(best.variables[2]),
    }
    return {
        "params": params,
        "loss": float(best.objectives[0]),
        "elapsed_sec": float(elapsed),
        "sample_size": int(len(calibration_df)),
    }


def time_grid_from_config(asm_config: dict[str, object]) -> np.ndarray:
    duration_min = float(asm_config["batch_time"]["duration_min"])
    dt_min = float(asm_config["batch_time"]["dt_sec"]) / 60.0
    steps = int(duration_min / dt_min) + 1
    return np.linspace(0.0, duration_min, num=steps)


def arange_inclusive(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step)) + 1
    return start + np.arange(count) * step


def generate_batch_datasets(
    kinetic_params: dict[str, float],
    asm_config: dict[str, object],
    batch_config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(asm_config["calibration"]["random_state"]))
    time_min = time_grid_from_config(asm_config)
    noise_frac = float(asm_config["preprocessing"]["sensor_noise_frac"])
    keep_ratio = float(asm_config["preprocessing"]["fourier_keep_ratio"])
    trim_points = int(asm_config["preprocessing"]["derivative_trim_points"])

    profile_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    batch_id = 0

    anoxic_grid = batch_config["anoxic_grid"]
    cn_values = arange_inclusive(anoxic_grid["cn_ratio_start"], anoxic_grid["cn_ratio_stop"], anoxic_grid["cn_ratio_step"])
    no3_values = arange_inclusive(anoxic_grid["no3_start"], anoxic_grid["no3_stop"], anoxic_grid["no3_step"])
    design_nrr = float(asm_config["design"]["nitrate_return_flow_m3h"])
    for cn_ratio in cn_values:
        for no3 in no3_values:
            bod = cn_ratio * no3
            mu, delta, final_value = anoxic_truth_parameters(bod, no3, design_nrr, kinetic_params, asm_config)
            amplitude = amplitude_from_endpoints(no3, final_value, time_min, mu, delta)
            concentration, _ = solve_profile_odeint(no3, amplitude, mu, delta, time_min)
            noisy = concentration + rng.normal(0.0, noise_frac * max(no3, 1.0), size=len(concentration))
            extracted = fit_gaussian_feature_parameters(time_min, noisy, keep_ratio, trim_points)
            current_batch_id = batch_id
            batch_id += 1
            for t_value, conc_value in zip(time_min, noisy, strict=True):
                profile_rows.append(
                    {
                        "batch_id": current_batch_id,
                        "phase": "anoxic_no3",
                        "substrate": "no3",
                        "time_min": float(t_value),
                        "concentration_mgL": float(max(conc_value, 0.0)),
                    }
                )
            summary_rows.append(
                {
                    "batch_id": current_batch_id,
                    "phase": "anoxic_no3",
                    "substrate": "no3",
                    "anoxic_cn_ratio": float(cn_ratio),
                    "anoxic_bod_mgL": float(bod),
                    "anoxic_no3_mgL": float(no3),
                    "nrr_m3h": float(design_nrr),
                    "A_true": float(amplitude),
                    "mu_true": float(mu),
                    "delta_true": float(delta),
                    "A": extracted["A"],
                    "mu": extracted["mu"],
                    "delta": extracted["delta"],
                    "gaussian_r2": extracted["gaussian_r2"],
                    "gaussian_rmse": extracted["gaussian_rmse"],
                }
            )

    aerobic_grid = batch_config["aerobic_grid"]
    kla_values = arange_inclusive(aerobic_grid["kla_start"], aerobic_grid["kla_stop"], aerobic_grid["kla_step"])
    nh3_values = arange_inclusive(aerobic_grid["nh3_start"], aerobic_grid["nh3_stop"], aerobic_grid["nh3_step"])
    bod_values = arange_inclusive(aerobic_grid["bod_start"], aerobic_grid["bod_stop"], aerobic_grid["bod_step"])
    pair_count = min(len(kla_values) * len(nh3_values), len(kla_values) * len(bod_values))
    for idx in range(pair_count):
        kla = float(kla_values[idx % len(kla_values)])
        nh3 = float(nh3_values[idx % len(nh3_values)])
        bod = float(bod_values[idx % len(bod_values)])

        mu_bod, delta_bod, final_bod = aerobic_bod_truth_parameters(bod, kla, kinetic_params)
        amp_bod = amplitude_from_endpoints(bod, final_bod, time_min, mu_bod, delta_bod)
        bod_concentration, _ = solve_profile_odeint(bod, amp_bod, mu_bod, delta_bod, time_min)
        bod_noisy = bod_concentration + rng.normal(0.0, noise_frac * max(bod, 1.0), size=len(bod_concentration))
        bod_extracted = fit_gaussian_feature_parameters(time_min, bod_noisy, keep_ratio, trim_points)

        mu_nh3, delta_nh3, final_nh3 = aerobic_nh3_truth_parameters(nh3, bod, kla, kinetic_params)
        amp_nh3 = amplitude_from_endpoints(nh3, final_nh3, time_min, mu_nh3, delta_nh3)
        nh3_concentration, _ = solve_profile_odeint(nh3, amp_nh3, mu_nh3, delta_nh3, time_min)
        nh3_noisy = nh3_concentration + rng.normal(0.0, noise_frac * max(nh3, 1.0), size=len(nh3_concentration))
        nh3_extracted = fit_gaussian_feature_parameters(time_min, nh3_noisy, keep_ratio, trim_points)

        current_bod_batch_id = batch_id
        batch_id += 1
        current_nh3_batch_id = batch_id
        batch_id += 1

        for t_value, conc_value in zip(time_min, bod_noisy, strict=True):
            profile_rows.append(
                {
                    "batch_id": current_bod_batch_id,
                    "phase": "aerobic_bod",
                    "substrate": "bod",
                    "time_min": float(t_value),
                    "concentration_mgL": float(max(conc_value, 0.0)),
                }
            )
        for t_value, conc_value in zip(time_min, nh3_noisy, strict=True):
            profile_rows.append(
                {
                    "batch_id": current_nh3_batch_id,
                    "phase": "aerobic_nh3",
                    "substrate": "nh3",
                    "time_min": float(t_value),
                    "concentration_mgL": float(max(conc_value, 0.0)),
                }
            )

        summary_rows.extend(
            [
                {
                    "batch_id": current_bod_batch_id,
                    "phase": "aerobic_bod",
                    "substrate": "bod",
                    "kla_d1": kla,
                    "aerobic_bod_mgL": bod,
                    "A_true": float(amp_bod),
                    "mu_true": float(mu_bod),
                    "delta_true": float(delta_bod),
                    "A": bod_extracted["A"],
                    "mu": bod_extracted["mu"],
                    "delta": bod_extracted["delta"],
                    "gaussian_r2": bod_extracted["gaussian_r2"],
                    "gaussian_rmse": bod_extracted["gaussian_rmse"],
                },
                {
                    "batch_id": current_nh3_batch_id,
                    "phase": "aerobic_nh3",
                    "substrate": "nh3",
                    "kla_d1": kla,
                    "aerobic_nh3_mgL": nh3,
                    "A_true": float(amp_nh3),
                    "mu_true": float(mu_nh3),
                    "delta_true": float(delta_nh3),
                    "A": nh3_extracted["A"],
                    "mu": nh3_extracted["mu"],
                    "delta": nh3_extracted["delta"],
                    "gaussian_r2": nh3_extracted["gaussian_r2"],
                    "gaussian_rmse": nh3_extracted["gaussian_rmse"],
                },
            ]
        )

    profiles = pd.DataFrame(profile_rows)
    summary = pd.DataFrame(summary_rows)

    train_fraction = float(batch_config["batch_split"]["train_fraction"])
    summary["batch_order"] = summary.groupby("phase").cumcount()
    summary["phase_count"] = summary.groupby("phase")["batch_id"].transform("count")
    summary["split"] = np.where(
        summary["batch_order"] < np.floor(summary["phase_count"] * train_fraction),
        "train",
        "test",
    )
    summary = summary.drop(columns=["batch_order", "phase_count"])
    return profiles, summary


def model_formula_string(poly: PolynomialFeatures, reg: LinearRegression, input_cols: list[str], name: str) -> str:
    feature_names = poly.get_feature_names_out(input_cols)
    coef_terms = []
    for coef, fname in zip(reg.coef_, feature_names, strict=True):
        coef_terms.append(f"{coef:.6f}*{fname}")
    return f"{name} = {reg.intercept_:.6f} + " + " + ".join(coef_terms)


def fit_surface_models(batch_summary: pd.DataFrame, batch_config: dict[str, object]) -> tuple[dict[str, SurfaceModel], pd.DataFrame]:
    models: dict[str, SurfaceModel] = {}
    report_rows: list[dict[str, object]] = []
    for phase, param_name in SURFACE_MODEL_ORDER:
        cfg = batch_config["feature_models"][phase][param_name]
        input_cols = list(cfg["inputs"])
        degree = int(cfg["degree"])
        train_df = batch_summary.loc[(batch_summary["phase"] == phase) & (batch_summary["split"] == "train")].copy()
        target = train_df[param_name].to_numpy()
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X = poly.fit_transform(train_df[input_cols])
        reg = LinearRegression()
        reg.fit(X, target)
        pred = reg.predict(X)
        key = f"{phase}:{param_name}"
        formula = model_formula_string(poly, reg, input_cols, key)
        metrics = {
            "r2": safe_r2(target, pred),
            "rmse": rmse(target, pred),
        }
        models[key] = SurfaceModel(
            name=key,
            input_cols=input_cols,
            degree=degree,
            poly=poly,
            reg=reg,
            metrics=metrics,
            formula=formula,
        )
        report_rows.append(
            {
                "name": key,
                "degree": degree,
                "inputs": ",".join(input_cols),
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "formula": formula,
            }
        )
    return models, pd.DataFrame(report_rows)


def predict_surface_models(models: dict[str, SurfaceModel], frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for key, model in models.items():
        X = model.poly.transform(frame[model.input_cols])
        out[key] = model.reg.predict(X)
    return out


def evaluate_batch_reconstruction(
    batch_profiles: pd.DataFrame,
    batch_summary: pd.DataFrame,
    surface_models: dict[str, SurfaceModel],
    asm_config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_min = time_grid_from_config(asm_config)
    results: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    summary_test = batch_summary.loc[batch_summary["split"] == "test"].copy()
    prediction_inputs = pd.DataFrame(index=summary_test.index)
    for col in ["anoxic_bod_mgL", "anoxic_no3_mgL", "kla_d1", "aerobic_bod_mgL", "aerobic_nh3_mgL"]:
        if col in summary_test.columns:
            prediction_inputs[col] = summary_test[col]
    predicted_params = predict_surface_models(surface_models, prediction_inputs.fillna(0.0))

    for row_idx, row in summary_test.iterrows():
        phase = row["phase"]
        if phase == "anoxic_no3":
            key_prefix = "anoxic_no3"
            initial_value = float(row["anoxic_no3_mgL"])
        elif phase == "aerobic_bod":
            key_prefix = "aerobic_bod"
            initial_value = float(row["aerobic_bod_mgL"])
        else:
            key_prefix = "aerobic_nh3"
            initial_value = float(row["aerobic_nh3_mgL"])

        amplitude = float(predicted_params.loc[row_idx, f"{key_prefix}:A"])
        mu = float(predicted_params.loc[row_idx, f"{key_prefix}:mu"])
        delta = float(predicted_params.loc[row_idx, f"{key_prefix}:delta"])
        profile_true = batch_profiles.loc[batch_profiles["batch_id"] == row["batch_id"]].sort_values("time_min")
        profile_pred, _ = solve_profile_odeint(initial_value, amplitude, mu, delta, time_min)
        y_true = profile_true["concentration_mgL"].to_numpy()
        y_pred = profile_pred[: len(y_true)]
        result_row = {
            "batch_id": int(row["batch_id"]),
            "phase": phase,
            "substrate": row["substrate"],
            "r2": safe_r2(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
        }
        results.append(result_row)
        if len(example_rows) < 40:
            for t_value, truth_value, pred_value in zip(profile_true["time_min"], y_true, y_pred, strict=True):
                example_rows.append(
                    {
                        "batch_id": int(row["batch_id"]),
                        "phase": phase,
                        "substrate": row["substrate"],
                        "time_min": float(t_value),
                        "truth_concentration_mgL": float(truth_value),
                        "pred_concentration_mgL": float(pred_value),
                    }
                )
    return pd.DataFrame(results), pd.DataFrame(example_rows)


def data_driven_predict_real(
    df: pd.DataFrame,
    surface_models: dict[str, SurfaceModel],
    gamma: float,
    asm_config: dict[str, object],
    horizon_min: float = 60.0,
) -> pd.DataFrame:
    prediction_frame = pd.DataFrame(
        {
            "anoxic_bod_mgL": df["influent_bod_mgL"],
            "anoxic_no3_mgL": df["anoxic_no3_proxy_mgL"],
            "kla_d1": df["kla_proxy_d1"],
            "aerobic_bod_mgL": df["aerobic_bod_proxy_mgL"],
            "aerobic_nh3_mgL": df["aerobic_nh3_proxy_mgL"],
        },
        index=df.index,
    )
    params_pred = predict_surface_models(surface_models, prediction_frame)
    time_min = time_grid_from_config(asm_config)

    batch_final_no3 = final_from_feature_params(
        df["anoxic_no3_proxy_mgL"].to_numpy(),
        params_pred["anoxic_no3:A"].to_numpy(),
        params_pred["anoxic_no3:mu"].to_numpy(),
        params_pred["anoxic_no3:delta"].to_numpy(),
        time_min,
    )
    batch_final_bod = final_from_feature_params(
        df["aerobic_bod_proxy_mgL"].to_numpy(),
        params_pred["aerobic_bod:A"].to_numpy(),
        params_pred["aerobic_bod:mu"].to_numpy(),
        params_pred["aerobic_bod:delta"].to_numpy(),
        time_min,
    )
    batch_final_nh3 = final_from_feature_params(
        df["aerobic_nh3_proxy_mgL"].to_numpy(),
        params_pred["aerobic_nh3:A"].to_numpy(),
        params_pred["aerobic_nh3:mu"].to_numpy(),
        params_pred["aerobic_nh3:delta"].to_numpy(),
        time_min,
    )
    gamma_scaled = float(gamma) * horizon_min / 60.0
    residual_current = np.maximum(
        0.05,
        df["effluent_tn_mgL"].to_numpy()
        - df["effluent_nh3n_mgL"].to_numpy()
        - 0.3 * df["reactor_no3_mgL"].fillna(df["anoxic_no3_proxy_mgL"]).to_numpy(),
    )
    tn_batch = batch_final_nh3 + batch_final_no3 + residual_current * np.exp(
        -0.15 * (df["nrr_proxy_m3h"].to_numpy() / float(asm_config["design"]["nitrate_return_flow_m3h"]) - 1.0)
    )

    calibration = asm_config.get("calibration", {})
    scale_factors = calibration.get("gamma_target_scale_factors", {})
    bridge_coeffs = calibration.get("gamma_bridge_coefficients", {})
    bod_scale = float(scale_factors.get("pred_bod_proxy_mgL", 1.0))
    nh3_scale = float(scale_factors.get("pred_nh3_mgL", 1.0))
    tn_scale = float(scale_factors.get("pred_tn_mgL", 1.0))
    bod_coeff = bridge_coeffs.get("pred_bod_proxy_mgL", {})
    nh3_coeff = bridge_coeffs.get("pred_nh3_mgL", {})
    tn_coeff = bridge_coeffs.get("pred_tn_mgL", {})
    bod_delta = bod_scale * (batch_final_bod - df["effluent_bod_proxy_mgL"]) if not bod_coeff else (
        float(bod_coeff.get("intercept", 0.0)) + float(bod_coeff.get("slope", 1.0)) * (batch_final_bod - df["effluent_bod_proxy_mgL"])
    )
    nh3_delta = nh3_scale * (batch_final_nh3 - df["effluent_nh3n_mgL"]) if not nh3_coeff else (
        float(nh3_coeff.get("intercept", 0.0)) + float(nh3_coeff.get("slope", 1.0)) * (batch_final_nh3 - df["effluent_nh3n_mgL"])
    )
    tn_delta = tn_scale * (tn_batch - df["effluent_tn_mgL"]) if not tn_coeff else (
        float(tn_coeff.get("intercept", 0.0)) + float(tn_coeff.get("slope", 1.0)) * (tn_batch - df["effluent_tn_mgL"])
    )

    pred = pd.DataFrame(index=df.index)
    pred["pred_bod_proxy_mgL"] = df["effluent_bod_proxy_mgL"] + gamma_scaled * bod_delta
    pred["pred_nh3_mgL"] = df["effluent_nh3n_mgL"] + gamma_scaled * nh3_delta
    pred["pred_tn_mgL"] = df["effluent_tn_mgL"] + gamma_scaled * tn_delta
    return pred.clip(lower=0.0)


def calibrate_gamma(
    train_df: pd.DataFrame,
    surface_models: dict[str, SurfaceModel],
    asm_config: dict[str, object],
) -> dict[str, object]:
    sample_size = int(asm_config["calibration"]["train_sample_size"])
    calibration_df = select_gamma_calibration_subset(train_df, sample_size, asm_config)
    gamma_bounds = asm_config["calibration"]["gamma_bounds"]
    calibration = asm_config["calibration"]
    scale_factors = estimate_gamma_target_scale_factors(calibration_df, surface_models, asm_config)
    bridge_coeffs = estimate_gamma_bridge_coefficients(calibration_df, surface_models, asm_config)
    calibration["gamma_target_scale_factors"] = scale_factors
    calibration["gamma_bridge_coefficients"] = bridge_coeffs
    coarse_points = int(calibration.get("gamma_grid_points", 81))
    refine_span = float(calibration.get("gamma_refine_span", 0.08))
    refine_points = int(calibration.get("gamma_refine_points", 21))

    start = time.perf_counter()
    coarse_candidates = np.linspace(float(gamma_bounds[0]), float(gamma_bounds[1]), num=coarse_points)
    best_gamma = float(coarse_candidates[0])
    best_loss = float("inf")
    for gamma in coarse_candidates:
        pred = data_driven_predict_real(calibration_df, surface_models, float(gamma), asm_config)
        loss = gamma_objective_loss(calibration_df, pred, float(gamma), asm_config)
        if loss < best_loss:
            best_gamma = float(gamma)
            best_loss = float(loss)

    lower = max(float(gamma_bounds[0]), best_gamma - refine_span)
    upper = min(float(gamma_bounds[1]), best_gamma + refine_span)
    refine_candidates = np.linspace(lower, upper, num=refine_points)
    for gamma in refine_candidates:
        pred = data_driven_predict_real(calibration_df, surface_models, float(gamma), asm_config)
        loss = gamma_objective_loss(calibration_df, pred, float(gamma), asm_config)
        if loss < best_loss:
            best_gamma = float(gamma)
            best_loss = float(loss)
    elapsed = time.perf_counter() - start
    return {
        "gamma": float(best_gamma),
        "loss": float(best_loss),
        "elapsed_sec": float(elapsed),
        "sample_size": int(len(calibration_df)),
        "objective": "level_plus_delta_with_dynamic_focus",
        "target_scale_factors": scale_factors,
        "bridge_coefficients": bridge_coeffs,
    }


def build_real_prediction_frame(
    real_df: pd.DataFrame,
    pred: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    out = real_df[["timestamp", "split", "source_flag"]].copy()
    out["model_name"] = model_name
    out["actual_bod_proxy_mgL"] = real_df["label_next_effluent_bod_proxy_mgL"]
    out["actual_nh3_mgL"] = real_df["label_next_effluent_nh3n_mgL"]
    out["actual_tn_mgL"] = real_df["label_next_effluent_tn_mgL"]
    out["pred_bod_proxy_mgL"] = pred["pred_bod_proxy_mgL"]
    out["pred_nh3_mgL"] = pred["pred_nh3_mgL"]
    out["pred_tn_mgL"] = pred["pred_tn_mgL"]
    out["abs_err_bod_proxy_mgL"] = (out["pred_bod_proxy_mgL"] - out["actual_bod_proxy_mgL"]).abs()
    out["abs_err_nh3_mgL"] = (out["pred_nh3_mgL"] - out["actual_nh3_mgL"]).abs()
    out["abs_err_tn_mgL"] = (out["pred_tn_mgL"] - out["actual_tn_mgL"]).abs()
    return out


def evaluate_real_predictions(real_df: pd.DataFrame, pred: pd.DataFrame) -> dict[str, object]:
    metrics: dict[str, dict[str, float]] = {}
    for target, pred_col in [
        ("label_next_effluent_bod_proxy_mgL", "pred_bod_proxy_mgL"),
        ("label_next_effluent_nh3n_mgL", "pred_nh3_mgL"),
        ("label_next_effluent_tn_mgL", "pred_tn_mgL"),
    ]:
        y_true = real_df[target].to_numpy()
        y_pred = pred[pred_col].to_numpy()
        metrics[target] = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse(y_true, y_pred),
            "r2": safe_r2(y_true, y_pred),
        }
    weighted_rmse = sum(REAL_TARGET_WEIGHTS[target] * metrics[target]["rmse"] for target in REAL_TARGET_COLUMNS)
    return {"targets": metrics, "weighted_rmse": float(weighted_rmse)}


def optimization_row_from_series(row: pd.Series, kla_d1: float, nrr_m3h: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "influent_bod_mgL": [float(row["influent_bod_mgL"])],
            "influent_nh3n_mgL": [float(row["influent_nh3n_mgL"])],
            "influent_tn_mgL": [float(row["influent_tn_mgL"])],
            "reactor_no3_mgL": [float(row["reactor_no3_mgL"])],
            "internal_nh3_mgL": [float(row["internal_nh3_mgL"])],
            "anoxic_no3_proxy_mgL": [float(row["anoxic_no3_proxy_mgL"])],
            "aerobic_bod_proxy_mgL": [float(row["aerobic_bod_proxy_mgL"])],
            "aerobic_nh3_proxy_mgL": [float(row["aerobic_nh3_proxy_mgL"])],
            "effluent_bod_proxy_mgL": [float(row["effluent_bod_proxy_mgL"])],
            "effluent_nh3n_mgL": [float(row["effluent_nh3n_mgL"])],
            "effluent_tn_mgL": [float(row["effluent_tn_mgL"])],
            "chemical_dose_kgph": [float(row["chemical_dose_kgph"])],
            "influent_flow_m3h": [float(row["influent_flow_m3h"])],
            "kla_proxy_d1": [float(kla_d1)],
            "nrr_proxy_m3h": [float(nrr_m3h)],
        }
    )


def operating_cost_from_frame(
    frame: pd.DataFrame,
    kla_col: str,
    nrr_col: str,
    asm_config: dict[str, object],
) -> pd.Series:
    costing = asm_config["costing"]
    aeration_pct = (
        float(costing["aeration_from_kla_intercept"])
        + float(costing["aeration_from_kla_coef"]) * frame[kla_col]
    ).clip(
        lower=float(costing["aeration_pct_min"]),
        upper=float(costing["aeration_pct_max"]),
    )
    flow_ratio = frame["influent_flow_m3h"] / max(float(costing["influent_flow_reference_m3h"]), 1e-6)
    aeration_term = float(costing["aeration_weight"]) * (aeration_pct / max(float(costing["aeration_pct_reference"]), 1e-6)) * flow_ratio
    recycle_term = float(costing["recycle_weight"]) * (frame[nrr_col] / max(float(costing["nrr_reference_m3h"]), 1e-6))
    chemical_term = float(costing["chemical_weight"]) * (
        frame["chemical_dose_kgph"] / max(float(costing["chemical_dose_reference_kgph"]), 1e-6)
    )
    return aeration_term + recycle_term + chemical_term


def operating_cost_from_row(
    row: pd.Series,
    kla_d1: float,
    nrr_m3h: float,
    asm_config: dict[str, object],
) -> float:
    candidate = optimization_row_from_series(row, kla_d1, nrr_m3h)
    return float(operating_cost_from_frame(candidate, "kla_proxy_d1", "nrr_proxy_m3h", asm_config).iloc[0])


def build_steady_state_problem(
    representative_row: pd.Series,
    surface_models: dict[str, SurfaceModel],
    gamma: float,
    asm_config: dict[str, object],
    online_config: dict[str, object],
) -> Problem:
    steady_cfg = online_config["steady_state_optimization"]
    constraints = steady_cfg["constraints"]
    design = asm_config["design"]
    problem = Problem(2, 4, 3)
    problem.types[:] = [
        Real(float(asm_config["design"]["kla_base_d1"]), 140.0),
        Real(float(design["nitrate_return_flow_m3h"]) * 0.7, float(design["nitrate_return_flow_m3h"]) * 1.3),
    ]
    problem.constraints[:] = "<=0"

    def objective(vars_: list[float]) -> tuple[list[float], list[float]]:
        kla_d1 = float(vars_[0])
        nrr_m3h = float(vars_[1])
        candidate = optimization_row_from_series(representative_row, kla_d1, nrr_m3h)
        pred = data_driven_predict_real(candidate, surface_models, gamma, asm_config)
        energy = operating_cost_from_row(representative_row, kla_d1, nrr_m3h, asm_config)
        bod = float(pred["pred_bod_proxy_mgL"].iloc[0])
        nh3 = float(pred["pred_nh3_mgL"].iloc[0])
        tn = float(pred["pred_tn_mgL"].iloc[0])
        return [energy, bod, nh3, tn], [
            nh3 - float(constraints["nh3_upper"]),
            tn - float(constraints["tn_upper"]),
            bod - float(constraints["bod_proxy_upper"]),
        ]

    problem.function = objective
    return problem


def select_steady_solution(pareto_df: pd.DataFrame, online_config: dict[str, object]) -> dict[str, float]:
    weights = online_config["steady_state_optimization"]["selection_weights"]
    scale_cols = ["energy", "pred_bod_proxy_mgL", "pred_nh3_mgL", "pred_tn_mgL"]
    scored = pareto_df.copy()
    for col in scale_cols:
        span = max(scored[col].max() - scored[col].min(), 1e-9)
        scored[f"{col}_scaled"] = (scored[col] - scored[col].min()) / span
    scored["selection_score"] = (
        float(weights["energy"]) * scored["energy_scaled"]
        + float(weights["bod_proxy"]) * scored["pred_bod_proxy_mgL_scaled"]
        + float(weights["nh3"]) * scored["pred_nh3_mgL_scaled"]
        + float(weights["tn"]) * scored["pred_tn_mgL_scaled"]
    )
    best = scored.sort_values("selection_score").iloc[0]
    return {
        "kla_d1": float(best["kla_d1"]),
        "nrr_m3h": float(best["nrr_m3h"]),
        "selection_score": float(best["selection_score"]),
    }


def run_steady_state_optimization(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    surface_models: dict[str, SurfaceModel],
    gamma: float,
    kinetic_params: dict[str, float],
    asm_config: dict[str, object],
    online_config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, float], dict[str, object]]:
    representative = train_df.median(numeric_only=True)
    representative["effluent_bod_proxy_mgL"] = float(train_df["effluent_bod_proxy_mgL"].median())
    representative["effluent_nh3n_mgL"] = float(train_df["effluent_nh3n_mgL"].median())
    representative["effluent_tn_mgL"] = float(train_df["effluent_tn_mgL"].median())
    problem = build_steady_state_problem(representative, surface_models, gamma, asm_config, online_config)

    steady_cfg = online_config["steady_state_optimization"]
    algorithm = NSGAII(problem, population_size=int(steady_cfg["population_size"]))
    start = time.perf_counter()
    algorithm.run(int(steady_cfg["evaluations"]))
    elapsed = time.perf_counter() - start

    pareto_rows: list[dict[str, object]] = []
    for solution in algorithm.result:
        if not solution.feasible:
            continue
        pareto_rows.append(
            {
                "kla_d1": float(solution.variables[0]),
                "nrr_m3h": float(solution.variables[1]),
                "energy": float(solution.objectives[0]),
                "pred_bod_proxy_mgL": float(solution.objectives[1]),
                "pred_nh3_mgL": float(solution.objectives[2]),
                "pred_tn_mgL": float(solution.objectives[3]),
            }
        )
    pareto_df = pd.DataFrame(pareto_rows).sort_values("energy").reset_index(drop=True)
    selected = select_steady_solution(pareto_df, online_config)

    baseline_energy = operating_cost_from_frame(test_df, "kla_proxy_d1", "nrr_proxy_m3h", asm_config)
    baseline_pred = mechanistic_predict_real(test_df, kinetic_params, asm_config)
    steady_candidate = test_df.copy()
    steady_candidate["kla_proxy_d1"] = selected["kla_d1"]
    steady_candidate["nrr_proxy_m3h"] = selected["nrr_m3h"]
    steady_energy = operating_cost_from_frame(steady_candidate, "kla_proxy_d1", "nrr_proxy_m3h", asm_config).mean()
    steady_pred = mechanistic_predict_real(steady_candidate, kinetic_params, asm_config)

    constraints = online_config["steady_state_optimization"]["constraints"]
    baseline_compliant = (
        (baseline_pred["pred_bod_proxy_mgL"] <= float(constraints["bod_proxy_upper"]))
        & (baseline_pred["pred_nh3_mgL"] <= float(constraints["nh3_upper"]))
        & (baseline_pred["pred_tn_mgL"] <= float(constraints["tn_upper"]))
    )
    steady_compliant = (
        (steady_pred["pred_bod_proxy_mgL"] <= float(constraints["bod_proxy_upper"]))
        & (steady_pred["pred_nh3_mgL"] <= float(constraints["nh3_upper"]))
        & (steady_pred["pred_tn_mgL"] <= float(constraints["tn_upper"]))
    )

    evaluation = {
        "pareto_points": int(len(pareto_df)),
        "selected_solution": selected,
        "elapsed_sec": float(elapsed),
        "baseline_energy_mean": float(baseline_energy.mean()),
        "steady_energy_mean": float(steady_energy),
        "steady_energy_saving_pct": float((baseline_energy.mean() - steady_energy) / max(baseline_energy.mean(), 1e-6) * 100.0),
        "baseline_compliance_rate": float(baseline_compliant.mean()),
        "steady_compliance_rate": float(steady_compliant.mean()),
    }
    return pareto_df, selected, evaluation


def resample_online_replay(real_df: pd.DataFrame, online_config: dict[str, object]) -> pd.DataFrame:
    history_days = int(online_config["dynamic_control"]["history_days"])
    replay_interval = int(online_config["dynamic_control"]["replay_interval_min"])
    replay_source = real_df.sort_values("timestamp").tail(history_days * 24).copy()
    numeric_cols = [col for col in replay_source.columns if pd.api.types.is_numeric_dtype(replay_source[col])]
    replay = replay_source.set_index("timestamp")[numeric_cols].resample(f"{replay_interval}min").interpolate(method="time")
    replay = replay.reset_index()
    replay["actual_next_effluent_bod_proxy_mgL"] = replay["effluent_bod_proxy_mgL"].shift(-1)
    replay["actual_next_effluent_nh3n_mgL"] = replay["effluent_nh3n_mgL"].shift(-1)
    replay["actual_next_effluent_tn_mgL"] = replay["effluent_tn_mgL"].shift(-1)
    replay = replay.dropna(subset=["actual_next_effluent_bod_proxy_mgL", "actual_next_effluent_nh3n_mgL", "actual_next_effluent_tn_mgL"]).reset_index(drop=True)
    return replay


def dynamic_objective_problem(
    current_row: pd.Series,
    current_state: dict[str, float],
    surface_models: dict[str, SurfaceModel],
    gamma: float,
    asm_config: dict[str, object],
    online_config: dict[str, object],
    kla_center: float,
    nrr_center: float,
) -> Problem:
    dynamic_cfg = online_config["dynamic_control"]
    steady_constraints = online_config["steady_state_optimization"]["constraints"]
    kla_step = float(dynamic_cfg["kla_local_step_d1"])
    nrr_step = float(dynamic_cfg["nrr_local_step_m3h"])
    problem = Problem(2, 4, 3)
    problem.types[:] = [
        Real(max(60.0, kla_center - kla_step), min(140.0, kla_center + kla_step)),
        Real(
            max(float(asm_config["design"]["nitrate_return_flow_m3h"]) * 0.7, nrr_center - nrr_step),
            min(float(asm_config["design"]["nitrate_return_flow_m3h"]) * 1.3, nrr_center + nrr_step),
        ),
    ]
    problem.constraints[:] = "<=0"

    def objective(vars_: list[float]) -> tuple[list[float], list[float]]:
        kla_d1 = float(vars_[0])
        nrr_m3h = float(vars_[1])
        candidate = optimization_row_from_series(current_row, kla_d1, nrr_m3h)
        candidate["effluent_bod_proxy_mgL"] = current_state["bod_proxy"]
        candidate["effluent_nh3n_mgL"] = current_state["nh3"]
        candidate["effluent_tn_mgL"] = current_state["tn"]
        pred = data_driven_predict_real(candidate, surface_models, gamma, asm_config, horizon_min=float(dynamic_cfg["replay_interval_min"]))
        energy = operating_cost_from_row(current_row, kla_d1, nrr_m3h, asm_config)
        bod = float(pred["pred_bod_proxy_mgL"].iloc[0])
        nh3 = float(pred["pred_nh3_mgL"].iloc[0])
        tn = float(pred["pred_tn_mgL"].iloc[0])
        return [energy, bod, nh3, tn], [
            nh3 - float(steady_constraints["nh3_upper"]),
            tn - float(steady_constraints["tn_upper"]),
            bod - float(steady_constraints["bod_proxy_upper"]),
        ]

    problem.function = objective
    return problem


def one_step_mechanistic_transition(
    current_row: pd.Series,
    current_state: dict[str, float],
    kla_d1: float,
    nrr_m3h: float,
    kinetic_params: dict[str, float],
    asm_config: dict[str, object],
    horizon_min: float,
) -> dict[str, float]:
    current_df = optimization_row_from_series(current_row, kla_d1, nrr_m3h)
    current_df["effluent_bod_proxy_mgL"] = current_state["bod_proxy"]
    current_df["effluent_nh3n_mgL"] = current_state["nh3"]
    current_df["effluent_tn_mgL"] = current_state["tn"]
    pred = mechanistic_predict_real(current_df, kinetic_params, asm_config, horizon_min=horizon_min)
    return {
        "bod_proxy": float(pred["pred_bod_proxy_mgL"].iloc[0]),
        "nh3": float(pred["pred_nh3_mgL"].iloc[0]),
        "tn": float(pred["pred_tn_mgL"].iloc[0]),
    }


def select_dynamic_solution(result_solutions: list, online_config: dict[str, object]) -> tuple[float, float]:
    feasible_rows = []
    for solution in result_solutions:
        if not solution.feasible:
            continue
        feasible_rows.append(
            {
                "kla_d1": float(solution.variables[0]),
                "nrr_m3h": float(solution.variables[1]),
                "energy": float(solution.objectives[0]),
                "bod": float(solution.objectives[1]),
                "nh3": float(solution.objectives[2]),
                "tn": float(solution.objectives[3]),
            }
        )
    if not feasible_rows:
        first = result_solutions[0]
        return float(first.variables[0]), float(first.variables[1])
    feasible_df = pd.DataFrame(feasible_rows)
    weights = online_config["dynamic_control"].get(
        "selection_weights",
        {"energy": 1.0, "bod": 0.8, "nh3": 1.2, "tn": 1.2},
    )
    feasible_df["score"] = (
        float(weights["energy"]) * feasible_df["energy"].rank(pct=True)
        + float(weights["bod"]) * feasible_df["bod"].rank(pct=True)
        + float(weights["nh3"]) * feasible_df["nh3"].rank(pct=True)
        + float(weights["tn"]) * feasible_df["tn"].rank(pct=True)
    )
    best = feasible_df.sort_values("score").iloc[0]
    return float(best["kla_d1"]), float(best["nrr_m3h"])


def counterfactual_next_state(
    current_row: pd.Series,
    current_state: dict[str, float],
    candidate_kla: float,
    candidate_nrr: float,
    baseline_kla: float,
    baseline_nrr: float,
    surface_models: dict[str, SurfaceModel],
    gamma: float,
    asm_config: dict[str, object],
    horizon_min: float,
) -> dict[str, float]:
    baseline_actual_current = {
        "bod_proxy": float(current_row["effluent_bod_proxy_mgL"]),
        "nh3": float(current_row["effluent_nh3n_mgL"]),
        "tn": float(current_row["effluent_tn_mgL"]),
    }
    actual_next = {
        "bod_proxy": float(current_row["actual_next_effluent_bod_proxy_mgL"]),
        "nh3": float(current_row["actual_next_effluent_nh3n_mgL"]),
        "tn": float(current_row["actual_next_effluent_tn_mgL"]),
    }
    natural_delta = {
        key: actual_next[key] - baseline_actual_current[key]
        for key in actual_next
    }

    baseline_candidate = optimization_row_from_series(current_row, baseline_kla, baseline_nrr)
    baseline_candidate["effluent_bod_proxy_mgL"] = current_state["bod_proxy"]
    baseline_candidate["effluent_nh3n_mgL"] = current_state["nh3"]
    baseline_candidate["effluent_tn_mgL"] = current_state["tn"]
    baseline_pred = data_driven_predict_real(baseline_candidate, surface_models, gamma, asm_config, horizon_min=horizon_min)

    candidate = optimization_row_from_series(current_row, candidate_kla, candidate_nrr)
    candidate["effluent_bod_proxy_mgL"] = current_state["bod_proxy"]
    candidate["effluent_nh3n_mgL"] = current_state["nh3"]
    candidate["effluent_tn_mgL"] = current_state["tn"]
    candidate_pred = data_driven_predict_real(candidate, surface_models, gamma, asm_config, horizon_min=horizon_min)

    next_state = {
        "bod_proxy": current_state["bod_proxy"] + natural_delta["bod_proxy"] + float(candidate_pred["pred_bod_proxy_mgL"].iloc[0] - baseline_pred["pred_bod_proxy_mgL"].iloc[0]),
        "nh3": current_state["nh3"] + natural_delta["nh3"] + float(candidate_pred["pred_nh3_mgL"].iloc[0] - baseline_pred["pred_nh3_mgL"].iloc[0]),
        "tn": current_state["tn"] + natural_delta["tn"] + float(candidate_pred["pred_tn_mgL"].iloc[0] - baseline_pred["pred_tn_mgL"].iloc[0]),
    }
    return {key: max(float(value), 0.0) for key, value in next_state.items()}


def recalibrate_gamma_window(
    replay_history: pd.DataFrame,
    surface_models: dict[str, SurfaceModel],
    gamma_start: float,
    asm_config: dict[str, object],
) -> float:
    if len(replay_history) < 12:
        return float(gamma_start)
    candidate = replay_history.copy()
    actual_bod_col = "actual_next_effluent_bod_proxy_mgL" if "actual_next_effluent_bod_proxy_mgL" in candidate.columns else "actual_bod_proxy_mgL"
    actual_nh3_col = "actual_next_effluent_nh3n_mgL" if "actual_next_effluent_nh3n_mgL" in candidate.columns else "actual_nh3_mgL"
    actual_tn_col = "actual_next_effluent_tn_mgL" if "actual_next_effluent_tn_mgL" in candidate.columns else "actual_tn_mgL"
    candidate["label_next_effluent_bod_proxy_mgL"] = candidate[actual_bod_col]
    candidate["label_next_effluent_nh3n_mgL"] = candidate[actual_nh3_col]
    candidate["label_next_effluent_tn_mgL"] = candidate[actual_tn_col]
    calibration = asm_config["calibration"]
    window_size = int(calibration.get("gamma_window_size", 48))
    calibration_df = candidate.iloc[-min(window_size, len(candidate)) :].reset_index(drop=True)
    calibration_df = select_gamma_calibration_subset(calibration_df, min(len(calibration_df), 36), asm_config)
    calibration["gamma_target_scale_factors"] = estimate_gamma_target_scale_factors(calibration_df, surface_models, asm_config)
    calibration["gamma_bridge_coefficients"] = estimate_gamma_bridge_coefficients(calibration_df, surface_models, asm_config)
    gamma_bounds = calibration["gamma_bounds"]
    search_span = float(calibration.get("gamma_recalibration_span", 0.15))
    candidate_count = int(calibration.get("gamma_recalibration_points", 13))
    candidates = np.linspace(
        max(float(gamma_bounds[0]), gamma_start - search_span),
        min(float(gamma_bounds[1]), gamma_start + search_span),
        num=candidate_count,
    )
    best_gamma = float(gamma_start)
    best_loss = float("inf")
    for gamma in candidates:
        pred = data_driven_predict_real(calibration_df, surface_models, float(gamma), asm_config, horizon_min=30.0)
        loss = gamma_objective_loss(calibration_df, pred, float(gamma), asm_config)
        if loss < best_loss:
            best_loss = loss
            best_gamma = float(gamma)
    return best_gamma


def run_dynamic_replay(
    replay_df: pd.DataFrame,
    selected_steady: dict[str, float],
    surface_models: dict[str, SurfaceModel],
    gamma_value: float,
    kinetic_params: dict[str, float],
    asm_config: dict[str, object],
    online_config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    replay_interval = float(online_config["dynamic_control"]["replay_interval_min"])
    update_every_steps = int(online_config["dynamic_control"]["model_update_interval_h"] * 60 / replay_interval)
    modbus = ModbusMock()

    initial_state = {
        "bod_proxy": float(replay_df["effluent_bod_proxy_mgL"].iloc[0]),
        "nh3": float(replay_df["effluent_nh3n_mgL"].iloc[0]),
        "tn": float(replay_df["effluent_tn_mgL"].iloc[0]),
    }
    states = {
        "baseline": initial_state.copy(),
        "steady": initial_state.copy(),
        "dynamic": initial_state.copy(),
    }
    current_gamma = float(gamma_value)
    dynamic_center = {
        "kla_d1": float(replay_df["kla_proxy_d1"].iloc[0]),
        "nrr_m3h": float(replay_df["nrr_proxy_m3h"].iloc[0]),
    }
    rows: list[dict[str, object]] = []
    timings: list[float] = []

    for step, row in replay_df.iterrows():
        timestamp = row["timestamp"]
        baseline_kla = float(row["kla_proxy_d1"])
        baseline_nrr = float(row["nrr_proxy_m3h"])

        baseline_next = {
            "bod_proxy": float(row["actual_next_effluent_bod_proxy_mgL"]),
            "nh3": float(row["actual_next_effluent_nh3n_mgL"]),
            "tn": float(row["actual_next_effluent_tn_mgL"]),
        }
        steady_next = counterfactual_next_state(
            current_row=row,
            current_state=states["steady"],
            candidate_kla=float(selected_steady["kla_d1"]),
            candidate_nrr=float(selected_steady["nrr_m3h"]),
            baseline_kla=baseline_kla,
            baseline_nrr=baseline_nrr,
            surface_models=surface_models,
            gamma=current_gamma,
            asm_config=asm_config,
            horizon_min=replay_interval,
        )

        if step % update_every_steps == 0 and step > 0:
            history_df = replay_df.iloc[:step].copy().reset_index(drop=True)
            current_gamma = recalibrate_gamma_window(history_df, surface_models, current_gamma, asm_config)

        problem = dynamic_objective_problem(
            current_row=row,
            current_state=states["dynamic"],
            surface_models=surface_models,
            gamma=current_gamma,
            asm_config=asm_config,
            online_config=online_config,
            kla_center=dynamic_center["kla_d1"],
            nrr_center=dynamic_center["nrr_m3h"],
        )
        dynamic_cfg = online_config["dynamic_control"]
        start = time.perf_counter()
        algorithm = NSGAII(problem, population_size=int(dynamic_cfg["population_size"]))
        algorithm.run(int(dynamic_cfg["evaluations"]))
        timings.append(time.perf_counter() - start)
        chosen_kla, chosen_nrr = select_dynamic_solution(algorithm.result, online_config)
        modbus.write_setpoints(timestamp, chosen_kla, chosen_nrr)
        dynamic_next = counterfactual_next_state(
            current_row=row,
            current_state=states["dynamic"],
            candidate_kla=chosen_kla,
            candidate_nrr=chosen_nrr,
            baseline_kla=baseline_kla,
            baseline_nrr=baseline_nrr,
            surface_models=surface_models,
            gamma=current_gamma,
            asm_config=asm_config,
            horizon_min=replay_interval,
        )

        rows.append(
            {
                "timestamp": timestamp,
                "actual_bod_proxy_mgL": float(row["actual_next_effluent_bod_proxy_mgL"]),
                "actual_nh3_mgL": float(row["actual_next_effluent_nh3n_mgL"]),
                "actual_tn_mgL": float(row["actual_next_effluent_tn_mgL"]),
                "baseline_kla_d1": baseline_kla,
                "baseline_nrr_m3h": baseline_nrr,
                "steady_kla_d1": float(selected_steady["kla_d1"]),
                "steady_nrr_m3h": float(selected_steady["nrr_m3h"]),
                "dynamic_kla_d1": chosen_kla,
                "dynamic_nrr_m3h": chosen_nrr,
                "baseline_energy": operating_cost_from_row(row, baseline_kla, baseline_nrr, asm_config),
                "steady_energy": operating_cost_from_row(row, float(selected_steady["kla_d1"]), float(selected_steady["nrr_m3h"]), asm_config),
                "dynamic_energy": operating_cost_from_row(row, chosen_kla, chosen_nrr, asm_config),
                "baseline_pred_bod_proxy_mgL": baseline_next["bod_proxy"],
                "baseline_pred_nh3_mgL": baseline_next["nh3"],
                "baseline_pred_tn_mgL": baseline_next["tn"],
                "steady_pred_bod_proxy_mgL": steady_next["bod_proxy"],
                "steady_pred_nh3_mgL": steady_next["nh3"],
                "steady_pred_tn_mgL": steady_next["tn"],
                "dynamic_pred_bod_proxy_mgL": dynamic_next["bod_proxy"],
                "dynamic_pred_nh3_mgL": dynamic_next["nh3"],
                "dynamic_pred_tn_mgL": dynamic_next["tn"],
                "gamma_used": current_gamma,
            }
        )

        states["baseline"] = baseline_next
        states["steady"] = steady_next
        states["dynamic"] = dynamic_next
        dynamic_center["kla_d1"] = chosen_kla
        dynamic_center["nrr_m3h"] = chosen_nrr

    replay_result = pd.DataFrame(rows)
    constraints = online_config["steady_state_optimization"]["constraints"]
    for prefix in ["baseline", "steady", "dynamic"]:
        replay_result[f"{prefix}_is_compliant"] = (
            (replay_result[f"{prefix}_pred_bod_proxy_mgL"] <= float(constraints["bod_proxy_upper"]))
            & (replay_result[f"{prefix}_pred_nh3_mgL"] <= float(constraints["nh3_upper"]))
            & (replay_result[f"{prefix}_pred_tn_mgL"] <= float(constraints["tn_upper"]))
        )

    summary = {
        "steps": int(len(replay_result)),
        "mean_control_execution_sec": float(np.mean(timings) if timings else 0.0),
        "baseline_energy_total": float(replay_result["baseline_energy"].sum()),
        "steady_energy_total": float(replay_result["steady_energy"].sum()),
        "dynamic_energy_total": float(replay_result["dynamic_energy"].sum()),
        "steady_energy_saving_pct": float(
            (replay_result["baseline_energy"].sum() - replay_result["steady_energy"].sum())
            / max(replay_result["baseline_energy"].sum(), 1e-6)
            * 100.0
        ),
        "dynamic_energy_saving_pct": float(
            (replay_result["baseline_energy"].sum() - replay_result["dynamic_energy"].sum())
            / max(replay_result["baseline_energy"].sum(), 1e-6)
            * 100.0
        ),
        "baseline_compliance_rate": float(replay_result["baseline_is_compliant"].mean()),
        "steady_compliance_rate": float(replay_result["steady_is_compliant"].mean()),
        "dynamic_compliance_rate": float(replay_result["dynamic_is_compliant"].mean()),
        "final_gamma": float(current_gamma),
    }
    return replay_result, modbus.to_frame(), summary


def save_pareto_plot(path: Path, pareto_df: pd.DataFrame, selected: dict[str, float]) -> None:
    if pareto_df.empty:
        return
    selected_idx = (pareto_df["kla_d1"] - selected["kla_d1"]).abs().add((pareto_df["nrr_m3h"] - selected["nrr_m3h"]).abs()).idxmin()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pareto_df["energy"], pareto_df["pred_tn_mgL"], s=28, alpha=0.7, label="Feasible Pareto")
    ax.scatter(
        pareto_df.loc[selected_idx, "energy"],
        pareto_df.loc[selected_idx, "pred_tn_mgL"],
        s=70,
        color="red",
        label="Selected steady-state",
    )
    ax.set_xlabel("Operating cost index")
    ax.set_ylabel("Predicted TN (mg/L)")
    ax.set_title("Steady-state Pareto frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_replay_plot(path: Path, replay_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(replay_df["timestamp"], replay_df["baseline_pred_tn_mgL"], label="Baseline TN", linewidth=1.2)
    axes[0].plot(replay_df["timestamp"], replay_df["steady_pred_tn_mgL"], label="Steady TN", linewidth=1.2)
    axes[0].plot(replay_df["timestamp"], replay_df["dynamic_pred_tn_mgL"], label="Dynamic TN", linewidth=1.2)
    axes[0].set_ylabel("TN (mg/L)")
    axes[0].set_title("Closed-loop replay: TN")
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].plot(replay_df["timestamp"], replay_df["baseline_energy"], label="Baseline energy", linewidth=1.2)
    axes[1].plot(replay_df["timestamp"], replay_df["steady_energy"], label="Steady energy", linewidth=1.2)
    axes[1].plot(replay_df["timestamp"], replay_df["dynamic_energy"], label="Dynamic energy", linewidth=1.2)
    axes[1].set_ylabel("Operating cost index")
    axes[1].set_title("Closed-loop replay: control cost")
    axes[1].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_batch_example_plot(path: Path, example_df: pd.DataFrame) -> None:
    if example_df.empty:
        return
    unique_batches = example_df["batch_id"].drop_duplicates().tolist()[:6]
    plot_df = example_df.loc[example_df["batch_id"].isin(unique_batches)].copy()
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()
    for ax, batch_id in zip(axes, unique_batches):
        batch = plot_df.loc[plot_df["batch_id"] == batch_id]
        title = f"{batch['phase'].iloc[0]} #{batch_id}"
        ax.plot(batch["time_min"], batch["truth_concentration_mgL"], label="Truth", linewidth=1.2)
        ax.plot(batch["time_min"], batch["pred_concentration_mgL"], label="Reconstructed", linewidth=1.2)
        ax.set_title(title, fontsize=9)
    for ax in axes[len(unique_batches) :]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Conc. (mg/L)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_report_markdown(
    output_dir: Path,
    summary: dict[str, object],
    feature_model_report: pd.DataFrame,
) -> str:
    lines = [
        "# Paper Reproduction Report",
        "",
        "## Positioning",
        "",
        "This run implements a paper-faithful reproduction pipeline with an ASM2D-inspired digital twin fallback.",
        "It follows the paper's method chain, but it cannot be an exact numerical replica because the original code, 2017 plant data, and online interfaces are not public.",
        "",
        "## Calibration",
        "",
        f"- Mechanistic proxy calibration loss: {summary['mechanistic_proxy_calibration']['loss']:.4f}",
        f"- Mechanistic proxy calibration time: {summary['mechanistic_proxy_calibration']['elapsed_sec']:.2f} s",
        f"- Calibrated proxy parameters: {json.dumps(summary['mechanistic_proxy_calibration']['params'], ensure_ascii=False)}",
        f"- Mechanistic calibration loss: {summary['mechanistic_calibration']['loss']:.4f}",
        f"- Mechanistic calibration time: {summary['mechanistic_calibration']['elapsed_sec']:.2f} s",
        f"- Calibrated parameters: {json.dumps(summary['mechanistic_calibration']['params'], ensure_ascii=False)}",
        f"- Gamma calibration: {summary['gamma_calibration']['gamma']:.4f} in {summary['gamma_calibration']['elapsed_sec']:.2f} s ({summary['gamma_calibration'].get('objective', 'level_only')})",
        "",
        "## Batch Feature Reconstruction",
        "",
        f"- Mean test reconstruction R2: {summary['phase1_batch_reconstruction']['mean_r2']:.4f}",
        f"- Mean test reconstruction RMSE: {summary['phase1_batch_reconstruction']['mean_rmse']:.4f}",
        "",
        "## Real-data Prediction",
        "",
        f"- Mechanistic weighted RMSE: {summary['phase2_real_prediction']['mechanistic']['weighted_rmse']:.4f}",
        f"- Data-driven weighted RMSE: {summary['phase2_real_prediction']['data_driven']['weighted_rmse']:.4f}",
        "",
        "## Optimization",
        "",
        f"- Steady-state operating-cost saving: {summary['phase3_steady_state']['steady_energy_saving_pct']:.2f}%",
        f"- Dynamic operating-cost saving: {summary['phase4_dynamic_replay']['dynamic_energy_saving_pct']:.2f}%",
        f"- Dynamic replay mean execution time: {summary['phase4_dynamic_replay']['mean_control_execution_sec']:.2f} s",
        "",
        "## Surface Models",
        "",
    ]
    for row in feature_model_report.itertuples(index=False):
        lines.append(f"- `{row.name}`: degree {row.degree}, R2={row.r2:.4f}, RMSE={row.rmse:.4f}")
    lines.extend(
        [
            "",
            "## Output Artifacts",
            "",
            "- `batch_profiles.csv`",
            "- `batch_feature_parameters.csv`",
            "- `feature_parameter_functions.json`",
            "- `real_prediction_comparison.csv`",
            "- `steady_state_pareto.csv`",
            "- `online_dynamic_replay.csv`",
            "- `summary.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)
    figures_dir = ensure_dir(output_dir / "figures")

    asm_config = load_json(args.asm_config)
    batch_config = load_json(args.batch_config)
    online_config = load_json(args.online_config)

    decision_raw, labels, plant = load_stage1_tables(args.stage1_output)
    base_observed_df = build_repro_observed_base_table(decision_raw, labels, plant, asm_config)
    mechanistic_proxy_calibration = calibrate_mechanistic_proxy_parameters(
        base_observed_df,
        asm_config,
        kinetic_params={key: float(value) for key, value in asm_config["kinetic_defaults"].items()},
    )
    asm_config = with_mechanistic_proxy_params(asm_config, mechanistic_proxy_calibration["params"])
    real_df = apply_mechanistic_proxy_mapping(base_observed_df, asm_config)
    real_df.to_csv(output_dir / "real_observed_hourly.csv", index=False, encoding="utf-8-sig")

    train_df = real_df.loc[real_df["split"] == "train"].reset_index(drop=True)
    test_df = real_df.loc[real_df["split"] == "test"].reset_index(drop=True)

    mechanistic_calibration = calibrate_mechanistic_parameters(train_df, asm_config)
    calibrated_params = mechanistic_calibration["params"]

    batch_profiles, batch_summary = generate_batch_datasets(calibrated_params, asm_config, batch_config)
    batch_profiles.to_csv(output_dir / "batch_profiles.csv", index=False, encoding="utf-8-sig")
    batch_summary.to_csv(output_dir / "batch_feature_parameters.csv", index=False, encoding="utf-8-sig")

    surface_models, feature_model_report = fit_surface_models(batch_summary, batch_config)
    feature_model_report.to_csv(output_dir / "feature_parameter_function_report.csv", index=False, encoding="utf-8-sig")
    feature_model_json = {
        key: {
            "inputs": model.input_cols,
            "degree": model.degree,
            "r2": model.metrics["r2"],
            "rmse": model.metrics["rmse"],
            "formula": model.formula,
        }
        for key, model in surface_models.items()
    }
    (output_dir / "feature_parameter_functions.json").write_text(
        json.dumps(json_ready(feature_model_json), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    batch_reconstruction, batch_examples = evaluate_batch_reconstruction(batch_profiles, batch_summary, surface_models, asm_config)
    batch_reconstruction.to_csv(output_dir / "batch_reconstruction_metrics.csv", index=False, encoding="utf-8-sig")
    batch_examples.to_csv(output_dir / "batch_reconstruction_examples.csv", index=False, encoding="utf-8-sig")

    gamma_calibration = calibrate_gamma(train_df, surface_models, asm_config)
    gamma_value = float(gamma_calibration["gamma"])

    mech_pred_train = mechanistic_predict_real(train_df, calibrated_params, asm_config)
    mech_pred_test = mechanistic_predict_real(test_df, calibrated_params, asm_config)
    data_pred_train = data_driven_predict_real(train_df, surface_models, gamma_value, asm_config)
    data_pred_test = data_driven_predict_real(test_df, surface_models, gamma_value, asm_config)

    real_prediction_comparison = pd.concat(
        [
            build_real_prediction_frame(train_df, mech_pred_train, "mechanistic"),
            build_real_prediction_frame(test_df, mech_pred_test, "mechanistic"),
            build_real_prediction_frame(train_df, data_pred_train, "data_driven"),
            build_real_prediction_frame(test_df, data_pred_test, "data_driven"),
        ],
        ignore_index=True,
    )
    real_prediction_comparison.to_csv(output_dir / "real_prediction_comparison.csv", index=False, encoding="utf-8-sig")

    pareto_df, selected_steady, steady_summary = run_steady_state_optimization(
        train_df=train_df,
        test_df=test_df,
        surface_models=surface_models,
        gamma=gamma_value,
        kinetic_params=calibrated_params,
        asm_config=asm_config,
        online_config=online_config,
    )
    pareto_df.to_csv(output_dir / "steady_state_pareto.csv", index=False, encoding="utf-8-sig")

    replay_df = resample_online_replay(real_df, online_config)
    replay_result, modbus_log, replay_summary = run_dynamic_replay(
        replay_df=replay_df,
        selected_steady=selected_steady,
        surface_models=surface_models,
        gamma_value=gamma_value,
        kinetic_params=calibrated_params,
        asm_config=asm_config,
        online_config=online_config,
    )
    replay_result.to_csv(output_dir / "online_dynamic_replay.csv", index=False, encoding="utf-8-sig")
    modbus_log.to_csv(output_dir / "online_modbus_mock_log.csv", index=False, encoding="utf-8-sig")

    if bool(online_config["reporting"]["save_plots"]):
        save_pareto_plot(figures_dir / "steady_state_pareto.png", pareto_df, selected_steady)
        save_replay_plot(figures_dir / "dynamic_replay.png", replay_result)
        save_batch_example_plot(figures_dir / "batch_fit_examples.png", batch_examples)

    config_snapshot = {
        "asm_config": asm_config,
        "batch_config": batch_config,
        "online_config": online_config,
    }
    (output_dir / "config_snapshot.json").write_text(json.dumps(json_ready(config_snapshot), ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "positioning": {
            "mode": "paper_faithful_reimplementation_with_asm2d_inspired_fallback",
            "deviation_note": "The workspace lacks the authors' original ASM2D code, 2017 plant data, and online Modbus system, so this run reproduces the method chain and trend-level results rather than exact paper values.",
        },
        "input_counts": {
            "real_hourly_rows": int(len(real_df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "batch_profile_rows": int(len(batch_profiles)),
            "batch_summary_rows": int(len(batch_summary)),
            "online_replay_rows": int(len(replay_df)),
        },
        "mechanistic_proxy_calibration": mechanistic_proxy_calibration,
        "mechanistic_calibration": mechanistic_calibration,
        "gamma_calibration": gamma_calibration,
        "phase1_batch_reconstruction": {
            "mean_r2": float(batch_reconstruction["r2"].mean()),
            "mean_rmse": float(batch_reconstruction["rmse"].mean()),
            "by_phase": {
                str(k): {
                    "mean_r2": float(v["r2"].mean()),
                    "mean_rmse": float(v["rmse"].mean()),
                }
                for k, v in batch_reconstruction.groupby("phase")
            },
        },
        "phase2_real_prediction": {
            "mechanistic": {
                "train": evaluate_real_predictions(train_df, mech_pred_train),
                "test": evaluate_real_predictions(test_df, mech_pred_test),
                "weighted_rmse": evaluate_real_predictions(test_df, mech_pred_test)["weighted_rmse"],
            },
            "data_driven": {
                "train": evaluate_real_predictions(train_df, data_pred_train),
                "test": evaluate_real_predictions(test_df, data_pred_test),
                "weighted_rmse": evaluate_real_predictions(test_df, data_pred_test)["weighted_rmse"],
            },
        },
        "phase3_steady_state": steady_summary,
        "phase4_dynamic_replay": replay_summary,
        "artifacts": {
            "batch_profiles": output_dir / "batch_profiles.csv",
            "batch_feature_parameters": output_dir / "batch_feature_parameters.csv",
            "feature_parameter_functions": output_dir / "feature_parameter_functions.json",
            "real_prediction_comparison": output_dir / "real_prediction_comparison.csv",
            "steady_state_pareto": output_dir / "steady_state_pareto.csv",
            "online_dynamic_replay": output_dir / "online_dynamic_replay.csv",
            "online_modbus_mock_log": output_dir / "online_modbus_mock_log.csv",
            "report": output_dir / "report.md",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    report = build_report_markdown(output_dir, summary, feature_model_report)
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    with (output_dir / "surface_models.pkl").open("wb") as handle:
        pickle.dump(surface_models, handle)

    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
