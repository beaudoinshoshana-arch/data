from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STATE_COLUMNS = [
    "influent_cod_mgL",
    "influent_bod_mgL",
    "influent_nh3n_mgL",
    "influent_tp_mgL",
    "influent_flow_m3h",
    "reactor_do_mgL",
    "sludge_mlss_mgL",
    "aeration_intensity_pct",
    "chemical_dose_pac_mgL",
]
ACTION_COLUMNS = ["aeration_delta_pct", "pac_delta_mgL"]
TARGET_COLUMNS = [
    "effluent_cod_mgL",
    "effluent_nh3n_mgL",
    "effluent_tp_mgL",
    "effluent_tn_mgL",
]
LABEL_TO_TARGET = {
    "label_next_effluent_cod_mgL": "effluent_cod_mgL",
    "label_next_effluent_nh3n_mgL": "effluent_nh3n_mgL",
    "label_next_effluent_tp_mgL": "effluent_tp_mgL",
    "label_next_effluent_tn_mgL": "effluent_tn_mgL",
}
TARGET_LIMITS = {
    "effluent_cod_mgL": 50.0,
    "effluent_nh3n_mgL": 5.0,
    "effluent_tp_mgL": 0.5,
    "effluent_tn_mgL": 15.0,
}
TARGET_WEIGHTS = {
    "effluent_cod_mgL": 0.30,
    "effluent_nh3n_mgL": 0.28,
    "effluent_tp_mgL": 0.24,
    "effluent_tn_mgL": 0.18,
}
SCENARIO_RISK_MULTIPLIER = {
    "observed": 1.0,
    "load_up": 1.25,
    "rain_dilution": 0.9,
    "external_prior": 1.1,
}
DEFAULT_CONFIG = {
    "decision_bounds": {
        "aeration_intensity_pct": {"lower": 10.0, "upper": 100.0, "max_step": 20.0},
        "chemical_dose_pac_mgL": {"lower": 2.0, "upper": 60.0, "max_step": 10.0},
    },
    "objective_weights": {
        "effluent_risk": 1.0,
        "energy": 0.10,
        "chemical": 0.12,
        "smoothness": 0.06,
        "constraint_violation": 6.0,
    },
    "grid_offsets": {
        "observed": {
            "aeration": [-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0],
            "pac": [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0],
        },
        "load_up": {
            "aeration": [0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
            "pac": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        },
        "rain_dilution": {
            "aeration": [-18.0, -14.0, -10.0, -6.0, -2.0, 0.0, 4.0],
            "pac": [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0],
        },
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_constraint_config(path: Path | str | None = None) -> dict[str, Any]:
    config = DEFAULT_CONFIG
    if path:
        candidate = Path(path)
        if candidate.exists():
            config = deep_merge(config, json.loads(candidate.read_text(encoding="utf-8")))
    return config


def row_float(row: pd.Series | dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pac_to_kgph(pac_mg_l: float, flow_m3h: float) -> float:
    return max(0.0, float(pac_mg_l) * max(float(flow_m3h), 0.0) / 1000.0)


def estimate_baseline_effluent(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    """Estimate next-hour effluent if labels are not present, using conservative engineering proxies."""
    out: dict[str, float] = {}
    for label, target in LABEL_TO_TARGET.items():
        if label in row and pd.notna(row.get(label)):
            out[target] = row_float(row, label)
    if len(out) == len(TARGET_COLUMNS):
        return out

    cod = row_float(row, "influent_cod_mgL", 120.0)
    bod = row_float(row, "influent_bod_mgL", cod * 0.45)
    nh3n = row_float(row, "influent_nh3n_mgL", 18.0)
    tp = row_float(row, "influent_tp_mgL", 0.3)
    flow = row_float(row, "influent_flow_m3h", 1200.0)
    do = row_float(row, "reactor_do_mgL", 3.5)
    mlss = row_float(row, "sludge_mlss_mgL", 6500.0)
    pac = row_float(row, "chemical_dose_pac_mgL", 8.0)
    loading = flow / 1200.0
    mlss_factor = np.clip(mlss / 6500.0, 0.7, 1.35)

    out.setdefault("effluent_cod_mgL", max(5.0, cod * 0.055 + bod * 0.012 + loading - do * 0.12))
    out.setdefault("effluent_nh3n_mgL", max(0.03, nh3n * (0.0065 / mlss_factor) + 0.08 / max(do, 0.35)))
    out.setdefault("effluent_tp_mgL", max(0.01, tp * 0.075 - pac * 0.0015 + loading * 0.003))
    out.setdefault("effluent_tn_mgL", max(1.0, nh3n * 0.095 + cod * 0.002 + loading * 0.7 - do * 0.05))
    return out


@dataclass(frozen=True)
class SafetyShield:
    config: dict[str, Any]

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "SafetyShield":
        return cls(config or DEFAULT_CONFIG)

    @property
    def aeration_bounds(self) -> dict[str, float]:
        return self.config["decision_bounds"]["aeration_intensity_pct"]

    @property
    def pac_bounds(self) -> dict[str, float]:
        return self.config["decision_bounds"]["chemical_dose_pac_mgL"]

    def apply(
        self,
        row: pd.Series | dict[str, Any],
        aeration_delta: float,
        pac_delta: float,
    ) -> dict[str, float | bool | str]:
        base_a = row_float(row, "aeration_intensity_pct", 50.0)
        base_p = row_float(row, "chemical_dose_pac_mgL", 8.0)
        flow = row_float(row, "influent_flow_m3h", 1200.0)
        raw_a = base_a + float(aeration_delta)
        raw_p = base_p + float(pac_delta)
        step_a = np.clip(raw_a, base_a - self.aeration_bounds["max_step"], base_a + self.aeration_bounds["max_step"])
        step_p = np.clip(raw_p, base_p - self.pac_bounds["max_step"], base_p + self.pac_bounds["max_step"])
        safe_a = float(np.clip(step_a, self.aeration_bounds["lower"], self.aeration_bounds["upper"]))
        safe_p = float(np.clip(step_p, self.pac_bounds["lower"], self.pac_bounds["upper"]))
        was_adjusted = bool(abs(raw_a - safe_a) > 1e-9 or abs(raw_p - safe_p) > 1e-9)
        return {
            "baseline_aeration_intensity_pct": base_a,
            "recommended_aeration_intensity_pct": safe_a,
            "baseline_chemical_dose_pac_mgL": base_p,
            "recommended_chemical_dose_pac_mgL": safe_p,
            "baseline_chemical_dose_kgph": pac_to_kgph(base_p, flow),
            "recommended_chemical_dose_kgph": pac_to_kgph(safe_p, flow),
            "shield_adjusted": was_adjusted,
            "shield_reason": "clipped_to_bounds_or_step" if was_adjusted else "within_bounds",
        }


def adjusted_effluent(
    row: pd.Series | dict[str, Any],
    recommended_aeration: float,
    recommended_pac: float,
) -> dict[str, float]:
    baseline = estimate_baseline_effluent(row)
    base_a = row_float(row, "aeration_intensity_pct", 50.0)
    base_p = row_float(row, "chemical_dose_pac_mgL", 8.0)
    delta_a = float(recommended_aeration) - base_a
    delta_p = float(recommended_pac) - base_p
    aer_up = max(delta_a, 0.0)
    aer_down = max(-delta_a, 0.0)
    pac_up = max(delta_p, 0.0)
    pac_down = max(-delta_p, 0.0)

    return {
        "effluent_cod_mgL": max(1.0, baseline["effluent_cod_mgL"] * (1.0 - 0.0040 * aer_up + 0.0026 * aer_down)),
        "effluent_nh3n_mgL": max(0.005, baseline["effluent_nh3n_mgL"] * (1.0 - 0.0120 * aer_up + 0.0080 * aer_down)),
        "effluent_tp_mgL": max(0.002, baseline["effluent_tp_mgL"] * (1.0 - 0.0180 * pac_up + 0.0100 * pac_down)),
        "effluent_tn_mgL": max(0.2, baseline["effluent_tn_mgL"] * (1.0 - 0.0032 * aer_up + 0.0022 * aer_down)),
    }


def reward_components(
    row: pd.Series | dict[str, Any],
    action_result: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, float | bool]:
    cfg = config or DEFAULT_CONFIG
    weights = cfg["objective_weights"]
    scenario = str(row.get("scenario_tag", "observed"))
    pred = adjusted_effluent(
        row,
        float(action_result["recommended_aeration_intensity_pct"]),
        float(action_result["recommended_chemical_dose_pac_mgL"]),
    )
    risk = 0.0
    violation = 0.0
    for target, value in pred.items():
        limit = TARGET_LIMITS[target]
        normalized = value / limit
        risk += TARGET_WEIGHTS[target] * normalized
        violation += max(value - limit, 0.0) / limit
    risk *= SCENARIO_RISK_MULTIPLIER.get(scenario, 1.0)
    energy = float(action_result["recommended_aeration_intensity_pct"]) / 100.0
    chemical = float(action_result["recommended_chemical_dose_pac_mgL"]) / 60.0
    smoothness = (
        abs(float(action_result["recommended_aeration_intensity_pct"]) - float(action_result["baseline_aeration_intensity_pct"])) / 20.0
        + abs(float(action_result["recommended_chemical_dose_pac_mgL"]) - float(action_result["baseline_chemical_dose_pac_mgL"])) / 10.0
    ) / 2.0
    objective = (
        weights["effluent_risk"] * risk
        + weights["energy"] * energy
        + weights["chemical"] * chemical
        + weights["smoothness"] * smoothness
        + weights["constraint_violation"] * violation
    )
    reward = -objective
    return {
        "reward": float(reward),
        "objective": float(objective),
        "effluent_risk": float(risk),
        "energy_term": float(energy),
        "chemical_term": float(chemical),
        "smoothness_term": float(smoothness),
        "constraint_violation": float(violation),
        "is_feasible": bool(violation <= 1e-9),
        **{f"pred_{target}": float(value) for target, value in pred.items()},
    }


def grid_search_recommendation(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or DEFAULT_CONFIG
    shield = SafetyShield.from_config(cfg)
    scenario = str(row.get("scenario_tag", "observed"))
    offsets = cfg["grid_offsets"].get(scenario, cfg["grid_offsets"]["observed"])
    best: dict[str, Any] | None = None
    for da in offsets["aeration"]:
        for dp in offsets["pac"]:
            action = shield.apply(row, float(da), float(dp))
            reward = reward_components(row, action, cfg)
            candidate = {**action, **reward, "aeration_delta_pct": float(da), "pac_delta_mgL": float(dp)}
            if best is None or candidate["reward"] > best["reward"]:
                best = candidate
    assert best is not None
    return best


def explain_action(row: pd.Series | dict[str, Any], action: dict[str, Any]) -> str:
    scenario = str(row.get("scenario_tag", "observed"))
    aer_delta = float(action["recommended_aeration_intensity_pct"]) - float(action["baseline_aeration_intensity_pct"])
    pac_delta = float(action["recommended_chemical_dose_pac_mgL"]) - float(action["baseline_chemical_dose_pac_mgL"])
    parts: list[str] = []
    if scenario == "load_up":
        parts.append("冲击负荷场景提高合规安全边际")
    elif scenario == "rain_dilution":
        parts.append("雨水稀释场景优先压低能耗药耗")
    else:
        parts.append("常规工况执行经济合规平衡")
    if aer_delta > 0.5:
        parts.append(f"曝气上调 {aer_delta:.1f} pct 以降低氨氮和总氮风险")
    elif aer_delta < -0.5:
        parts.append(f"曝气下调 {abs(aer_delta):.1f} pct 以节约能耗")
    else:
        parts.append("曝气维持平稳")
    if pac_delta > 0.3:
        parts.append(f"PAC 上调 {pac_delta:.1f} mg/L 以压低总磷风险")
    elif pac_delta < -0.3:
        parts.append(f"PAC 下调 {abs(pac_delta):.1f} mg/L 以减少药耗")
    else:
        parts.append("PAC 维持平稳")
    if action.get("shield_adjusted"):
        parts.append("安全盾已裁剪到设备边界或单次步长")
    return "；".join(parts) + "。"
