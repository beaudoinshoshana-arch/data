from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.backend.services import recommend_with_policy  # noqa: E402
from wwtp_decision.safe_marl import (  # noqa: E402
    LABEL_TO_TARGET,
    STATE_COLUMNS,
    TARGET_LIMITS,
    SafetyShield,
    load_constraint_config,
    pac_to_kgph,
    reward_components,
)

OUT = ROOT / "outputs" / "decision_benefit"


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def state_from_row(row: pd.Series) -> dict[str, Any]:
    state = {column: finite(row.get(column)) for column in STATE_COLUMNS}
    state["scenario_tag"] = str(row.get("scenario_tag", "observed"))
    state["source_flag"] = str(row.get("source_flag", "local_real"))
    for label in LABEL_TO_TARGET:
        if label in row:
            state[label] = finite(row.get(label))
    return state


def perturb(state: dict[str, Any], variant: str, idx: int, medians: dict[str, float]) -> dict[str, Any]:
    out = dict(state)
    rng = np.random.default_rng(20260524 + idx)
    if variant == "monitoring_noise":
        for column, sigma in {
            "influent_cod_mgL": 0.04,
            "influent_bod_mgL": 0.04,
            "influent_nh3n_mgL": 0.05,
            "influent_tp_mgL": 0.05,
            "influent_flow_m3h": 0.03,
            "sludge_mlss_mgL": 0.05,
        }.items():
            out[column] = max(0.0, finite(out[column]) * (1.0 + rng.normal(0.0, sigma)))
        out["reactor_do_mgL"] = max(0.1, finite(out["reactor_do_mgL"]) + rng.normal(0.0, 0.18))
    elif variant == "device_fluctuation":
        out["aeration_intensity_pct"] = float(np.clip(finite(out["aeration_intensity_pct"]) + rng.normal(0.0, 2.0), 10.0, 100.0))
        out["chemical_dose_pac_mgL"] = float(np.clip(finite(out["chemical_dose_pac_mgL"]) + rng.normal(0.0, 0.6), 2.0, 60.0))
    elif variant == "sensor_flatline":
        out["reactor_do_mgL"] = medians["reactor_do_mgL"]
        out["sludge_mlss_mgL"] = medians["sludge_mlss_mgL"]
    out["robustness_variant"] = variant
    return out


def compliance_rate(frame: pd.DataFrame) -> float:
    checks = []
    for target, limit in TARGET_LIMITS.items():
        column = f"pred_{target}"
        if column in frame:
            checks.append(frame[column].astype(float) <= limit)
    if not checks:
        return 0.0
    return float(pd.concat(checks, axis=1).all(axis=1).mean())


def summarize_records(frame: pd.DataFrame) -> dict[str, float]:
    baseline_a = frame["baseline_aeration_intensity_pct"].mean()
    recommended_a = frame["recommended_aeration_intensity_pct"].mean()
    baseline_kg = frame["baseline_chemical_dose_kgph"].mean()
    recommended_kg = frame["recommended_chemical_dose_kgph"].mean()
    return {
        "rows": int(len(frame)),
        "feasible_rate": float(frame["is_feasible"].mean()),
        "predicted_compliance_rate": compliance_rate(frame),
        "fallback_rate": float(frame["mode"].astype(str).str.contains("grid_safe_expert").mean()),
        "objective_improvement_pct": float(frame["objective_improvement_pct"].mean()),
        "energy_saving_vs_current_pct": float((baseline_a - recommended_a) / max(baseline_a, 1e-9) * 100.0),
        "chemical_saving_vs_current_pct": float((baseline_kg - recommended_kg) / max(baseline_kg, 1e-9) * 100.0),
        "mean_response_ms": float(frame["response_ms"].mean()),
        "p95_response_ms": float(frame["response_ms"].quantile(0.95)),
        "max_response_ms": float(frame["response_ms"].max()),
    }


def evaluate_state(state: dict[str, Any], cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    start = time.perf_counter()
    result = recommend_with_policy(state)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    rec = result["recommendation"]
    shield = SafetyShield.from_config(cfg)
    baseline = shield.apply(state, 0.0, 0.0)
    baseline_reward = reward_components(state, baseline, cfg)
    improvement = (baseline_reward["objective"] - rec["objective"]) / max(baseline_reward["objective"], 1e-9) * 100.0
    record = {
        "scenario_tag": state.get("scenario_tag", "observed"),
        "robustness_variant": variant,
        "mode": result["mode"],
        "response_ms": elapsed_ms,
        "objective_improvement_pct": float(improvement),
        "baseline_objective": baseline_reward["objective"],
        "final_objective": rec["objective"],
        "baseline_aeration_intensity_pct": rec["baseline_aeration_intensity_pct"],
        "recommended_aeration_intensity_pct": rec["recommended_aeration_intensity_pct"],
        "baseline_chemical_dose_kgph": rec["baseline_chemical_dose_kgph"],
        "recommended_chemical_dose_kgph": rec["recommended_chemical_dose_kgph"],
        "is_feasible": rec["is_feasible"],
        "explanation": result["explanation"],
    }
    for target in TARGET_LIMITS:
        record[f"pred_{target}"] = rec[f"pred_{target}"]
        record[f"{target}_upper_limit"] = TARGET_LIMITS[target]
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_constraint_config(ROOT / "configs" / "safe_marl.default.json")
    raw = pd.read_csv(ROOT / "outputs" / "stage1_data" / "decision_dataset" / "decision_dataset_raw.csv", parse_dates=["timestamp"])
    raw = raw.dropna(subset=STATE_COLUMNS).sort_values("timestamp").reset_index(drop=True)
    train = raw[raw["split"].isin(["train", "val"])].copy()
    test = raw[raw["split"].eq("test")].copy()
    medians = {column: float(train[column].median()) for column in ["reactor_do_mgL", "sludge_mlss_mgL"]}

    states = [state_from_row(row) for _, row in test.iterrows()]
    sample = test.groupby("scenario_tag", group_keys=False).head(80).reset_index(drop=True)
    robust_states: list[tuple[str, dict[str, Any]]] = [("test_set", state) for state in states]
    for variant in ["monitoring_noise", "device_fluctuation", "sensor_flatline"]:
        robust_states.extend((variant, perturb(state_from_row(row), variant, idx, medians)) for idx, (_, row) in enumerate(sample.iterrows()))

    if states:
        recommend_with_policy(states[0])

    records = [evaluate_state(state, cfg, variant) for variant, state in robust_states]
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "decision_benefit_records.csv", index=False, encoding="utf-8-sig")

    test_frame = frame[frame["robustness_variant"].eq("test_set")].copy()
    fixed_aer = float(train["aeration_intensity_pct"].quantile(0.90))
    fixed_pac = float(train["chemical_dose_pac_mgL"].quantile(0.90))
    fixed_kg = test.apply(lambda row: pac_to_kgph(fixed_pac, row["influent_flow_m3h"]), axis=1)
    response = frame["response_ms"]
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policy": "Safe-MARL policy plus safety/objective arbitration",
        "current_control_baseline": summarize_records(test_frame),
        "traditional_fixed_baseline": {
            "definition": "Training-period P90 conservative manual setpoints for continuous operation.",
            "fixed_aeration_intensity_pct": fixed_aer,
            "fixed_pac_mgL": fixed_pac,
            "recommended_aeration_mean_pct": float(test_frame["recommended_aeration_intensity_pct"].mean()),
            "recommended_chemical_mean_kgph": float(test_frame["recommended_chemical_dose_kgph"].mean()),
            "fixed_chemical_mean_kgph": float(fixed_kg.mean()),
            "energy_saving_pct": float((fixed_aer - test_frame["recommended_aeration_intensity_pct"].mean()) / max(fixed_aer, 1e-9) * 100.0),
            "chemical_saving_pct": float((fixed_kg.mean() - test_frame["recommended_chemical_dose_kgph"].mean()) / max(fixed_kg.mean(), 1e-9) * 100.0),
        },
        "by_scenario": {
            name: summarize_records(group)
            for name, group in test_frame.groupby("scenario_tag")
        },
        "robustness": {
            name: summarize_records(group)
            for name, group in frame.groupby("robustness_variant")
        },
        "response_time": {
            "requirement_sec": 1.0,
            "mean_ms": float(response.mean()),
            "p95_ms": float(response.quantile(0.95)),
            "max_ms": float(response.max()),
            "pass": bool((response.max() / 1000.0) <= 1.0),
        },
        "reflection": {
            "result": "Decision benefit is now reported against current plant controls and a conservative traditional fixed baseline.",
            "risk": "P90 fixed baseline is a defensible manual-control proxy, not a measured operator trial.",
            "improvement": "A future plant pilot should log actual manual setpoints and closed-loop actuator feedback for direct A/B validation.",
            "next_step": "Use the benefit summary in the dashboard KPI, completion audit, and competition report.",
        },
    }
    (OUT / "decision_benefit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# 决策收益、鲁棒性与响应时间评估",
        "",
        f"- 当前控制对照：目标函数改善 {summary['current_control_baseline']['objective_improvement_pct']:.2f}%，曝气节能 {summary['current_control_baseline']['energy_saving_vs_current_pct']:.2f}%，PAC 节药 {summary['current_control_baseline']['chemical_saving_vs_current_pct']:.2f}%。",
        f"- 传统保守定值对照：曝气节能 {summary['traditional_fixed_baseline']['energy_saving_pct']:.2f}%，PAC 节药 {summary['traditional_fixed_baseline']['chemical_saving_pct']:.2f}%。",
        f"- 响应时间：平均 {summary['response_time']['mean_ms']:.1f} ms，P95 {summary['response_time']['p95_ms']:.1f} ms，最大 {summary['response_time']['max_ms']:.1f} ms。",
        "",
        "## 鲁棒性结果",
        "",
        "| 场景 | 样本 | 可行率 | 预测达标率 | 当前曝气节能 | 当前PAC节药 | P95响应ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in summary["robustness"].items():
        report.append(
            f"| {name} | {item['rows']} | {item['feasible_rate']:.1%} | {item['predicted_compliance_rate']:.1%} | "
            f"{item['energy_saving_vs_current_pct']:.2f}% | {item['chemical_saving_vs_current_pct']:.2f}% | {item['p95_response_ms']:.1f} |"
        )
    (OUT / "decision_benefit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(OUT / "decision_benefit_summary.json"), "records": len(frame)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
