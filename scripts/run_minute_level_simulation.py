from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wwtp_decision.safe_marl import (  # noqa: E402
    TARGET_LIMITS,
    SafetyShield,
    grid_search_recommendation,
    load_constraint_config,
    pac_to_kgph,
    reward_components,
)

OUT = ROOT / "outputs" / "minute_simulation"
FUSION_PATH = ROOT / "outputs" / "fusion_data" / "fusion_long.csv"
STAGE1_DECISION_PATH = ROOT / "outputs" / "stage1_data" / "decision_dataset" / "decision_dataset_raw.csv"


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return default


def load_medians() -> dict[str, float]:
    frame = pd.read_csv(STAGE1_DECISION_PATH)
    return {
        "cod": float(frame["influent_cod_mgL"].median()),
        "bod": float(frame["influent_bod_mgL"].median()),
        "nh3n": float(frame["influent_nh3n_mgL"].median()),
        "tp": float(frame["influent_tp_mgL"].median()),
        "flow": float(frame["influent_flow_m3h"].median()),
        "do": float(frame["reactor_do_mgL"].median()),
        "mlss": float(frame["sludge_mlss_mgL"].median()),
        "aeration": float(frame["aeration_intensity_pct"].median()),
        "pac": float(frame["chemical_dose_pac_mgL"].median()),
    }


def load_agtrup_wide(max_rows: int = 1440) -> pd.DataFrame:
    columns = [
        "timestamp",
        "source_domain",
        "metric_code",
        "value",
        "native_frequency_min",
    ]
    fusion = pd.read_csv(FUSION_PATH, usecols=columns, parse_dates=["timestamp"])
    subset = fusion[
        fusion["source_domain"].eq("agtrup_bluekolding_2min")
        & fusion["metric_code"].isin(["FLOW", "DO", "NH3N", "TP", "PAC", "TEMP", "COAGULANT_IN"])
    ].copy()
    if subset.empty:
        raise RuntimeError("Agtrup 2-minute source is missing. Run scripts/build_external_fusion_dataset.py first.")
    wide = (
        subset.pivot_table(index="timestamp", columns="metric_code", values="value", aggfunc="mean")
        .sort_index()
        .reset_index()
    )
    wide = wide.dropna(subset=["FLOW", "NH3N", "TP"], how="any").head(max_rows).copy()
    wide["native_frequency_min"] = float(subset["native_frequency_min"].dropna().min())
    wide["shock_active"] = False
    return wide


def apply_variant(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = frame.copy()
    if variant == "native_2min":
        return out
    start = int(len(out) * 0.34)
    width = max(30, int(90 / max(finite(out["native_frequency_min"].iloc[0], 2.0), 1.0)))
    window = out.index[start : start + width]
    if variant == "load_spike_90min":
        out.loc[window, "NH3N"] = out.loc[window, "NH3N"] * 1.85 + 0.25
        out.loc[window, "TP"] = out.loc[window, "TP"] * 1.65 + 0.05
        out.loc[window, "FLOW"] = out.loc[window, "FLOW"] * 1.22
        out.loc[window, "shock_active"] = True
    elif variant == "sensor_flatline_40min":
        flat_width = max(20, int(40 / max(finite(out["native_frequency_min"].iloc[0], 2.0), 1.0)))
        flat_window = out.index[start : start + flat_width]
        for column in ["DO", "NH3N", "TP"]:
            if column in out:
                out.loc[flat_window, column] = out.loc[flat_window[0], column]
    return out


def state_from_row(row: pd.Series, medians: dict[str, float], aeration: float, pac: float, variant: str) -> dict[str, Any]:
    flow = max(finite(row.get("FLOW"), medians["flow"]), 1.0)
    nh3n = max(finite(row.get("NH3N"), medians["nh3n"]), 0.01)
    tp = max(finite(row.get("TP"), medians["tp"]), 0.002)
    do = max(finite(row.get("DO"), medians["do"]), 0.05)
    flow_factor = np.clip(flow / max(medians["flow"], 1.0), 0.45, 1.85)
    nutrient_factor = np.clip(0.55 * nh3n / max(medians["nh3n"], 0.1) + 0.45 * tp / max(medians["tp"], 0.01), 0.35, 2.4)
    cod = float(np.clip(medians["cod"] * (0.72 + 0.28 * flow_factor) * (0.80 + 0.20 * nutrient_factor), 35.0, 320.0))
    scenario = "load_up" if bool(row.get("shock_active", False)) else "observed"
    return {
        "influent_cod_mgL": cod,
        "influent_bod_mgL": cod * 0.45,
        "influent_nh3n_mgL": nh3n,
        "influent_tp_mgL": tp,
        "influent_flow_m3h": flow,
        "reactor_do_mgL": do,
        "sludge_mlss_mgL": medians["mlss"],
        "aeration_intensity_pct": aeration,
        "chemical_dose_pac_mgL": pac,
        "scenario_tag": scenario,
        "source_flag": "agtrup_2min_simulation",
    }


def compliance(row: dict[str, Any]) -> bool:
    return all(finite(row.get(f"pred_{target}")) <= limit for target, limit in TARGET_LIMITS.items())


def simulate_cadence(
    frame: pd.DataFrame,
    variant: str,
    cadence_min: int,
    medians: dict[str, float],
    config: dict[str, Any],
    lag_min: float = 6.0,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    shield = SafetyShield.from_config(config)
    actual_a = medians["aeration"]
    actual_p = medians["pac"]
    target_a = actual_a
    target_p = actual_p
    last_decision_time: pd.Timestamp | None = None
    decision_ms = 0.0
    decision_reason = "initial"

    for _, row in frame.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        dt_min = finite(row.get("native_frequency_min"), 2.0)
        state = state_from_row(row, medians, actual_a, actual_p, variant)
        should_decide = last_decision_time is None or (timestamp - last_decision_time).total_seconds() / 60.0 >= cadence_min - 1e-9
        if should_decide:
            start = time.perf_counter()
            rec = grid_search_recommendation(state, config)
            decision_ms = (time.perf_counter() - start) * 1000.0
            target_a = finite(rec["recommended_aeration_intensity_pct"], actual_a)
            target_p = finite(rec["recommended_chemical_dose_pac_mgL"], actual_p)
            last_decision_time = timestamp
            decision_reason = "safe_grid_refresh"
        alpha = float(np.clip(dt_min / max(lag_min, 1e-6), 0.0, 1.0))
        actual_a += (target_a - actual_a) * alpha
        actual_p += (target_p - actual_p) * alpha
        bounded = shield.apply(state, actual_a - finite(state["aeration_intensity_pct"]), actual_p - finite(state["chemical_dose_pac_mgL"]))
        reward = reward_components(state, bounded, config)
        record = {
            "timestamp": timestamp,
            "source_domain": "agtrup_bluekolding_2min",
            "scenario_variant": variant,
            "shock_active": bool(row.get("shock_active", False)),
            "control_cadence_min": cadence_min,
            "decision_refreshed": should_decide,
            "decision_reason": decision_reason if should_decide else "held_previous_target",
            "decision_ms": decision_ms if should_decide else 0.0,
            "influent_flow_m3h": finite(state["influent_flow_m3h"]),
            "influent_nh3n_mgL": finite(state["influent_nh3n_mgL"]),
            "influent_tp_mgL": finite(state["influent_tp_mgL"]),
            "reactor_do_mgL": finite(state["reactor_do_mgL"]),
            "target_aeration_intensity_pct": target_a,
            "actual_aeration_intensity_pct": finite(bounded["recommended_aeration_intensity_pct"]),
            "target_chemical_dose_pac_mgL": target_p,
            "actual_chemical_dose_pac_mgL": finite(bounded["recommended_chemical_dose_pac_mgL"]),
            "chemical_dose_kgph": finite(bounded["recommended_chemical_dose_kgph"]),
            "objective": finite(reward["objective"]),
            "reward": finite(reward["reward"]),
            "effluent_risk": finite(reward["effluent_risk"]),
            "constraint_violation": finite(reward["constraint_violation"]),
            "is_feasible": bool(reward["is_feasible"]),
        }
        for target in TARGET_LIMITS:
            record[f"pred_{target}"] = finite(reward[f"pred_{target}"])
            record[f"{target}_limit"] = TARGET_LIMITS[target]
        record["predicted_compliant"] = compliance(record)
        records.append(record)
    return pd.DataFrame(records)


def summarize(frame: pd.DataFrame, source_frequency_min: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (variant, cadence), group in frame.groupby(["scenario_variant", "control_cadence_min"]):
        decision_rows = group[group["decision_refreshed"]]
        rows.append(
            {
                "scenario_variant": str(variant),
                "control_cadence_min": int(cadence),
                "rows": int(len(group)),
                "mean_objective": float(group["objective"].mean()),
                "p95_effluent_risk": float(group["effluent_risk"].quantile(0.95)),
                "predicted_compliance_rate": float(group["predicted_compliant"].mean()),
                "feasible_rate": float(group["is_feasible"].mean()),
                "exceedance_minutes": float((~group["predicted_compliant"]).sum() * source_frequency_min),
                "mean_aeration_pct": float(group["actual_aeration_intensity_pct"].mean()),
                "mean_chemical_kgph": float(group["chemical_dose_kgph"].mean()),
                "decision_count": int(len(decision_rows)),
                "p95_decision_ms": float(decision_rows["decision_ms"].quantile(0.95)) if not decision_rows.empty else 0.0,
            }
        )
    result = pd.DataFrame(rows)
    comparisons: dict[str, dict[str, float]] = {}
    for variant, group in result.groupby("scenario_variant"):
        two = group[group["control_cadence_min"].eq(2)]
        hourly = group[group["control_cadence_min"].eq(60)]
        if two.empty or hourly.empty:
            continue
        two_row = two.iloc[0]
        hour_row = hourly.iloc[0]
        comparisons[str(variant)] = {
            "objective_reduction_vs_60min_pct": float((hour_row["mean_objective"] - two_row["mean_objective"]) / max(hour_row["mean_objective"], 1e-9) * 100.0),
            "exceedance_minutes_delta_vs_60min": float(two_row["exceedance_minutes"] - hour_row["exceedance_minutes"]),
            "chemical_delta_kgph_vs_60min": float(two_row["mean_chemical_kgph"] - hour_row["mean_chemical_kgph"]),
            "aeration_delta_pct_vs_60min": float(two_row["mean_aeration_pct"] - hour_row["mean_aeration_pct"]),
        }
    best = result.sort_values(["scenario_variant", "mean_objective"]).groupby("scenario_variant").head(1)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "2-minute Agtrup SCADA replay with safety-shielded local control cadence simulation",
        "source": {
            "source_domain": "agtrup_bluekolding_2min",
            "native_frequency_min": source_frequency_min,
            "rows_used": int(frame["timestamp"].nunique()),
            "metrics": ["FLOW", "DO", "NH3N", "TP", "PAC", "TEMP", "COAGULANT_IN"],
        },
        "cadences_min": sorted(int(v) for v in frame["control_cadence_min"].unique()),
        "scenario_variants": sorted(str(v) for v in frame["scenario_variant"].unique()),
        "results": rows,
        "best_by_scenario": best.to_dict(orient="records"),
        "two_minute_vs_60min": comparisons,
        "headline": {
            "mean_objective_reduction_vs_60min_pct": float(np.mean([item["objective_reduction_vs_60min_pct"] for item in comparisons.values()])) if comparisons else 0.0,
            "max_p95_decision_ms": float(result["p95_decision_ms"].max()) if not result.empty else 0.0,
            "min_compliance_rate": float(result["predicted_compliance_rate"].min()) if not result.empty else 0.0,
        },
        "reflection": {
            "result": "Agtrup 2 分钟 SCADA 回放已经形成分钟级控制周期实验，不再只依赖小时级离线验证。",
            "risk": "Agtrup 过程变量丰富，但 COD/BOD/出水标签与本地厂站不完全一致，因此 COD/BOD 当前仅作为控制仿真的派生代理。",
            "improvement": "下一步应接入本地 PLC/SCADA 分钟级日志，用在线分析仪或化验校准值替代 COD/BOD 代理。",
            "next_step": "在大屏 KPI 和参赛报告中展示分钟级数据量与 2 分钟相对 60 分钟控制收益。",
        },
    }


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# 分钟级数据融合与控制周期仿真实验",
        "",
        "## 数据来源",
        "",
        f"- 主时间轴：{summary['source']['source_domain']}，原始频率 {summary['source']['native_frequency_min']:.0f} 分钟，仿真唯一时间点 {summary['source']['rows_used']:,} 个。",
        "- 指标：FLOW、DO、NH3N、TP、PAC、TEMP、COAGULANT_IN，并派生 Safe-MARL 所需 COD/BOD 代理状态。",
        "",
        "## 结果摘要",
        "",
        f"- 2 分钟控制相对 60 分钟控制的平均目标函数改善：{summary['headline']['mean_objective_reduction_vs_60min_pct']:.2f}%。",
        f"- 所有场景最低预测达标率：{summary['headline']['min_compliance_rate']:.1%}。",
        f"- 控制决策 P95 最大耗时：{summary['headline']['max_p95_decision_ms']:.2f} ms。",
        "",
        "## 场景与控制周期",
        "",
        "| 场景 | 周期min | 平均目标 | P95风险 | 达标率 | 超限分钟 | 平均曝气% | 平均药耗kg/h | P95决策ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(summary["results"], key=lambda x: (x["scenario_variant"], x["control_cadence_min"])):
        lines.append(
            f"| {item['scenario_variant']} | {item['control_cadence_min']} | {item['mean_objective']:.4f} | "
            f"{item['p95_effluent_risk']:.4f} | {item['predicted_compliance_rate']:.1%} | "
            f"{item['exceedance_minutes']:.1f} | {item['mean_aeration_pct']:.2f} | "
            f"{item['mean_chemical_kgph']:.2f} | {item['p95_decision_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 分步反思",
            "",
            f"- 结果：{summary['reflection']['result']}",
            f"- 风险：{summary['reflection']['risk']}",
            f"- 改进：{summary['reflection']['improvement']}",
            f"- 下一步：{summary['reflection']['next_step']}",
        ]
    )
    (OUT / "minute_simulation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_constraint_config(ROOT / "configs" / "safe_marl.default.json")
    medians = load_medians()
    base = load_agtrup_wide()
    source_frequency_min = finite(base["native_frequency_min"].iloc[0], 2.0)
    variants = ["native_2min", "load_spike_90min", "sensor_flatline_40min"]
    cadences = [2, 5, 15, 60]
    records: list[pd.DataFrame] = []
    for variant in variants:
        scenario_frame = apply_variant(base, variant)
        for cadence in cadences:
            records.append(simulate_cadence(scenario_frame, variant, cadence, medians, config))
    replay = pd.concat(records, ignore_index=True)
    replay.to_csv(OUT / "minute_control_replay.csv", index=False, encoding="utf-8-sig")
    sample = replay.groupby(["scenario_variant", "control_cadence_min"], group_keys=False).head(80)
    sample.to_csv(OUT / "minute_control_replay_sample.csv", index=False, encoding="utf-8-sig")
    summary = summarize(replay, source_frequency_min)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps({"summary": str(OUT / "summary.json"), "records": int(len(replay)), "headline": summary["headline"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
