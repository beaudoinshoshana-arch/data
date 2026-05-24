from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wwtp_decision.safe_marl import (  # noqa: E402
    STATE_COLUMNS,
    SafetyShield,
    estimate_baseline_effluent,
    explain_action,
    grid_search_recommendation,
    load_constraint_config,
    reward_components,
)


class Agent(nn.Module):
    def __init__(self, input_dim: int, max_delta: float) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1) * self.max_delta


class DualAgentPolicy(nn.Module):
    def __init__(self, input_dim: int, aeration_step: float, pac_step: float) -> None:
        super().__init__()
        self.aeration_agent = Agent(input_dim, aeration_step)
        self.dose_agent = Agent(input_dim, pac_step)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.aeration_agent(x), self.dose_agent(x)


def read_json(path: Path, default: Any = None) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


@lru_cache(maxsize=32)
def read_csv_cached(path: str) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        return pd.DataFrame()
    return pd.read_csv(file, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(file, nrows=0).columns else None)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return default


def records(frame: pd.DataFrame, limit: int = 600) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    slim = frame.tail(limit).copy()
    for column in slim.columns:
        if pd.api.types.is_datetime64_any_dtype(slim[column]):
            slim[column] = slim[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    return slim.replace({np.nan: None}).to_dict(orient="records")


@lru_cache(maxsize=1)
def load_policy() -> tuple[DualAgentPolicy | None, dict[str, Any] | None]:
    path = ROOT / "outputs" / "safe_marl" / "safe_marl_policy.pt"
    if not path.exists():
        return None, None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        policy = DualAgentPolicy(
            len(payload["state_columns"]),
            cfg["decision_bounds"]["aeration_intensity_pct"]["max_step"],
            cfg["decision_bounds"]["chemical_dose_pac_mgL"]["max_step"],
        )
        policy.load_state_dict(payload["state_dict"])
        policy.eval()
        return policy, payload
    except Exception:
        return None, None


def dashboard_summary() -> dict[str, Any]:
    stage1 = read_json(ROOT / "outputs" / "stage1_data" / "reports" / "stage1_summary.json", {})
    stage2 = read_json(ROOT / "outputs" / "stage2_model" / "summary.json", {})
    safe = read_json(ROOT / "outputs" / "safe_marl" / "summary.json", {})
    benefit = read_json(ROOT / "outputs" / "decision_benefit" / "decision_benefit_summary.json", {})
    fusion = read_json(ROOT / "outputs" / "fusion_data" / "source_registry.json", {})
    paper = read_json(ROOT / "outputs" / "paper_repro_integrated_control" / "summary.json", {})
    rec = read_csv_cached(str(ROOT / "outputs" / "safe_marl" / "rl_recommendations_test.csv"))
    latest = records(rec.tail(1), 1)
    return {
        "project": {
            "name": "污水厂曝气加药 Safe-MARL 智能决策系统",
            "positioning": "深度融合数据 + 双智能体强化学习 + 约束安全盾 + 可视化大屏",
        },
        "kpis": {
            "fusion_rows": int(fusion.get("summary", {}).get("fusion_rows", 0)),
            "public_monitor_rows": int(stage1.get("public_monitor_long", {}).get("row_count", 0)),
            "decision_rows": int(stage1.get("decision_dataset", {}).get("decision_rows", 0)),
            "stage2_test_weighted_mae": finite(stage2.get("test", {}).get("weighted_normalized_mae")),
            "stage2_compliance_rate": finite(stage2.get("test", {}).get("overall_compliance", {}).get("predicted_rate")),
            "safe_marl_feasible_rate": finite(safe.get("feasible_rate")),
            "safe_marl_fallback_rate": finite(safe.get("fallback_rate")),
            "safe_marl_energy_saving_vs_current_pct": finite(safe.get("mean_energy_saving_vs_current_pct")),
            "safe_marl_chemical_saving_vs_current_pct": finite(safe.get("mean_chemical_saving_vs_current_pct")),
            "fixed_energy_saving_pct": finite(benefit.get("traditional_fixed_baseline", {}).get("energy_saving_pct")),
            "fixed_chemical_saving_pct": finite(benefit.get("traditional_fixed_baseline", {}).get("chemical_saving_pct")),
            "recommend_response_p95_ms": finite(benefit.get("response_time", {}).get("p95_ms")),
            "dynamic_energy_saving_pct": finite(paper.get("phase4_dynamic_replay", {}).get("dynamic_energy_saving_pct")),
            "dynamic_mean_execution_sec": finite(paper.get("phase4_dynamic_replay", {}).get("mean_control_execution_sec")),
        },
        "sources": fusion.get("sources", {}),
        "model": {
            "selected_surrogate": stage2.get("selected_model"),
            "feature_count": stage2.get("feature_count"),
            "surrogate_test_targets": stage2.get("test", {}).get("targets", {}),
            "safe_marl_mode": safe.get("mode"),
            "safe_marl_rows": safe.get("recommendation_rows"),
            "safe_marl_by_scenario": safe.get("by_scenario", {}),
        },
        "evaluation": benefit,
        "latest_recommendation": latest[0] if latest else None,
        "reflection": safe.get("reflection", {}),
    }


def get_timeseries(metric: str = "COD", source: str = "prediction", scenario: str | None = None) -> dict[str, Any]:
    metric = metric.upper()
    if source == "fusion":
        fusion = read_csv_cached(str(ROOT / "outputs" / "fusion_data" / "fusion_long.csv"))
        if fusion.empty:
            return {"series": []}
        subset = fusion[fusion["metric_code"].astype(str).str.upper().eq(metric)].copy()
        subset = subset.tail(700)
        subset = subset.groupby(["timestamp", "source_domain"], as_index=False)["value"].mean()
        return {"series": records(subset, 700)}
    if metric in {"AERATION", "PAC", "REWARD", "ENERGY"}:
        rec = read_csv_cached(str(ROOT / "outputs" / "safe_marl" / "rl_recommendations_test.csv"))
        if scenario:
            rec = rec[rec["scenario_tag"].eq(scenario)]
        columns = ["timestamp", "scenario_tag"]
        if metric == "AERATION":
            columns += ["baseline_aeration_intensity_pct", "recommended_aeration_intensity_pct"]
        elif metric == "PAC":
            columns += ["baseline_chemical_dose_kgph", "recommended_chemical_dose_kgph"]
        elif metric == "REWARD":
            columns += ["baseline_reward", "rl_reward", "final_reward"]
        else:
            columns += ["energy_delta_pct"]
        return {"series": records(rec[columns], 700)}
    pred = read_csv_cached(str(ROOT / "outputs" / "stage2_model" / "test_predictions.csv"))
    map_name = {"COD": "cod", "NH3N": "nh3n", "TP": "tp", "TN": "tn"}
    key = map_name.get(metric, "cod")
    columns = [
        "timestamp",
        f"actual_effluent_{key}_mgL",
        f"pred_effluent_{key}_mgL",
        f"abs_err_effluent_{key}_mgL",
        f"effluent_{key}_mgL_upper_limit",
    ]
    return {"series": records(pred[columns], 700)}


def get_recommendations(scenario: str | None = None, limit: int = 240) -> dict[str, Any]:
    rec = read_csv_cached(str(ROOT / "outputs" / "safe_marl" / "rl_recommendations_test.csv"))
    if scenario:
        rec = rec[rec["scenario_tag"].eq(scenario)]
    return {"items": records(rec, limit)}


def infer_effluent(state: dict[str, Any]) -> dict[str, Any]:
    pred = estimate_baseline_effluent(state)
    risk_score = sum(pred[name] / limit for name, limit in {
        "effluent_cod_mgL": 50.0,
        "effluent_nh3n_mgL": 5.0,
        "effluent_tp_mgL": 0.5,
        "effluent_tn_mgL": 15.0,
    }.items()) / 4.0
    confidence = "high" if risk_score < 0.55 else ("medium" if risk_score < 0.85 else "low")
    return {
        "prediction": pred,
        "risk_score": float(risk_score),
        "confidence": confidence,
        "note": "使用当前在线状态的工程代理预测；批量评估以 Stage-2 监督模型结果为准。",
    }


def recommend_with_policy(state: dict[str, Any]) -> dict[str, Any]:
    cfg = load_constraint_config(ROOT / "configs" / "safe_marl.default.json")
    shield = SafetyShield.from_config(cfg)
    policy, payload = load_policy()
    if policy is None or payload is None:
        action = grid_search_recommendation(state, cfg)
        return {"mode": "grid_safe_expert", "recommendation": action, "explanation": explain_action(state, action)}
    values = np.array([[finite(state.get(column), 0.0) for column in STATE_COLUMNS]], dtype=np.float32)
    center = np.array(payload["center"], dtype=np.float32)
    scale = np.array(payload["scale"], dtype=np.float32)
    x = torch.tensor((values - center) / scale)
    with torch.no_grad():
        aer_delta, pac_delta = policy(x)
    action = shield.apply(state, float(aer_delta.item()), float(pac_delta.item()))
    reward = reward_components(state, action, cfg)
    grid = grid_search_recommendation(state, cfg)
    if not reward["is_feasible"]:
        final = grid_search_recommendation(state, cfg)
        mode = "grid_safe_expert_fallback"
    elif grid["reward"] > reward["reward"] + 1e-9:
        final = grid
        mode = "grid_safe_expert_better_objective"
    else:
        final = {**action, **reward}
        mode = "safe_marl_policy"
    baseline = shield.apply(state, 0.0, 0.0)
    baseline_reward = reward_components(state, baseline, cfg)
    return {
        "mode": mode,
        "raw_policy_delta": {"aeration_delta_pct": float(aer_delta.item()), "pac_delta_mgL": float(pac_delta.item())},
        "reward_compare": {
            "baseline_reward": baseline_reward["reward"],
            "rl_reward": reward["reward"],
            "grid_reward": grid["reward"],
            "final_reward": final["reward"],
        },
        "recommendation": final,
        "explanation": explain_action(state, final),
    }


def ai_summary() -> dict[str, Any]:
    summary = dashboard_summary()
    kpis = summary["kpis"]
    lines = [
        "系统已形成单厂真实数据、国内公开监测数据和外部数据适配器的融合数据底座。",
        f"当前融合长表约 {kpis['fusion_rows']:,} 条，决策样本 {kpis['decision_rows']:,} 条。",
        f"监督代理模型测试集综合归一化 MAE 为 {kpis['stage2_test_weighted_mae']:.3f}，预测达标率 {kpis['stage2_compliance_rate']:.1%}。",
        f"Safe-MARL 推荐通过安全盾后可行率 {kpis['safe_marl_feasible_rate']:.1%}，相对当前控制节能 {kpis['safe_marl_energy_saving_vs_current_pct']:.2f}%、节药 {kpis['safe_marl_chemical_saving_vs_current_pct']:.2f}%。",
        f"相对传统保守定值策略，曝气能耗降低 {kpis['fixed_energy_saving_pct']:.2f}%、PAC 药耗降低 {kpis['fixed_chemical_saving_pct']:.2f}%，P95 决策响应 {kpis['recommend_response_p95_ms']:.1f} ms。",
        "建议答辩时强调：RL 是安全约束下的推荐层，真实执行前仍经过边界裁剪、步长约束和局部专家回退。",
    ]
    return {
        "title": "运行分析摘要",
        "bullets": lines,
        "innovation": ["双智能体协同决策", "融合场景库", "约束安全盾", "可解释动作建议", "动态控制复现对照"],
        "risk": summary["reflection"].get("risk", "离线 RL 仍需现场闭环验证。"),
        "next_step": summary["reflection"].get("next_step", "接入高频在线数据和真实执行反馈。"),
    }
