from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wwtp_decision.safe_marl import (
    LABEL_TO_TARGET,
    STATE_COLUMNS,
    TARGET_LIMITS,
    SafetyShield,
    explain_action,
    grid_search_recommendation,
    load_constraint_config,
    pac_to_kgph,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight safe multi-agent RL policy for WWTP control.")
    parser.add_argument("--stage1-output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "stage1_data")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "safe_marl.default.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "safe_marl")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(stage1_output: Path) -> pd.DataFrame:
    path = stage1_output / "decision_dataset" / "decision_dataset_raw.csv"
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = [*STATE_COLUMNS, *LABEL_TO_TARGET.keys(), "timestamp", "split", "scenario_tag", "source_flag", "chemical_dose_kgph"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame.dropna(subset=[*STATE_COLUMNS, *LABEL_TO_TARGET.keys()]).copy()
    return frame.sort_values("timestamp").reset_index(drop=True)


def fit_normalizer(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = train[STATE_COLUMNS].astype(float).to_numpy()
    center = np.nanmedian(values, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    scale = np.where((q75 - q25) < 1e-6, 1.0, q75 - q25)
    return center.astype(np.float32), scale.astype(np.float32)


def normalize(frame: pd.DataFrame, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((frame[STATE_COLUMNS].astype(float).to_numpy() - center) / scale).astype(np.float32)


def torch_reward(frame: pd.DataFrame, aer_delta: torch.Tensor, pac_delta: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    device = aer_delta.device
    base_a = torch.tensor(frame["aeration_intensity_pct"].to_numpy(dtype=np.float32), device=device)
    base_p = torch.tensor(frame["chemical_dose_pac_mgL"].to_numpy(dtype=np.float32), device=device)
    scenario = frame["scenario_tag"].astype(str).to_numpy()
    aer_bounds = config["decision_bounds"]["aeration_intensity_pct"]
    pac_bounds = config["decision_bounds"]["chemical_dose_pac_mgL"]
    new_a = torch.clamp(
        torch.clamp(base_a + aer_delta, base_a - aer_bounds["max_step"], base_a + aer_bounds["max_step"]),
        aer_bounds["lower"],
        aer_bounds["upper"],
    )
    new_p = torch.clamp(
        torch.clamp(base_p + pac_delta, base_p - pac_bounds["max_step"], base_p + pac_bounds["max_step"]),
        pac_bounds["lower"],
        pac_bounds["upper"],
    )
    da = new_a - base_a
    dp = new_p - base_p
    aer_up = torch.relu(da)
    aer_down = torch.relu(-da)
    pac_up = torch.relu(dp)
    pac_down = torch.relu(-dp)
    labels = {
        "effluent_cod_mgL": torch.tensor(frame["label_next_effluent_cod_mgL"].to_numpy(dtype=np.float32), device=device),
        "effluent_nh3n_mgL": torch.tensor(frame["label_next_effluent_nh3n_mgL"].to_numpy(dtype=np.float32), device=device),
        "effluent_tp_mgL": torch.tensor(frame["label_next_effluent_tp_mgL"].to_numpy(dtype=np.float32), device=device),
        "effluent_tn_mgL": torch.tensor(frame["label_next_effluent_tn_mgL"].to_numpy(dtype=np.float32), device=device),
    }
    pred = {
        "effluent_cod_mgL": torch.clamp(labels["effluent_cod_mgL"] * (1.0 - 0.0040 * aer_up + 0.0026 * aer_down), min=1.0),
        "effluent_nh3n_mgL": torch.clamp(labels["effluent_nh3n_mgL"] * (1.0 - 0.0120 * aer_up + 0.0080 * aer_down), min=0.005),
        "effluent_tp_mgL": torch.clamp(labels["effluent_tp_mgL"] * (1.0 - 0.0180 * pac_up + 0.0100 * pac_down), min=0.002),
        "effluent_tn_mgL": torch.clamp(labels["effluent_tn_mgL"] * (1.0 - 0.0032 * aer_up + 0.0022 * aer_down), min=0.2),
    }
    risk = (
        0.30 * pred["effluent_cod_mgL"] / TARGET_LIMITS["effluent_cod_mgL"]
        + 0.28 * pred["effluent_nh3n_mgL"] / TARGET_LIMITS["effluent_nh3n_mgL"]
        + 0.24 * pred["effluent_tp_mgL"] / TARGET_LIMITS["effluent_tp_mgL"]
        + 0.18 * pred["effluent_tn_mgL"] / TARGET_LIMITS["effluent_tn_mgL"]
    )
    multiplier = torch.tensor(
        np.where(scenario == "load_up", 1.25, np.where(scenario == "rain_dilution", 0.9, 1.0)).astype(np.float32),
        device=device,
    )
    risk = risk * multiplier
    violation = sum(torch.relu(value - TARGET_LIMITS[name]) / TARGET_LIMITS[name] for name, value in pred.items())
    energy = new_a / 100.0
    chemical = new_p / 60.0
    smooth = (torch.abs(da) / 20.0 + torch.abs(dp) / 10.0) / 2.0
    weights = config["objective_weights"]
    objective = (
        weights["effluent_risk"] * risk
        + weights["energy"] * energy
        + weights["chemical"] * chemical
        + weights["smoothness"] * smooth
        + weights["constraint_violation"] * violation
    )
    return -objective.mean()


def evaluate_policy(
    policy: DualAgentPolicy,
    frame: pd.DataFrame,
    center: np.ndarray,
    scale: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    shield = SafetyShield.from_config(config)
    x = torch.tensor(normalize(frame, center, scale))
    policy.eval()
    with torch.no_grad():
        aer_delta, pac_delta = policy(x)
    records: list[dict[str, Any]] = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        rl_action = shield.apply(row, float(aer_delta[idx]), float(pac_delta[idx]))
        rl_reward = reward_components(row, rl_action, config)
        baseline_action = shield.apply(row, 0.0, 0.0)
        baseline_reward = reward_components(row, baseline_action, config)
        grid = grid_search_recommendation(row, config)
        rl_underperformed = bool(rl_reward["reward"] < baseline_reward["reward"])
        expert_better = bool(grid["reward"] > rl_reward["reward"] + 1e-9)
        used_fallback = bool(not rl_reward["is_feasible"] or expert_better)
        final = grid if used_fallback else {**rl_action, **rl_reward}
        if not rl_reward["is_feasible"]:
            fallback_reason = "grid_safe_expert_infeasible"
        elif expert_better:
            fallback_reason = "grid_safe_expert_better_objective"
        else:
            fallback_reason = "rl_policy_accepted"
        baseline_kg = float(baseline_action["baseline_chemical_dose_kgph"])
        rec_kg = float(final["recommended_chemical_dose_kgph"])
        rec = {
            "timestamp": row["timestamp"],
            "split": row["split"],
            "scenario_tag": row["scenario_tag"],
            "source_flag": row["source_flag"],
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "rl_underperformed_baseline": rl_underperformed,
            "baseline_reward": baseline_reward["reward"],
            "rl_reward": rl_reward["reward"],
            "grid_reward": grid["reward"],
            "final_reward": final["reward"],
            "baseline_objective": baseline_reward["objective"],
            "rl_objective": rl_reward["objective"],
            "grid_objective": grid["objective"],
            "final_objective": final["objective"],
            "baseline_aeration_intensity_pct": baseline_action["baseline_aeration_intensity_pct"],
            "recommended_aeration_intensity_pct": final["recommended_aeration_intensity_pct"],
            "baseline_chemical_dose_pac_mgL": baseline_action["baseline_chemical_dose_pac_mgL"],
            "recommended_chemical_dose_pac_mgL": final["recommended_chemical_dose_pac_mgL"],
            "baseline_chemical_dose_kgph": baseline_kg,
            "recommended_chemical_dose_kgph": rec_kg,
            "energy_delta_pct": float(final["recommended_aeration_intensity_pct"] - baseline_action["baseline_aeration_intensity_pct"]),
            "chemical_delta_kgph": float(rec_kg - baseline_kg),
            "objective_improvement_pct": float((baseline_reward["objective"] - final["objective"]) / max(baseline_reward["objective"], 1e-9) * 100.0),
            "reward_effluent_risk": final["effluent_risk"],
            "reward_energy_term": final["energy_term"],
            "reward_chemical_term": final["chemical_term"],
            "reward_smoothness_term": final["smoothness_term"],
            "constraint_violation": final["constraint_violation"],
            "is_feasible": final["is_feasible"],
            "explanation": explain_action(row, final),
        }
        for target in TARGET_LIMITS:
            rec[f"pred_{target}"] = final[f"pred_{target}"]
            rec[f"{target}_upper_limit"] = TARGET_LIMITS[target]
        records.append(rec)
    return pd.DataFrame(records)


def aggregate_recommendations(frame: pd.DataFrame) -> dict[str, float]:
    baseline_aer = frame["baseline_aeration_intensity_pct"].mean()
    recommended_aer = frame["recommended_aeration_intensity_pct"].mean()
    baseline_kg = frame["baseline_chemical_dose_kgph"].mean()
    recommended_kg = frame["recommended_chemical_dose_kgph"].mean()
    return {
        "objective_improvement_pct": float(frame["objective_improvement_pct"].mean()),
        "energy_delta_pct_points": float(frame["energy_delta_pct"].mean()),
        "chemical_delta_kgph": float(frame["chemical_delta_kgph"].mean()),
        "energy_saving_vs_current_pct": float((baseline_aer - recommended_aer) / max(baseline_aer, 1e-9) * 100.0),
        "chemical_saving_vs_current_pct": float((baseline_kg - recommended_kg) / max(baseline_kg, 1e-9) * 100.0),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_constraint_config(args.config)
    data = load_data(args.stage1_output)
    train = data[data["split"].isin(["train", "val"])].copy()
    test = data[data["split"].eq("test")].copy()
    center, scale = fit_normalizer(train)
    policy = DualAgentPolicy(
        input_dim=len(STATE_COLUMNS),
        aeration_step=config["decision_bounds"]["aeration_intensity_pct"]["max_step"],
        pac_step=config["decision_bounds"]["chemical_dose_pac_mgL"]["max_step"],
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=2.5e-3, weight_decay=1e-4)
    curve: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        shuffled = train.sample(frac=1.0, random_state=args.seed + epoch).reset_index(drop=True)
        losses: list[float] = []
        for offset in range(0, len(shuffled), args.batch_size):
            batch = shuffled.iloc[offset : offset + args.batch_size]
            x = torch.tensor(normalize(batch, center, scale))
            aer_delta, pac_delta = policy(x)
            reward = torch_reward(batch, aer_delta, pac_delta, config)
            loss = -reward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            curve.append({"epoch": epoch, "loss": float(np.mean(losses)), "mean_reward": float(-np.mean(losses))})
    elapsed = time.perf_counter() - start

    recommendations = evaluate_policy(policy, test, center, scale, config)
    recommendations.to_csv(args.output / "rl_recommendations_test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve).to_csv(args.output / "training_curve.csv", index=False, encoding="utf-8-sig")
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "state_columns": STATE_COLUMNS,
            "center": center.tolist(),
            "scale": scale.tolist(),
            "config": config,
        },
        args.output / "safe_marl_policy.pt",
    )
    summary = {
        "mode": "safe_dual_agent_offline_rl",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "epochs": int(args.epochs),
        "elapsed_sec": float(elapsed),
        "recommendation_rows": int(len(recommendations)),
        "fallback_rate": float(recommendations["used_fallback"].mean()),
        "policy_acceptance_rate": float(1.0 - recommendations["used_fallback"].mean()),
        "feasible_rate": float(recommendations["is_feasible"].mean()),
        "mean_objective_improvement_pct": aggregate_recommendations(recommendations)["objective_improvement_pct"],
        "mean_energy_delta_pct": aggregate_recommendations(recommendations)["energy_delta_pct_points"],
        "mean_chemical_delta_kgph": aggregate_recommendations(recommendations)["chemical_delta_kgph"],
        "mean_energy_saving_vs_current_pct": aggregate_recommendations(recommendations)["energy_saving_vs_current_pct"],
        "mean_chemical_saving_vs_current_pct": aggregate_recommendations(recommendations)["chemical_saving_vs_current_pct"],
        "by_scenario": {
            scenario: {
                "rows": int(len(group)),
                "feasible_rate": float(group["is_feasible"].mean()),
                "fallback_rate": float(group["used_fallback"].mean()),
                "policy_acceptance_rate": float(1.0 - group["used_fallback"].mean()),
                **aggregate_recommendations(group),
            }
            for scenario, group in recommendations.groupby("scenario_tag")
        },
        "artifacts": {
            "policy": str(args.output / "safe_marl_policy.pt"),
            "recommendations": str(args.output / "rl_recommendations_test.csv"),
            "training_curve": str(args.output / "training_curve.csv"),
        },
        "reflection": {
            "result": "RL policy is trained against a safety-constrained surrogate reward; final actions are selected by a safety-and-objective arbitration layer.",
            "risk": "Offline RL cannot prove real-plant optimality without closed-loop plant trials.",
            "improvement": "Next iteration should reduce expert fallback rate by adding simulator rollouts or behavior-cloned warm starts.",
            "next_step": "Expose recommendation benefit, robustness, and response-time evidence in the dashboard and competition report.",
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    model_card = f"""# Safe-MARL 决策层模型卡

## 模型用途

本模型用于污水厂曝气强度和 PAC 投加量的离线推荐。曝气 agent 输出曝气调节量，投药 agent 输出 PAC 调节量；最终动作需经过安全盾和目标函数仲裁后才能展示或对接自控系统。

## 输入与输出

- 输入：COD、BOD、氨氮、总磷、流量、DO、MLSS、当前曝气强度、当前 PAC 投加量。
- 输出：推荐曝气强度、推荐 PAC 投加量、预测出水风险、reward 分解、可行性、回退原因和中文动作解释。

## 训练与评估数据

- 训练/验证样本：{summary["train_rows"]:,} 行。
- 测试推荐样本：{summary["recommendation_rows"]:,} 行。
- 工况：observed、load_up、rain_dilution。

## 关键指标

- 安全可行率：{summary["feasible_rate"]:.1%}。
- 目标函数改善：{summary["mean_objective_improvement_pct"]:.2f}%。
- 相对当前控制曝气节能：{summary["mean_energy_saving_vs_current_pct"]:.2f}%。
- 相对当前控制 PAC 节药：{summary["mean_chemical_saving_vs_current_pct"]:.2f}%。
- 策略直接接受率：{summary["policy_acceptance_rate"]:.1%}。

## 安全机制

1. 动作边界：曝气强度、PAC 投加量均受设备上下限约束。
2. 单步限制：单次调节不得超过配置文件中的最大步长。
3. 合规风险：COD、NH3-N、TP、TN 超限会进入高权重违规惩罚。
4. 目标仲裁：若 RL 动作不可行，或局部专家搜索在同一目标函数下更优，则展示专家安全动作。

## 适用边界

本模型是基于历史数据、场景扩增和工程代理响应函数的离线推荐层，不能替代现场闭环试验。正式接入 PLC/SCADA 前，应在旁路模式运行，记录操作员接受率、真实执行反馈和出水变化，再进行参数复标定。
"""
    (args.output / "model_card.md").write_text(model_card, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
