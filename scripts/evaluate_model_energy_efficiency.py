from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_stage2_model import (  # noqa: E402
    TARGET_COLUMNS,
    assemble_feature_frame,
    build_observed_training_table,
    build_temporal_context,
    evaluate_predictions,
    load_constraint_config,
    load_stage1_tables,
    select_feature_columns,
)


OUT = ROOT / "outputs" / "model_efficiency"
STAGE2 = ROOT / "outputs" / "stage2_model"
DECISION_BENEFIT = ROOT / "outputs" / "decision_benefit" / "decision_benefit_summary.json"
DEADLINES_MS = [20.0, 50.0, 100.0, 250.0, 1000.0]
CANDIDATE_BATCH_SIZE = 49


ModelFactory = Callable[[], Pipeline]


def read_json(path: Path, default: Any = None) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def set_single_thread(model: Pipeline) -> None:
    estimator = model.named_steps.get("model")
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1
    estimators = getattr(estimator, "estimators_", None)
    if estimators is not None:
        for member in np.ravel(estimators):
            if hasattr(member, "n_jobs"):
                member.n_jobs = 1


def extra_trees_factory(n_estimators: int, max_depth: int | None, min_samples_leaf: int, seed: int) -> ModelFactory:
    return lambda: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def random_forest_factory(n_estimators: int, max_depth: int | None, min_samples_leaf: int, seed: int) -> ModelFactory:
    return lambda: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def gradient_boosting_factory(n_estimators: int, max_depth: int, seed: int) -> ModelFactory:
    return lambda: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                MultiOutputRegressor(
                    GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=0.045,
                        subsample=0.85,
                        random_state=seed,
                    ),
                    n_jobs=1,
                ),
            ),
        ]
    )


def ridge_factory(alpha: float) -> ModelFactory:
    return lambda: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def predict_profile(profile: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if "members" not in profile:
        return profile["model"].predict(frame[profile["features"]])
    pred: np.ndarray | None = None
    for member, weight in profile["members"]:
        member_pred = predict_profile(member, frame)
        pred = member_pred * weight if pred is None else pred + member_pred * weight
    assert pred is not None
    return pred


def timed_predict_ms(profile: dict[str, Any], frame: pd.DataFrame) -> float:
    predict_profile(profile, frame.head(min(10, len(frame))))
    start = time.perf_counter()
    predict_profile(profile, frame)
    return float((time.perf_counter() - start) * 1000.0)


def latency_samples_ms(profile: dict[str, Any], frame: pd.DataFrame, batch_size: int, repeats: int = 80) -> list[float]:
    if frame.empty:
        return []
    predict_profile(profile, frame.head(min(batch_size, len(frame))))
    values: list[float] = []
    for idx in range(repeats):
        start_idx = (idx * 17) % len(frame)
        batch = frame.iloc[start_idx : start_idx + batch_size]
        if len(batch) < batch_size:
            batch = pd.concat([batch, frame.iloc[: batch_size - len(batch)]], axis=0)
        start = time.perf_counter()
        predict_profile(profile, batch)
        values.append(float((time.perf_counter() - start) * 1000.0))
    return values


def timing_summary(profile: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    single = latency_samples_ms(profile, frame, 1, repeats=120)
    batch = latency_samples_ms(profile, frame, CANDIDATE_BATCH_SIZE, repeats=80)
    deadline_rates = {
        f"miss_rate_{int(deadline)}ms": float(np.mean(np.asarray(batch) > deadline)) if batch else 0.0
        for deadline in DEADLINES_MS
    }
    delay_penalty = {
        f"delay_penalty_{int(deadline)}ms": float(np.mean(np.maximum(np.asarray(batch) - deadline, 0.0) / deadline)) if batch else 0.0
        for deadline in DEADLINES_MS
    }
    return {
        "single_p50_ms": percentile(single, 50),
        "single_p95_ms": percentile(single, 95),
        "candidate_batch_size": CANDIDATE_BATCH_SIZE,
        "candidate_batch_p50_ms": percentile(batch, 50),
        "candidate_batch_p95_ms": percentile(batch, 95),
        "candidate_batch_max_ms": max(batch) if batch else 0.0,
        **deadline_rates,
        **delay_penalty,
    }


def metric_row(name: str, metrics: dict[str, Any], reference: float) -> dict[str, Any]:
    wnmae = float(metrics["weighted_normalized_mae"])
    return {
        "model_name": name,
        "weighted_normalized_mae": wnmae,
        "mae_delta_vs_full_baseline": float(wnmae - reference) if np.isfinite(reference) else None,
        "predicted_compliance_rate": float(metrics.get("overall_compliance", {}).get("predicted_rate", 0.0)),
    }


def train_profile(
    name: str,
    family: str,
    features: list[str],
    factory: ModelFactory,
    complexity_score: float,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_train_val: pd.DataFrame,
    y_train_val: pd.DataFrame,
    x_val: pd.DataFrame,
    y_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    reference_std: pd.Series,
    constraint_config: dict[str, Any],
    full_test_wnmae: float,
) -> dict[str, Any]:
    val_model = factory()
    start = time.perf_counter()
    val_model.fit(x_train[features], y_train)
    val_fit_sec = float(time.perf_counter() - start)
    val_pred = val_model.predict(x_val[features])
    val_metrics = evaluate_predictions(y_val, val_pred, reference_std, constraint_config)

    model = factory()
    start = time.perf_counter()
    model.fit(x_train_val[features], y_train_val)
    final_fit_sec = float(time.perf_counter() - start)
    profile = {
        "name": name,
        "family": family,
        "features": features,
        "feature_count": len(features),
        "model": model,
        "fit_sec": val_fit_sec + final_fit_sec,
        "validation_weighted_normalized_mae": float(val_metrics["weighted_normalized_mae"]),
        "complexity_score": float(complexity_score),
        "member_names": [],
        "member_weights": [],
    }
    test_pred = predict_profile(profile, x_test)
    test_metrics = evaluate_predictions(y_test, test_pred, reference_std, constraint_config)
    profile.update(metric_row(name, test_metrics, full_test_wnmae))
    profile["targets"] = test_metrics["targets"]
    profile["test_predict_ms"] = timed_predict_ms(profile, x_test)
    profile.update(timing_summary(profile, x_test))
    return profile


def make_ensemble(
    name: str,
    family: str,
    members: list[dict[str, Any]],
    weights: list[float],
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    reference_std: pd.Series,
    constraint_config: dict[str, Any],
    full_test_wnmae: float,
) -> dict[str, Any]:
    total = float(sum(weights))
    normalized = [float(weight / total) for weight in weights]
    profile = {
        "name": name,
        "family": family,
        "features": sorted({feature for member in members for feature in member["features"]}),
        "feature_count": int(sum(member["feature_count"] for member in members)),
        "members": list(zip(members, normalized)),
        "fit_sec": float(sum(member["fit_sec"] for member in members)),
        "validation_weighted_normalized_mae": float(sum(w * member["validation_weighted_normalized_mae"] for w, member in zip(normalized, members))),
        "complexity_score": float(sum(w * member["complexity_score"] for w, member in zip(normalized, members))),
        "member_names": [member["name"] for member in members],
        "member_weights": normalized,
    }
    test_pred = predict_profile(profile, x_test)
    test_metrics = evaluate_predictions(y_test, test_pred, reference_std, constraint_config)
    profile.update(metric_row(name, test_metrics, full_test_wnmae))
    profile["targets"] = test_metrics["targets"]
    profile["test_predict_ms"] = timed_predict_ms(profile, x_test)
    profile.update(timing_summary(profile, x_test))
    return profile


def json_profile(profile: dict[str, Any]) -> dict[str, Any]:
    skip = {"model", "members", "features", "targets"}
    return {key: value for key, value in profile.items() if key not in skip}


def choose_online_profile(profiles: list[dict[str, Any]], baseline_wnmae: float) -> dict[str, Any]:
    eligible = [
        profile
        for profile in profiles
        if profile["predicted_compliance_rate"] >= 0.99
        and profile["candidate_batch_p95_ms"] <= 100.0
        and profile["weighted_normalized_mae"] <= baseline_wnmae + 0.02
    ]
    if not eligible:
        eligible = profiles
    return min(
        eligible,
        key=lambda profile: (
            profile["weighted_normalized_mae"],
            profile["candidate_batch_p95_ms"],
            profile["complexity_score"],
        ),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    constraint_config, _ = load_constraint_config(ROOT / "configs" / "stage2_constraints.default.json")
    decision_raw, labels, plant = load_stage1_tables(ROOT / "outputs" / "stage1_data")
    observed = build_observed_training_table(decision_raw, labels, plant)
    temporal_context = build_temporal_context(observed)
    observed_features = assemble_feature_frame(observed, temporal_context)
    observed_features = observed_features.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    feature_columns = select_feature_columns(observed_features)

    train_mask = observed_features["split"].eq("train")
    val_mask = observed_features["split"].eq("val")
    test_mask = observed_features["split"].eq("test")
    train_val_mask = train_mask | val_mask

    x_train = observed_features.loc[train_mask, feature_columns]
    y_train = observed_features.loc[train_mask, TARGET_COLUMNS]
    x_val = observed_features.loc[val_mask, feature_columns]
    y_val = observed_features.loc[val_mask, TARGET_COLUMNS]
    x_train_val = observed_features.loc[train_val_mask, feature_columns]
    y_train_val = observed_features.loc[train_val_mask, TARGET_COLUMNS]
    x_test = observed_features.loc[test_mask, feature_columns]
    y_test = observed_features.loc[test_mask, TARGET_COLUMNS]
    reference_std = y_train.std(ddof=0).replace(0.0, 1.0)

    ranker = extra_trees_factory(80, 18, 3, 20260524)()
    rank_start = time.perf_counter()
    ranker.fit(x_train, y_train)
    rank_fit_sec = float(time.perf_counter() - rank_start)
    importances = ranker.named_steps["model"].feature_importances_
    ranked_features = [
        {"feature": str(feature), "importance": float(importance)}
        for importance, feature in sorted(zip(importances, feature_columns), reverse=True)
    ]

    stage2_summary = read_json(STAGE2 / "summary.json", {})
    full_features = int(stage2_summary.get("feature_count", len(feature_columns)))
    full_test_wnmae = float(stage2_summary.get("test", {}).get("weighted_normalized_mae", np.nan))
    full_predict_ms: float | None = None
    full_profile: dict[str, Any] | None = None
    model_path = STAGE2 / "best_model.pkl"
    if model_path.exists():
        try:
            with model_path.open("rb") as handle:
                full_model = pickle.load(handle)
            set_single_thread(full_model)
            full_profile = {
                "name": "full_extra_trees_current",
                "family": "baseline_full_model",
                "features": feature_columns,
                "feature_count": full_features,
                "model": full_model,
                "fit_sec": None,
                "validation_weighted_normalized_mae": None,
                "complexity_score": float(full_features * 700),
                "member_names": [],
                "member_weights": [],
                "weighted_normalized_mae": full_test_wnmae,
                "mae_delta_vs_full_baseline": 0.0,
                "predicted_compliance_rate": float(stage2_summary.get("test", {}).get("overall_compliance", {}).get("predicted_rate", 0.0)),
            }
            full_predict_ms = timed_predict_ms(full_profile, x_test)
            full_profile["test_predict_ms"] = full_predict_ms
            full_profile.update(timing_summary(full_profile, x_test))
        except Exception:
            full_profile = None

    feature_sets = {
        24: [item["feature"] for item in ranked_features[:24]],
        40: [item["feature"] for item in ranked_features[:40]],
        64: [item["feature"] for item in ranked_features[:64]],
        96: [item["feature"] for item in ranked_features[:96]],
    }

    base_specs = [
        ("ecolite_extra_trees_24", "compact_extra_trees", 24, extra_trees_factory(140, 16, 3, 20260524), 24 * 140),
        ("ecolite_extra_trees_40", "compact_extra_trees", 40, extra_trees_factory(140, 16, 3, 20260540), 40 * 140),
        ("ecolite_extra_trees_64", "compact_extra_trees", 64, extra_trees_factory(140, 16, 3, 20260564), 64 * 140),
        ("compact_random_forest_40", "bagging_random_forest", 40, random_forest_factory(120, 16, 3, 20260640), 40 * 120),
        ("compact_gradient_boosting_40", "boosting_gradient", 40, gradient_boosting_factory(120, 3, 20260740), 40 * 120 * len(TARGET_COLUMNS)),
        ("fast_ridge_40", "linear_ridge", 40, ridge_factory(4.0), 40 * len(TARGET_COLUMNS)),
        ("balanced_extra_trees_96", "compact_extra_trees", 96, extra_trees_factory(110, 18, 2, 20260596), 96 * 110),
    ]

    profiles: list[dict[str, Any]] = []
    for name, family, top_k, factory, complexity in base_specs:
        profiles.append(
            train_profile(
                name,
                family,
                feature_sets[top_k],
                factory,
                complexity,
                x_train,
                y_train,
                x_train_val,
                y_train_val,
                x_val,
                y_val,
                x_test,
                y_test,
                reference_std,
                constraint_config,
                full_test_wnmae,
            )
        )

    by_name = {profile["name"]: profile for profile in profiles}
    ensemble_members = [
        by_name["ecolite_extra_trees_40"],
        by_name["compact_random_forest_40"],
        by_name["compact_gradient_boosting_40"],
    ]
    inverse_error_weights = [
        1.0 / max(member["validation_weighted_normalized_mae"], 1e-9)
        for member in ensemble_members
    ]
    profiles.append(
        make_ensemble(
            "validation_weighted_ensemble",
            "ensemble_error_weighted",
            ensemble_members,
            inverse_error_weights,
            x_test,
            y_test,
            reference_std,
            constraint_config,
            full_test_wnmae,
        )
    )

    latency_members = [
        by_name["ecolite_extra_trees_40"],
        by_name["compact_gradient_boosting_40"],
        by_name["fast_ridge_40"],
    ]
    latency_weights = [
        1.0 / (max(member["validation_weighted_normalized_mae"], 1e-9) * max(member["candidate_batch_p95_ms"], 1e-3))
        for member in latency_members
    ]
    profiles.append(
        make_ensemble(
            "latency_penalized_ensemble",
            "ensemble_latency_penalized",
            latency_members,
            latency_weights,
            x_test,
            y_test,
            reference_std,
            constraint_config,
            full_test_wnmae,
        )
    )

    all_profiles = ([full_profile] if full_profile else []) + profiles
    best_accuracy = min(profiles, key=lambda profile: profile["weighted_normalized_mae"])
    best_latency = min(profiles, key=lambda profile: profile["candidate_batch_p95_ms"])
    recommended = choose_online_profile(profiles, full_test_wnmae)

    timing_rows = []
    for profile in all_profiles:
        for deadline in DEADLINES_MS:
            timing_rows.append(
                {
                    "model": profile["name"],
                    "candidate_batch_size": CANDIDATE_BATCH_SIZE,
                    "deadline_ms": deadline,
                    "candidate_batch_p95_ms": profile["candidate_batch_p95_ms"],
                    "miss_rate": profile[f"miss_rate_{int(deadline)}ms"],
                    "delay_penalty": profile[f"delay_penalty_{int(deadline)}ms"],
                }
            )

    decision_benefit = read_json(DECISION_BENEFIT, {})
    plant_energy = decision_benefit.get("current_control_baseline", {})
    fixed_energy = decision_benefit.get("traditional_fixed_baseline", {})
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "EcoLite multi-model benchmark with timing simulation",
        "ranking_fit_sec": rank_fit_sec,
        "baseline_model": {
            "feature_count": full_features,
            "test_weighted_normalized_mae": full_test_wnmae,
            "predict_ms": full_predict_ms,
            "candidate_batch_p95_ms": full_profile.get("candidate_batch_p95_ms") if full_profile else None,
        },
        "best_compact_model": {
            "name": recommended["name"],
            "family": recommended["family"],
            "top_k": int(len(recommended["features"])) if "members" not in recommended else None,
            "feature_count": recommended["feature_count"],
            "feature_reduction_pct": float((full_features - min(recommended["feature_count"], full_features)) / max(full_features, 1) * 100.0),
            "weighted_normalized_mae": recommended["weighted_normalized_mae"],
            "mae_delta_vs_baseline": recommended["mae_delta_vs_full_baseline"],
            "predicted_compliance_rate": recommended["predicted_compliance_rate"],
            "predict_ms": recommended["test_predict_ms"],
            "candidate_batch_p95_ms": recommended["candidate_batch_p95_ms"],
            "single_p95_ms": recommended["single_p95_ms"],
            "speedup_vs_full": float(full_predict_ms / recommended["test_predict_ms"]) if full_predict_ms else None,
            "serving_time_reduction_pct": float((full_predict_ms - recommended["test_predict_ms"]) / full_predict_ms * 100.0) if full_predict_ms else None,
            "deadline_miss_rate_100ms": recommended["miss_rate_100ms"],
            "member_names": recommended["member_names"],
            "member_weights": recommended["member_weights"],
        },
        "best_accuracy_model": json_profile(best_accuracy),
        "best_latency_model": json_profile(best_latency),
        "recommended_online_profile": json_profile(recommended),
        "plant_energy_evidence": {
            "energy_saving_vs_current_pct": plant_energy.get("energy_saving_vs_current_pct"),
            "chemical_saving_vs_current_pct": plant_energy.get("chemical_saving_vs_current_pct"),
            "energy_saving_vs_fixed_pct": fixed_energy.get("energy_saving_pct"),
            "chemical_saving_vs_fixed_pct": fixed_energy.get("chemical_saving_pct"),
        },
        "model_search": [json_profile(profile) for profile in all_profiles],
        "timing_simulation": {
            "candidate_batch_size": CANDIDATE_BATCH_SIZE,
            "deadlines_ms": DEADLINES_MS,
            "rows": timing_rows,
            "interpretation": "Each row simulates one online control cycle that scores a 49-action local search grid before sending aeration/PAC setpoints.",
        },
        "reflection": {
            "result": f"Benchmarked {len(profiles)} optimized and ensemble profiles; recommended `{recommended['name']}` balances error, compliance, and candidate-grid latency.",
            "risk": "Validation-weighted ensembles can improve error but may multiply inference cost; online promotion should keep a hard latency budget.",
            "improvement": "Use latency-penalized ensemble scoring for offline analysis and keep the fastest eligible profile for real-time dosing.",
            "next_step": "Persist the recommended profile as a selectable online model after action-consistency regression against Safe-MARL recommendations.",
        },
    }

    (OUT / "model_efficiency_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "compact_surrogate_features.json").write_text(
        json.dumps(recommended["features"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([json_profile(profile) for profile in all_profiles]).to_csv(
        OUT / "compact_surrogate_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(timing_rows).to_csv(OUT / "timing_simulation.csv", index=False, encoding="utf-8-sig")

    report = [
        "# EcoLite 多模型效率、集成与投药时机验证",
        "",
        "## 结论",
        "",
        f"- 推荐在线 profile：`{recommended['name']}`（{recommended['family']}），测试加权标准化 MAE {recommended['weighted_normalized_mae']:.4f}，预测达标率 {recommended['predicted_compliance_rate']:.1%}。",
        f"- 全量基准模型：{full_features} 特征，测试加权标准化 MAE {full_test_wnmae:.4f}，测试集预测耗时 {full_predict_ms:.2f} ms。" if full_predict_ms else f"- 全量基准模型：{full_features} 特征，测试加权标准化 MAE {full_test_wnmae:.4f}。",
        f"- 推荐 profile 候选动作批量 P95：{recommended['candidate_batch_p95_ms']:.2f} ms，100ms 投药时机 miss rate {recommended['miss_rate_100ms']:.1%}。",
        f"- 最低误差 profile：`{best_accuracy['name']}`，MAE {best_accuracy['weighted_normalized_mae']:.4f}，候选批量 P95 {best_accuracy['candidate_batch_p95_ms']:.2f} ms。",
        f"- 最低延迟 profile：`{best_latency['name']}`，候选批量 P95 {best_latency['candidate_batch_p95_ms']:.2f} ms，MAE {best_latency['weighted_normalized_mae']:.4f}。",
        f"- Safe-MARL 当前控制对照曝气节能 {plant_energy.get('energy_saving_vs_current_pct', 0):.2f}%，PAC 节药 {plant_energy.get('chemical_saving_vs_current_pct', 0):.2f}%。",
        "",
        "## 模型组合对比",
        "",
        "| 模型 | 类型 | 特征/复杂度 | MAE | 达标率 | 单点P95ms | 49动作P95ms | 100ms miss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in all_profiles:
        report.append(
            f"| {profile['name']} | {profile['family']} | {profile['feature_count']} / {profile['complexity_score']:.0f} | "
            f"{profile['weighted_normalized_mae']:.4f} | {profile['predicted_compliance_rate']:.1%} | "
            f"{profile['single_p95_ms']:.2f} | {profile['candidate_batch_p95_ms']:.2f} | {profile['miss_rate_100ms']:.1%} |"
        )
    report.extend(
        [
            "",
            "## 投药时机仿真",
            "",
            f"- 仿真假设：每个控制周期需要一次 49 动作局部搜索，再下发曝气/PAC 设定值；若候选批量推理超过 deadline，则视为有投药时机不足风险。",
            f"- 推荐 profile 在 20/50/100/250/1000ms deadline 下的 miss rate 分别为 "
            f"{recommended['miss_rate_20ms']:.1%}、{recommended['miss_rate_50ms']:.1%}、{recommended['miss_rate_100ms']:.1%}、"
            f"{recommended['miss_rate_250ms']:.1%}、{recommended['miss_rate_1000ms']:.1%}。",
            "",
            "## 分步反思",
            "",
            "- 结果：本轮不再只验证单一 ExtraTrees 特征裁剪，而是同时比较袋装树、随机森林、梯度提升、线性 Ridge、误差加权融合和延迟惩罚融合。",
            "- 风险：集成融合虽然可能降低误差，但需要多次模型推理；在短 deadline 工况下应选择低延迟 profile 或延迟惩罚融合。",
            "- 改进：下一步可把推荐 profile 固化为在线推理 artifact，并在 `/api/rl/recommend` 中记录真实候选搜索耗时。",
        ]
    )
    (OUT / "model_efficiency_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(OUT / "model_efficiency_summary.json"),
                "profiles": len(all_profiles),
                "recommended": recommended["name"],
                "candidate_batch_p95_ms": recommended["candidate_batch_p95_ms"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
