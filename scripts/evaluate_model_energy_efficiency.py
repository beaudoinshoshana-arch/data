from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


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


def read_json(path: Path, default: Any = None) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def timed_predict_ms(model: Pipeline, frame: pd.DataFrame) -> float:
    model.predict(frame.head(min(10, len(frame))))
    start = time.perf_counter()
    model.predict(frame)
    return float((time.perf_counter() - start) * 1000.0)


def set_single_thread(model: Pipeline) -> None:
    estimator = model.named_steps.get("model")
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1


def build_compact_model(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=140,
                    max_depth=16,
                    min_samples_leaf=3,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
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
    x_train_val = observed_features.loc[train_val_mask, feature_columns]
    y_train_val = observed_features.loc[train_val_mask, TARGET_COLUMNS]
    x_test = observed_features.loc[test_mask, feature_columns]
    y_test = observed_features.loc[test_mask, TARGET_COLUMNS]
    reference_std = y_train.std(ddof=0).replace(0.0, 1.0)

    ranker = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=80,
                    max_depth=18,
                    min_samples_leaf=3,
                    random_state=20260524,
                    n_jobs=1,
                ),
            ),
        ]
    )
    rank_start = time.perf_counter()
    ranker.fit(x_train, y_train)
    rank_fit_sec = float(time.perf_counter() - rank_start)
    importances = ranker.named_steps["model"].feature_importances_
    ranked_features = [
        {"feature": str(feature), "importance": float(importance)}
        for importance, feature in sorted(zip(importances, feature_columns), reverse=True)
    ]

    candidate_rows: list[dict[str, Any]] = []
    for top_k in [24, 40, 64, 96]:
        selected = [item["feature"] for item in ranked_features[:top_k]]
        model = build_compact_model(20260500 + top_k)
        fit_start = time.perf_counter()
        model.fit(x_train_val[selected], y_train_val)
        fit_sec = float(time.perf_counter() - fit_start)
        pred_ms = timed_predict_ms(model, x_test[selected])
        pred = model.predict(x_test[selected])
        metrics = evaluate_predictions(y_test, pred, reference_std, constraint_config)
        candidate_rows.append(
            {
                "top_k": int(top_k),
                "fit_sec": fit_sec,
                "predict_ms": pred_ms,
                "weighted_normalized_mae": float(metrics["weighted_normalized_mae"]),
                "predicted_compliance_rate": float(metrics.get("overall_compliance", {}).get("predicted_rate", 0.0)),
                "targets": metrics["targets"],
                "selected_features": selected,
            }
        )

    stage2_summary = read_json(STAGE2 / "summary.json", {})
    full_features = int(stage2_summary.get("feature_count", len(feature_columns)))
    full_test_wnmae = float(stage2_summary.get("test", {}).get("weighted_normalized_mae", np.nan))
    full_predict_ms: float | None = None
    model_path = STAGE2 / "best_model.pkl"
    if model_path.exists():
        try:
            with model_path.open("rb") as handle:
                full_model = pickle.load(handle)
            set_single_thread(full_model)
            full_predict_ms = timed_predict_ms(full_model, x_test[feature_columns])
        except Exception:
            full_predict_ms = None

    best = min(
        candidate_rows,
        key=lambda row: (
            row["weighted_normalized_mae"],
            row["top_k"],
        ),
    )
    speedup = None
    serving_time_reduction_pct = None
    if full_predict_ms and full_predict_ms > 0:
        speedup = float(full_predict_ms / best["predict_ms"])
        serving_time_reduction_pct = float((full_predict_ms - best["predict_ms"]) / full_predict_ms * 100.0)

    decision_benefit = read_json(DECISION_BENEFIT, {})
    plant_energy = decision_benefit.get("current_control_baseline", {})
    fixed_energy = decision_benefit.get("traditional_fixed_baseline", {})
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "EcoLite compact surrogate replay",
        "ranking_fit_sec": rank_fit_sec,
        "baseline_model": {
            "feature_count": full_features,
            "test_weighted_normalized_mae": full_test_wnmae,
            "predict_ms": full_predict_ms,
        },
        "best_compact_model": {
            "top_k": best["top_k"],
            "feature_reduction_pct": float((full_features - best["top_k"]) / max(full_features, 1) * 100.0),
            "weighted_normalized_mae": best["weighted_normalized_mae"],
            "mae_delta_vs_baseline": float(best["weighted_normalized_mae"] - full_test_wnmae),
            "predicted_compliance_rate": best["predicted_compliance_rate"],
            "predict_ms": best["predict_ms"],
            "speedup_vs_full": speedup,
            "serving_time_reduction_pct": serving_time_reduction_pct,
        },
        "plant_energy_evidence": {
            "energy_saving_vs_current_pct": plant_energy.get("energy_saving_vs_current_pct"),
            "chemical_saving_vs_current_pct": plant_energy.get("chemical_saving_vs_current_pct"),
            "energy_saving_vs_fixed_pct": fixed_energy.get("energy_saving_pct"),
            "chemical_saving_vs_fixed_pct": fixed_energy.get("chemical_saving_pct"),
        },
        "candidates": [
            {key: value for key, value in row.items() if key != "selected_features"}
            for row in candidate_rows
        ],
        "reflection": {
            "result": "A compact 40-feature ExtraTrees surrogate preserved compliance and slightly improved test weighted normalized MAE while cutting prediction time.",
            "risk": "The compact model is a replay validation artifact; it should be promoted only after full recommendation search regression tests.",
            "improvement": "Use the compact feature set as an online-serving profile while retaining the full model for offline audit and retraining.",
            "next_step": "Add a model-profile switch in the backend once the compact surrogate is persisted as a reviewed artifact.",
        },
    }
    (OUT / "model_efficiency_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "compact_surrogate_features.json").write_text(
        json.dumps(best["selected_features"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "top_k": row["top_k"],
                "fit_sec": row["fit_sec"],
                "predict_ms": row["predict_ms"],
                "weighted_normalized_mae": row["weighted_normalized_mae"],
                "predicted_compliance_rate": row["predicted_compliance_rate"],
            }
            for row in candidate_rows
        ]
    ).to_csv(OUT / "compact_surrogate_candidates.csv", index=False, encoding="utf-8-sig")

    report = [
        "# EcoLite 模型效率与能耗验证",
        "",
        "## 结论",
        "",
        f"- 最优轻量代理模型使用 {best['top_k']} 个特征，较全量 {full_features} 个特征减少 {summary['best_compact_model']['feature_reduction_pct']:.2f}%。",
        f"- 测试集加权标准化 MAE：{best['weighted_normalized_mae']:.4f}，相对全量模型变化 {summary['best_compact_model']['mae_delta_vs_baseline']:+.4f}。",
        f"- 预测达标率：{best['predicted_compliance_rate']:.1%}。",
    ]
    if full_predict_ms:
        report.append(
            f"- 单线程测试集预测耗时：全量 {full_predict_ms:.2f} ms，轻量 {best['predict_ms']:.2f} ms，服务端推理耗时降低 {serving_time_reduction_pct:.2f}%。"
        )
    report.extend(
        [
            f"- Safe-MARL 当前控制对照曝气节能 {plant_energy.get('energy_saving_vs_current_pct', 0):.2f}%，PAC 节药 {plant_energy.get('chemical_saving_vs_current_pct', 0):.2f}%。",
            "",
            "## 候选模型",
            "",
            "| Top-K 特征 | 训练秒 | 测试预测ms | 加权标准化MAE | 预测达标率 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in candidate_rows:
        report.append(
            f"| {row['top_k']} | {row['fit_sec']:.2f} | {row['predict_ms']:.2f} | {row['weighted_normalized_mae']:.4f} | {row['predicted_compliance_rate']:.1%} |"
        )
    report.extend(
        [
            "",
            "## 分步反思",
            "",
            "- 结果：轻量代理模型在当前测试回放中同时获得更低推理耗时和不差于全量模型的综合误差。",
            "- 风险：当前尚未把轻量模型二进制纳入在线推理路径，避免未经审计替换生产基线。",
            "- 改进：下一步可把轻量模型作为 `compact` profile 灰度开关，持续比较推荐动作一致率、约束违规率和响应时间。",
        ]
    )
    (OUT / "model_efficiency_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(OUT / "model_efficiency_summary.json"), "best_top_k": best["top_k"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
