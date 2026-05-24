from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PLANT_METRIC_MAP = {
    "influent_cod_mgL": ("COD", "influent", "mg/L"),
    "influent_bod_mgL": ("BOD", "influent", "mg/L"),
    "influent_nh3n_mgL": ("NH3N", "influent", "mg/L"),
    "influent_tp_mgL": ("TP", "influent", "mg/L"),
    "influent_tn_mgL": ("TN", "influent", "mg/L"),
    "influent_flow_m3h": ("FLOW", "influent", "m3/h"),
    "reactor_do_mgL": ("DO", "reactor", "mg/L"),
    "sludge_mlss_mgL": ("MLSS", "reactor", "mg/L"),
    "effluent_cod_mgL": ("COD", "effluent", "mg/L"),
    "effluent_nh3n_mgL": ("NH3N", "effluent", "mg/L"),
    "effluent_tp_mgL": ("TP", "effluent", "mg/L"),
    "effluent_tn_mgL": ("TN", "effluent", "mg/L"),
    "effluent_flow_m3h": ("FLOW", "effluent", "m3/h"),
    "effluent_ph": ("pH", "effluent", "pH"),
    "effluent_temp_c": ("TEMP", "effluent", "degC"),
}
WQP_CHARACTERISTIC_MAP = {
    "Ammonia": "NH3N",
    "Ammonia-nitrogen": "NH3N",
    "Nitrogen, ammonia": "NH3N",
    "Chemical oxygen demand": "COD",
    "Biochemical oxygen demand": "BOD",
    "Phosphorus": "TP",
    "Phosphate": "TP",
    "Total nitrogen": "TN",
    "pH": "pH",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build external fusion datasets for the WWTP competition system.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage1-output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "stage1_data")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "external_sources.default.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "fusion_data")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_local_plant(stage1_output: Path) -> pd.DataFrame:
    plant = pd.read_csv(stage1_output / "plant_real_hourly" / "plant_real_hourly.csv", parse_dates=["timestamp"])
    pieces: list[pd.DataFrame] = []
    for column, (metric, role, unit) in PLANT_METRIC_MAP.items():
        if column not in plant.columns:
            continue
        frame = plant[["timestamp", column]].rename(columns={column: "value"}).copy()
        frame["source_domain"] = "local_plant"
        frame["source_type"] = "local_real_hourly"
        frame["plant_id"] = "local_06yzzx"
        frame["plant_name"] = "06yzzx 单厂运行数据"
        frame["metric_code"] = metric
        frame["metric_role"] = role
        frame["unit"] = unit
        frame["scenario_tag"] = "observed"
        frame["quality_grade"] = "gold"
        frame["data_use"] = "supervised_training"
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True).dropna(subset=["value"])


def normalize_public_monitor(stage1_output: Path) -> pd.DataFrame:
    public = pd.read_csv(stage1_output / "public_monitor_long" / "public_monitor_long.csv", parse_dates=["timestamp"])
    out = public.rename(columns={"metric_value": "value"}).copy()
    out["source_domain"] = "china_public_monitor"
    out["plant_id"] = out["source_city"].astype(str) + "::" + out["enterprise_name"].astype(str)
    out["plant_name"] = out["enterprise_name"].astype(str)
    out["metric_role"] = "effluent"
    out["unit"] = out["metric_code"].map(lambda x: "pH" if x == "pH" else ("m3/h" if x == "FLOW" else "mg/L"))
    out["scenario_tag"] = "external_prior"
    out["quality_grade"] = "silver"
    out["data_use"] = "domain_prior_and_robustness"
    return out[
        [
            "timestamp",
            "source_domain",
            "source_type",
            "plant_id",
            "plant_name",
            "metric_code",
            "metric_role",
            "value",
            "unit",
            "scenario_tag",
            "quality_grade",
            "data_use",
        ]
    ].dropna(subset=["timestamp", "metric_code", "value"])


def parse_wqp_response(text: str, url: str, max_rows: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    from io import StringIO

    raw = pd.read_csv(StringIO(text), low_memory=False)
    if raw.empty:
        return pd.DataFrame(), {"status": "empty", "rows": 0, "url": url}
    raw = raw.head(max_rows).copy()
    time_col = "ActivityStartDate"
    value_col = "ResultMeasureValue"
    characteristic_col = "CharacteristicName"
    if value_col not in raw.columns or characteristic_col not in raw.columns:
        return pd.DataFrame(), {"status": "schema_mismatch", "columns": list(raw.columns)[:30], "url": url}
    site_id = raw["MonitoringLocationIdentifier"].astype(str) if "MonitoringLocationIdentifier" in raw.columns else pd.Series(["unknown"] * len(raw))
    site_name = raw["MonitoringLocationName"].astype(str) if "MonitoringLocationName" in raw.columns else site_id
    unit = raw["ResultMeasure/MeasureUnitCode"].astype(str) if "ResultMeasure/MeasureUnitCode" in raw.columns else pd.Series(["unknown"] * len(raw))
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw.get(time_col), errors="coerce"),
            "source_domain": "us_wqp",
            "source_type": "epa_usgs_wqp_api",
            "plant_id": site_id,
            "plant_name": site_name,
            "metric_code": raw[characteristic_col].map(WQP_CHARACTERISTIC_MAP).fillna(raw[characteristic_col].astype(str)),
            "metric_role": "ambient_or_effluent",
            "value": pd.to_numeric(raw[value_col], errors="coerce"),
            "unit": unit,
            "scenario_tag": "external_prior",
            "quality_grade": "bronze",
            "data_use": "domain_prior_and_robustness",
        }
    ).dropna(subset=["timestamp", "value"])
    if out.empty:
        return out, {"status": "empty_after_parse", "rows": 0, "url": url}
    return out, {"status": "ok", "rows": int(len(out)), "url": url}


def fetch_wqp(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not config.get("enabled", False):
        return pd.DataFrame(), {"status": "disabled"}
    attempts = [config.get("params", {})] + list(config.get("fallback_queries", []))
    errors: list[dict[str, Any]] = []
    try:
        for params in attempts:
            try:
                response = requests.get(config["url"], params=params, timeout=float(config.get("timeout_sec", 20)))
                response.raise_for_status()
                out, status = parse_wqp_response(response.text, response.url, int(config.get("max_rows", 2000)))
                if not out.empty:
                    return out, {**status, "attempts": len(errors) + 1}
                errors.append(status)
            except Exception as exc:  # noqa: BLE001 - continue to fallback query.
                errors.append({"status": "failed", "error": str(exc), "params": params})
        return pd.DataFrame(), {"status": "failed", "attempts": len(attempts), "errors": errors}
    except Exception as exc:  # noqa: BLE001 - keep ETL resilient for offline demos.
        return pd.DataFrame(), {"status": "failed", "error": str(exc)}


def read_kaggle_local(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not config.get("enabled", False):
        return pd.DataFrame(), {"status": "disabled"}
    pattern = str(root / config.get("local_glob", "external_data/kaggle/**/*.csv"))
    files = glob.glob(pattern, recursive=True)
    if not files:
        return pd.DataFrame(), {"status": "not_found", "pattern": pattern}
    pieces: list[pd.DataFrame] = []
    for file in files[:5]:
        raw = pd.read_csv(file, nrows=5000)
        lower = {column.lower(): column for column in raw.columns}
        time_col = next((lower[key] for key in lower if "time" in key or "date" in key), None)
        if time_col is None:
            continue
        for metric, aliases in {
            "COD": ["cod"],
            "BOD": ["bod"],
            "NH3N": ["nh3", "ammonia"],
            "TP": ["tp", "phosphorus"],
            "TN": ["tn", "nitrogen"],
            "FLOW": ["flow"],
        }.items():
            col = next((name for key, name in lower.items() if any(alias in key for alias in aliases)), None)
            if col is None:
                continue
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(raw[time_col], errors="coerce"),
                    "source_domain": "kaggle_open_dataset",
                    "source_type": "kaggle_local_csv",
                    "plant_id": Path(file).stem,
                    "plant_name": Path(file).stem,
                    "metric_code": metric,
                    "metric_role": "mixed",
                    "value": pd.to_numeric(raw[col], errors="coerce"),
                    "unit": "unknown",
                    "scenario_tag": "external_prior",
                    "quality_grade": "bronze",
                    "data_use": "domain_prior_and_robustness",
                }
            )
            pieces.append(frame)
    if not pieces:
        return pd.DataFrame(), {"status": "schema_unmapped", "files": files[:5]}
    return pd.concat(pieces, ignore_index=True).dropna(subset=["timestamp", "value"]), {"status": "ok", "files": files[:5]}


def build_scenario_library(fusion: pd.DataFrame, output: Path) -> pd.DataFrame:
    summary = (
        fusion.groupby(["source_domain", "metric_code", "metric_role"], dropna=False)
        .agg(rows=("value", "size"), plants=("plant_id", "nunique"), mean=("value", "mean"), p05=("value", lambda s: s.quantile(0.05)), p95=("value", lambda s: s.quantile(0.95)))
        .reset_index()
    )
    scenario = summary.copy()
    scenario["domain_shift_hint"] = scenario.apply(
        lambda r: "high_range_prior" if r["p95"] > r["mean"] * 1.8 and r["mean"] > 0 else "normal_range_prior",
        axis=1,
    )
    scenario["recommended_use"] = scenario["source_domain"].map(
        {
            "local_plant": "train_validate_test",
            "china_public_monitor": "public_distribution_and_shock_library",
            "us_wqp": "external_domain_generalization_check",
            "kaggle_open_dataset": "external_domain_generalization_check",
        }
    ).fillna("external_reference")
    scenario.to_csv(output / "scenario_library.csv", index=False, encoding="utf-8-sig")
    return scenario


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    registry: dict[str, Any] = {"generated_at": datetime.now().isoformat(timespec="seconds"), "sources": {}}

    local = normalize_local_plant(args.stage1_output)
    public = normalize_public_monitor(args.stage1_output)
    registry["sources"]["local_plant"] = {"status": "ok", "rows": int(len(local))}
    registry["sources"]["china_public_monitor"] = {"status": "ok", "rows": int(len(public))}

    wqp, wqp_status = fetch_wqp(config.get("wqp", {}))
    kaggle, kaggle_status = read_kaggle_local(args.root, config.get("kaggle", {}))
    registry["sources"]["wqp"] = wqp_status
    registry["sources"]["echo"] = {"status": "metadata_reference", "url": config.get("echo", {}).get("url")}
    registry["sources"]["kaggle"] = kaggle_status

    frames = [frame for frame in [local, public, wqp, kaggle] if not frame.empty]
    fusion = pd.concat(frames, ignore_index=True)
    fusion["value"] = pd.to_numeric(fusion["value"], errors="coerce")
    fusion = fusion.dropna(subset=["timestamp", "metric_code", "value"]).sort_values("timestamp")
    fusion.to_csv(args.output / "fusion_long.csv", index=False, encoding="utf-8-sig")
    scenario = build_scenario_library(fusion, args.output)

    registry["artifacts"] = {
        "fusion_long": str(args.output / "fusion_long.csv"),
        "scenario_library": str(args.output / "scenario_library.csv"),
    }
    registry["summary"] = {
        "fusion_rows": int(len(fusion)),
        "source_domains": {str(k): int(v) for k, v in fusion["source_domain"].value_counts().items()},
        "metric_counts": {str(k): int(v) for k, v in fusion["metric_code"].value_counts().items()},
        "scenario_rows": int(len(scenario)),
    }
    (args.output / "source_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(registry["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
