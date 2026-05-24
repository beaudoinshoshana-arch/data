from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


WASTEWATER_PATTERN = re.compile(r"污水|水务|净化|排水")
TAIZHOU_ALLOWED_METRICS = {"COD", "NH3N", "TP", "TN", "pH", "FLOW"}
CORE_FEATURES = [
    "influent_cod_mgL",
    "influent_bod_mgL",
    "influent_nh3n_mgL",
    "influent_tp_mgL",
    "reactor_do_mgL",
    "sludge_mlss_mgL",
    "aeration_intensity_pct",
    "chemical_dose_kgph",
]
BASE_CORE_INPUTS = [
    "influent_cod_mgL",
    "influent_nh3n_mgL",
    "influent_tp_mgL",
    "influent_flow_m3h",
    "reactor_do_mgL",
    "sludge_mlss_mgL",
]
ENGINEERING_RANGES = {
    "influent_cod_mgL": (0.0, 1000.0),
    "influent_bod_mgL": (0.0, 600.0),
    "influent_nh3n_mgL": (0.0, 100.0),
    "influent_tp_mgL": (0.0, 20.0),
    "reactor_do_mgL": (0.0, 20.0),
    "sludge_mlss_mgL": (500.0, 20000.0),
    "aeration_intensity_pct": (0.0, 100.0),
    "chemical_dose_kgph": (0.0, 500.0),
}
TAIZHOU_METRIC_ALIASES = {
    "化学需氧量": ("COD", "COD"),
    "氨氮": ("氨氮", "NH3N"),
    "总磷": ("总磷", "TP"),
    "总氮": ("总氮", "TN"),
    "pH值": ("pH", "pH"),
    "PH值": ("pH", "pH"),
    "废水瞬时流量": ("流量", "FLOW"),
}
TAIZHOU_SUFFIX_ALIASES = {
    "COD": ("COD", "COD"),
    "NH3": ("氨氮", "NH3N"),
    "NH4": ("氨氮", "NH3N"),
    "TP": ("总磷", "TP"),
    "TN": ("总氮", "TN"),
    "PH": ("pH", "pH"),
    "FLOW": ("流量", "FLOW"),
    "ALLFLOW": ("流量", "FLOW"),
}
JINAN_METRIC_TAGS = {
    "codauditvalue": ("COD", "COD"),
    "nh4auditvalue": ("氨氮", "NH3N"),
    "tpauditvalue": ("总磷", "TP"),
    "tnauditvalue": ("总氮", "TN"),
    "phauditvalue": ("pH", "pH"),
}


@dataclass(frozen=True)
class PlantSeriesSpec:
    column: str
    relative_path: str


PLANT_SERIES_SPECS = [
    PlantSeriesSpec("influent_cod_mgL", r"水厂数据-06yzzx\水厂数据\进水\进水COD.xlsx"),
    PlantSeriesSpec("influent_nh3n_mgL", r"水厂数据-06yzzx\水厂数据\进水\进水氨氮.xlsx"),
    PlantSeriesSpec("influent_tp_mgL", r"水厂数据-06yzzx\水厂数据\进水\进水总磷.xlsx"),
    PlantSeriesSpec("influent_flow_m3h", r"水厂数据-06yzzx\水厂数据\进水\进水流量.xlsx"),
    PlantSeriesSpec("influent_tn_mgL", r"水厂数据-06yzzx\水厂数据\进水\进水总氮.xlsx"),
    PlantSeriesSpec("reactor_do_mgL", r"水厂数据-06yzzx\水厂数据\3号\三号池2#DO.xlsx"),
    PlantSeriesSpec("reactor_orp_mv", r"水厂数据-06yzzx\水厂数据\4号\四号池1#ORP.xlsx"),
    PlantSeriesSpec("reactor_no3_mgL", r"水厂数据-06yzzx\水厂数据\4号\四号池1#硝氮.xlsx"),
    PlantSeriesSpec("internal_nh3_mgL", r"水厂数据-06yzzx\水厂数据\4号\四号池1#氨氮.xlsx"),
    PlantSeriesSpec("sludge_mlss_mgL", r"水厂数据-06yzzx\水厂数据\4号\四号池污泥浓度.xlsx"),
    PlantSeriesSpec("effluent_cod_mgL", r"水厂数据-06yzzx\水厂数据\出水\出水COD.xlsx"),
    PlantSeriesSpec("effluent_nh3n_mgL", r"水厂数据-06yzzx\水厂数据\出水\出水NH3N.xlsx"),
    PlantSeriesSpec("effluent_tp_mgL", r"水厂数据-06yzzx\水厂数据\出水\出水TP.xlsx"),
    PlantSeriesSpec("effluent_tn_mgL", r"水厂数据-06yzzx\水厂数据\出水\出水TN.xlsx"),
    PlantSeriesSpec("effluent_flow_m3h", r"水厂数据-06yzzx\水厂数据\出水\出水流量.xlsx"),
    PlantSeriesSpec("effluent_ph", r"水厂数据-06yzzx\水厂数据\出水\出水PH.xlsx"),
    PlantSeriesSpec("effluent_temp_c", r"水厂数据-06yzzx\水厂数据\出水\出水温度.xlsx"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage-1 wastewater datasets.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Workspace root that contains the raw data folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "stage1_data",
        help="Output directory for processed datasets.",
    )
    return parser.parse_args()


def round_to_nearest_hour(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    shifted = ts.where(ts.dt.minute < 30, ts + pd.Timedelta(hours=1))
    return shifted.dt.floor("h")


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "plant": output_dir / "plant_real_hourly",
        "public": output_dir / "public_monitor_long",
        "decision": output_dir / "decision_dataset",
        "reports": output_dir / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_excel_series(path: Path, column_name: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "时间" not in df.columns or "数值" not in df.columns:
        raise ValueError(f"{path} does not contain expected 时间/数值 columns.")
    out = pd.DataFrame(
        {
            "timestamp": round_to_nearest_hour(df["时间"]),
            column_name: pd.to_numeric(df["数值"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp"])
    out = out.groupby("timestamp", as_index=False)[column_name].median()
    return out.sort_values("timestamp").reset_index(drop=True)


def build_plant_hourly(root: Path) -> tuple[pd.DataFrame, dict]:
    timeline = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01 00:00:00",
                "2025-07-10 13:00:00",
                freq="h",
            )
        }
    )
    plant = timeline.copy()
    source_files: dict[str, str] = {}
    for spec in PLANT_SERIES_SPECS:
        path = root / spec.relative_path
        source_files[spec.column] = str(path)
        series_df = read_excel_series(path, spec.column)
        plant = plant.merge(series_df, on="timestamp", how="left")

    plant["source_flag"] = "local_real"
    plant = add_next_hour_labels(plant)
    plant, qc_summary = apply_quality_rules(plant)
    plant["core_complete_before_fill"] = plant[[
        "influent_cod_mgL",
        "influent_nh3n_mgL",
        "influent_tp_mgL",
        "influent_flow_m3h",
        "reactor_do_mgL",
        "sludge_mlss_mgL",
        "label_next_effluent_tn_mgL",
    ]].notna().all(axis=1)
    meta = {
        "source_files": source_files,
        "row_count": int(len(plant)),
        "timestamp_min": plant["timestamp"].min().isoformat(),
        "timestamp_max": plant["timestamp"].max().isoformat(),
        "core_complete_before_fill_rows": int(plant["core_complete_before_fill"].sum()),
        "qc_summary": qc_summary,
    }
    return plant, meta


def add_next_hour_labels(df: pd.DataFrame) -> pd.DataFrame:
    label_map = {
        "label_next_effluent_cod_mgL": "effluent_cod_mgL",
        "label_next_effluent_nh3n_mgL": "effluent_nh3n_mgL",
        "label_next_effluent_tp_mgL": "effluent_tp_mgL",
        "label_next_effluent_tn_mgL": "effluent_tn_mgL",
    }
    out = df.copy()
    for label_col, base_col in label_map.items():
        out[label_col] = out[base_col].shift(-1)
    return out


def hampel_mask(series: pd.Series, window: int = 12, n_sigmas: float = 3.0) -> pd.Series:
    x = series.astype(float)
    rolling_median = x.rolling(window=window * 2 + 1, center=True, min_periods=window).median()
    diff = (x - rolling_median).abs()
    mad = diff.rolling(window=window * 2 + 1, center=True, min_periods=window).median()
    threshold = n_sigmas * 1.4826 * mad
    mask = diff > threshold
    mask = mask & x.notna() & rolling_median.notna()
    mask = mask.fillna(False)
    return mask


def flatline_mask(series: pd.Series, min_run_length: int = 25) -> pd.Series:
    rounded = series.round(6)
    change = rounded.ne(rounded.shift()) | rounded.isna() | rounded.shift().isna()
    group_id = change.cumsum()
    run_lengths = rounded.groupby(group_id).transform("size")
    return rounded.notna() & (run_lengths >= min_run_length)


def apply_engineering_range(series: pd.Series, lower: float | None, upper: float | None) -> tuple[pd.Series, pd.Series]:
    out = pd.to_numeric(series, errors="coerce")
    negative_mask = out < 0
    out = out.mask(negative_mask)
    range_mask = pd.Series(False, index=out.index)
    if lower is not None:
        range_mask = range_mask | (out < lower)
    if upper is not None:
        range_mask = range_mask | (out > upper)
    out = out.mask(range_mask)
    return out, (negative_mask | range_mask)


def apply_quality_rules(plant: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = plant.copy()
    qc_summary: dict[str, dict[str, int]] = {}
    numeric_columns = [col for col in df.columns if col not in {"timestamp", "source_flag"} and pd.api.types.is_numeric_dtype(df[col])]
    for column in numeric_columns:
        lower, upper = ENGINEERING_RANGES.get(column, (None, None))
        cleaned, invalid_mask = apply_engineering_range(df[column], lower, upper)
        negative_mask = pd.to_numeric(df[column], errors="coerce") < 0
        range_mask = invalid_mask & ~negative_mask
        hampel = hampel_mask(cleaned)
        flatline = flatline_mask(cleaned)
        df[column] = cleaned
        df[f"flag_negative_or_range_{column}"] = invalid_mask
        df[f"flag_hampel_{column}"] = hampel
        df[f"flag_flatline_{column}"] = flatline
        qc_summary[column] = {
            "invalid_or_range_rows": int(invalid_mask.sum()),
            "hampel_rows": int(hampel.sum()),
            "flatline_rows": int(flatline.sum()),
        }
    return df, qc_summary


def fill_short_and_medium_gaps(series: pd.Series, index: pd.Series) -> pd.Series:
    data = pd.Series(series.to_numpy(), index=pd.DatetimeIndex(index), dtype=float)
    original_missing = data.isna()
    group_id = original_missing.ne(original_missing.shift(fill_value=False)).cumsum()
    gap_sizes = original_missing.groupby(group_id).transform("sum")

    interpolated = data.interpolate(method="time", limit=3, limit_area="inside")
    filled = data.copy()
    short_gap_mask = original_missing & gap_sizes.between(1, 3)
    filled.loc[short_gap_mask] = interpolated.loc[short_gap_mask]

    history = pd.concat([filled.shift(24 * day) for day in range(1, 8)], axis=1)
    historical_median = history.median(axis=1, skipna=True)
    medium_gap_mask = filled.isna() & original_missing & gap_sizes.between(4, 24)
    filled.loc[medium_gap_mask] = historical_median.loc[medium_gap_mask]
    return pd.Series(filled.to_numpy(), index=series.index)


def compute_bod_ratio(df: pd.DataFrame) -> pd.Series:
    ratio = pd.Series(0.45, index=df.index, dtype=float)
    return ratio.rolling(window=6, min_periods=1).mean().clip(0.35, 0.60)


def compute_decision_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ratio = compute_bod_ratio(out)
    out["influent_bod_mgL"] = (out["influent_cod_mgL"] * ratio).clip(*ENGINEERING_RANGES["influent_bod_mgL"])
    out["aeration_intensity_pct"] = np.clip(
        35
        + 12 * np.maximum(0, 2.5 - out["reactor_do_mgL"])
        + 0.6 * out["influent_nh3n_mgL"]
        + 0.04 * out["influent_cod_mgL"]
        + 0.003 * out["influent_flow_m3h"]
        - 0.002 * (out["sludge_mlss_mgL"] - 5000),
        10,
        100,
    )
    out["chemical_dose_pac_mgL"] = np.clip(
        6
        + 10 * out["influent_tp_mgL"]
        + 0.015 * np.maximum(out["influent_cod_mgL"] - 120, 0)
        + 0.04 * np.maximum(out["influent_nh3n_mgL"] - 15, 0),
        2,
        60,
    )
    out["chemical_dose_kgph"] = (
        out["chemical_dose_pac_mgL"] * out["influent_flow_m3h"] / 1000.0
    ).clip(*ENGINEERING_RANGES["chemical_dose_kgph"])
    return out


def split_by_time(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    total = len(ordered)
    train_end = int(np.floor(total * 0.70))
    val_end = int(np.floor(total * 0.85))
    split = np.full(total, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    ordered["split"] = split
    return ordered


def scenario_transform(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    out = df.copy()
    out["scenario_tag"] = tag
    out["is_simulated"] = tag != "observed"
    if tag == "load_up":
        out["influent_cod_mgL"] *= 1.20
        out["influent_nh3n_mgL"] *= 1.25
        out["influent_tp_mgL"] *= 1.15
        out["influent_flow_m3h"] *= 1.10
        out["reactor_do_mgL"] *= 0.92
    elif tag == "rain_dilution":
        out["influent_cod_mgL"] *= 0.85
        out["influent_nh3n_mgL"] *= 0.90
        out["influent_tp_mgL"] *= 0.90
        out["influent_flow_m3h"] *= 1.20
        out["sludge_mlss_mgL"] *= 0.97
    return compute_decision_fields(out)


def build_decision_dataset(
    plant: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    work = plant.copy()
    work["exclude_due_to_flatline"] = False
    work["exclude_due_to_hampel"] = False
    for column in BASE_CORE_INPUTS:
        flat_col = f"flag_flatline_{column}"
        hampel_col = f"flag_hampel_{column}"
        work["exclude_due_to_flatline"] = work["exclude_due_to_flatline"] | work.get(flat_col, False)
        work["exclude_due_to_hampel"] = work["exclude_due_to_hampel"] | work.get(hampel_col, False)

        work[column] = work[column].mask(work.get(hampel_col, False))
        work[column] = fill_short_and_medium_gaps(work[column], work["timestamp"])

    work["exclude_due_to_long_gap"] = work[BASE_CORE_INPUTS].isna().any(axis=1)
    work["exclude_from_model"] = work["exclude_due_to_flatline"] | work["exclude_due_to_long_gap"]

    base_rows = work.loc[
        ~work["exclude_from_model"],
        [
            "timestamp",
            "source_flag",
            "influent_cod_mgL",
            "influent_nh3n_mgL",
            "influent_tp_mgL",
            "influent_flow_m3h",
            "reactor_do_mgL",
            "sludge_mlss_mgL",
            "label_next_effluent_cod_mgL",
            "label_next_effluent_nh3n_mgL",
            "label_next_effluent_tp_mgL",
            "label_next_effluent_tn_mgL",
        ],
    ].copy()
    base_rows = split_by_time(base_rows)

    observed = scenario_transform(base_rows, "observed")
    load_up = scenario_transform(base_rows, "load_up")
    rain_dilution = scenario_transform(base_rows, "rain_dilution")
    decision_raw = pd.concat([observed, load_up, rain_dilution], ignore_index=True)
    decision_raw = decision_raw.sort_values(["timestamp", "scenario_tag"]).reset_index(drop=True)

    label_table = base_rows[
        [
            "timestamp",
            "split",
            "source_flag",
            "label_next_effluent_cod_mgL",
            "label_next_effluent_nh3n_mgL",
            "label_next_effluent_tp_mgL",
            "label_next_effluent_tn_mgL",
        ]
    ].copy()

    feature_columns = CORE_FEATURES + ["chemical_dose_pac_mgL"]
    scaler = RobustScaler()
    train_mask = decision_raw["split"] == "train"
    scaler.fit(decision_raw.loc[train_mask, feature_columns])
    decision_scaled = decision_raw.copy()
    decision_scaled[feature_columns] = scaler.transform(decision_raw[feature_columns])

    split_span = {
        split_name: {
            "rows": int((base_rows["split"] == split_name).sum()),
            "timestamp_min": (
                base_rows.loc[base_rows["split"] == split_name, "timestamp"].min().isoformat()
                if (base_rows["split"] == split_name).any()
                else None
            ),
            "timestamp_max": (
                base_rows.loc[base_rows["split"] == split_name, "timestamp"].max().isoformat()
                if (base_rows["split"] == split_name).any()
                else None
            ),
        }
        for split_name in ["train", "val", "test"]
    }
    meta = {
        "base_rows_after_fill": int(len(base_rows)),
        "decision_rows": int(len(decision_raw)),
        "split_span": split_span,
        "feature_columns": feature_columns,
        "scaler_center": dict(zip(feature_columns, scaler.center_.tolist(), strict=True)),
        "scaler_scale": dict(zip(feature_columns, scaler.scale_.tolist(), strict=True)),
        "excluded_due_to_flatline_rows": int(work["exclude_due_to_flatline"].sum()),
        "excluded_due_to_long_gap_rows": int(work["exclude_due_to_long_gap"].sum()),
    }
    return decision_raw, decision_scaled, label_table, meta


def infer_taizhou_metric_name(row: pd.Series) -> tuple[str | None, str | None]:
    yzmc_raw = row.get("YZMC", "")
    yzmc = "" if pd.isna(yzmc_raw) else str(yzmc_raw).strip()
    if yzmc in TAIZHOU_METRIC_ALIASES:
        return TAIZHOU_METRIC_ALIASES[yzmc]

    xh_raw = row.get("XH", "")
    xh = "" if pd.isna(xh_raw) else str(xh_raw).strip()
    suffix_match = re.search(r"([A-Za-z]+)$", xh)
    if suffix_match:
        suffix = suffix_match.group(1).upper()
        if suffix in TAIZHOU_SUFFIX_ALIASES:
            return TAIZHOU_SUFFIX_ALIASES[suffix]
    return None, None


def read_csv_with_fallbacks(path: Path, usecols: Iterable[str]) -> pd.DataFrame:
    last_error: Exception | None = None
    expected = list(usecols)
    for encoding in ["utf-8", "utf-8-sig", "gb18030", "gbk"]:
        for encoding_errors in ["strict", "ignore"]:
            try:
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    encoding_errors=encoding_errors,
                    low_memory=False,
                )
                for column in expected:
                    if column not in df.columns:
                        df[column] = pd.NA
                return df[expected]
            except Exception as exc:
                last_error = exc
    raise RuntimeError(f"Failed to read {path}") from last_error


def build_taizhou_public_dataset(root: Path) -> tuple[pd.DataFrame, dict]:
    folder = root / "台州市污染源在线监测废水数据信息-csv(2026-04-16)"
    files = sorted(folder.glob("*.csv"))
    frames: list[pd.DataFrame] = []
    for path in files:
        raw = read_csv_with_fallbacks(
            path,
            usecols=["XH", "JCZ", "JCSJ", "PKMC", "QYMC", "YZMC", "YZBH"],
        )
        raw["enterprise_name"] = raw["QYMC"].astype(str).str.strip()
        raw["outlet_name"] = raw["PKMC"].astype(str).str.strip()
        keep = raw["enterprise_name"].str.contains(WASTEWATER_PATTERN, na=False) | raw[
            "outlet_name"
        ].str.contains(WASTEWATER_PATTERN, na=False)
        raw = raw.loc[keep].copy()
        metric_info = raw.apply(infer_taizhou_metric_name, axis=1, result_type="expand")
        raw["metric_name"] = metric_info[0]
        raw["metric_code"] = metric_info[1]
        raw = raw.loc[raw["metric_code"].isin(TAIZHOU_ALLOWED_METRICS)].copy()
        raw["timestamp"] = pd.to_datetime(raw["JCSJ"], errors="coerce")
        raw["metric_value"] = pd.to_numeric(raw["JCZ"], errors="coerce")
        raw = raw.dropna(subset=["timestamp", "metric_value"])
        raw = raw.assign(
            source_city="台州",
            source_file=path.name,
            source_type="government_public_csv",
        )[
            [
                "source_city",
                "source_file",
                "enterprise_name",
                "outlet_name",
                "timestamp",
                "metric_name",
                "metric_value",
                "metric_code",
                "source_type",
            ]
        ]
        frames.append(raw)

    taizhou = pd.concat(frames, ignore_index=True).drop_duplicates()
    meta = {
        "source_files": len(files),
        "rows": int(len(taizhou)),
        "enterprise_count": int(taizhou["enterprise_name"].nunique()),
        "metric_counts": {
            str(k): int(v) for k, v in taizhou["metric_code"].value_counts().sort_index().items()
        },
    }
    return taizhou, meta


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def build_jinan_public_dataset(root: Path) -> tuple[pd.DataFrame, dict]:
    folder = root / "污水厂数据"
    files = sorted(folder.glob("*.rdf"))
    frames: list[pd.DataFrame] = []
    for path in files:
        records: list[dict] = []
        for _, elem in ET.iterparse(path, events=("end",)):
            if strip_namespace(elem.tag) != "Description":
                continue
            fields = {strip_namespace(child.tag): (child.text or "").strip() for child in elem}
            enterprise_name = fields.get("ent_name", "")
            outlet_name = fields.get("sub_name", "")
            if not (
                WASTEWATER_PATTERN.search(enterprise_name or "")
                or WASTEWATER_PATTERN.search(outlet_name or "")
            ):
                elem.clear()
                continue

            timestamp = pd.to_datetime(fields.get("datetime"), errors="coerce")
            if pd.isna(timestamp):
                elem.clear()
                continue

            for tag, (metric_name, metric_code) in JINAN_METRIC_TAGS.items():
                value = pd.to_numeric(fields.get(tag), errors="coerce")
                if pd.isna(value):
                    continue
                records.append(
                    {
                        "source_city": "济南",
                        "source_file": path.name,
                        "enterprise_name": enterprise_name,
                        "outlet_name": outlet_name,
                        "timestamp": timestamp,
                        "metric_name": metric_name,
                        "metric_value": float(value),
                        "metric_code": metric_code,
                        "source_type": "government_public_rdf",
                    }
                )
            elem.clear()
        frames.append(pd.DataFrame.from_records(records))

    jinan = pd.concat(frames, ignore_index=True).drop_duplicates()
    meta = {
        "source_files": len(files),
        "rows": int(len(jinan)),
        "enterprise_count": int(jinan["enterprise_name"].nunique()),
        "metric_counts": {
            str(k): int(v) for k, v in jinan["metric_code"].value_counts().sort_index().items()
        },
    }
    return jinan, meta


def build_public_monitor_long(root: Path) -> tuple[pd.DataFrame, dict]:
    taizhou, taizhou_meta = build_taizhou_public_dataset(root)
    jinan, jinan_meta = build_jinan_public_dataset(root)
    public_long = pd.concat([taizhou, jinan], ignore_index=True).drop_duplicates()
    public_long = public_long.sort_values(
        ["source_city", "enterprise_name", "timestamp", "metric_code"]
    ).reset_index(drop=True)
    meta = {
        "row_count": int(len(public_long)),
        "source_counts": {
            "taizhou_csv": taizhou_meta,
            "jinan_rdf": jinan_meta,
        },
        "metric_counts": {
            str(k): int(v) for k, v in public_long["metric_code"].value_counts().sort_index().items()
        },
    }
    return public_long, meta


def save_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_paths = ensure_output_dirs(args.output)

    plant_df, plant_meta = build_plant_hourly(args.root)
    decision_raw, decision_scaled, label_table, decision_meta = build_decision_dataset(plant_df)
    public_long, public_meta = build_public_monitor_long(args.root)

    plant_path = output_paths["plant"] / "plant_real_hourly.csv"
    public_path = output_paths["public"] / "public_monitor_long.csv"
    decision_raw_path = output_paths["decision"] / "decision_dataset_raw.csv"
    decision_scaled_path = output_paths["decision"] / "decision_dataset_scaled.csv"
    decision_labels_path = output_paths["decision"] / "decision_labels_observed.csv"

    plant_df.to_csv(plant_path, index=False, encoding="utf-8-sig")
    public_long.to_csv(public_path, index=False, encoding="utf-8-sig")
    decision_raw.to_csv(decision_raw_path, index=False, encoding="utf-8-sig")
    decision_scaled.to_csv(decision_scaled_path, index=False, encoding="utf-8-sig")
    label_table.to_csv(decision_labels_path, index=False, encoding="utf-8-sig")

    summary = {
        "plant_real_hourly": plant_meta,
        "public_monitor_long": public_meta,
        "decision_dataset": decision_meta,
        "artifacts": {
            "plant_real_hourly": str(plant_path),
            "public_monitor_long": str(public_path),
            "decision_dataset_raw": str(decision_raw_path),
            "decision_dataset_scaled": str(decision_scaled_path),
            "decision_labels_observed": str(decision_labels_path),
        },
    }
    save_summary(output_paths["reports"] / "stage1_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
