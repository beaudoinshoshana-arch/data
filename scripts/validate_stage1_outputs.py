from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_DECISION_COLUMNS = {
    "timestamp",
    "split",
    "scenario_tag",
    "source_flag",
    "is_simulated",
    "influent_cod_mgL",
    "influent_bod_mgL",
    "influent_nh3n_mgL",
    "influent_tp_mgL",
    "reactor_do_mgL",
    "sludge_mlss_mgL",
    "aeration_intensity_pct",
    "chemical_dose_kgph",
}
ALLOWED_PUBLIC_METRICS = {"COD", "NH3N", "TP", "TN", "pH", "FLOW"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stage-1 wastewater datasets.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "stage1_data",
        help="Output directory produced by build_stage1_datasets.py",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    plant_path = args.output / "plant_real_hourly" / "plant_real_hourly.csv"
    public_path = args.output / "public_monitor_long" / "public_monitor_long.csv"
    decision_raw_path = args.output / "decision_dataset" / "decision_dataset_raw.csv"
    decision_scaled_path = args.output / "decision_dataset" / "decision_dataset_scaled.csv"
    labels_path = args.output / "decision_dataset" / "decision_labels_observed.csv"
    summary_path = args.output / "reports" / "stage1_summary.json"

    for path in [plant_path, public_path, decision_raw_path, decision_scaled_path, labels_path, summary_path]:
        require(path.exists(), f"Missing expected artifact: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    plant = pd.read_csv(plant_path, parse_dates=["timestamp"])
    public_long = pd.read_csv(public_path, parse_dates=["timestamp"])
    decision_raw = pd.read_csv(decision_raw_path, parse_dates=["timestamp"])
    decision_scaled = pd.read_csv(decision_scaled_path, parse_dates=["timestamp"])
    labels = pd.read_csv(labels_path, parse_dates=["timestamp"])

    require(4550 <= len(plant) <= 4600, f"plant_real_hourly row count out of range: {len(plant)}")
    require(plant["timestamp"].is_unique, "plant_real_hourly must have unique timestamps")
    require(
        summary["plant_real_hourly"]["core_complete_before_fill_rows"] >= 4300,
        "Expected at least 4300 fully observed core rows before fill",
    )

    require(len(public_long) >= 100000, f"public_monitor_long too small: {len(public_long)}")
    require(
        set(public_long["metric_code"].dropna().unique()).issubset(ALLOWED_PUBLIC_METRICS),
        "public_monitor_long contains unsupported metric codes",
    )

    require(REQUIRED_DECISION_COLUMNS.issubset(decision_raw.columns), "decision_dataset_raw missing required columns")
    require(len(decision_raw) >= 12000, f"decision_dataset_raw too small: {len(decision_raw)}")
    require(len(decision_raw) == len(decision_scaled), "raw/scaled decision datasets should have identical row counts")
    require(
        set(decision_raw["scenario_tag"].unique()) == {"observed", "load_up", "rain_dilution"},
        "Scenario tags are incomplete",
    )
    require(
        decision_raw.loc[decision_raw["scenario_tag"] == "observed", "is_simulated"].eq(False).all(),
        "Observed rows must not be simulated",
    )
    require(
        decision_raw.loc[decision_raw["scenario_tag"] != "observed", "is_simulated"].eq(True).all(),
        "Scenario rows must be marked simulated",
    )

    split_order = {"train": 0, "val": 1, "test": 2}
    observed = decision_raw.loc[decision_raw["scenario_tag"] == "observed", ["timestamp", "split"]].drop_duplicates()
    observed = observed.sort_values("timestamp").reset_index(drop=True)
    require(observed["timestamp"].is_unique, "Observed decision rows should have unique timestamps")
    split_sequence = observed["split"].map(split_order)
    require(split_sequence.is_monotonic_increasing, "Decision splits must follow chronological order")

    numeric_check_cols = [
        "influent_cod_mgL",
        "influent_bod_mgL",
        "influent_nh3n_mgL",
        "influent_tp_mgL",
        "reactor_do_mgL",
        "sludge_mlss_mgL",
        "aeration_intensity_pct",
        "chemical_dose_kgph",
    ]
    require(not decision_raw[numeric_check_cols].isna().any().any(), "Decision raw dataset should not contain NaNs in core features")
    require(len(labels) == len(observed), "Observed label table should align with observed decision rows")

    report = {
        "plant_rows": int(len(plant)),
        "plant_complete_before_fill_rows": int(summary["plant_real_hourly"]["core_complete_before_fill_rows"]),
        "public_rows": int(len(public_long)),
        "decision_rows": int(len(decision_raw)),
        "label_rows": int(len(labels)),
        "decision_split_counts": {
            str(k): int(v) for k, v in decision_raw["split"].value_counts().sort_index().items()
        },
        "public_metric_counts": {
            str(k): int(v) for k, v in public_long["metric_code"].value_counts().sort_index().items()
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
