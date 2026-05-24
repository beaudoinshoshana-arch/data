import json
from pathlib import Path

import pandas as pd


def test_minute_level_simulation_outputs() -> None:
    summary_path = Path("outputs/minute_simulation/summary.json")
    sample_path = Path("outputs/minute_simulation/minute_control_replay_sample.csv")
    report_path = Path("outputs/minute_simulation/minute_simulation_report.md")
    assert summary_path.exists()
    assert sample_path.exists()
    assert report_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["source"]["native_frequency_min"] <= 2.0
    assert summary["source"]["rows_used"] >= 1000
    assert {2, 5, 15, 60}.issubset(set(summary["cadences_min"]))
    assert len(summary["scenario_variants"]) >= 3
    assert summary["headline"]["min_compliance_rate"] >= 0.99
    assert summary["headline"]["max_p95_decision_ms"] < 1000
    assert "native_2min" in summary["two_minute_vs_60min"]

    sample = pd.read_csv(sample_path)
    assert {"timestamp", "scenario_variant", "control_cadence_min", "decision_ms", "predicted_compliant"}.issubset(sample.columns)
    assert sample["control_cadence_min"].nunique() >= 4
