import json
from pathlib import Path

import pandas as pd


def test_fusion_registry_and_scenario_library_exist() -> None:
    registry_path = Path("outputs/fusion_data/source_registry.json")
    scenario_path = Path("outputs/fusion_data/scenario_library.csv")
    assert registry_path.exists()
    assert scenario_path.exists()

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["summary"]["fusion_rows"] > 10000
    assert "local_plant" in registry["summary"]["source_domains"]
    assert registry["summary"]["minute_level_rows"] >= 200000
    assert registry["summary"]["min_native_frequency_min"] <= 2.0
    assert "agtrup_bluekolding_2min" in registry["summary"]["source_domains"]
    assert "iwa_bsm1_dynamic_influent" in registry["summary"]["source_domains"]

    scenario = pd.read_csv(scenario_path)
    assert {"source_domain", "metric_code", "recommended_use", "min_native_frequency_min"}.issubset(scenario.columns)
    assert len(scenario) >= 5

    high_frequency_catalog = pd.read_csv("outputs/fusion_data/high_frequency_source_catalog.csv")
    assert {"source_domain", "min_frequency_min", "metrics"}.issubset(high_frequency_catalog.columns)
    assert high_frequency_catalog["min_frequency_min"].min() <= 2.0
