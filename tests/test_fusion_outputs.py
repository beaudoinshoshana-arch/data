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

    scenario = pd.read_csv(scenario_path)
    assert {"source_domain", "metric_code", "recommended_use"}.issubset(scenario.columns)
    assert len(scenario) >= 5
