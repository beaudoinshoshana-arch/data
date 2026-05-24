import json
from pathlib import Path


def test_decision_benefit_meets_competition_gates() -> None:
    path = Path("outputs/decision_benefit/decision_benefit_summary.json")
    assert path.exists()
    summary = json.loads(path.read_text(encoding="utf-8"))

    current = summary["current_control_baseline"]
    fixed = summary["traditional_fixed_baseline"]
    response = summary["response_time"]

    assert current["predicted_compliance_rate"] >= 0.95
    assert fixed["energy_saving_pct"] >= 10.0
    assert fixed["chemical_saving_pct"] >= 10.0
    assert response["pass"] is True
    assert response["max_ms"] <= 1000.0
