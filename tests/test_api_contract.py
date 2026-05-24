from fastapi.testclient import TestClient

from dashboard.backend.main import app


def test_dashboard_api_contracts() -> None:
    client = TestClient(app)
    for path in [
        "/api/health",
        "/api/summary",
        "/api/timeseries?metric=COD",
        "/api/recommendations?limit=5",
        "/api/report/ai-summary",
    ]:
        response = client.get(path)
        assert response.status_code == 200
    summary = client.get("/api/summary").json()["data"]
    assert summary["kpis"]["fixed_energy_saving_pct"] >= 10.0
    assert summary["kpis"]["recommend_response_p95_ms"] < 1000.0
    assert summary["kpis"]["compact_feature_reduction_pct"] >= 50.0
    assert summary["kpis"]["compact_serving_time_reduction_pct"] >= 50.0
    assert summary["kpis"]["compact_candidate_batch_p95_ms"] < 100.0
    assert summary["kpis"]["compact_deadline_miss_rate_100ms"] == 0.0
    assert summary["efficiency"]["best_compact_model"]["predicted_compliance_rate"] >= 0.99
    assert len(summary["efficiency"]["model_search"]) >= 6

    state = {
        "scenario_tag": "load_up",
        "influent_cod_mgL": 135,
        "influent_bod_mgL": 60,
        "influent_nh3n_mgL": 18,
        "influent_tp_mgL": 0.35,
        "influent_flow_m3h": 1250,
        "reactor_do_mgL": 3.8,
        "sludge_mlss_mgL": 6800,
        "aeration_intensity_pct": 52,
        "chemical_dose_pac_mgL": 8.5,
    }
    infer = client.post("/api/infer", json=state)
    rl = client.post("/api/rl/recommend", json=state)
    assert infer.status_code == 200
    assert rl.status_code == 200
    assert "prediction" in infer.json()["data"]
    assert "recommendation" in rl.json()["data"]
