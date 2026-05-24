from wwtp_decision.safe_marl import SafetyShield, grid_search_recommendation, load_constraint_config


def test_safety_shield_clips_bounds_and_steps() -> None:
    config = load_constraint_config("configs/safe_marl.default.json")
    shield = SafetyShield.from_config(config)
    state = {
        "scenario_tag": "load_up",
        "influent_flow_m3h": 1200,
        "aeration_intensity_pct": 50,
        "chemical_dose_pac_mgL": 8,
    }
    action = shield.apply(state, 999, -999)
    assert action["recommended_aeration_intensity_pct"] <= 70
    assert action["recommended_chemical_dose_pac_mgL"] >= 2
    assert action["shield_adjusted"] is True


def test_grid_recommendation_is_feasible() -> None:
    config = load_constraint_config("configs/safe_marl.default.json")
    state = {
        "scenario_tag": "observed",
        "influent_cod_mgL": 120,
        "influent_bod_mgL": 54,
        "influent_nh3n_mgL": 16,
        "influent_tp_mgL": 0.3,
        "influent_flow_m3h": 1200,
        "reactor_do_mgL": 3.5,
        "sludge_mlss_mgL": 6500,
        "aeration_intensity_pct": 50,
        "chemical_dose_pac_mgL": 8,
    }
    recommendation = grid_search_recommendation(state, config)
    assert recommendation["is_feasible"] is True
    assert recommendation["recommended_aeration_intensity_pct"] >= 10
    assert recommendation["recommended_chemical_dose_pac_mgL"] >= 2
