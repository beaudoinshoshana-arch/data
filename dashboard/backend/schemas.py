from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PlantState(BaseModel):
    timestamp: str | None = None
    scenario_tag: str = "observed"
    influent_cod_mgL: float = Field(default=120.0, ge=0)
    influent_bod_mgL: float = Field(default=55.0, ge=0)
    influent_nh3n_mgL: float = Field(default=16.0, ge=0)
    influent_tp_mgL: float = Field(default=0.3, ge=0)
    influent_flow_m3h: float = Field(default=1200.0, ge=0)
    reactor_do_mgL: float = Field(default=3.5, ge=0)
    sludge_mlss_mgL: float = Field(default=6500.0, ge=0)
    aeration_intensity_pct: float = Field(default=50.0, ge=0, le=120)
    chemical_dose_pac_mgL: float = Field(default=8.0, ge=0)


class ApiResponse(BaseModel):
    data: Any
    meta: dict[str, Any] = {}
