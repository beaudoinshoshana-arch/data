from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .schemas import ApiResponse, PlantState
from .services import (
    ai_summary,
    dashboard_summary,
    get_recommendations,
    get_timeseries,
    infer_effluent,
    recommend_with_policy,
)


app = FastAPI(
    title="WWTP Safe-MARL Decision Dashboard API",
    version="1.0.0",
    description="污水厂曝气加药智能决策可视化大屏后端接口",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/summary", response_model=ApiResponse)
def summary() -> ApiResponse:
    return ApiResponse(data=dashboard_summary())


@app.get("/api/timeseries", response_model=ApiResponse)
def timeseries(
    metric: str = Query(default="COD"),
    source: str = Query(default="prediction"),
    scenario: str | None = Query(default=None),
) -> ApiResponse:
    return ApiResponse(data=get_timeseries(metric=metric, source=source, scenario=scenario))


@app.get("/api/recommendations", response_model=ApiResponse)
def recommendations(
    scenario: str | None = Query(default=None),
    limit: int = Query(default=240, ge=1, le=1000),
) -> ApiResponse:
    return ApiResponse(data=get_recommendations(scenario=scenario, limit=limit))


@app.post("/api/infer", response_model=ApiResponse)
def infer(state: PlantState) -> ApiResponse:
    return ApiResponse(data=infer_effluent(state.model_dump()))


@app.post("/api/rl/recommend", response_model=ApiResponse)
def rl_recommend(state: PlantState) -> ApiResponse:
    return ApiResponse(data=recommend_with_policy(state.model_dump()))


@app.get("/api/report/ai-summary", response_model=ApiResponse)
def report_ai_summary() -> ApiResponse:
    return ApiResponse(data=ai_summary())


frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
