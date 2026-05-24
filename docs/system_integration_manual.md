# 系统对接手册

## 接口概览

后端服务地址默认：

```text
http://127.0.0.1:8000
```

## 1. 系统摘要

```http
GET /api/summary
```

返回数据规模、模型指标、Safe-MARL 可行率、节能节药收益、响应时间、最新推荐和数据源状态。

## 2. 时序曲线

```http
GET /api/timeseries?metric=COD&source=prediction
GET /api/timeseries?metric=AERATION&source=rl&scenario=load_up
```

支持指标：`COD`、`NH3N`、`TP`、`TN`、`AERATION`、`PAC`、`REWARD`、`ENERGY`。

## 3. 推荐列表

```http
GET /api/recommendations?scenario=observed&limit=100
```

返回基线动作、推荐动作、reward 分解、可行性、回退原因和中文解释。

## 4. 单点出水预测

```http
POST /api/infer
Content-Type: application/json

{
  "scenario_tag": "load_up",
  "influent_cod_mgL": 135,
  "influent_bod_mgL": 60,
  "influent_nh3n_mgL": 18,
  "influent_tp_mgL": 0.35,
  "influent_flow_m3h": 1250,
  "reactor_do_mgL": 3.8,
  "sludge_mlss_mgL": 6800,
  "aeration_intensity_pct": 52,
  "chemical_dose_pac_mgL": 8.5
}
```

## 5. RL 推荐

```http
POST /api/rl/recommend
```

请求体同 `/api/infer`。返回内容包括：

- `mode`：`safe_marl_policy`、`grid_safe_expert_better_objective` 或 `grid_safe_expert_fallback`。
- `raw_policy_delta`：RL 原始动作增量。
- `reward_compare`：当前基线、RL、专家搜索和最终动作的 reward 对比。
- `recommendation`：安全盾后的可执行动作。
- `explanation`：中文动作解释。

## 6. 自控系统适配建议

- PLC/SCADA 每小时或每 30 分钟推送一组当前状态到 `/api/rl/recommend`。
- 系统只输出推荐，不直接写 PLC；由操作员或上位机确认后执行。
- 执行后将实际曝气、投药、出水反馈写回历史库，用于后续再训练。
- 若接口超时、传感器缺失或动作不可行，现场控制保持原规则或使用专家网格推荐。
