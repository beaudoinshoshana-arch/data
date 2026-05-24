import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import * as echarts from "echarts";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bolt,
  BrainCircuit,
  Clock3,
  Database,
  Droplets,
  Gauge,
  RefreshCcw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import "./styles.css";

type Summary = {
  project: { name: string; positioning: string };
  kpis: Record<string, number>;
  sources: Record<string, { status: string; rows?: number; url?: string; error?: string }>;
  model: {
    selected_surrogate: string;
    feature_count: number;
    surrogate_test_targets: Record<string, TargetMetric>;
    safe_marl_mode: string;
    safe_marl_rows: number;
    safe_marl_by_scenario: Record<string, Record<string, number>>;
  };
  evaluation?: {
    robustness?: Record<string, Record<string, number>>;
  };
  efficiency?: EfficiencySummary;
  minute_simulation?: MinuteSimulation;
  latest_recommendation: Recommendation | null;
  reflection: Record<string, string>;
};

type TargetMetric = {
  mae?: number;
  rmse?: number;
  r2?: number;
  normalized_mae?: number;
  predicted_compliance_rate?: number;
  actual_compliance_rate?: number;
  limit_upper?: number;
};

type Recommendation = {
  timestamp: string;
  scenario_tag: string;
  fallback_reason: string;
  final_reward: number;
  objective_improvement_pct: number;
  baseline_aeration_intensity_pct: number;
  recommended_aeration_intensity_pct: number;
  baseline_chemical_dose_kgph: number;
  recommended_chemical_dose_kgph: number;
  reward_effluent_risk: number;
  reward_energy_term: number;
  reward_chemical_term: number;
  reward_smoothness_term: number;
  constraint_violation: number;
  explanation: string;
};

type EfficiencySummary = {
  method?: string;
  baseline_model?: {
    feature_count?: number;
    test_weighted_normalized_mae?: number;
    predict_ms?: number;
  };
  best_compact_model?: {
    name?: string;
    family?: string;
    top_k?: number;
    feature_count?: number;
    feature_reduction_pct?: number;
    weighted_normalized_mae?: number;
    mae_delta_vs_baseline?: number;
    predicted_compliance_rate?: number;
    predict_ms?: number;
    candidate_batch_p95_ms?: number;
    single_p95_ms?: number;
    speedup_vs_full?: number;
    serving_time_reduction_pct?: number;
    deadline_miss_rate_100ms?: number;
    member_names?: string[];
  };
  recommended_online_profile?: {
    name?: string;
    family?: string;
    candidate_batch_p95_ms?: number;
    weighted_normalized_mae?: number;
    miss_rate_100ms?: number;
  };
  plant_energy_evidence?: {
    energy_saving_vs_current_pct?: number;
    chemical_saving_vs_current_pct?: number;
    energy_saving_vs_fixed_pct?: number;
    chemical_saving_vs_fixed_pct?: number;
  };
};

type MinuteSimulation = {
  source?: {
    source_domain?: string;
    native_frequency_min?: number;
    rows_used?: number;
  };
  headline?: {
    mean_objective_reduction_vs_60min_pct?: number;
    max_p95_decision_ms?: number;
    min_compliance_rate?: number;
  };
};

type AiSummary = {
  title: string;
  bullets: string[];
  innovation: string[];
  risk: string;
  next_step: string;
};

const API = "";
const CHART_TEXT = "#52616b";
const CHART_MUTED = "#82919c";
const CHART_GRID = "#e1ebee";
const CHART_AXIS = "#cbd9df";
const TEAL = "#0f766e";
const BLUE = "#2563eb";
const AMBER = "#d97706";
const GREEN = "#16a34a";
const ROSE = "#e11d48";
const METRICS = ["COD", "NH3N", "TP", "TN"] as const;
const SCENARIOS = ["", "observed", "load_up", "rain_dilution"];
const TARGET_ORDER = [
  "label_next_effluent_cod_mgL",
  "label_next_effluent_nh3n_mgL",
  "label_next_effluent_tp_mgL",
  "label_next_effluent_tn_mgL",
];

type MetricCode = (typeof METRICS)[number];

function getInitialMetric(): MetricCode {
  if (typeof window === "undefined") return "COD";
  const value = new URLSearchParams(window.location.search).get("metric")?.toUpperCase();
  return METRICS.includes(value as MetricCode) ? (value as MetricCode) : "COD";
}

function getInitialScenario() {
  if (typeof window === "undefined") return "";
  const value = new URLSearchParams(window.location.search).get("scenario") || "";
  return SCENARIOS.includes(value) ? value : "";
}

async function getData<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  const body = await res.json();
  return body.data as T;
}

function fmt(value?: number, digits = 2) {
  if (value === undefined || Number.isNaN(value)) return "-";
  if (Math.abs(value) >= 1000000) return `${(value / 1000000).toFixed(2)}M`;
  if (Math.abs(value) >= 10000) return `${(value / 10000).toFixed(1)}万`;
  return value.toFixed(digits);
}

function pct(value?: number) {
  if (value === undefined || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function targetLabel(key: string) {
  const labels: Record<string, string> = {
    label_next_effluent_cod_mgL: "COD",
    label_next_effluent_nh3n_mgL: "NH3-N",
    label_next_effluent_tp_mgL: "TP",
    label_next_effluent_tn_mgL: "TN",
  };
  return labels[key] || key.replace("label_next_effluent_", "").replace("_mgL", "").toUpperCase();
}

function useChart(id: string, option: echarts.EChartsOption | null) {
  useEffect(() => {
    if (!option) return;
    const el = document.getElementById(id);
    if (!el) return;
    const chart = echarts.init(el);
    chart.setOption(option);
    const resize = () => chart.resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      chart.dispose();
    };
  }, [id, option]);
}

function Kpi({ icon, label, value, hint }: { icon: React.ReactNode; label: string; value: string; hint: string }) {
  return (
    <section className="kpi">
      <div className="kpiIcon" aria-hidden="true">{icon}</div>
      <div>
        <div className="kpiLabel">{label}</div>
        <div className="kpiValue">{value}</div>
        <div className="kpiHint">{hint}</div>
      </div>
    </section>
  );
}

function ChartPanel({ title, icon, id, children }: { title: string; icon: React.ReactNode; id?: string; children?: React.ReactNode }) {
  return (
    <section className="panel">
      <header className="panelHeader">
        <span aria-hidden="true">{icon}</span>
        <h2>{title}</h2>
      </header>
      {id ? <div id={id} className="chart" /> : children}
    </section>
  );
}

function App() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [ai, setAi] = useState<AiSummary | null>(null);
  const [predictionSeries, setPredictionSeries] = useState<any[]>([]);
  const [aerationSeries, setAerationSeries] = useState<any[]>([]);
  const [pacSeries, setPacSeries] = useState<any[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [metric, setMetric] = useState<MetricCode>(getInitialMetric);
  const [scenario, setScenario] = useState(getInitialScenario);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setError("");
    setLoading(true);
    try {
      const [summaryData, aiData, predData, aerData, pacData, recData] = await Promise.all([
        getData<Summary>("/api/summary"),
        getData<AiSummary>("/api/report/ai-summary"),
        getData<{ series: any[] }>(`/api/timeseries?metric=${encodeURIComponent(metric)}&source=prediction`),
        getData<{ series: any[] }>(`/api/timeseries?metric=AERATION&source=rl&scenario=${encodeURIComponent(scenario)}`),
        getData<{ series: any[] }>(`/api/timeseries?metric=PAC&source=rl&scenario=${encodeURIComponent(scenario)}`),
        getData<{ items: Recommendation[] }>(`/api/recommendations?limit=80${scenario ? `&scenario=${encodeURIComponent(scenario)}` : ""}`),
      ]);
      setSummary(summaryData);
      setAi(aiData);
      setPredictionSeries(predData.series);
      setAerationSeries(aerData.series);
      setPacSeries(pacData.series);
      setRecommendations(recData.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set("metric", metric);
    if (scenario) params.set("scenario", scenario);
    else params.delete("scenario");
    window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}${window.location.hash}`);
  }, [metric, scenario]);

  useEffect(() => {
    load();
  }, [metric, scenario]);

  const predictionOption = useMemo<echarts.EChartsOption | null>(() => {
    if (!predictionSeries.length) return null;
    const actualKey = Object.keys(predictionSeries[0]).find((k) => k.startsWith("actual_"));
    const predKey = Object.keys(predictionSeries[0]).find((k) => k.startsWith("pred_"));
    const limitKey = Object.keys(predictionSeries[0]).find((k) => k.endsWith("_upper_limit"));
    const x = predictionSeries.map((d) => d.timestamp?.slice(5, 16));
    return {
      tooltip: { trigger: "axis" },
      legend: { top: 0, textStyle: { color: CHART_TEXT } },
      grid: { left: 48, right: 24, top: 42, bottom: 34 },
      xAxis: { type: "category", data: x, axisLabel: { color: CHART_MUTED }, axisLine: { lineStyle: { color: CHART_AXIS } } },
      yAxis: { type: "value", axisLabel: { color: CHART_MUTED }, splitLine: { lineStyle: { color: CHART_GRID } } },
      series: [
        { name: "实测", type: "line", smooth: true, symbol: "none", data: predictionSeries.map((d) => d[actualKey || ""]), lineStyle: { color: TEAL, width: 2 } },
        { name: "预测", type: "line", smooth: true, symbol: "none", data: predictionSeries.map((d) => d[predKey || ""]), lineStyle: { color: AMBER, width: 2 } },
        { name: "限值", type: "line", symbol: "none", data: predictionSeries.map((d) => d[limitKey || ""]), lineStyle: { color: ROSE, type: "dashed", width: 1.5 } },
      ],
    };
  }, [predictionSeries]);

  const actionOption = useMemo<echarts.EChartsOption | null>(() => {
    if (!aerationSeries.length) return null;
    const x = aerationSeries.map((d) => d.timestamp?.slice(5, 16));
    return {
      tooltip: { trigger: "axis" },
      legend: { top: 0, textStyle: { color: CHART_TEXT } },
      grid: { left: 46, right: 24, top: 42, bottom: 34 },
      xAxis: { type: "category", data: x, axisLabel: { color: CHART_MUTED }, axisLine: { lineStyle: { color: CHART_AXIS } } },
      yAxis: { type: "value", axisLabel: { color: CHART_MUTED }, splitLine: { lineStyle: { color: CHART_GRID } } },
      series: [
        { name: "基线曝气%", type: "line", symbol: "none", data: aerationSeries.map((d) => d.baseline_aeration_intensity_pct), lineStyle: { color: "#94a3b8", width: 1.5 } },
        { name: "RL曝气%", type: "line", symbol: "none", data: aerationSeries.map((d) => d.recommended_aeration_intensity_pct), lineStyle: { color: BLUE, width: 2.2 } },
        { name: "基线药耗kg/h", type: "line", symbol: "none", data: pacSeries.map((d) => d.baseline_chemical_dose_kgph), lineStyle: { color: AMBER, width: 1.5 } },
        { name: "RL药耗kg/h", type: "line", symbol: "none", data: pacSeries.map((d) => d.recommended_chemical_dose_kgph), lineStyle: { color: GREEN, width: 2.2 } },
      ],
    };
  }, [aerationSeries, pacSeries]);

  const rewardOption = useMemo<echarts.EChartsOption | null>(() => {
    const latest = recommendations[recommendations.length - 1];
    if (!latest) return null;
    return {
      tooltip: {},
      radar: {
        indicator: [
          { name: "出水风险", max: 1.4 },
          { name: "能耗", max: 1 },
          { name: "药耗", max: 1 },
          { name: "平滑", max: 1 },
          { name: "违规", max: 1 },
        ],
        axisName: { color: CHART_TEXT },
        splitLine: { lineStyle: { color: CHART_GRID } },
        splitArea: { areaStyle: { color: ["rgba(236,248,247,.88)", "rgba(248,252,252,.9)"] } },
      },
      series: [
        {
          type: "radar",
          data: [
            {
              name: "Reward分解",
              value: [
                latest.reward_effluent_risk,
                latest.reward_energy_term,
                latest.reward_chemical_term,
                latest.reward_smoothness_term,
                latest.constraint_violation,
              ],
              areaStyle: { color: "rgba(15,118,110,.16)" },
              lineStyle: { color: TEAL, width: 2 },
            },
          ],
        },
      ],
    };
  }, [recommendations]);

  const robustnessOption = useMemo<echarts.EChartsOption | null>(() => {
    const robustness = summary?.evaluation?.robustness || {};
    const rows = Object.entries(robustness);
    if (!rows.length) return null;
    const metrics = [
      ["feasible_rate", "可行率"],
      ["predicted_compliance_rate", "达标率"],
      ["energy_saving_vs_current_pct", "曝气节能"],
      ["chemical_saving_vs_current_pct", "PAC节药"],
    ] as const;
    const data = rows.flatMap(([_, item], y) =>
      metrics.map(([key], x) => [x, y, key.endsWith("_rate") ? (item[key] || 0) * 100 : item[key] || 0])
    );
    return {
      tooltip: {
        position: "top",
        formatter: (params: any) => `${rows[params.value[1]][0]}<br/>${metrics[params.value[0]][1]}: ${params.value[2].toFixed(1)}%`,
      },
      grid: { left: 88, right: 18, top: 20, bottom: 34 },
      xAxis: { type: "category", data: metrics.map((m) => m[1]), axisLabel: { color: CHART_MUTED }, axisLine: { show: false } },
      yAxis: { type: "category", data: rows.map(([name]) => name), axisLabel: { color: CHART_MUTED }, axisLine: { show: false } },
      visualMap: {
        min: 0,
        max: 100,
        show: false,
        inRange: { color: ["#e0f2fe", "#67e8f9", "#16a34a"] },
      },
      series: [{ type: "heatmap", data, label: { show: true, formatter: (p: any) => `${p.value[2].toFixed(0)}%`, color: "#0f172a" } }],
    };
  }, [summary]);

  useChart("predictionChart", predictionOption);
  useChart("actionChart", actionOption);
  useChart("rewardChart", rewardOption);
  useChart("robustnessChart", robustnessOption);

  const latest = recommendations[recommendations.length - 1] || summary?.latest_recommendation;
  const byScenario = summary?.model.safe_marl_by_scenario || {};
  const targetRows = useMemo(
    () =>
      TARGET_ORDER.map((key) => [key, summary?.model.surrogate_test_targets?.[key]] as const).filter((row) => Boolean(row[1])),
    [summary]
  );
  const sourceOkCount = Object.values(summary?.sources || {}).filter((source) => source.status === "ok").length;
  const compact = summary?.efficiency?.best_compact_model;
  const baseline = summary?.efficiency?.baseline_model;
  const minute = summary?.minute_simulation;

  return (
    <main className="screen">
      <a className="skipLink" href="#dashboard-content">跳转到监控内容</a>
      <header className="topbar">
        <div>
          <div className="eyebrow">Safe-MARL WWTP Control Center</div>
          <h1>污水厂曝气加药智能决策大屏</h1>
        </div>
        <div className="toolbar">
          <select value={metric} onChange={(e) => setMetric(e.target.value as MetricCode)} aria-label="选择水质指标">
            <option value="COD">COD</option>
            <option value="NH3N">NH3-N</option>
            <option value="TP">TP</option>
            <option value="TN">TN</option>
          </select>
          <select value={scenario} onChange={(e) => setScenario(e.target.value)} aria-label="选择工况">
            <option value="">全部工况</option>
            <option value="observed">常规</option>
            <option value="load_up">冲击负荷</option>
            <option value="rain_dilution">雨水稀释</option>
          </select>
          <button onClick={load} title="刷新数据" aria-label="刷新数据" aria-busy={loading} disabled={loading}>
            <RefreshCcw size={18} aria-hidden="true" className={loading ? "spin" : ""} />
          </button>
        </div>
      </header>

      <div className="srOnly" role="status" aria-live="polite">
        {loading ? "正在刷新数据…" : error ? `数据加载失败：${error}` : summary ? "监控数据已更新。" : "等待监控数据加载…"}
      </div>

      <section className="statusStrip" aria-label="系统运行状态">
        <div className="statusPill">
          <span>安全经济模式</span>
          <strong>{summary?.model.safe_marl_mode || "-"}</strong>
        </div>
        <div className="statusPill">
          <span>推荐 P95</span>
          <strong>{fmt(summary?.kpis.recommend_response_p95_ms, 1)} ms</strong>
        </div>
        <div className="statusPill">
          <span>动作样本</span>
          <strong>{fmt(summary?.model.safe_marl_rows, 0)}</strong>
        </div>
        <div className="statusPill">
          <span>外部数据源</span>
          <strong>{sourceOkCount}/{Object.keys(summary?.sources || {}).length || 0} ok</strong>
        </div>
        <div className="statusPill">
          <span>最小粒度</span>
          <strong>{fmt(summary?.kpis.min_native_frequency_min, 0)} min</strong>
        </div>
        <div className="statusPill">
          <span>EcoLite 推理</span>
          <strong>{fmt(compact?.speedup_vs_full, 2)}x</strong>
        </div>
      </section>

      {error && <div className="error" role="alert"><AlertTriangle size={18} aria-hidden="true" /> {error}</div>}

      <section className="kpiGrid" id="dashboard-content">
        <Kpi icon={<Database size={22} />} label="融合数据" value={fmt(summary?.kpis.fusion_rows, 0)} hint="本地 + 公开监测 + 外部适配" />
        <Kpi icon={<BrainCircuit size={22} />} label="决策样本" value={fmt(summary?.kpis.decision_rows, 0)} hint={`${fmt(summary?.model.feature_count, 0)} 个模型特征`} />
        <Kpi icon={<ShieldCheck size={22} />} label="安全可行率" value={pct(summary?.kpis.safe_marl_feasible_rate)} hint="RL 动作经过约束盾" />
        <Kpi
          icon={<Bolt size={22} />}
          label="定值对照节能"
          value={`${fmt(summary?.kpis.fixed_energy_saving_pct)}%`}
          hint={`PAC节药 ${fmt(summary?.kpis.fixed_chemical_saving_pct)}%，P95 ${fmt(summary?.kpis.recommend_response_p95_ms, 1)}ms`}
        />
        <Kpi icon={<Gauge size={22} />} label="预测误差" value={fmt(summary?.kpis.stage2_test_weighted_mae, 3)} hint={`达标率 ${pct(summary?.kpis.stage2_compliance_rate)}`} />
        <Kpi
          icon={<Clock3 size={22} />}
          label="分钟级数据"
          value={fmt(summary?.kpis.minute_level_rows, 0)}
          hint={`${fmt(summary?.kpis.min_native_frequency_min, 0)}min Agtrup SCADA + BSM1 15min`}
        />
        <Kpi
          icon={<Activity size={22} />}
          label="2min 仿真收益"
          value={`${fmt(summary?.kpis.minute_simulation_objective_reduction_vs_60min_pct)}%`}
          hint={`达标 ${pct(summary?.kpis.minute_simulation_min_compliance_rate)}，P95 ${fmt(summary?.kpis.minute_simulation_p95_decision_ms, 2)}ms`}
        />
        <Kpi
          icon={<BrainCircuit size={22} />}
          label="轻量代理"
          value={`${fmt(compact?.speedup_vs_full, 2)}x`}
          hint={`49动作P95 ${fmt(compact?.candidate_batch_p95_ms, 2)}ms，MAE变化 ${fmt(compact?.mae_delta_vs_baseline, 4)}`}
        />
      </section>

      <section className="dashboardGrid">
        <ChartPanel title={`${metric} 未来出水预测`} icon={<Activity size={18} />} id="predictionChart" />
        <ChartPanel title="曝气与药耗动作对比" icon={<BarChart3 size={18} />} id="actionChart" />
        <ChartPanel title="最新 Reward 分解" icon={<Sparkles size={18} />} id="rewardChart" />
        <ChartPanel title="当前推荐" icon={<ShieldCheck size={18} />}>
          <div className="recommendBox">
            <div className="recTime">{latest?.timestamp || "-"}</div>
            <div className="recScenario">{latest?.scenario_tag || "-"}</div>
            <div className="recActions">
              <div>
                <span>曝气</span>
                <strong>{fmt(latest?.baseline_aeration_intensity_pct)} → {fmt(latest?.recommended_aeration_intensity_pct)}%</strong>
              </div>
              <div>
                <span>药耗</span>
                <strong>{fmt(latest?.baseline_chemical_dose_kgph)} → {fmt(latest?.recommended_chemical_dose_kgph)} kg/h</strong>
              </div>
              <div>
                <span>目标函数改善</span>
                <strong>{fmt(latest?.objective_improvement_pct, 3)}%</strong>
              </div>
            </div>
            <p>{latest?.explanation || "等待推荐结果。"}</p>
          </div>
        </ChartPanel>
      </section>

      <section className="lowerGrid">
        <section className="panel">
          <header className="panelHeader"><span aria-hidden="true"><Droplets size={18} /></span><h2>工况鲁棒性</h2></header>
          <div id="robustnessChart" className="smallChart" />
          <div className="scenarioRows">
            {Object.entries(byScenario).map(([name, item]) => (
              <div className="scenarioRow" key={name}>
                <span>{name}</span>
                <b>{fmt(item.objective_improvement_pct, 3)}%</b>
                <em>可行率 {pct(item.feasible_rate)}</em>
              </div>
            ))}
          </div>
        </section>

        <section className="panel modelPanel">
          <header className="panelHeader"><span aria-hidden="true"><Gauge size={18} /></span><h2>基准模型评估</h2></header>
          <div className="modelMeta">
            <span>{summary?.model.selected_surrogate || "-"} surrogate</span>
            <span>{fmt(summary?.model.feature_count, 0)} features</span>
            <span>{compact?.name || "EcoLite"} profile</span>
            <span>{fmt(compact?.feature_count || compact?.top_k, 0)} effective features</span>
          </div>
          <div className="efficiencyGrid" aria-label="轻量模型效率验证">
            <div>
              <span>推理耗时</span>
              <strong>{fmt(baseline?.predict_ms, 1)} → {fmt(compact?.predict_ms, 1)} ms</strong>
            </div>
            <div>
              <span>耗时降低</span>
              <strong>{fmt(compact?.serving_time_reduction_pct)}%</strong>
            </div>
            <div>
              <span>轻量达标率</span>
              <strong>{pct(compact?.predicted_compliance_rate)}</strong>
            </div>
            <div>
              <span>49动作P95</span>
              <strong>{fmt(compact?.candidate_batch_p95_ms, 2)} ms</strong>
            </div>
            <div>
              <span>100ms miss</span>
              <strong>{pct(compact?.deadline_miss_rate_100ms)}</strong>
            </div>
          </div>
          <table className="modelTable">
            <caption className="srOnly">Stage-2 测试集分目标误差</caption>
            <thead>
              <tr>
                <th scope="col">指标</th>
                <th scope="col">MAE</th>
                <th scope="col">RMSE</th>
                <th scope="col">R²</th>
                <th scope="col">达标</th>
              </tr>
            </thead>
            <tbody>
              {targetRows.map(([key, item]) => (
                <tr key={key}>
                  <th scope="row">{targetLabel(key)}</th>
                  <td>{fmt(item?.mae, key.includes("_tp_") ? 3 : 2)}</td>
                  <td>{fmt(item?.rmse, key.includes("_tp_") ? 3 : 2)}</td>
                  <td>{fmt(item?.r2, 3)}</td>
                  <td>{pct(item?.predicted_compliance_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="modelNote">以 ExtraTrees 代理模型作为安全基线；EcoLite 已比较树模型、梯度提升、Ridge 与两类融合，在线推荐优先选择满足候选动作搜索时限的 profile。</p>
        </section>

        <section className="panel">
          <header className="panelHeader"><span aria-hidden="true"><Database size={18} /></span><h2>数据源状态</h2></header>
          <div className="minuteSummary">
            <span>{minute?.source?.source_domain || "minute simulation"}</span>
            <strong>{fmt(minute?.source?.rows_used, 0)} points · {fmt(minute?.source?.native_frequency_min, 0)}min</strong>
            <em>2min vs 60min 目标函数改善 {fmt(minute?.headline?.mean_objective_reduction_vs_60min_pct)}%</em>
          </div>
          <div className="sourceList">
            {Object.entries(summary?.sources || {}).map(([name, source]) => (
              <div className="sourceItem" key={name}>
                <span>{name}</span>
                <b>{source.status}</b>
                <em>{source.rows ? `${fmt(source.rows, 0)} rows` : source.url || source.error || "configured"}</em>
              </div>
            ))}
          </div>
        </section>

        <section className="panel aiPanel">
          <header className="panelHeader"><span aria-hidden="true"><Sparkles size={18} /></span><h2>{ai?.title || "运行分析摘要"}</h2></header>
          <ul>
            {(ai?.bullets || []).map((line) => <li key={line}>{line}</li>)}
          </ul>
          <div className="chips">
            {(ai?.innovation || []).map((tag) => <span key={tag}>{tag}</span>)}
          </div>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
