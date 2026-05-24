const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  Packer,
  PageNumber,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableOfContents,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("../dashboard/frontend/node_modules/docx");

const root = path.resolve(__dirname, "..");
const docsDir = path.join(root, "docs");
const outputsDir = path.join(root, "outputs");
fs.mkdirSync(docsDir, { recursive: true });

function readJson(rel, fallback = {}) {
  const file = path.join(root, rel);
  if (!fs.existsSync(file)) return fallback;
  return JSON.parse(fs.readFileSync(file, "utf-8"));
}

const stage1 = readJson("outputs/stage1_data/reports/stage1_summary.json");
const stage2 = readJson("outputs/stage2_model/summary.json");
const fusion = readJson("outputs/fusion_data/source_registry.json");
const safe = readJson("outputs/safe_marl/summary.json");
const paper = readJson("outputs/paper_repro_integrated_control/summary.json");
const benefit = readJson("outputs/decision_benefit/decision_benefit_summary.json");

function fmt(value, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "-";
  const n = Number(value);
  if (Math.abs(n) >= 1000000) return `${(n / 1000000).toFixed(digits)}M`;
  if (Math.abs(n) >= 10000) return `${(n / 10000).toFixed(digits)}万`;
  return n.toFixed(digits);
}

function p(text, options = {}) {
  return new Paragraph({
    ...options,
    children: [new TextRun({ text: String(text), size: options.size || 22 })],
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}

function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullet-list", level: 0 },
    children: [new TextRun({ text, size: 22 })],
  });
}

function table(rows) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "B8CDD4" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const widths = [3100, 3150, 3110];
  return new Table({
    columnWidths: widths,
    margins: { top: 100, bottom: 100, left: 140, right: 140 },
    rows: rows.map((row, index) =>
      new TableRow({
        tableHeader: index === 0,
        children: row.map(
          (cell, cellIndex) =>
            new TableCell({
              borders,
              width: { size: widths[cellIndex] || 3120, type: WidthType.DXA },
              verticalAlign: VerticalAlign.CENTER,
              shading: index === 0 ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
              children: [
                new Paragraph({
                  alignment: index === 0 ? AlignmentType.CENTER : AlignmentType.LEFT,
                  children: [new TextRun({ text: String(cell), bold: index === 0, size: 20 })],
                }),
              ],
            })
        ),
      })
    ),
  });
}

const kpis = [
  ["指标", "结果", "说明"],
  ["融合长表规模", fmt(fusion.summary?.fusion_rows, 0), "本地单厂 + 国内公开监测 + WQP 联网补充"],
  ["决策样本", fmt(stage1.decision_dataset?.decision_rows, 0), "observed/load_up/rain_dilution 三类工况"],
  ["监督模型", stage2.selected_model || "-", `${stage2.feature_count || "-"} 个特征，多输出下一小时出水预测`],
  ["测试综合误差", fmt(stage2.test?.weighted_normalized_mae, 3), "污染物重要性加权归一化 MAE"],
  ["预测达标率", `${fmt((stage2.test?.overall_compliance?.predicted_rate || 0) * 100, 1)}%`, "COD/NH3-N/TP/TN 四目标"],
  ["Safe-MARL 可行率", `${fmt((safe.feasible_rate || 0) * 100, 1)}%`, "所有动作经过安全盾"],
  ["当前控制节能/节药", `${fmt(safe.mean_energy_saving_vs_current_pct, 2)}% / ${fmt(safe.mean_chemical_saving_vs_current_pct, 2)}%`, "安全仲裁后最终推荐"],
  ["传统定值节能/节药", `${fmt(benefit.traditional_fixed_baseline?.energy_saving_pct, 2)}% / ${fmt(benefit.traditional_fixed_baseline?.chemical_saving_pct, 2)}%`, "训练期 P90 保守定值对照"],
  ["P95 响应时间", `${fmt(benefit.response_time?.p95_ms, 1)} ms`, "单组推荐响应 ≤1s"],
  ["论文复现动态节能", `${fmt(paper.phase4_dynamic_replay?.dynamic_energy_saving_pct, 2)}%`, "Water Research 方法链复现对照"],
];

const markdown = `# 污水厂曝气加药智能决策开发参赛报告

## 1. 项目概述

本项目面向污水厂曝气加药运行中响应滞后、精准度低、药耗能耗高和复杂工况适应不足的问题，构建“多源数据融合 + 下一小时出水预测 + Safe-MARL 双智能体决策 + 约束安全盾 + 可视化大屏”的智能体系统。

## 2. 数据采集与融合

- 本地单厂小时级运行数据：${fmt(stage1.plant_real_hourly?.row_count, 0)} 行。
- 国内公开监测长表：${fmt(stage1.public_monitor_long?.row_count, 0)} 行。
- 融合数据长表：${fmt(fusion.summary?.fusion_rows, 0)} 行，来源包括 ${Object.keys(fusion.summary?.source_domains || {}).join("、")}。
- WQP 联网补充状态：${fusion.sources?.wqp?.status || "-"}，用于外部域分布参考，不直接混入单厂监督训练。

## 3. 模型与智能体设计

监督代理模型采用 ${stage2.selected_model || "-"}，输入包括进水水质、水量、DO、MLSS、当前控制量、当前出水、滞后特征、滚动特征、代理动态特征和质量标记。Safe-MARL 采用曝气智能体与投药智能体协同输出动作，共享出水风险、曝气能耗、药耗、平滑性和约束违规 reward。

## 4. 安全约束与决策闭环

所有 RL 推荐动作先进入安全盾，检查曝气强度上下限、PAC 投加上下限、单次最大调节步长和出水合规风险。随后进入“安全 + 目标函数”仲裁层：若 RL 动作不可行，或局部有界专家搜索能给出更优的合规动作，则回退到专家动作。大屏同时显示 reward 分解、可行率、回退率和中文动作解释。

核心目标函数可写为：

\\[
J = w_q R_{water} + w_e E_{aeration} + w_c C_{PAC} + w_s S_{smooth} + w_v V_{violation}
\\]

系统最小化 \\(J\\)，并将 reward 定义为 \\(-J\\)。其中 \\(V_{violation}\\) 对 COD、NH3-N、TP、TN 超限进行高权重惩罚，保证节能节药只在达标边界内发生。

## 5. 系统实现

后端采用 FastAPI，前端采用 React + ECharts。系统提供 /api/summary、/api/timeseries、/api/recommendations、/api/infer、/api/rl/recommend 和 /api/report/ai-summary 等接口，可支撑实时监测、智能推荐、AI 分析摘要和系统对接。

## 6. 结果与价值

- 测试集综合归一化 MAE：${fmt(stage2.test?.weighted_normalized_mae, 3)}。
- 预测达标率：${fmt((stage2.test?.overall_compliance?.predicted_rate || 0) * 100, 1)}%。
- Safe-MARL 推荐可行率：${fmt((safe.feasible_rate || 0) * 100, 1)}%。
- 相对当前控制：曝气节能 ${fmt(safe.mean_energy_saving_vs_current_pct, 2)}%，PAC 节药 ${fmt(safe.mean_chemical_saving_vs_current_pct, 2)}%。
- 相对传统保守定值：曝气节能 ${fmt(benefit.traditional_fixed_baseline?.energy_saving_pct, 2)}%，PAC 节药 ${fmt(benefit.traditional_fixed_baseline?.chemical_saving_pct, 2)}%。
- 单组推荐响应：P95 ${fmt(benefit.response_time?.p95_ms, 1)} ms，最大 ${fmt(benefit.response_time?.max_ms, 1)} ms。
- 论文复现动态回放节能：${fmt(paper.phase4_dynamic_replay?.dynamic_energy_saving_pct, 2)}%。

## 7. 创新点

1. 外部数据深度融合但保持监督训练域隔离，避免跨域混训风险。
2. 双智能体协同控制曝气与投药，兼顾合规、能耗、药耗和平滑性。
3. 约束安全盾把 RL 动作转化为可执行、可解释、可回退的工程建议。
4. 结合 Water Research 实时控制论文的动态特征思想，形成小时级代理动态特征和在线控制展示。
5. 大屏将预测、推荐、reward、数据源、工况鲁棒性和报告摘要放在同一操作界面。
`;

fs.writeFileSync(path.join(docsDir, "competition_report.md"), markdown, "utf-8");

const children = [
  new Paragraph({
    heading: HeadingLevel.TITLE,
    children: [new TextRun("污水厂曝气加药智能决策开发参赛报告")],
  }),
  p("Safe-MARL 双智能体协同控制与可视化大屏系统", { alignment: AlignmentType.CENTER, size: 24 }),
  new TableOfContents("目录", { hyperlink: true, headingStyleRange: "1-2" }),
  h1("1. 项目概述"),
  p("本项目面向污水厂曝气加药运行中响应滞后、精准度低、药耗能耗高和复杂工况适应不足的问题，构建多源数据融合、下一小时出水预测、Safe-MARL 双智能体决策、约束安全盾和可视化大屏的一体化智能体系统。"),
  h1("2. 数据采集与融合"),
  bullet(`本地单厂小时级运行数据 ${fmt(stage1.plant_real_hourly?.row_count, 0)} 行。`),
  bullet(`国内公开监测长表 ${fmt(stage1.public_monitor_long?.row_count, 0)} 行。`),
  bullet(`融合数据长表 ${fmt(fusion.summary?.fusion_rows, 0)} 行，覆盖 ${Object.keys(fusion.summary?.source_domains || {}).join("、")}。`),
  bullet(`WQP 联网补充状态为 ${fusion.sources?.wqp?.status || "-"}，用于外部域分布参考。`),
  h1("3. 模型与智能体设计"),
  p(`监督代理模型采用 ${stage2.selected_model || "-"}，共 ${stage2.feature_count || "-"} 个特征，用于同时预测下一小时 COD、NH3-N、TP 和 TN。`),
  p("Safe-MARL 采用曝气智能体与投药智能体协同输出动作，共享出水风险、曝气能耗、药耗、平滑性和约束违规 reward。"),
  h1("4. 安全约束与决策闭环"),
  p("所有 RL 推荐动作先进入安全盾，检查曝气强度上下限、PAC 投加上下限、单次最大调节步长和出水合规风险。随后进入安全与目标函数仲裁层：若 RL 动作不可行，或局部有界专家搜索能给出更优的合规动作，则回退到专家动作。"),
  p("核心目标函数：J = wq * 水质风险 + we * 曝气能耗 + wc * PAC药耗 + ws * 平滑项 + wv * 违规惩罚；系统最小化 J，并将 reward 定义为 -J。"),
  h1("5. 系统实现"),
  p("后端采用 FastAPI，前端采用 React + ECharts。系统提供摘要、时序、推荐、单点推理、RL 推荐和 AI 摘要接口。"),
  h2("核心指标"),
  table(kpis),
  h1("6. 结果与应用价值"),
  bullet(`测试集综合归一化 MAE 为 ${fmt(stage2.test?.weighted_normalized_mae, 3)}。`),
  bullet(`预测达标率为 ${fmt((stage2.test?.overall_compliance?.predicted_rate || 0) * 100, 1)}%。`),
  bullet(`Safe-MARL 推荐可行率为 ${fmt((safe.feasible_rate || 0) * 100, 1)}%。`),
  bullet(`相对当前控制，曝气节能 ${fmt(safe.mean_energy_saving_vs_current_pct, 2)}%，PAC 节药 ${fmt(safe.mean_chemical_saving_vs_current_pct, 2)}%。`),
  bullet(`相对传统保守定值，曝气节能 ${fmt(benefit.traditional_fixed_baseline?.energy_saving_pct, 2)}%，PAC 节药 ${fmt(benefit.traditional_fixed_baseline?.chemical_saving_pct, 2)}%。`),
  bullet(`单组推荐响应 P95 为 ${fmt(benefit.response_time?.p95_ms, 1)} ms，最大 ${fmt(benefit.response_time?.max_ms, 1)} ms。`),
  bullet(`论文复现动态回放节能为 ${fmt(paper.phase4_dynamic_replay?.dynamic_energy_saving_pct, 2)}%。`),
  h1("7. 创新点"),
  bullet("外部数据深度融合但保持监督训练域隔离，避免跨域混训风险。"),
  bullet("双智能体协同控制曝气与投药，兼顾合规、能耗、药耗和平滑性。"),
  bullet("约束安全盾把 RL 动作转化为可执行、可解释、可回退的工程建议。"),
  bullet("结合实时智能控制论文的动态特征思想，形成小时级代理动态特征和在线控制展示。"),
  bullet("大屏将预测、推荐、reward、数据源、工况鲁棒性和报告摘要放在同一操作界面。"),
];

const verifiedScreenshot = path.join(outputsDir, "dashboard_verified.png");
const fallbackScreenshot = path.join(outputsDir, "dashboard_fullpage.png");
const screenshot = fs.existsSync(verifiedScreenshot) ? verifiedScreenshot : fallbackScreenshot;
if (fs.existsSync(screenshot)) {
  children.push(h1("8. 可视化大屏截图"));
  children.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new ImageRun({
          type: "png",
          data: fs.readFileSync(screenshot),
          transformation: { width: 560, height: 454 },
          altText: { title: "Dashboard", description: "Safe-MARL wastewater dashboard screenshot", name: "dashboard" },
        }),
      ],
    })
  );
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      {
        id: "Title",
        name: "Title",
        basedOn: "Normal",
        run: { size: 40, bold: true, color: "12343B", font: "Arial" },
        paragraph: { spacing: { before: 240, after: 160 }, alignment: AlignmentType.CENTER },
      },
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 30, bold: true, color: "12343B", font: "Arial" },
        paragraph: { spacing: { before: 260, after: 160 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 25, bold: true, color: "1B5966", font: "Arial" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullet-list",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: { page: { margin: { top: 1200, right: 1200, bottom: 1200, left: 1200 } } },
      headers: {
        default: new Header({
          children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "污水厂曝气加药智能决策", size: 18, color: "667A82" })] })],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: "Page ", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], size: 18 })],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(path.join(docsDir, "competition_report.docx"), buffer);
  console.log("generated docs/competition_report.md and docs/competition_report.docx");
});
