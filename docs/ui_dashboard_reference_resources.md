# UI 大屏参考资源与本轮优化记录

## 新增 Skills

| Skill | 来源 | 安装命令 | 采用要点 |
| --- | --- | --- | --- |
| web-design-guidelines | https://github.com/vercel-labs/agent-skills；https://www.skills.sh/vercel-labs/agent-skills/web-design-guidelines | `npx skills add vercel-labs/agent-skills@web-design-guidelines -g -y` | 图标按钮补 `aria-label`、异步状态 `aria-live`、URL 反映筛选状态、focus-visible、reduced-motion |
| build-dashboard | https://github.com/anthropics/knowledge-work-plugins | `npx skills add anthropics/knowledge-work-plugins@build-dashboard -g -y` | 顶部 KPI、核心趋势图、操作建议、筛选器与明细面板分层 |
| frontend-accessibility-best-practices | https://www.skills.sh/sergiodxa/agent-skills/frontend-accessibility-best-practices | `npx skills add sergiodxa/agent-skills@frontend-accessibility-best-practices -g -y` | 语义化结构、键盘焦点、屏幕阅读器状态、触摸目标、用户动效偏好 |
| ui-ux-design | https://www.skills.sh/arvindand/agent-skills/ui-ux-design | `npx skills add arvindand/agent-skills@ui-ux-design -g -y` | 选择明确审美方向、明亮语义色、WCAG 对比度、状态完整性 |
| react-performance-optimization | https://www.skills.sh/nickcrew/claude-ctx-plugin/react-performance-optimization | `npx skills add nickcrew/claude-ctx-plugin@react-performance-optimization -g -y` | 用可测量的特征裁剪、推理耗时对比和 bundle/渲染检查支撑效率优化 |

说明：skills 已下载到本机全局 skills 目录；Codex 重启后会在技能列表中自动拾取。当前仓库仍保留项目专属 `.codex/skills/wwtp-safe-marl/SKILL.md` 与 `.codex/rules/wwtp-safe-marl-workflow.md` 作为参赛复现 workflow。

## 视觉参考

| 参考 | 链接 | 对本项目的借鉴 |
| --- | --- | --- |
| Apache ECharts 示例与产品说明 | https://echarts.apache.org/examples/en/index.html；https://echarts.apache.org/en/index.html | 保持 Canvas 图表、雷达图、热力图和趋势线组合；利用响应式容器与高对比色避免大屏拥挤 |
| Grafana Dashboard Gallery / Docs | https://grafana.com/grafana/dashboards/；https://grafana.com/docs/grafana/latest/visualizations/dashboards/ | 采用“同一屏内先回答运维问题”的面板组织：KPI、筛选、主趋势、推荐动作和诊断明细 |
| Ant Design Pro Analysis Dashboard | https://preview.pro.ant.design/dashboard/analysis/；https://github.com/ant-design/ant-design-pro | 借鉴企业级后台的浅色指标卡、图表区和表格明细层级，用清爽留白替代过重暗色 |
| Elastic Kibana Dashboard | https://www.elastic.co/kibana/kibana-dashboard/ | 强化交互筛选和状态 drill-down 思路，把工况、推荐动作和数据源状态放在同一决策闭环 |
| Alibaba Cloud DataV 场景说明 | https://www.alibabacloud.com/help/en/datav/latest/application-scenario | 借鉴指挥中心式信息分区，但避免纯装饰化视觉，优先展示可执行控制建议 |

## 本轮落地

- 新增顶部运行状态条：安全经济模式、推荐响应 P95、动作样本数、外部数据源连通状态。
- 筛选器状态写入 URL 查询参数，例如 `?metric=TN&scenario=load_up`，便于评审直接打开指定工况。
- 刷新按钮加入加载禁用、旋转反馈、`aria-busy` 和屏幕阅读器 `aria-live` 状态。
- 增加跳转链接、focus-visible、触摸目标、长文本换行、数字等宽和 reduced-motion 适配。
- 新增“基准模型评估”面板，直接展示 Stage-2 ExtraTrees 代理模型测试集 COD/NH3-N/TP/TN 的 MAE、RMSE、R² 和预测达标率。
- 响应式布局从 4 列逐级折叠到 2 列/1 列，避免移动端文本和图表重叠。
- 将整体主题从深色控制室调整为明亮工业数据产品风：低饱和水务背景、白色信息面、深色正文、语义色仅用于状态和图表。
- 接入 EcoLite 多模型验证：树模型、随机森林、梯度提升、Ridge 与融合模型的精度/时延权衡进入大屏模型面板，突出 49 动作候选搜索 P95 和 100ms 投药时机 miss rate。

## 分步反思

- 结果：大屏从“图表可展示”推进到“评审可解释、运维可扫读、筛选可复现”的状态；基准模型指标进入页面后，RL 推荐不再只依赖摘要数字。
- 风险：当前设计仍以静态 CSV 结果驱动，真实厂站接入后需要更细的实时告警、权限和数据延迟提示。
- 改进：下一步可把模型评估扩展为可切换的验证集/测试集/场景集视图，并接入更多传感器健康指标。
- 本轮再反思：浅色化后可读性和评审友好度提升，但大屏投影环境可能需要可切换深浅主题；后续可增加主题切换并保存到 URL 参数。
