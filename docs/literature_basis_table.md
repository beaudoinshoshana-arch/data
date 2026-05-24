# 论文依据与本项目创新对照表

本表用于说明项目如何借鉴论文，并在哪些环节做出工程化创新。在线条目已在开发过程中核对，完整文献笔记见 `文献/文献索引与建模公式笔记.md` 与 `文献/结构性研究空白_GAPS.md`。

| 依据类别 | 代表文献/来源 | 本项目采用方式 | 创新延伸 |
| --- | --- | --- | --- |
| 活性污泥机理 | A Comprehensive View of the ASM1 Dynamic Model: Study on a Practical Case, 2022, DOI: 10.3390/w14071046 | 用于确定进水负荷、DO、MLSS、出水指标和质量平衡解释。 | 没有直接复刻 ASM1，而是把机理变量压缩成监督代理模型和安全 reward 输入。 |
| 污水厂建模仿真 | Systematic Modeling of Municipal Wastewater Activated Sludge Process and Treatment Plant Capacity Analysis Using GPS-X, 2020, DOI: 10.3390/su12198182 | 支撑“工况场景 + 代理环境”的设计思路。 | 用本地真实小时级数据和公开监测分布构建轻量场景库。 |
| 曝气控制综述 | A Review of AI-Driven Control Strategies in the Activated Sludge Process with Emphasis on Aeration Control, 2024, DOI: 10.3390/w16020305 | 支撑曝气强度作为主要能耗控制量。 | 将曝气 agent 与投药 agent 拆成双智能体，而不是只优化 DO。 |
| 曝气能耗与氧传递 | Modeling and Control Strategies for Energy Management in a Wastewater Center: A Review on Aeration, 2024, DOI: 10.3390/en17133162 | 支撑能耗项和曝气节能指标。 | 大屏直接展示曝气节能、动作平滑和合规风险。 |
| 化学除磷投药 | Research on Intelligent Chemical Dosing System for Phosphorus Removal in Wastewater Treatment Plants, 2024, DOI: 10.3390/w16111623 | 支撑 PAC 投加量、总磷风险和药耗优化。 | 把 PAC 投加从单独回归问题提升为与曝气协同的多目标动作。 |
| LSTM/MPC 合理性验证 | LSTM-Based Model-Predictive Control with Rationality Verification for Bioreactors in Wastewater Treatment, 2023, DOI: 10.3390/w15091779 | 支撑预测模型 + 合理性/边界校验的闭环框架。 | 增加安全盾与目标函数仲裁：RL 不可行或不优时回退专家搜索。 |
| 经济 MPC | Optimal Control of Wastewater Treatment Plants Using Economic-Oriented Model Predictive Dynamic Strategies, 2017, DOI: 10.3390/app7080813 | 支撑合规、能耗、药耗、平滑性的多目标形式。 | 将经济目标写成 Safe-MARL reward，并在 API 中返回 reward 分解。 |
| 出水时序预测 | RNN、IFFNN、Attention CNN-LSTM 等出水水质预测论文 | 支撑滞后、滚动、差分和当前出水状态特征。 | 使用 ExtraTrees 代理模型作为快速安全基线，并与 RL 推荐层解耦。 |
| 多智能体污水优化 | Optimal control towards sustainable wastewater treatment plants based on multi-agent reinforcement learning, arXiv:2008.10417 | 支撑多控制量、多 agent 协同优化思路。 | 针对赛题拆分为曝气 agent 与投药 agent，并显式加入工程安全盾。 |
| SAC 延迟控制 | Application of Soft Actor-Critic Algorithms in Optimizing Wastewater Treatment with Time Delays Integration, arXiv:2411.18305 | 支撑污水控制中慢时滞和随机扰动需要 RL/安全约束。 | 当前版本采用轻量 PyTorch 离线 RL，不依赖 stable-baselines3，便于比赛环境复现。 |
| Water Research 实时智能控制 | Integrated real-time intelligent control for wastewater treatment plants: Data-driven modeling | 支撑动态特征、在线回放和控制链路复现。 | 本项目在 `outputs/paper_repro_integrated_control/` 中复现方法链，并把结果接入大屏 KPI。 |
| WQP/EPA 外部数据 | Water Quality Portal、EPA ECHO、EPA DMAP | 支撑联网开放水质数据、监管数据和外部域标签。 | 外部数据默认用于场景库和鲁棒性，不直接混入单厂监督训练，降低域偏移风险。 |

## 创新归纳

1. **Safe-MARL + 双智能体**：将曝气和 PAC 投药拆成协同 agent，共享合规、能耗、药耗和平滑 reward。
2. **深度融合场景库**：本地单厂、国内公开监测、WQP 联网补充统一到长表规范，但保留域标签。
3. **约束安全盾**：动作边界、单步调节、出水风险和专家回退形成工程可执行层。
4. **目标函数仲裁**：不是盲目采纳 RL，而是在安全和目标函数上与专家搜索比较。
5. **可解释大屏**：把 reward 分解、动作解释、数据源状态、鲁棒性热力图和 AI 摘要同时展示。
