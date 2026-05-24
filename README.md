# 污水厂曝气加药 Safe-MARL 智能决策系统

本仓库是“数据要素素质大赛：污水厂曝气加药智能决策开发”的工程实现。项目以本地单厂运行数据和公开监测数据为底座，构建数据融合、监督代理模型、Safe-MARL 双智能体推荐、安全约束盾和 React/FastAPI 可视化大屏。

## 主要模块

- `scripts/build_stage1_datasets.py`：构建单厂小时级主表、公开监测长表和决策数据集。
- `scripts/train_stage2_model.py`：训练多输出出水水质代理模型并生成局部约束推荐。
- `scripts/build_external_fusion_dataset.py`：融合本地、国内公开监测、WQP/Kaggle 适配数据。
- `scripts/train_safe_marl.py`：训练轻量 PyTorch 双智能体策略，并通过安全盾生成推荐。
- `scripts/evaluate_decision_benefits.py`：评估节能药耗收益、复杂工况鲁棒性和单组响应时间。
- `wwtp_decision/`：Safe-MARL、安全盾、奖励函数、动作解释等共享逻辑。
- `dashboard/backend/`：FastAPI 接口，提供摘要、时序、推荐、推理和报告 API。
- `configs/`：排放阈值、控制边界、外部数据源和 Safe-MARL 配置。
- `docs/final_scoring_matrix.md`：按赛题 20/50/30/20 评分口径整理证据索引。

## 快速运行

```powershell
python scripts/build_stage1_datasets.py
python scripts/validate_stage1_outputs.py
python scripts/train_stage2_model.py
python scripts/build_external_fusion_dataset.py
python scripts/train_safe_marl.py
python scripts/evaluate_decision_benefits.py
uvicorn dashboard.backend.main:app --reload --host 127.0.0.1 --port 8000
```

另开一个终端启动前端：

```powershell
cd dashboard/frontend
npm install
npm run dev
```

浏览器打开 `http://127.0.0.1:5173` 查看可视化大屏。前端通过 Vite 代理访问 `http://127.0.0.1:8000/api/*`。

如果已经执行过 `npm run build`，也可以只启动后端并访问 `http://127.0.0.1:8000`，FastAPI 会自动托管 `dashboard/frontend/dist`。

## 验证

```powershell
python scripts/validate_stage1_outputs.py
python scripts/build_external_fusion_dataset.py
python scripts/train_safe_marl.py --epochs 40
python scripts/evaluate_decision_benefits.py
pytest
cd dashboard/frontend
npm run build
```

## 交付打包

```powershell
python scripts/package_submission.py
```

生成的作品包位于 `outputs/submission/wwtp_safe_marl_submission.zip`。原始大数据、PDF/Office 源材料、模型二进制和 `node_modules` 不进入压缩包。

评审可先阅读 `docs/final_scoring_matrix.md`，再查看 `docs/competition_report.docx`、`docs/literature_basis_table.md` 和 Web 大屏。

## 已保留的轻量结果

仓库只保留可复现和演示所需的轻量结果文件，例如 `summary.json`、模型卡、测试预测表、Safe-MARL 推荐表和场景库。原始数据、PDF、Word、Excel、模型二进制和大规模融合长表已通过 `.gitignore` 排除，避免把大文件或敏感材料上传到 GitHub。

## 核心创新

1. 深度融合数据：单厂真实运行数据 + 国内公开监测 + 外部开放数据适配器。
2. Safe-MARL：曝气智能体和加药智能体协同输出动作。
3. 安全约束盾：所有 RL 动作经过设备边界、单次步长和合规风险检查。
4. 可解释推荐：输出 reward 分解、节能药耗变化和中文动作解释。
5. 工程闭环：保留监督代理模型、局部专家优化和论文复现结果作为对照。
