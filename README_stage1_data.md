# 污水厂曝气加药项目第一阶段

这个目录下新增了一套可直接复用的“数据采集与预处理”流水线，目标是把本地单厂数据、台州公开 CSV、济南 RDF 公开数据统一加工成后续建模可直接接入的三类资产。

## 文件

- `scripts/build_stage1_datasets.py`
  生成 `plant_real_hourly`、`public_monitor_long`、`decision_dataset` 及汇总报告。
- `scripts/validate_stage1_outputs.py`
  对生成结果做结构、数量、字段和切分规则校验。
- `outputs/stage1_data/`
  默认输出目录。

## 运行

```powershell
python scripts/build_stage1_datasets.py
python scripts/validate_stage1_outputs.py
```

## 主要产物

- `outputs/stage1_data/plant_real_hourly/plant_real_hourly.csv`
  单厂小时级主表，包含核心传感器、可选运行指标、未来一小时出水标签和质量标记列。
- `outputs/stage1_data/public_monitor_long/public_monitor_long.csv`
  台州 CSV 与济南 RDF 解析后的公开长表。
- `outputs/stage1_data/decision_dataset/decision_dataset_raw.csv`
  按赛题 8 个核心字段组织的原始决策数据集，含 `observed/load_up/rain_dilution` 三类场景。
- `outputs/stage1_data/decision_dataset/decision_dataset_scaled.csv`
  仅用训练集拟合 `RobustScaler` 后得到的缩放版数据集。
- `outputs/stage1_data/decision_dataset/decision_labels_observed.csv`
  观测场景对应的未来一小时出水标签表。
- `outputs/stage1_data/reports/stage1_summary.json`
  记录样本量、质量规则命中数、分割区间和输出路径。

## 默认规则

- 本地 Excel 时间按“最近整点”对齐，而不是直接向下取整。
- 公开数据只保留污水处理相关厂站，指标只保留 `COD/NH3N/TP/TN/pH/FLOW`。
- `BOD、曝气强度、药剂投加量` 按既定工程规则补全，并保留 `is_simulated` 与 `scenario_tag`。
- 训练、验证、测试按时间顺序切分，缩放参数只由训练集拟合。
