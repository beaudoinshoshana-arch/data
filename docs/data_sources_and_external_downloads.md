# 数据来源与联网补充说明

## 本地与公开数据

- 本地单厂运行数据：`水厂数据-06yzzx/`，用于监督训练、模型验证和大屏实时回放。
- 台州公开 CSV 与济南 RDF：由 `scripts/build_stage1_datasets.py` 解析为 `public_monitor_long.csv`，用于公开监测分布、异常阈值和鲁棒性场景库。

## 联网补充数据

- Water Quality Portal (WQP)：`scripts/build_external_fusion_dataset.py` 通过 `https://www.waterqualitydata.us/data/Result/search` 下载小范围水质记录，当前配置为 Wisconsin、2024-01-01 到 2024-01-02、Ammonia/pH 短查询，避免大范围请求超时。
- EPA ECHO：当前纳入元数据来源说明，ECHO DMR/ICIS-NPDES 文件较大，适合后续用 Release/Git LFS 或单独数据目录接入。
- Kaggle：支持 `external_data/kaggle/**/*.csv` 本地适配；若用户下载 Kaggle 数据放入该目录，融合脚本会尝试按列名映射 COD/BOD/NH3N/TP/TN/FLOW。
- IWA BSM1：联网读取 `Inf_dry_2006.txt` 与 `Inf_rain_2006.txt`，转换为 15 分钟动态进水先验，映射 COD、BOD、NH3N、TN、DO、ALK、FLOW。
- Agtrup / BlueKolding：通过 Mendeley Data `10.17632/34rpmsxc4z.1` 的公开文件 API 下载 `IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv`，保留 2 分钟 SCADA 高频变量 FLOW、DO、NH3N、TP、PAC、TEMP、COAGULANT_IN 等。

## 高频数据产物

- `outputs/fusion_data/source_registry.json`：记录融合长表 1,597,529 行，其中高频行 258,830 行、分钟级行 240,000 行、最小原始粒度 2 分钟。
- `outputs/fusion_data/high_frequency_source_catalog.csv`：列出 Agtrup 2 分钟 SCADA 与 BSM1 15 分钟动态进水的时间范围、指标和粒度。
- `outputs/fusion_data/high_frequency_sample.csv`：仅保留轻量样本用于 GitHub 与评审检查；Agtrup 原始 43MB CSV 缓存在被忽略的 `external_data/agtrup/`。
- `outputs/minute_simulation/summary.json` 与 `outputs/minute_simulation/minute_simulation_report.md`：基于 2 分钟 SCADA 回放 2/5/15/60 分钟控制周期，并比较 2 分钟控制相对 60 分钟控制的收益。

## 使用边界

外部数据默认标记为 `external_prior`，用途是分布边界、鲁棒性评估和场景扩增，不直接与本地单厂监督标签混训。只有确认厂站、单位、标签、控制变量语义一致时，才允许进入监督训练。

分钟级和 15 分钟级外部数据默认标记为 `high_frequency_simulation_and_domain_prior`，用于仿真、场景扰动和控制周期对比；Agtrup 数据包含真实高频过程变量，但没有与本地厂完全一致的 COD/BOD/出水标签，因此当前不混入单厂监督训练。

## 参考链接

- WQP Web Services Guide: https://www.waterqualitydata.us/webservices_documentation/
- EPA ECHO Data Downloads: https://echo.epa.gov/tools/data-downloads
- EPA DMAP: https://data.epa.gov/
- IWA BSM1 Benchmarking: https://iwa-mia.org/benchmarking/
- Agtrup / BlueKolding Mendeley Data: https://data.mendeley.com/datasets/34rpmsxc4z/1
