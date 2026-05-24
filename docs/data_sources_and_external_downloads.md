# 数据来源与联网补充说明

## 本地与公开数据

- 本地单厂运行数据：`水厂数据-06yzzx/`，用于监督训练、模型验证和大屏实时回放。
- 台州公开 CSV 与济南 RDF：由 `scripts/build_stage1_datasets.py` 解析为 `public_monitor_long.csv`，用于公开监测分布、异常阈值和鲁棒性场景库。

## 联网补充数据

- Water Quality Portal (WQP)：`scripts/build_external_fusion_dataset.py` 通过 `https://www.waterqualitydata.us/data/Result/search` 下载小范围水质记录，当前配置为 Wisconsin、2024-01-01 到 2024-01-02、Ammonia/pH 短查询，避免大范围请求超时。
- EPA ECHO：当前纳入元数据来源说明，ECHO DMR/ICIS-NPDES 文件较大，适合后续用 Release/Git LFS 或单独数据目录接入。
- Kaggle：支持 `external_data/kaggle/**/*.csv` 本地适配；若用户下载 Kaggle 数据放入该目录，融合脚本会尝试按列名映射 COD/BOD/NH3N/TP/TN/FLOW。

## 使用边界

外部数据默认标记为 `external_prior`，用途是分布边界、鲁棒性评估和场景扩增，不直接与本地单厂监督标签混训。只有确认厂站、单位、标签、控制变量语义一致时，才允许进入监督训练。

## 参考链接

- WQP Web Services Guide: https://www.waterqualitydata.us/webservices_documentation/
- EPA ECHO Data Downloads: https://echo.epa.gov/tools/data-downloads
- EPA DMAP: https://data.epa.gov/
