# EcoLite 多模型效率、集成与投药时机验证

## 结论

- 推荐在线 profile：`compact_gradient_boosting_40`（boosting_gradient），测试加权标准化 MAE 0.1077，预测达标率 100.0%。
- 全量基准模型：349 特征，测试加权标准化 MAE 0.2410，测试集预测耗时 83.69 ms。
- 推荐 profile 候选动作批量 P95：2.29 ms，100ms 投药时机 miss rate 0.0%。
- 最低误差 profile：`compact_gradient_boosting_40`，MAE 0.1077，候选批量 P95 2.29 ms。
- 最低延迟 profile：`fast_ridge_40`，候选批量 P95 1.01 ms，MAE 0.1316。
- Safe-MARL 当前控制对照曝气节能 38.67%，PAC 节药 69.63%。

## 模型组合对比

| 模型 | 类型 | 特征/复杂度 | MAE | 达标率 | 单点P95ms | 49动作P95ms | 100ms miss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_extra_trees_current | baseline_full_model | 349 / 244300 | 0.2410 | 100.0% | 31.91 | 37.93 | 0.0% |
| ecolite_extra_trees_24 | compact_extra_trees | 24 / 3360 | 0.2524 | 100.0% | 7.35 | 8.29 | 0.0% |
| ecolite_extra_trees_40 | compact_extra_trees | 40 / 5600 | 0.2367 | 100.0% | 7.71 | 8.28 | 0.0% |
| ecolite_extra_trees_64 | compact_extra_trees | 64 / 8960 | 0.2483 | 100.0% | 7.53 | 9.53 | 0.0% |
| compact_random_forest_40 | bagging_random_forest | 40 / 4800 | 0.2421 | 100.0% | 6.04 | 7.41 | 0.0% |
| compact_gradient_boosting_40 | boosting_gradient | 40 / 19200 | 0.1077 | 100.0% | 2.07 | 2.29 | 0.0% |
| fast_ridge_40 | linear_ridge | 40 / 160 | 0.1316 | 100.0% | 1.00 | 1.01 | 0.0% |
| balanced_extra_trees_96 | compact_extra_trees | 96 / 10560 | 0.2407 | 100.0% | 6.24 | 7.21 | 0.0% |
| validation_weighted_ensemble | ensemble_error_weighted | 120 / 12471 | 0.1596 | 100.0% | 17.67 | 17.43 | 0.0% |
| latency_penalized_ensemble | ensemble_latency_penalized | 120 / 13954 | 0.1130 | 100.0% | 10.48 | 11.55 | 0.0% |

## 投药时机仿真

- 仿真假设：每个控制周期需要一次 49 动作局部搜索，再下发曝气/PAC 设定值；若候选批量推理超过 deadline，则视为有投药时机不足风险。
- 推荐 profile 在 20/50/100/250/1000ms deadline 下的 miss rate 分别为 0.0%、0.0%、0.0%、0.0%、0.0%。

## 分步反思

- 结果：本轮不再只验证单一 ExtraTrees 特征裁剪，而是同时比较袋装树、随机森林、梯度提升、线性 Ridge、误差加权融合和延迟惩罚融合。
- 风险：集成融合虽然可能降低误差，但需要多次模型推理；在短 deadline 工况下应选择低延迟 profile 或延迟惩罚融合。
- 改进：下一步可把推荐 profile 固化为在线推理 artifact，并在 `/api/rl/recommend` 中记录真实候选搜索耗时。
