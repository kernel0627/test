# 第一问 EDA 总结

## 1. 数据清洗口径

小时级原值保留在 `per_raw`，分析值保留在 `per_analysis`，第一问主分析使用日中位透水率。

小维护约每 3 至 5 天进行一次，但附件 2 未提供小维护日期，因此本文不将小维护作为显式维护事件，也不计算 `G_small` 或 `rho_small`。小维护被视为高频、低成本、未记录的背景维护机制，其影响包含在日常波动和两次显式维护之间的净衰减率 `r_i*` 中。

## 2. 缺失类型说明

维护日整天无数据定义为 `maintenance_gap`，非维护日整天无数据定义为 `random_gap`。

## 3. 异常值处理

仅剔除物理异常和确认孤立异常，维护日前后跳升、连续下降和真实低值不因数值低而删除。

## 4. 设备差异

设备差异由 `q1_table_05_device_core_metrics.csv` 中的 `alpha_raw` 和 `alpha_reg_fe` 描述。

## 5. 相关性分析

Spearman 相关矩阵输出到 `q1_table_correlation_matrix.csv`，热力图输出到 `q1_fig_09_correlation_heatmap.png`。相关性只说明变量关系方向，不作为因果结论。

## 6. 季节周期

月份季节指数输出到 `q1_table_02_monthly_seasonal_index.csv`，后续建模优先使用 `seasonal_index_adjusted`。

## 7. 当前维护规律

当前维护规律输出到 `q1_current_maintenance_rule.csv`，包括维护间隔中位数、大维护比例和历史维护序列，是第二问生成未来维护日程的直接输入。

## 8. 维护周期衰减

周期衰减由 `q1_table_07_cycle_decay_raw.csv` 和 `q1_table_08_cycle_decay_centered.csv` 描述，中心化版本更适合论文解释。这里的维护周期定义为相邻有记录的中维护或大维护之间的周期，不是大维护年度周期；衰减率解释为小维护背景下的净衰减率。

## 9. 窗口敏感性

窗口敏感性输出到 `q1_table_window_sensitivity_gain.csv` 和 `q1_fig_10_window_sensitivity_gain.png`。主窗口仍采用 3 天，1 天窗口易受波动影响，5 天和 7 天窗口可能混入自然衰减并减少有效事件数。

## 10. 中/大维护影响

维护即时效果以维护事件表中的 `gain_abs` 和 `gain_rel` 为主，不用日尺度回归窗口变量替代维护效果主结论。

恢复比例 `rho` 输出到 `q1_maintenance_recovery_ratio.csv`。事件级 `rho_raw` 允许小于 0 或大于 1，模型使用 `rho_clipped` 及第二问收缩后的设备级 `rho_use`。大维护每年允许 0 至 4 次，并非必然年度固定维护，因此 `G_major` 和 `rho_major` 的解释需要强调样本有限和参数收缩。

## 11. 恢复高点趋势

恢复高点趋势输出到 `q1_table_09_post_recovery_trend.csv`，并进入第二问恢复上限模型。

## 12. 回归验证

回归结果输出到 `q1_table_04_regression_summary.csv` 和 `q1_table_10_maintenance_gain_regression.csv`，只作为解释性验证。

## 13. 第一问核心指标

核心影响指标为 `alpha_i, S_m, r_i, G_medium, G_major, b_H`；承接第二问的维护规律为 `R_i=(I_i,p_i_major,interval_sequence,type_sequence)`。

## 14. 第二问输入文件说明

第二问读取 `q1_cleaned_filter_daily_features.csv`、`q1_cleaned_maintenance_event_effects.csv`、`q1_cleaned_maintenance_cycles.csv`、`q1_table_02_monthly_seasonal_index.csv`、`q1_table_05_device_core_metrics.csv`、`q1_current_maintenance_rule.csv`。

第二问主模型读取设备级收缩参数 `rho_medium_use` 和 `rho_major_use`，不直接使用事件级 `rho_raw`。

## 附：新增结果规模

- 相关矩阵变量数：6
- 窗口敏感性记录数：8
- 当前维护规律设备数：10
