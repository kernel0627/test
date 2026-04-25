# 第一问论文交付索引

## 1. 最终清洗数据

### 1.1 来源型清洗数据

- `cleaned/csv/cleaned_filter_hourly_long.csv`
  - 对应附件 1 的小时级透水率清洗长表
- `cleaned/csv/cleaned_maintenance_records.csv`
  - 对应附件 2 的维护记录清洗表
  - 保留维护类型，并补充维护顺序与前后维护间隔

### 1.2 分析型派生数据

- `cleaned/csv/cleaned_filter_daily_features.csv`
  - 第一问主分析日特征表
- `cleaned/csv/cleaned_maintenance_event_effects.csv`
  - 维护前后窗口效果表，用于 `G_medium`、`G_major`
- `cleaned/csv/cleaned_maintenance_cycles.csv`
  - 维护周期表，用于 `r_i`

对应 Parquet 版本位于：`cleaned/parquet/`

## 2. 检查表

- `tables/q1_check_duplicate_timestamps.csv`
  - 小时级重复时间戳检查
- `tables/q1_check_per_value_range.csv`
  - 透水率取值范围检查
- `tables/q1_check_row_counts.csv`
  - 清洗前后行数与质量分层计数
- `tables/q1_check_maintenance_gap_summary.csv`
  - 维护性缺失统计
- `tables/q1_check_random_gap_summary.csv`
  - 随机缺失统计
- `tables/q1_check_random_long_gap_summary.csv`
  - 随机长缺口统计
- `tables/q1_check_event_eligibility_summary.csv`
  - 维护事件可分析性统计
- `tables/q1_check_cycle_eligibility_summary.csv`
  - 维护周期可分析性统计

## 3. 第一问核心表

- `tables/q1_table_01_data_overview.csv`
  - 数据概况与设备差异
- `tables/q1_table_02_monthly_seasonal_index.csv`
  - 原始季节指数与修正季节指数
- `tables/q1_table_03_maintenance_effect_summary.csv`
  - 中维护/大维护增益统计
- `tables/q1_table_04_regression_summary.csv`
  - 回归验证结果
- `tables/q1_table_05_device_core_metrics.csv`
  - `alpha_i`、`r_i`、`G_medium`、`G_major`、`b_H`
- `tables/q1_table_06_maintenance_event_alignment.csv`
  - 维护事件对齐数据
- `tables/q1_table_07_cycle_decay_raw.csv`
  - 周期衰减原始水平
- `tables/q1_table_08_cycle_decay_centered.csv`
  - 周期衰减中心化结果
- `tables/q1_table_09_post_recovery_trend.csv`
  - 维护后恢复高点趋势
- `tables/q1_table_10_maintenance_gain_regression.csv`
  - 维护增益控制回归
- `tables/q1_maintenance_recovery_ratio.csv`
  - 中维护和大维护恢复比例 rho 统计
- `tables/q1_table_correlation_matrix.csv`
  - Spearman 相关矩阵
- `tables/q1_table_window_sensitivity_gain.csv`
  - 维护窗口敏感性分析
- `tables/q1_current_maintenance_rule.csv`
  - 当前固定维护规律

## 4. 第一问最终 PNG 图

- `figures/png/q1_fig_01_all_devices_trend_with_maintenance.png`
  - 图 1：设备差异与维护标记
- `figures/png/q1_fig_02_monthly_seasonal_index.png`
  - 图 2：季节指数
- `figures/png/q1_fig_03_maintenance_event_alignment.png`
  - 图 3：维护事件对齐
- `figures/png/q1_fig_04_cycle_decay_raw.png`
  - 图 4：周期衰减原始水平
- `figures/png/q1_fig_05_cycle_decay_centered.png`
  - 图 5：周期衰减中心化
- `figures/png/q1_fig_06_post_recovery_trend.png`
  - 图 6：恢复高点趋势
- `figures/png/q1_fig_07_missingness_structure.png`
  - 图 7：缺失结构图
- `figures/png/q1_fig_08_maintenance_gain_comparison.png`
  - 图 8：维护增益比较
- `figures/png/q1_fig_09_correlation_heatmap.png`
  - 图 9：相关性热力图
- `figures/png/q1_fig_10_window_sensitivity_gain.png`
  - 图 10：窗口敏感性
- `figures/png/q1_fig_11_current_maintenance_rule.png`
  - 图 11：当前维护规律

## 5. 主分析口径

- 主分析不是回归
- 主分析使用：
  - `cleaned_filter_daily_features.csv`
  - `cleaned_maintenance_event_effects.csv`
  - `cleaned_maintenance_cycles.csv`
- 回归仅用于解释性验证

## 6. 文字说明

- `markdown/q1_cleaning_notes.md`
- `markdown/q1_findings_summary.md`
- `markdown/q1_regression_notes.md`
- `markdown/q1_outputs_index.md`
- `markdown/q1_eda_summary.md`

## 7. 说明

- 第一问最终图像只保留 `png`
- `html/svg` 不作为最终交付物保留
