# 第一问结论摘要

## 1. 数据处理结论

- 已经将缺失值区分为 `maintenance_gap` 和 `random_gap`
- 小时级原值保留在 `cleaned/csv/cleaned_filter_hourly_long.csv`
- 主分析不做插值，只有 `daily_median_plot` 用于绘图连贯性
- 第一问来源型清洗数据位于：
  - `cleaned/csv/cleaned_filter_hourly_long.csv`
  - `cleaned/csv/cleaned_maintenance_records.csv`
- 第一问分析型派生数据位于：
  - `cleaned/csv/cleaned_filter_daily_features.csv`
  - `cleaned/csv/cleaned_maintenance_event_effects.csv`
  - `cleaned/csv/cleaned_maintenance_cycles.csv`

## 2. 主分析口径

- 主分析不是回归
- 主分析使用 3 张分析型派生数据表：
  - `cleaned_filter_daily_features.csv`
  - `cleaned_maintenance_event_effects.csv`
  - `cleaned_maintenance_cycles.csv`
- 日特征表用于设备差异、季节性、主趋势和回归样本
- 维护事件表用于中维护/大维护即时增益
- 维护周期表用于周期衰减率
- 恢复高点趋势来自维护事件表中的维护后水平
- 回归只负责验证，不负责定义核心指标

## 3. 周期性结论

- 季节指数最高月份：`8`，修正指数为 `13.588`
- 季节指数最低月份：`1`，修正指数为 `-13.845`
- 月份季节指数对应表：`tables/q1_table_02_monthly_seasonal_index.csv`
- 其中 `seasonal_index_raw` 用于直观展示，`seasonal_index_adjusted` 用于后续分析

## 4. 下降趋势结论

- 维护周期内下降由 `r_i` 描述，对应：
  - `tables/q1_table_07_cycle_decay_raw.csv`
  - `tables/q1_table_08_cycle_decay_centered.csv`
- 长期恢复能力变化由 `b_H` 描述，对应：
  - `tables/q1_table_09_post_recovery_trend.csv`
  - `figures/png/q1_fig_06_post_recovery_trend.png`

## 5. 维护影响结论

- 中维护即时增益中位数：`17.630`
- 大维护即时增益中位数：`16.258`
- 中维护相对增益中位数：`0.238`
- 大维护相对增益中位数：`0.243`
- 维护影响统计表：`tables/q1_table_03_maintenance_effect_summary.csv`
- 控制回归中的 `is_major` 系数：`-3.17521`
- 维护增益控制回归表：`tables/q1_table_10_maintenance_gain_regression.csv`

## 6. 回归验证

- `days_since_last_maintenance` 系数：`-0.29039`
- `days_from_observation_start` 系数：`-0.02995`
- 回归样本只使用 `high_quality` 和 `low_quality` 的非维护日样本
- 维护日缺失不进入回归
- 回归仅用于验证影响因素，不作为寿命预测模型

## 7. 第一问交给第二问的核心指标

- 设备基准水平 `alpha_i`
- 月份季节指数 `S_m`
- 维护周期衰减率 `r_i`
- 中维护即时增益 `G_medium`
- 大维护即时增益 `G_major`
- 维护后恢复高点趋势 `b_H`
