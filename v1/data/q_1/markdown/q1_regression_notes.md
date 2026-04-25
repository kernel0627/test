# 回归解释说明

## 模型口径

- 因变量：`daily_median`
- 样本：`daily_quality` 为 `high_quality` 或 `low_quality`，且 `daily_median` 非空，并排除维护日
- 协变量：
  - 设备固定效应
  - 月份效应
  - 距上次维护天数
  - 观测窗口内时间趋势
  - 中维护后 1 到 3 天窗口
  - 大维护后 1 到 3 天窗口

## 解释边界

- `days_from_observation_start` 仅表示观测窗口内长期变化
- 不将其直接解释为完整寿命阶段的全部老化过程
- 标准误采用 HC3 稳健标准误
- 该回归用于解释性验证，不用于严格因果识别
