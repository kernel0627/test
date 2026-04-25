# 第二问寿命预测解释

## 判定规则

主判据为 `rolling365_pred < 37` 加强制大维护恢复测试失败。恢复测试统计量为 `M_i_at_life_test`，即强制大维护后未来一年内滚动 365 天均值的最大值。

## 预测结果

canonical 结果表为 `q2_life_prediction_results.csv`。剩余寿命较短的设备如下：

```text
device_id predicted_life_end_date  remaining_life_years                               life_end_reason
      a10     2060-05-25 00:00:00             34.142466 rolling365_below_37_and_major_recovery_failed
       a1                                           NaN                        not_reached_within_50y
       a2                                           NaN                        not_reached_within_50y
```

## 注意事项

50 年为计算截断，不代表对 50 年后状态作可靠预测。`not_reached_within_50y` 表示在截断期内未触发寿命终止条件。
