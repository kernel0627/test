# 第二问回测总结

## 对照模型

`fixed_gain_baseline` 使用固定增益恢复；`recovery_ratio_main` 使用恢复比例恢复当前可恢复空间的一部分。

## 指标文件

回测结果输出到 `q2_backtest_metrics.csv`。当前 MAE 最小模型为 `fixed_gain_baseline`。

```text
           model_name       mae      rmse     mape  n_validation_samples
baseline_season_trend 12.190142 14.964227 0.207370                  1533
  fixed_gain_baseline  9.116254 11.147085 0.142850                  1533
  recovery_ratio_main 10.684833 12.887928 0.181734                  1533
```
