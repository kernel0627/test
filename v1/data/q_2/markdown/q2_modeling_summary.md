# 第二问建模总结

第二问采用恢复比例主模型预测寿命，并保留固定增益模型作为 baseline。小维护作为背景维护；中维护和大维护作为显式维护事件；大维护每年允许 0 至 4 次，不强制年度必做。

canonical 输出包括：

- `q2_model_parameters.csv`
- `q2_life_prediction_results.csv`
- `q2_backtest_metrics.csv`

旧表继续保留兼容，但论文和后续分析以上述 canonical 表为准。

寿命判定采用 rolling365 低于 37 加强制大维护恢复测试。强制大维护只用于判定，不改变主预测路径。
