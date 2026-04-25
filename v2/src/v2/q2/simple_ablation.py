from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["MAE", "RMSE", "post_maintenance_7d_MAE", "rolling365_MAE"]


def _weighted_metric(rows: pd.DataFrame, metric: str, count_col: str) -> float:
    data = rows[[metric, count_col]].dropna().copy()
    if data.empty or float(data[count_col].sum()) <= 0:
        return math.nan
    values = data[metric].astype(float)
    weights = data[count_col].astype(float)
    if metric == "RMSE":
        return float(np.sqrt(np.average(values**2, weights=weights)))
    return float(np.average(values, weights=weights))


def _metric_count_col(metric: str) -> str:
    if metric == "post_maintenance_7d_MAE":
        return "n_post_maintenance_7d_days"
    if metric == "rolling365_MAE":
        return "n_rolling365_days"
    return "n_validation_days"


def _summary_from_rows(rows: pd.DataFrame) -> dict[str, float]:
    return {metric: _weighted_metric(rows, metric, _metric_count_col(metric)) for metric in METRICS}


def _interpret(row: dict[str, object]) -> str:
    model = str(row["model_name"])
    if model == "M0_trend_season_baseline":
        return "未运行；保留为只含趋势/季节的可选基准，不影响当前最小消融结论。"
    if model == "M1_fixed_gain_baseline":
        return "固定增益模型作为显式维护恢复机制的基准。"
    return "按表06推荐模型逐设备汇总，用于代表第二问最终推荐口径。"


def build_simple_ablation(backtest: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    m0 = {
        "model_level": "M0",
        "model_name": "M0_trend_season_baseline",
        "MAE": math.nan,
        "RMSE": math.nan,
        "post_maintenance_7d_MAE": math.nan,
        "rolling365_MAE": math.nan,
        "improvement_over_M0": "optional_not_run",
    }
    m0["interpretation"] = _interpret(m0)
    rows.append(m0)

    fixed_all = backtest[
        (backtest["device_id"].astype(str) == "all")
        & (backtest["model_name"].astype(str) == "fixed_gain_baseline")
    ]
    if len(fixed_all):
        fixed_metrics = {metric: float(fixed_all[metric].iloc[0]) for metric in METRICS}
    else:
        fixed_metrics = _summary_from_rows(backtest[backtest["model_name"].astype(str) == "fixed_gain_baseline"])
    m1 = {
        "model_level": "M1",
        "model_name": "M1_fixed_gain_baseline",
        **fixed_metrics,
        "improvement_over_M0": "not_applicable",
    }
    m1["interpretation"] = _interpret(m1)
    rows.append(m1)

    selected_rows: list[pd.Series] = []
    device_backtest = backtest[backtest["device_id"].astype(str) != "all"].copy()
    for _, item in comparison.iterrows():
        device_id = str(item["device_id"])
        preferred = str(item["preferred_model"])
        model = "fixed_gain_baseline" if preferred == "fixed_gain_baseline" else "recovery_ratio_main"
        match = device_backtest[
            (device_backtest["device_id"].astype(str) == device_id)
            & (device_backtest["model_name"].astype(str) == model)
        ]
        if len(match):
            selected_rows.append(match.iloc[0])
    selected = pd.DataFrame(selected_rows)
    final_metrics = _summary_from_rows(selected)
    m2 = {
        "model_level": "M2",
        "model_name": "M2_final_preferred_model",
        **final_metrics,
        "improvement_over_M0": "not_applicable",
    }
    m2["interpretation"] = _interpret(m2)
    rows.append(m2)
    return pd.DataFrame(rows)


def write_ablation_markdown(path: Path) -> None:
    content = """# 第二问检验与消融说明

## 1. 历史回测

第二问采用历史回测检验模型预测能力。每台设备最后 180 个有效日作为验证期，验证期使用真实维护日程，避免把维护触发误差混入状态转移比较。

评价指标包括 MAE、RMSE、`post_maintenance_7d_MAE` 和 `rolling365_MAE`。模型选择依据表03回测结果和表06推荐逻辑。

## 2. 简化消融

本次只做最小消融，不重写主模型，也不增加复杂模型层级。表10比较三档口径：

- `M0_trend_season_baseline`：只使用趋势/季节的可选基准，若暂未运行则标记为 `optional_not_run`。
- `M1_fixed_gain_baseline`：固定增益维护恢复基准。
- `M2_final_preferred_model`：按表06推荐模型逐设备汇总后的最终口径。

简化消融用于证明显式加入维护恢复机制后，模型对维护后短期恢复和寿命相关 rolling365 指标的刻画更合理。表10不替代表06，第二问正文主结果仍以表06为准。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
