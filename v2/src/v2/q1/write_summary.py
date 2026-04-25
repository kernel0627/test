from __future__ import annotations

import pandas as pd

from .paths import Q1Paths


def write_markdown_summary(
    paths: Q1Paths,
    overview: pd.DataFrame,
    season_decay: pd.DataFrame,
    rule: pd.DataFrame,
    decay_summary: pd.DataFrame,
    effect_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    seasonal_effect: pd.DataFrame,
    core: pd.DataFrame,
) -> None:
    n_devices = int(core["device_id"].nunique())
    n_valid_days = int(overview["n_valid_days"].sum())
    median_interval = rule["medium_interval_median"].dropna().median()
    medium_gain = effect_summary.loc[effect_summary["maintenance_type"] == "medium", "plateau_gain_median"]
    major_gain = effect_summary.loc[effect_summary["maintenance_type"] == "major", "plateau_gain_median"]
    regression = season_decay[season_decay["section"] == "regression"]
    time_decay = regression.loc[regression["term"] == "days_from_observation_start", "coef"]
    cycle_decay = regression.loc[regression["term"] == "days_since_last_maintenance", "coef"]
    monthly_decay_rows = int((season_decay["section"] == "monthly_decay_intensity").sum())
    unreliable_rho = int(
        (~recovery_summary["medium_rho_reliable"].astype(bool)).sum()
        + (~recovery_summary["major_rho_reliable"].astype(bool)).sum()
    )

    content = f"""# 第一问分析总结

## 1. 使用的数据是什么

本次读取附件 1 的 10 台设备小时级透水率数据，以及附件 2 的中维护和大维护记录。设备编号统一为 `a1` 到 `a10`，维护类型统一为 `medium` 和 `major`。

小维护没有具体记录，因此作为背景维护处理；第一问不从差分中重新识别维护点，维护事件以附件 2 的正式记录为准。

## 2. 数据做了什么处理

小时级数据转为日尺度，共得到 {n_devices} 台设备、{n_valid_days} 个有效分析日。主变量为日中位透水率 `daily_median`。

维护性缺失视为结构性缺失，不做普通插值。异常处理只剔除明显物理非法值和保守识别出的孤立异常，不删除真实低值。

## 3. 透水率有哪些变化规律

数据存在设备基础差异，因此用 `alpha_median` 描述观测期设备基础水平。月份季节水平项 `S_m` 由控制设备差异、维护周期和长期时间后的残差估计。

季节还可能影响衰减速度，因此从纯净衰减片段中计算了 {monthly_decay_rows} 行月度衰减强度。控制季节和维护周期后，`days_from_observation_start` 系数为 `{float(time_decay.iloc[0]) if len(time_decay) else float("nan"):.6f}`；`days_since_last_maintenance` 系数为 `{float(cycle_decay.iloc[0]) if len(cycle_decay) else float("nan"):.6f}`。

## 4. 当前固定维护规律是什么

每台设备分别提取当前维护触发规律。中维护用“间隔触发 + 近期下降速度触发”刻画，大维护用“长周期触发 + 多次中维护后触发”刻画。全设备中维护典型间隔中位数约为 `{float(median_interval) if pd.notna(median_interval) else float("nan"):.2f}` 天。

历史没有大维护的设备，第二问主方案中不主动安排大维护；增加大维护属于第三问优化内容。

## 5. 衰减指标是什么

维护周期衰减率表示两次中/大维护之间、日常小维护作为背景时的净衰减率。纯净衰减片段用于检验衰减趋势，避免只看周期首尾。

`表04_衰减指标汇总.csv` 同时给出周期净衰减、纯净片段衰减、早晚期衰减差异和月度衰减敏感性。

## 6. 维护事件影响是什么

维护后主要体现为第 `+1` 天即时跳升，`+1` 到 `+3` 天用于估计短期平台。维护效果同时用绝对提升和比例提升刻画，不能只凭绝对提升判断大维护一定强于中维护。

中维护平台提升中位数为 `{float(medium_gain.iloc[0]) if len(medium_gain) else float("nan"):.6f}`，大维护平台提升中位数为 `{float(major_gain.iloc[0]) if len(major_gain) else float("nan"):.6f}`。维护后 7 天用维持率刻画平台是否回落，不把短窗口斜率作为核心指标。

## 7. Hmax 和恢复比例是什么

`Hmax` 表示设备当前最大可恢复水平，并强制不低于当前状态 `current_state_level`。恢复比例 `rho` 表示维护恢复当前可恢复空间的比例；若某类 rho 大量被截断，说明该类 rho 不可靠，第二问需要使用全局 rho 或固定增益兜底。

本次共有 {unreliable_rho} 个设备-维护类型的 rho 可靠性标记为不可靠。

## 8. 季节如何影响维护效果

题目指出维护效果受季节影响，因此按春、夏、秋、冬统计维护恢复效率，并计算固定增益修正因子和恢复比例修正因子。样本不足或截断比例过高时，不硬估季节参数，第二问回退为修正因子 1。

## 9. 第一问如何回答题目

透水率变化受设备差异、季节水平项、月度衰减强度、长期时间趋势、维护周期衰减和维护事件恢复共同影响。中/大维护均具有恢复作用，但两者差异需要结合绝对提升、比例提升、恢复上限和维持率判断。

第一问已经提取出第二问寿命预测所需参数。

## 10. 第二问读取哪些表

主要读取：

- `表03_当前固定维护触发规律.csv`
- `表07_设备核心指标汇总.csv`

必要时读取：

- `表02_季节水平项与月度衰减强度.csv`
- `表06_Hmax与恢复比例质量评估.csv`
- `表08_季节对维护效应影响.csv`
"""
    paths.markdown_dir.joinpath("第一问分析总结.md").write_text(content, encoding="utf-8")
