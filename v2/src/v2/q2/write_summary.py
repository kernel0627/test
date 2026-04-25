from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import Q2Paths


def _fmt(value: object, digits: int = 2, empty: str = "-") -> str:
    if value is None or pd.isna(value):
        return empty
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _date(value: object) -> str:
    if value is None or pd.isna(value) or str(value) in {"", "nan", "NaT"}:
        return "-"
    return str(value)[:10]


def _device_sort_key(device_id: object) -> int:
    text = str(device_id).lower().replace("a", "")
    return int(text) if text.isdigit() else 999


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        rows = [["-" for _ in headers]]
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _sort_devices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "device_id" not in df.columns:
        return df
    return df.sort_values("device_id", key=lambda s: s.map(_device_sort_key))


def _table_path(paths: Q2Paths, filename: str) -> str:
    return f"`v2/q_2/tables/{filename}`"


def _backtest_rows(backtest: pd.DataFrame) -> list[list[object]]:
    if backtest.empty:
        return []
    summary = backtest[backtest["device_id"].astype(str) == "all"]
    if summary.empty:
        summary = backtest
    rows: list[list[object]] = []
    for _, row in summary.iterrows():
        rows.append(
            [
                row["model_name"],
                _fmt(row.get("MAE"), 3),
                _fmt(row.get("RMSE"), 3),
                _fmt(row.get("post_maintenance_7d_MAE"), 3),
                _fmt(row.get("rolling365_MAE"), 3),
            ]
        )
    return rows


def _comparison_rows(comparison: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = []
    for _, row in _sort_devices(comparison).iterrows():
        rows.append(
            [
                row["device_id"],
                _date(row.get("preferred_life_date")),
                _fmt(row.get("preferred_remaining_years"), 2),
                row.get("preferred_model", "-"),
                _fmt(row.get("fixed_gain_MAE"), 3),
                _fmt(row.get("recovery_ratio_MAE"), 3),
                row.get("selection_reason", "-"),
            ]
        )
    return rows


def _scenario_rows(lifetime: pd.DataFrame) -> list[list[object]]:
    if lifetime.empty:
        return []
    grouped = (
        lifetime.groupby(["model_name", "hmax_scenario"], dropna=False)
        .agg(n_devices=("device_id", "nunique"), n_lifetime_end=("status", lambda s: int((s == "lifetime_end").sum())))
        .reset_index()
    )
    rows: list[list[object]] = []
    for _, row in grouped.iterrows():
        rows.append([row["model_name"], row["hmax_scenario"], int(row["n_devices"]), int(row["n_lifetime_end"])])
    return rows


def _table_status(paths: Q2Paths) -> str:
    names = [
        "表01_模型参数表.csv",
        "表02_未来维护日程表.csv",
        "表03_历史回测指标表.csv",
        "表04_未来模拟路径表.csv",
        "表05_寿命预测结果表.csv",
        "表06_模型对比汇总表.csv",
        "表07_长期外推参考表.csv",
        "表08_寿命过长诊断表.csv",
        "表09_Hmax工程保守敏感性表.csv",
    ]
    rows = [[name, "存在" if (paths.q2_tables_dir / name).exists() else "缺失"] for name in names]
    return _md_table(["正式表", "状态"], rows)


def write_q2_summary(
    paths: Q2Paths,
    params: pd.DataFrame,
    schedule: pd.DataFrame,
    backtest: pd.DataFrame,
    lifetime: pd.DataFrame,
    comparison: pd.DataFrame,
    future_paths: pd.DataFrame,
    extended_lifetime: pd.DataFrame,
    diagnostics: pd.DataFrame,
    engineering_lifetime: pd.DataFrame,
) -> None:
    n_devices = int(params["device_id"].nunique()) if "device_id" in params else 0
    prediction_start = _date(lifetime["prediction_start_date"].dropna().iloc[0]) if len(lifetime) else "-"
    n_future_rows = len(future_paths)
    n_schedule_rows = len(schedule)

    content = f"""# 第二问分析总结

## 1. 第二问解决什么问题

第二问解决的是在当前维护触发规律和现有维护效果口径下，对设备未来功能寿命进行预测。这里的寿命是功能寿命，主要依据预测透水率及 rolling365 判据是否触发失效。

本问不做维护策略优化，不新增经济寿命硬终止，也不把维护负担作为主寿命硬终止条件。维护频率、维护负担、是否提前更换等策略问题放到第三问处理。

## 2. 使用了哪些第一问结果

Q2 从第一问读取当前维护触发规律、设备核心指标、季节项、长期衰减项、Hmax 与恢复比例质量评估等结果，并据此生成参数、未来维护日程、未来模拟路径和寿命预测表。

当前预测起点为 {prediction_start}，涉及设备 {n_devices} 台。未来维护日程共 {n_schedule_rows} 行，未来模拟路径共 {n_future_rows} 行。

{_table_status(paths)}

## 3. 两个模型是什么

`fixed_gain_baseline` 是基准模型：维护日按固定平台增益修复设备状态，用于提供稳定、可解释的对照结果。

`recovery_ratio_main` 是主模型：维护日按恢复比例机制修复设备状态，优先使用设备级或全局恢复比例；当恢复比例来源不足时，内部可以使用 fixed gain fallback。

两者的核心差异只在维护日恢复机制。第二问本轮不修改衰减机制、Hmax 主情景、维护日程生成规则或寿命终止机制。

## 4. 回测结果

历史回测结果来自 {_table_path(paths, "表03_历史回测指标表.csv")}。回测指标包括 MAE、RMSE、`post_maintenance_7d_MAE` 和 `rolling365_MAE`，分别衡量整体误差、维护后短期误差和 rolling365 判据相关误差。

模型选择主要依据 neutral 主情景下的设备级回测 MAE，并结合恢复比例来源是否可靠判断。

{_md_table(["模型", "MAE", "RMSE", "post_maintenance_7d_MAE", "rolling365_MAE"], _backtest_rows(backtest))}

## 5. 寿命预测主结果

第二问正文主结果使用 {_table_path(paths, "表06_模型对比汇总表.csv")}。该表是“第二问主寿命预测结果与模型选择汇总表”，在 neutral 主情景下结合历史回测误差选择推荐模型。

`preferred_model` 规则如下：

- 若 `recovery_ratio_main` 的 MAE 不比 `fixed_gain_baseline` 差超过 5%，且恢复比例来源可靠，则推荐 `recovery_ratio_main`。
- 若 `recovery_ratio_main` 内部使用 fixed gain fallback，但 MAE 不比 baseline 差超过 5%，则推荐 `recovery_ratio_with_fixed_gain_fallback`。
- 若 `recovery_ratio_main` 的 MAE 比 `fixed_gain_baseline` 差超过 5%，则推荐 `fixed_gain_baseline`。

表06中的 `preferred_life_date` 和 `preferred_remaining_years` 是正文引用的主寿命日期和剩余寿命年限。`selection_reason` 与 `note` 必须和 `preferred_model` 保持一致。

{_md_table(["设备", "推荐寿命日期", "推荐剩余年限", "推荐模型", "固定增益MAE", "恢复比例MAE", "选择原因"], _comparison_rows(comparison))}

## 6. 完整情景与敏感性分析

{_table_path(paths, "表05_寿命预测结果表.csv")} 给出了不同模型与 Hmax 情景下的完整寿命预测结果，包括 `fixed_gain_baseline`、`recovery_ratio_main` 以及 optimistic、neutral、pessimistic 等 Hmax 情景。表05是完整情景结果，不把所有行混作第二问正文主寿命结果。

{_table_path(paths, "表07_长期外推参考表.csv")} 只作为超过 30 年预测窗口后的长期外推参考，不参与 `preferred_model` 判断，不作为第二问主结论。

{_table_path(paths, "表08_寿命过长诊断表.csv")} 用于解释某些设备为什么在功能寿命判据下预测较长，辅助判断 Hmax、衰减率、维护频率或 rolling365 判据的影响，不直接改变主寿命结果。

{_table_path(paths, "表09_Hmax工程保守敏感性表.csv")} 只作为 Hmax 更保守下降假设下的敏感性分析，不替代 neutral 主情景，不参与 `preferred_model` 判断。

表05情景覆盖情况如下：

{_md_table(["模型", "Hmax情景", "设备数", "触发寿命终止数"], _scenario_rows(lifetime))}

## 7. 结果解释

如果某设备在表05中出现 `>30 年` 或 `not_reached_within_30y`，并不表示其精确寿命就是几十年，而是表示在当前功能寿命判据和 30 年预测窗口内没有触发失效。

正文引用时以表06推荐模型结果为主口径。表05、表07、表08、表09分别用于补充展示完整情景、长期外推、原因诊断和 Hmax 敏感性。

## 8. 与第三问的衔接

维护负担、维护频率、提前更换时点和维护策略优化不在第二问中作为硬判寿命条件。第二问只给出当前维护规律下的功能寿命预测；第三问再把维护负担和策略成本收益纳入优化目标。

第二问最终采用表06作为主结果表。表05保留全部模型与 Hmax 情景下的寿命预测结果，表07提供长期外推参考，表08用于解释寿命较长设备的原因，表09用于展示 Hmax 更保守假设下的敏感性。由于第二问目标是当前维护规律下的功能寿命预测，维护负担不作为主寿命硬终止，而作为第三问维护策略优化的重要依据。
"""
    out_path: Path = paths.q2_markdown_dir / "第二问分析总结.md"
    out_path.write_text(content, encoding="utf-8")
