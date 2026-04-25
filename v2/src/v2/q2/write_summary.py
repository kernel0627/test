from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import Q2Paths


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def _fmt(value: object, digits: int = 2, empty: str = "-") -> str:
    if value is None or pd.isna(value):
        return empty
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _date(value: object) -> str:
    if value is None or pd.isna(value) or str(value) in {"", "nan", "NaT"}:
        return "未触发"
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


def _list_devices(values: pd.Series) -> str:
    devices = sorted([str(v) for v in values.dropna().unique()], key=_device_sort_key)
    return "、".join(devices) if devices else "无"


def _metric_rows(backtest: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = []
    if backtest.empty:
        return rows
    summary = backtest[backtest["device_id"].astype(str) == "all"]
    if summary.empty:
        summary = backtest
    for _, row in summary.iterrows():
        rows.append(
            [
                row["device_id"],
                row["model_name"],
                _fmt(row["MAE"], 3),
                _fmt(row["RMSE"], 3),
                _fmt(row["post_maintenance_7d_MAE"], 3),
                _fmt(row["rolling365_MAE"], 3),
                int(row["n_validation_days"]),
                int(row["n_post_maintenance_7d_days"]),
                int(row["n_rolling365_days"]),
            ]
        )
    return rows


def _path_sample_rows(future_paths: pd.DataFrame) -> list[list[object]]:
    if future_paths.empty:
        return []
    main = future_paths[
        (future_paths["model_name"] == "recovery_ratio_main")
        & (future_paths["hmax_scenario"] == "neutral")
    ].copy()
    if main.empty:
        main = future_paths.copy()

    samples: list[pd.DataFrame] = []
    start_date = main["date"].min()
    start_rows = main[main["date"] == start_date].sort_values("device_id").head(3)
    if len(start_rows):
        samples.append(start_rows)

    maintenance_rows = main[main["is_maintenance_day"].astype(bool)].sort_values(["date", "device_id"]).head(4)
    if len(maintenance_rows):
        samples.append(maintenance_rows)

    risk_rows = main[main["rolling365_pred"].astype(float) < 37].sort_values(["date", "device_id"]).head(4)
    if len(risk_rows):
        samples.append(risk_rows)

    if not samples:
        samples = [main.sort_values(["date", "device_id"]).head(8)]
    sample = pd.concat(samples, ignore_index=True).drop_duplicates(
        subset=["model_name", "hmax_scenario", "device_id", "date"]
    ).head(10)
    rows: list[list[object]] = []
    for _, row in sample.iterrows():
        rows.append(
            [
                row["model_name"],
                row["hmax_scenario"],
                row["device_id"],
                _date(row["date"]),
                _fmt(row["predicted_permeability"], 2),
                _fmt(row["rolling365_pred"], 2),
                bool(row["is_maintenance_day"]),
                row["maintenance_type"] if pd.notna(row["maintenance_type"]) else "",
                _fmt(row["hmax_t"], 2),
                row["rho_used_source"] if pd.notna(row["rho_used_source"]) else "",
            ]
        )
    return rows


def write_q2_summary(
    paths: Q2Paths,
    params: pd.DataFrame,
    schedule: pd.DataFrame,
    backtest: pd.DataFrame,
    lifetime: pd.DataFrame,
    comparison: pd.DataFrame,
    future_paths: pd.DataFrame,
    extended_lifetime: pd.DataFrame,
) -> None:
    overview = _read_csv(paths.q1_tables_dir / "表01_数据质量与设备基础概览.csv")
    season = _read_csv(paths.q1_tables_dir / "表02_季节水平项与月度衰减强度.csv")
    rule = _read_csv(paths.q1_tables_dir / "表03_当前固定维护触发规律.csv")
    effect = _read_csv(paths.q1_tables_dir / "表05_维护事件效应汇总.csv")
    hmax = _read_csv(paths.q1_tables_dir / "表06_Hmax与恢复比例质量评估.csv")

    prediction_start = _date(lifetime["prediction_start_date"].dropna().iloc[0]) if len(lifetime) else "未生成"
    n_devices = int(params["device_id"].nunique()) if "device_id" in params else 0
    n_valid_days = int(pd.to_numeric(overview.get("n_valid_days", pd.Series(dtype=float)), errors="coerce").sum())
    n_low_days = int(pd.to_numeric(overview.get("n_days_below_37", pd.Series(dtype=float)), errors="coerce").sum())

    season_level = season[season.get("section", pd.Series(dtype=str)) == "seasonal_level"].copy()
    month_decay = season[season.get("section", pd.Series(dtype=str)) == "monthly_decay_intensity"].copy()
    regression = season[season.get("section", pd.Series(dtype=str)) == "regression"].copy()
    season_max = season_level.loc[pd.to_numeric(season_level["value"], errors="coerce").idxmax()] if len(season_level) else None
    season_min = season_level.loc[pd.to_numeric(season_level["value"], errors="coerce").idxmin()] if len(season_level) else None
    beta_tau = regression[regression.get("term", "") == "days_since_last_maintenance"]
    beta_time = regression[regression.get("term", "") == "days_from_observation_start"]

    n_medium = int(pd.to_numeric(rule.get("n_medium", pd.Series(dtype=float)), errors="coerce").sum())
    n_major = int(pd.to_numeric(rule.get("n_major", pd.Series(dtype=float)), errors="coerce").sum())
    global_major_ratio = n_major / (n_medium + n_major) if (n_medium + n_major) else float("nan")
    no_major_devices = _list_devices(rule.loc[rule.get("n_major", 0) == 0, "device_id"]) if len(rule) else "无"
    single_major_devices = _list_devices(rule.loc[rule.get("n_major", 0) == 1, "device_id"]) if len(rule) else "无"

    medium_effect = effect[effect.get("maintenance_type", "") == "medium"]
    major_effect = effect[effect.get("maintenance_type", "") == "major"]

    main_schedule = schedule[schedule["model_name"].isin(["fixed_gain_baseline", "recovery_ratio_main"])] if len(schedule) else schedule
    a4a8_main_major = len(
        main_schedule[
            main_schedule["device_id"].astype(str).isin(["a4", "a8"])
            & (main_schedule["maintenance_type"].astype(str) == "major")
        ]
    ) if len(main_schedule) else 0
    sensitivity_schedule = schedule[schedule["model_name"].astype(str).str.contains("major_sensitivity", na=False)] if len(schedule) else schedule
    a4a8_first = sensitivity_schedule[
        sensitivity_schedule["device_id"].astype(str).isin(["a4", "a8"])
        & (sensitivity_schedule["maintenance_type"].astype(str) == "major")
    ].groupby("device_id")["date"].min().reset_index() if len(sensitivity_schedule) else pd.DataFrame(columns=["device_id", "date"])

    main_lifetime = lifetime[lifetime["model_name"] == "recovery_ratio_main"].copy() if len(lifetime) else pd.DataFrame()
    ended_main = int((main_lifetime.get("status", pd.Series(dtype=str)) == "lifetime_end").sum()) if len(main_lifetime) else 0
    not_reached_main = _list_devices(main_lifetime.loc[main_lifetime.get("status", "") != "lifetime_end", "device_id"]) if len(main_lifetime) else "无"

    lifetime_rows: list[list[object]] = []
    if len(main_lifetime):
        for _, row in main_lifetime.sort_values("device_id", key=lambda s: s.map(_device_sort_key)).iterrows():
            lifetime_rows.append(
                [
                    row["device_id"],
                    row["hmax_scenario"],
                    _fmt(row["current_real_rolling365"], 2),
                    _date(row["lifetime_end_date"]),
                    _fmt(row["remaining_life_years"], 2, "未触发"),
                    _fmt(row["predicted_total_life_years"], 2, "未触发"),
                    row["status"],
                ]
            )

    comparison_rows: list[list[object]] = []
    if len(comparison):
        for _, row in comparison.sort_values("device_id", key=lambda s: s.map(_device_sort_key)).iterrows():
            comparison_rows.append(
                [
                    row["device_id"],
                    _date(row["fixed_gain_life_date"]),
                    _date(row["recovery_ratio_life_date"]),
                    _fmt(row["fixed_gain_MAE"], 3),
                    _fmt(row["recovery_ratio_MAE"], 3),
                    row["rho_used_source_summary"],
                    row["preferred_model"],
                    row["selection_reason"],
                ]
            )

    scenario_rows: list[list[object]] = []
    if len(lifetime):
        scenario_summary = lifetime.groupby(["model_name", "hmax_scenario"])["status"].apply(
            lambda s: int((s == "lifetime_end").sum())
        ).reset_index(name="n_lifetime_end")
        for _, row in scenario_summary.iterrows():
            scenario_rows.append([row["model_name"], row["hmax_scenario"], int(row["n_lifetime_end"])])

    extended_rows: list[list[object]] = []
    if len(extended_lifetime):
        extended_main = extended_lifetime[extended_lifetime["model_name"] == "recovery_ratio_main"].copy()
        for _, row in extended_main.sort_values("device_id", key=lambda s: s.map(_device_sort_key)).iterrows():
            extended_rows.append(
                [
                    row["device_id"],
                    row["hmax_scenario"],
                    _date(row["extended_lifetime_end_date"]),
                    _fmt(row["extended_remaining_life_years"], 2),
                    row["extended_status"],
                ]
            )

    hmax_rows: list[list[object]] = []
    if len(params):
        for _, row in params.sort_values("device_id", key=lambda s: s.map(_device_sort_key)).iterrows():
            hmax_rows.append(
                [
                    row["device_id"],
                    _fmt(row["hmax_trend_raw"], 6),
                    _fmt(row["hmax_trend_limited"], 6),
                    _fmt(row["hmax_trend_used"], 6),
                    _fmt(row["hmax_annual_drop_ratio_used"], 3),
                ]
            )

    archive_files = sorted((paths.q1_tables_dir / "archive").glob("*.csv"))
    archive_rows = [[path.name] for path in archive_files]
    path_sample_rows = _path_sample_rows(future_paths)
    table04_rows = len(future_paths)
    table04_devices = future_paths["device_id"].nunique() if "device_id" in future_paths else 0
    table04_models = future_paths["model_name"].nunique() if "model_name" in future_paths else 0

    content = f"""# 第二问分析总结

## 0. archive 是哪里来的

`v2/q_1/tables/archive/` 是本次修正 Q2 之前主动生成的旧表归档目录。它不是新的数据源，也不是第二问模型输入。

之前 `v2/q_1/tables/` 顶层同时存在新版表和旧版表，例如旧的 `表03_当前固定维护规律.csv`、旧的 `表06_Hmax与恢复比例汇总.csv`。这些旧表字段和现在的 Q2 口径不一致，如果继续放在顶层，后续代码或论文整理时很容易误读旧表。

所以 Q1 pipeline 重跑时会把旧版表从 `v2/q_1/tables/` 顶层移动到 `v2/q_1/tables/archive/`，只作为历史对照保留，不删除。Q2 只读取顶层新版 8 张表，绝不读取 archive 里的文件。

当前 archive 中的旧表为：

{_md_table(["归档旧表"], archive_rows)}

## 1. 当前完成状态

Q2 已经从第一问新版表中读取参数，并完成两类主模型：`fixed_gain_baseline` 和 `recovery_ratio_main`。主预测统一采用 Hmax 的 `neutral` 情景；`optimistic`、`pessimistic` 和全局大维护兜底方案只作为敏感性分析。

已读取的核心输入包括：

- `表03_当前固定维护触发规律.csv`
- `表07_设备核心指标汇总.csv`
- `表02_季节水平项与月度衰减强度.csv`
- `表06_Hmax与恢复比例质量评估.csv`
- `表08_季节对维护效应影响.csv`
- `日尺度特征表.csv`

数据规模：设备 {n_devices} 台，有效日记录 {n_valid_days} 条，历史低于 37 的日记录 {n_low_days} 条。

## 2. 第一问参数读数

季节水平项最高月份为 {int(season_max["month"]) if season_max is not None else "-"} 月，值约 {_fmt(season_max["value"] if season_max is not None else None, 2)}；最低月份为 {int(season_min["month"]) if season_min is not None else "-"} 月，值约 {_fmt(season_min["value"] if season_min is not None else None, 2)}。

解释型回归中，`days_since_last_maintenance` 系数为 {_fmt(beta_tau["coef"].iloc[0] if len(beta_tau) else None, 6)}，`days_from_observation_start` 系数为 {_fmt(beta_time["coef"].iloc[0] if len(beta_time) else None, 6)}。前者是维护周期内净衰减，后者是长期日历时间趋势。

维护记录中，中维护 {n_medium} 次，大维护 {n_major} 次，全局大维护比例约 {_fmt(global_major_ratio, 3)}。历史无大维护设备为 {no_major_devices}；只有一次大维护、主方案不直接外推设备级大维护周期的设备为 {single_major_devices}。

维护效应方面，中维护平台提升中位数为 {_fmt(medium_effect["plateau_gain_median"].iloc[0] if len(medium_effect) else None, 3)}，平台回升率中位数为 {_fmt(medium_effect["plateau_gain_ratio_median"].iloc[0] if len(medium_effect) else None, 3)}；大维护平台提升中位数为 {_fmt(major_effect["plateau_gain_median"].iloc[0] if len(major_effect) else None, 3)}，平台回升率中位数为 {_fmt(major_effect["plateau_gain_ratio_median"].iloc[0] if len(major_effect) else None, 3)}。

## 3. Hmax 情景与限幅

Hmax 趋势不再直接把两年样本的原始负斜率外推 30 年，而是先限制年度下降比例，再生成三情景：

- `optimistic`: Hmax 不下降。
- `neutral`: 使用限幅后斜率的一半，作为第二问主结果。
- `pessimistic`: 使用限幅后斜率，作为敏感性分析。

限幅规则为 `hmax_annual_drop_ratio <= 0.20`。主参数表中的 neutral 情景如下：

{_md_table(["设备", "原始斜率", "限幅斜率", "主用斜率", "主用年下降比例"], hmax_rows)}

## 4. 回测口径

回测阶段使用验证期真实中/大维护日程，目的是检验两种状态转移机制，而不是检验维护触发规则。未来预测阶段才使用当前维护触发规则生成维护日程。

验证期为每台设备最后 180 个有效日。`post_maintenance_7d_MAE` 对维护后 +1 到 +7 天日期去重后统计；`rolling365_MAE` 使用“验证期前 364 天真实 daily_median + 验证期预测值”的拼接序列计算预测 rolling365，再和真实 rolling365 对齐比较。

{_md_table(["设备", "模型", "MAE", "RMSE", "维护后7日MAE", "rolling365_MAE", "验证日数", "维护后7日数", "rolling365日数"], _metric_rows(backtest))}

## 5. 大维护触发修正

未来预测中，大维护优先级仍高于中维护。新的触发规则修正了三个边界：

- `medium_count_between_major_median <= 0` 时关闭“中维护次数触发大维护”，不再强行转成 1。
- `n_major == 1` 的设备主方案不直接使用设备级大维护间隔外推，敏感性方案才使用全局 fallback。
- `n_major == 0` 的设备主方案不安排大维护。

主方案中 A4/A8 的大维护事件数为 {a4a8_main_major}。全局大维护兜底敏感性方案中，A4/A8 第一次大维护如下：

{_md_table(["设备", "敏感性方案第一次大维护"], a4a8_first.values.tolist())}

## 6. 表04：未来模拟路径表

`表04_未来模拟路径表.csv` 现在恢复为完整单表输出，路径为 `v2/q_2/tables/表04_未来模拟路径表.csv`。由于该表体积较大，已经写入 `.gitignore`，用于本地复现和检查，不建议提交进 Git。

表04保存的是第二问逐日模拟明细，共 {table04_rows} 行，覆盖 {table04_devices} 台设备和 {table04_models} 个模型/情景组合。主要字段含义如下：

- `model_name`：模型名称，包括固定增益、恢复比例、Hmax 乐观/悲观敏感性、大维护兜底敏感性。
- `hmax_scenario`：Hmax 情景，主结果为 `neutral`。
- `device_id`、`date`：设备和模拟日期。
- `x_state`：扣除季节水平项后的设备状态。
- `seasonal_level`：当月季节水平项。
- `predicted_permeability`：预测透水率，即 `x_state + seasonal_level`。
- `rolling365_pred`：预测路径上的滚动 365 天平均透水率。
- `is_maintenance_day`、`maintenance_type`：当天是否触发维护以及维护类型。
- `rho_used_source`：恢复比例模型当天维护使用的 rho 来源，或固定增益 fallback 标记。
- `hmax_t`、`hmax_trend_used`、`hmax_annual_drop_ratio_used`：该日 Hmax 水平和情景斜率信息。

下面只抽取少量代表性行展示表04内容，完整逐日路径请直接查看 CSV：

{_md_table(["模型", "Hmax情景", "设备", "日期", "预测透水率", "rolling365", "维护日", "维护类型", "Hmax", "rho来源"], path_sample_rows)}

## 7. 寿命预测主结果

寿命判定仍使用滚动 365 天平均透水率低于 37，并继续按当前触发规则模拟一年；若一年内最高 rolling365 仍低于 37，则判定寿命终止。

预测起点为 {prediction_start}。主模型 `recovery_ratio_main` 的 neutral 情景中，{ended_main} 台设备在 30 年预测期内触发寿命终止；未触发设备为 {not_reached_main}。

{_md_table(["设备", "Hmax情景", "当前真实rolling365", "寿命终止日", "剩余寿命年", "总寿命年", "状态"], lifetime_rows)}

## 8. 情景敏感性

以下表格统计每个模型和 Hmax 情景下触发生命终止的设备数。主结论只使用 `neutral` 情景；`optimistic` 和 `pessimistic` 用于说明 Hmax 外推假设对结果的影响。

{_md_table(["模型", "Hmax情景", "触发终止设备数"], scenario_rows)}

`表07_长期外推参考表.csv` 额外给出 100 年 extended horizon 的远期外推参考。该表只用于附录或敏感性讨论，不参与 `preferred_model` 判断，也不改变表05的 30 年主预测结论。

{_md_table(["设备", "Hmax情景", "100年参考寿命日", "100年参考剩余寿命", "extended_status"], extended_rows)}

## 9. 模型比较与推荐

`preferred_model` 不再只看 rho 可靠性标记，而是同时看实际使用的 rho 来源和设备级回测 MAE。若恢复比例模型 MAE 超过固定增益模型 5%，推荐固定增益；若恢复比例模型内部使用固定增益 fallback，则推荐名写为 `recovery_ratio_with_fixed_gain_fallback`。

{_md_table(["设备", "固定增益寿命日", "恢复比例寿命日", "固定增益MAE", "恢复比例MAE", "rho来源", "推荐模型", "选择原因"], comparison_rows)}

## 10. 当前结论

这一版 Q2 已经修正了四个关键问题：回测是真比较，Hmax 是限幅后三情景，大维护触发不再把 K=0 误转成 1，模型推荐逻辑也和 fallback 行为一致。

第二问仍然不是维护策略优化。A4/A8 是否应加入大维护、维护频率是否要改变，应放到第三问优化模型中处理。
"""
    paths.q2_markdown_dir.joinpath("第二问分析总结.md").write_text(content, encoding="utf-8")
