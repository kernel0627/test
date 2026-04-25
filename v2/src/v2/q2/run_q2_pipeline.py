from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from .backtest_models import build_backtest_metrics
from .build_parameters import build_model_parameters
from .lifetime_prediction import build_extended_lifetime_predictions, build_lifetime_predictions
from .load_q1_outputs import load_q1_outputs
from .paths import ensure_output_dirs, get_paths
from .simple_ablation import build_simple_ablation, write_ablation_markdown
from .simulate_models import (
    MODEL_FIXED,
    MODEL_FIXED_HMAX_ENG,
    MODEL_MAIN,
    MODEL_MAIN_HMAX_ENG,
    simulate_future_paths,
)
from .write_summary import write_q2_summary


DATE_COLUMNS = {
    "date",
    "commission_date",
    "prediction_start_date",
    "first_date_rolling365_below_37",
    "lifetime_end_date",
    "functional_failure_date",
    "recovery_failure_date",
    "maintenance_burden_failure_date",
    "maintenance_burden_warning_date",
    "burden_trigger_date",
    "service_cap_date",
    "final_lifetime_end_date",
    "extended_first_date_rolling365_below_37",
    "extended_lifetime_end_date",
    "extended_functional_failure_date",
    "extended_recovery_failure_date",
    "extended_maintenance_burden_failure_date",
    "extended_maintenance_burden_warning_date",
    "extended_service_cap_date",
    "extended_final_lifetime_end_date",
    "preferred_life_date",
}


def _format_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in DATE_COLUMNS.intersection(out.columns):
        original = out[column].copy()
        text = original.astype("string")
        keep_text = text.str.startswith(">", na=False)
        converted = pd.to_datetime(original.where(~keep_text), errors="coerce")
        formatted = converted.dt.strftime("%Y-%m-%d")
        out[column] = original.where(converted.isna(), formatted).fillna("")
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _format_for_csv(df).to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_new_{timestamp}{path.suffix}")
        _format_for_csv(df).to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"Target CSV is locked, wrote fallback file instead: {fallback.name.encode('unicode_escape').decode('ascii')}")


def _remove_stale_csv(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.unlink()
    except PermissionError:
        print(f"Stale CSV is locked and could not be removed: {path.name.encode('unicode_escape').decode('ascii')}")


def _write_table04_full(df: pd.DataFrame, tables_dir: Path) -> None:
    _write_csv(df, tables_dir / "表04_未来模拟路径表.csv")
    _remove_stale_csv(tables_dir / "表04_未来模拟路径表_part1.csv")
    _remove_stale_csv(tables_dir / "表04_未来模拟路径表_part2.csv")


def _rho_source_summary(param: pd.Series, schedule: pd.DataFrame, device_id: str) -> str:
    device_events = schedule[
        (schedule["model_name"] == "recovery_ratio_main")
        & (schedule["device_id"] == device_id)
        & (schedule["hmax_scenario"] == "neutral")
    ]
    maintenance_types = sorted(set(device_events["maintenance_type"].dropna().astype(str)))
    if not maintenance_types:
        maintenance_types = ["medium"]
    parts = []
    for maintenance_type in maintenance_types:
        if maintenance_type not in {"medium", "major"}:
            continue
        parts.append(f"{maintenance_type}:{param.get(f'{maintenance_type}_rho_used_source', 'fixed_gain_fallback')}")
    return ";".join(parts)


def _device_backtest_mae(backtest: pd.DataFrame, device_id: str, model_name: str) -> float:
    row = backtest[(backtest["device_id"] == device_id) & (backtest["model_name"] == model_name)]
    if row.empty:
        return np.nan
    return float(row["MAE"].iloc[0])


def _numeric_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _build_comparison(
    lifetime: pd.DataFrame,
    params: pd.DataFrame,
    backtest: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device_id in sorted(lifetime["device_id"].unique(), key=lambda x: int(str(x).replace("a", ""))):
        fixed = lifetime[(lifetime["device_id"] == device_id) & (lifetime["model_name"] == "fixed_gain_baseline")]
        main = lifetime[(lifetime["device_id"] == device_id) & (lifetime["model_name"] == "recovery_ratio_main")]
        param = params[params["device_id"] == device_id].iloc[0]
        fixed_date = fixed["lifetime_end_date"].iloc[0] if len(fixed) else pd.NaT
        main_date = main["lifetime_end_date"].iloc[0] if len(main) else pd.NaT
        fixed_years = fixed["remaining_life_years"].iloc[0] if len(fixed) else np.nan
        main_years = main["remaining_life_years"].iloc[0] if len(main) else np.nan
        fixed_years_num = _numeric_or_nan(fixed_years)
        main_years_num = _numeric_or_nan(main_years)
        if np.isfinite(fixed_years_num) and np.isfinite(main_years_num):
            diff = main_years_num - fixed_years_num
        else:
            diff = np.nan
        fixed_mae = _device_backtest_mae(backtest, device_id, "fixed_gain_baseline")
        main_mae = _device_backtest_mae(backtest, device_id, "recovery_ratio_main")
        source_summary = _rho_source_summary(param, schedule, device_id)
        uses_fixed_fallback = "fixed_gain_fallback" in source_summary
        if np.isfinite(fixed_mae) and np.isfinite(main_mae) and main_mae > fixed_mae * 1.05:
            preferred = "fixed_gain_baseline"
            reason = "恢复比例模型回测 MAE 超过固定增益 5%，采用固定增益。"
        elif uses_fixed_fallback:
            preferred = "recovery_ratio_with_fixed_gain_fallback"
            reason = "恢复比例模型部分维护类型使用固定增益兜底，且回测未显著差于固定增益。"
        else:
            preferred = "recovery_ratio_main"
            reason = "恢复比例模型使用 device/global rho，且回测未显著差于固定增益。"
        if preferred == "fixed_gain_baseline":
            preferred_date = fixed_date
            preferred_years = fixed_years
        else:
            preferred_date = main_date
            preferred_years = main_years
        note = reason
        rows.append(
            {
                "device_id": device_id,
                "fixed_gain_life_date": fixed_date,
                "recovery_ratio_life_date": main_date,
                "fixed_gain_remaining_years": fixed_years,
                "recovery_ratio_remaining_years": main_years,
                "difference_years": diff,
                "fixed_gain_MAE": fixed_mae,
                "recovery_ratio_MAE": main_mae,
                "rho_reliable_flag": bool(param["rho_reliable_flag"]),
                "rho_used_source_summary": source_summary,
                "preferred_model": preferred,
                "preferred_life_date": preferred_date,
                "preferred_remaining_years": preferred_years,
                "selection_reason": reason,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def _maintenance_burden(intervals: pd.Series, count: int) -> tuple[float, float, float, bool]:
    annual_count = count / 30.0
    avg_interval = float(intervals.mean()) if len(intervals) else np.nan
    min_interval = float(intervals.min()) if len(intervals) else np.nan
    flag = bool(
        annual_count > 8
        or (np.isfinite(avg_interval) and avg_interval < 45)
        or (np.isfinite(min_interval) and min_interval < 30)
    )
    return annual_count, avg_interval, min_interval, flag


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _burden_warning_values(life_row: pd.DataFrame) -> dict[str, object]:
    if life_row.empty or "maintenance_burden_flag" not in life_row.columns:
        return {
            "flag": False,
            "date": pd.NaT,
            "reason": "",
            "annual_count": np.nan,
            "avg_interval": np.nan,
            "min_interval": np.nan,
        }
    row = life_row.iloc[0]
    return {
        "flag": _truthy(row.get("maintenance_burden_flag", False)),
        "date": row.get("maintenance_burden_warning_date", pd.NaT),
        "reason": row.get("maintenance_burden_reason", ""),
        "annual_count": row.get("burden_trigger_annual_count", np.nan),
        "avg_interval": row.get("burden_trigger_avg_interval", np.nan),
        "min_interval": row.get("burden_trigger_min_interval", np.nan),
    }


def _long_life_reason(
    status: str,
    current_rolling: float,
    cycle_decay: float,
    annual_maintenance: float,
    hmax_trend: float,
    rolling_last: float,
    maintenance_burden_flag: bool,
) -> str:
    if status == "lifetime_end":
        return "30年内已触发寿命终止"
    reasons: list[str] = []
    if maintenance_burden_flag:
        reasons.append("维护负担已触发预警，需要第三问优化")
    if np.isfinite(current_rolling) and current_rolling >= 80:
        reasons.append("当前rolling365水平较高")
    if np.isfinite(cycle_decay) and cycle_decay > -0.2:
        reasons.append("周期净衰减较慢")
    if np.isfinite(annual_maintenance) and annual_maintenance >= 5:
        reasons.append("当前触发规则下维护频率较高")
    if np.isfinite(hmax_trend) and hmax_trend > -0.01:
        reasons.append("Hmax下降趋势较缓")
    if np.isfinite(rolling_last) and rolling_last >= 37:
        reasons.append("30年末rolling365仍高于阈值")
    return "；".join(reasons) if reasons else "30年内未触发寿命终止，综合状态与维护规则仍可维持阈值以上"


def _build_long_life_diagnostics(
    future_paths: pd.DataFrame,
    schedule: pd.DataFrame,
    params: pd.DataFrame,
    lifetime: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    path_main = future_paths[
        (future_paths["model_name"] == MODEL_MAIN)
        & (future_paths["hmax_scenario"] == "neutral")
    ].copy()
    sched_main = schedule[
        (schedule["model_name"] == MODEL_MAIN)
        & (schedule["hmax_scenario"] == "neutral")
    ].copy()
    for _, param in params.sort_values("device_id", key=lambda s: s.map(lambda x: int(str(x).replace("a", "")))).iterrows():
        device_id = str(param["device_id"])
        device_path = path_main[path_main["device_id"] == device_id].sort_values("date").head(30 * 365)
        device_sched = sched_main[sched_main["device_id"] == device_id].sort_values("date").copy()
        device_sched_30y = device_sched[
            device_sched["date"].isin(set(device_path["date"]))
        ].sort_values("date")
        event_count = int(len(device_sched_30y))
        intervals = pd.to_datetime(device_sched_30y["date"]).diff().dt.days.dropna().astype(float)
        annual_count, avg_interval, min_interval, burden_flag = _maintenance_burden(intervals, event_count)
        current_rolling = float(device_path["rolling365_pred"].iloc[0]) if len(device_path) else np.nan
        rolling_min = float(device_path["rolling365_pred"].min()) if len(device_path) else np.nan
        rolling_last = float(device_path["rolling365_pred"].iloc[-1]) if len(device_path) else np.nan
        hmax_at_10y = float(param["h_max_initial"]) + float(param["hmax_trend_used"]) * 3650
        hmax_at_30y = float(param["h_max_initial"]) + float(param["hmax_trend_used"]) * (30 * 365 - 1)
        life_row = lifetime[(lifetime["device_id"] == device_id) & (lifetime["model_name"] == MODEL_MAIN)]
        status = str(life_row["status"].iloc[0]) if len(life_row) else ""
        warning = _burden_warning_values(life_row)
        warning_flag = bool(warning["flag"])
        rows.append(
            {
                "device_id": device_id,
                "current_state_level": param["current_state_level"],
                "current_real_rolling365": life_row["current_real_rolling365"].iloc[0] if len(life_row) else np.nan,
                "cycle_decay_rate_used": param["cycle_decay_rate_used"],
                "maintenance_interval_median": param["maintenance_interval_median"],
                "future_maintenance_count_30y": event_count,
                "annual_maintenance_count_30y": annual_count,
                "avg_maintenance_interval_30y": avg_interval,
                "min_maintenance_interval_30y": min_interval,
                "maintenance_burden_flag": warning_flag,
                "burden_trigger_date": warning["date"],
                "burden_trigger_reason": warning["reason"],
                "burden_trigger_annual_count": warning["annual_count"],
                "burden_trigger_avg_interval": warning["avg_interval"],
                "burden_trigger_min_interval": warning["min_interval"],
                "h_max_initial": param["h_max_initial"],
                "hmax_trend_used": param["hmax_trend_used"],
                "hmax_at_10y": max(0.0, hmax_at_10y),
                "hmax_at_30y": max(0.0, hmax_at_30y),
                "rolling365_min_30y": rolling_min,
                "rolling365_last_30y": rolling_last,
                "long_life_reason": _long_life_reason(
                    status,
                    current_rolling,
                    float(param["cycle_decay_rate_used"]),
                    annual_count,
                    float(param["hmax_trend_used"]),
                    rolling_last,
                    warning_flag,
                ),
            }
        )
    return pd.DataFrame(rows)


def _assert_outputs(paths, outputs: dict[str, pd.DataFrame]) -> None:
    expected = [
        paths.q2_tables_dir / "表01_模型参数表.csv",
        paths.q2_tables_dir / "表02_未来维护日程表.csv",
        paths.q2_tables_dir / "表03_历史回测指标表.csv",
        paths.q2_tables_dir / "表04_未来模拟路径表.csv",
        paths.q2_tables_dir / "表05_寿命预测结果表.csv",
        paths.q2_tables_dir / "表06_模型对比汇总表.csv",
        paths.q2_tables_dir / "表07_长期外推参考表.csv",
        paths.q2_tables_dir / "表08_寿命过长诊断表.csv",
        paths.q2_tables_dir / "表09_Hmax工程保守敏感性表.csv",
        paths.q2_markdown_dir / "第二问分析总结.md",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise AssertionError("Missing q2 outputs: " + ", ".join(missing))
    schedule = outputs["schedule"]
    main_schedule = schedule[~schedule["model_name"].astype(str).str.contains("sensitivity", na=False)]
    if main_schedule.empty or main_schedule["device_id"].nunique() != 10:
        raise AssertionError("Q2 schedule must contain trigger-generated events for each device.")
    lifetime = outputs["lifetime"]
    if lifetime["current_real_rolling365"].isna().all():
        raise AssertionError("Q2 lifetime table current_real_rolling365 is empty.")
    for pattern in ["*.png", "*.html"]:
        found = list(paths.q2_dir.rglob(pattern))
        if found:
            raise AssertionError(f"Unexpected q2 visual outputs found: {found[:3]}")
    if (paths.q2_dir / "figures").exists():
        raise AssertionError("Unexpected figures directory under v2/q_2.")


def main() -> None:
    paths = get_paths()
    ensure_output_dirs(paths)
    q1 = load_q1_outputs(paths)
    params, helpers = build_model_parameters(q1)
    future_paths, schedule = simulate_future_paths(params, q1["daily"], helpers)
    extended_specs = [
        (MODEL_FIXED, "fixed", "neutral", False),
        (MODEL_MAIN, "main", "neutral", False),
    ]
    extended_paths, _ = simulate_future_paths(
        params,
        q1["daily"],
        helpers,
        horizon_years=100,
        include_lifetime_test_extra=True,
        model_specs=extended_specs,
    )
    backtest = build_backtest_metrics(q1["daily"], params, helpers)
    lifetime = build_lifetime_predictions(future_paths, params, q1["daily"], helpers["prediction_start"])
    extended_lifetime = build_extended_lifetime_predictions(
        extended_paths,
        params,
        q1["daily"],
        helpers["prediction_start"],
        extended_horizon_years=100,
    )
    comparison = _build_comparison(lifetime, params, backtest, schedule)
    ablation = build_simple_ablation(backtest, comparison)
    diagnostics = _build_long_life_diagnostics(future_paths, schedule, params, lifetime)
    engineering_specs = [
        (MODEL_FIXED_HMAX_ENG, "fixed", "engineering_conservative", False),
        (MODEL_MAIN_HMAX_ENG, "main", "engineering_conservative", False),
    ]
    engineering_paths, _ = simulate_future_paths(
        params,
        q1["daily"],
        helpers,
        horizon_years=30,
        include_lifetime_test_extra=True,
        model_specs=engineering_specs,
    )
    engineering_lifetime = build_lifetime_predictions(
        engineering_paths,
        params,
        q1["daily"],
        helpers["prediction_start"],
        horizon_years=30,
    )

    _write_csv(params[[
        "device_id",
        "current_state_level",
        "initial_x_state",
        "maintenance_interval_median",
        "major_ratio",
        "cycle_decay_rate_used",
        "h_max_initial",
        "hmax_trend_raw",
        "hmax_trend_limited",
        "hmax_trend_used",
        "hmax_annual_drop_ratio_used",
        "hmax_damage_delta_medium",
        "hmax_damage_delta_major",
        "hmax_damage_source",
        "service_cap_scenario",
        "service_cap_years",
        "hmax_scenario",
        "hmax_main_scenario",
        "medium_plateau_gain_used",
        "major_plateau_gain_used",
        "medium_recovery_ratio_used",
        "major_recovery_ratio_used",
        "medium_rho_used_source",
        "major_rho_used_source",
        "rho_reliable_flag",
    ]], paths.q2_tables_dir / "表01_模型参数表.csv")
    schedule_columns = [
        "model_name",
        "hmax_scenario",
        "device_id",
        "future_event_index",
        "date",
        "maintenance_type",
        "rule_type",
        "rho_used_source",
        "service_cap_scenario",
        "year",
    ]
    _write_csv(schedule[schedule_columns], paths.q2_tables_dir / "表02_未来维护日程表.csv")
    _write_csv(backtest, paths.q2_tables_dir / "表03_历史回测指标表.csv")
    _write_table04_full(future_paths, paths.q2_tables_dir)
    _write_csv(lifetime, paths.q2_tables_dir / "表05_寿命预测结果表.csv")
    _write_csv(comparison, paths.q2_tables_dir / "表06_模型对比汇总表.csv")
    _write_csv(extended_lifetime, paths.q2_tables_dir / "表07_长期外推参考表.csv")
    _write_csv(diagnostics, paths.q2_tables_dir / "表08_寿命过长诊断表.csv")
    _write_csv(engineering_lifetime, paths.q2_tables_dir / "表09_Hmax工程保守敏感性表.csv")
    _write_csv(ablation, paths.q2_tables_dir / "表10_简化消融实验表.csv")
    write_q2_summary(
        paths,
        params,
        schedule,
        backtest,
        lifetime,
        comparison,
        future_paths,
        extended_lifetime,
        diagnostics,
        engineering_lifetime,
    )
    write_ablation_markdown(paths.q2_markdown_dir / "第二问检验与消融说明.md")
    _assert_outputs(paths, {"schedule": schedule, "lifetime": lifetime})
    print("v2 q2 pipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"v2 q2 pipeline failed: {exc}", file=sys.stderr)
        raise
