from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from .backtest_models import build_backtest_metrics
from .build_parameters import build_model_parameters
from .lifetime_prediction import build_lifetime_predictions
from .load_q1_outputs import load_q1_outputs
from .paths import ensure_output_dirs, get_paths
from .simulate_models import simulate_future_paths
from .write_summary import write_q2_summary


DATE_COLUMNS = {
    "date",
    "commission_date",
    "prediction_start_date",
    "first_date_rolling365_below_37",
    "lifetime_end_date",
}


def _format_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in DATE_COLUMNS.intersection(out.columns):
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
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
        if pd.notna(fixed_years) and pd.notna(main_years):
            diff = float(main_years) - float(fixed_years)
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
                "selection_reason": reason,
                "note": note,
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
    backtest = build_backtest_metrics(q1["daily"], params, helpers)
    lifetime = build_lifetime_predictions(future_paths, params, q1["daily"], helpers["prediction_start"])
    comparison = _build_comparison(lifetime, params, backtest, schedule)

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
        "year",
    ]
    _write_csv(schedule[schedule_columns], paths.q2_tables_dir / "表02_未来维护日程表.csv")
    _write_csv(backtest, paths.q2_tables_dir / "表03_历史回测指标表.csv")
    _write_table04_full(future_paths, paths.q2_tables_dir)
    _write_csv(lifetime, paths.q2_tables_dir / "表05_寿命预测结果表.csv")
    _write_csv(comparison, paths.q2_tables_dir / "表06_模型对比汇总表.csv")
    write_q2_summary(paths, params, schedule, backtest, lifetime, comparison, future_paths)
    _assert_outputs(paths, {"schedule": schedule, "lifetime": lifetime})
    print("v2 q2 pipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"v2 q2 pipeline failed: {exc}", file=sys.stderr)
        raise
