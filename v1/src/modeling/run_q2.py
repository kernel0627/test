from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from common.device_order import sort_by_device_id, sorted_device_ids
from common.io_utils import write_dataframe, write_markdown
from common.pillow_plotting import PanelGeometry, draw_line_chart, draw_text, get_font, panel_rect, save_canvas
from common.paths import ProjectPaths


DEFAULT_SHRINKAGE_C = 5
SHRINKAGE_GRID = [1, 3, 5, 8, 10]
FAILURE_THRESHOLD = 37.0
MAX_SIMULATION_DAYS = 365 * 50
MAX_MAJOR_PER_DEVICE_YEAR = 4


def _seasonal_map(seasonal: pd.DataFrame) -> dict[int, float]:
    return seasonal.set_index("month")["seasonal_index_adjusted"].astype(float).to_dict()


def _seasonal_value(date: pd.Timestamp, seasonal: dict[int, float]) -> float:
    return float(seasonal.get(int(pd.Timestamp(date).month), 0.0))


def _parse_sequence(value: object, *, as_float: bool = False) -> list:
    if pd.isna(value) or str(value).strip() == "":
        return []
    parts = [part for part in str(value).split(";") if part != ""]
    if as_float:
        return [float(part) for part in parts]
    return parts


def _shrink(raw: float, global_value: float, n: int, c: int) -> float:
    if not np.isfinite(raw):
        raw = global_value
    if not np.isfinite(global_value):
        global_value = raw
    weight = n / (n + c) if n + c > 0 else 0.0
    return float(weight * raw + (1.0 - weight) * global_value)


def _load_q1(paths: ProjectPaths) -> dict[str, pd.DataFrame]:
    q1_csv = paths.q1_cleaned_csv_dir
    q1_tables = paths.q1_tables_dir
    data = {
        "daily": pd.read_csv(q1_csv / "q1_cleaned_filter_daily_features.csv"),
        "events": pd.read_csv(q1_csv / "q1_cleaned_maintenance_event_effects.csv"),
        "cycles": pd.read_csv(q1_csv / "q1_cleaned_maintenance_cycles.csv"),
        "seasonal": pd.read_csv(q1_tables / "q1_table_02_monthly_seasonal_index.csv"),
        "core": pd.read_csv(q1_tables / "q1_table_05_device_core_metrics.csv"),
        "rule": pd.read_csv(q1_tables / "q1_current_maintenance_rule.csv"),
    }
    for key in ["daily", "events", "cycles"]:
        for col in data[key].columns:
            if "date" in col:
                data[key][col] = pd.to_datetime(data[key][col], errors="coerce")
    return data


def _usable_daily(daily: pd.DataFrame) -> pd.DataFrame:
    return daily[
        daily["daily_quality"].isin(["high_quality", "low_quality"])
        & daily["daily_median"].notna()
        & (~daily["is_maintenance_day"].astype(bool))
    ].copy()


def _build_device_parameters(data: dict[str, pd.DataFrame], c: int = DEFAULT_SHRINKAGE_C) -> pd.DataFrame:
    daily = data["daily"]
    events = data["events"]
    cycles = data["cycles"]
    core = data["core"]
    seasonal = _seasonal_map(data["seasonal"])
    usable = _usable_daily(daily)
    eligible_events = events[events["eligible_for_effect_analysis"].astype(bool)].copy()
    rho_events = events[events["eligible_for_rho_estimation"].astype(bool)].copy()
    eligible_cycles = cycles[cycles["eligible_for_cycle_analysis"].astype(bool)].copy()

    global_r = float(eligible_cycles["cycle_decay_rate"].dropna().median())
    global_medium_gain = float(eligible_events.loc[eligible_events["maintenance_type"] == "medium", "gain_abs"].dropna().median())
    global_major_gain = float(eligible_events.loc[eligible_events["maintenance_type"] == "major", "gain_abs"].dropna().median())
    global_medium_rho = float(rho_events.loc[rho_events["maintenance_type"] == "medium", "rho_clipped"].dropna().median())
    global_major_rho = float(rho_events.loc[rho_events["maintenance_type"] == "major", "rho_clipped"].dropna().median())
    if not np.isfinite(global_medium_rho):
        global_medium_rho = float(rho_events["rho_clipped"].dropna().median()) if not rho_events.empty else 0.5
    if not np.isfinite(global_major_rho):
        global_major_rho = float(rho_events["rho_clipped"].dropna().median()) if not rho_events.empty else global_medium_rho
    global_bh = float(core["post_recovery_trend_slope"].dropna().median())
    recent_h_values = eligible_events["H_max_est"].dropna() if "H_max_est" in eligible_events.columns else eligible_events["post_level_median_3d"].dropna()
    global_h_recent = float(recent_h_values.median()) if not recent_h_values.empty else float(usable["daily_median"].median())

    rows: list[dict[str, object]] = []
    for device_id in sorted_device_ids(core["device_id"].unique()):
        core_row = core[core["device_id"] == device_id].iloc[0]
        device_usable = usable[usable["device_id"] == device_id].sort_values("date")
        device_events = eligible_events[eligible_events["device_id"] == device_id].sort_values("event_date")
        device_rho_events = rho_events[rho_events["device_id"] == device_id].sort_values("event_date")
        device_cycles = eligible_cycles[eligible_cycles["device_id"] == device_id]

        r_raw = float(core_row["r_net_small_maintenance_background"]) if "r_net_small_maintenance_background" in core_row.index and pd.notna(core_row["r_net_small_maintenance_background"]) else float(core_row["cycle_decay_rate_median"]) if pd.notna(core_row["cycle_decay_rate_median"]) else np.nan
        n_cycles = int(core_row["n_valid_cycles"])
        r_shrink = _shrink(r_raw, global_r, n_cycles, c)

        medium_gain_raw = float(core_row["medium_gain_median"]) if pd.notna(core_row["medium_gain_median"]) else np.nan
        major_gain_raw = float(core_row["major_gain_median"]) if pd.notna(core_row["major_gain_median"]) else np.nan
        n_medium = int(core_row["n_medium_events"])
        n_major = int(core_row["n_major_events"])
        medium_gain_shrink = _shrink(medium_gain_raw, global_medium_gain, n_medium, c)
        major_gain_shrink = _shrink(major_gain_raw, global_major_gain, n_major, c)
        medium_rho_raw = float(core_row["medium_recovery_ratio_median"]) if "medium_recovery_ratio_median" in core_row.index and pd.notna(core_row["medium_recovery_ratio_median"]) else np.nan
        major_rho_raw = float(core_row["major_recovery_ratio_median"]) if "major_recovery_ratio_median" in core_row.index and pd.notna(core_row["major_recovery_ratio_median"]) else np.nan
        n_medium_rho = int(core_row["n_medium_rho_events"]) if "n_medium_rho_events" in core_row.index and pd.notna(core_row["n_medium_rho_events"]) else int(len(device_rho_events[device_rho_events["maintenance_type"] == "medium"]))
        n_major_rho = int(core_row["n_major_rho_events"]) if "n_major_rho_events" in core_row.index and pd.notna(core_row["n_major_rho_events"]) else int(len(device_rho_events[device_rho_events["maintenance_type"] == "major"]))
        medium_rho_use = _shrink(medium_rho_raw, global_medium_rho, n_medium_rho, c)
        major_rho_use = _shrink(major_rho_raw, global_major_rho, n_major_rho, c)

        bh_raw = float(core_row["post_recovery_trend_slope"]) if pd.notna(core_row["post_recovery_trend_slope"]) else np.nan
        n_h = int(core_row["post_recovery_trend_n_events"])
        bh_shrink = _shrink(bh_raw, global_bh, n_h, c)
        bh_use = min(0.0, bh_shrink)

        h_source_col = "H_max_est" if "H_max_est" in device_events.columns else "post_level_median_3d"
        recent = device_events.dropna(subset=[h_source_col]).tail(3)
        if not recent.empty:
            h_device = float(recent[h_source_col].median())
        else:
            h_device = float(device_usable["daily_median"].tail(14).median()) if not device_usable.empty else global_h_recent
        h_weight = n_h / (n_h + c) if n_h + c > 0 else 0.0
        h_max_initial = float(h_weight * h_device + (1.0 - h_weight) * global_h_recent)

        recent_daily = device_usable.tail(14).copy()
        if recent_daily.empty:
            initial_state = h_max_initial
        else:
            initial_state = float(
                np.median(
                    [
                        row.daily_median - _seasonal_value(row.date, seasonal)
                        for row in recent_daily.itertuples(index=False)
                    ]
                )
            )
        residuals = [
            row.daily_median - _seasonal_value(row.date, seasonal) - float(core_row["alpha_raw"])
            for row in device_usable.itertuples(index=False)
        ]
        residual_std = float(np.std(residuals)) if residuals else 0.0

        rows.append(
            {
                "device_id": device_id,
                "initial_state": initial_state,
                "r_i_raw": r_raw,
                "r_i_shrink": r_shrink,
                "r_net": r_raw,
                "r_net_use": r_shrink,
                "medium_gain_raw": medium_gain_raw,
                "medium_gain_shrink": medium_gain_shrink,
                "major_gain_raw": major_gain_raw,
                "major_gain_shrink": major_gain_shrink,
                "rho_medium_raw": medium_rho_raw,
                "rho_major_raw": major_rho_raw,
                "rho_medium_use": medium_rho_use,
                "rho_major_use": major_rho_use,
                "h_max_initial": h_max_initial,
                "H0_max": h_max_initial,
                "b_h_raw": bh_raw,
                "b_h_shrink": bh_shrink,
                "b_h_use": bh_use,
                "residual_std": residual_std,
                "n_valid_cycles": n_cycles,
                "n_medium_events": n_medium,
                "n_major_events": n_major,
                "n_medium_rho_events": n_medium_rho,
                "n_major_rho_events": n_major_rho,
                "shrinkage_c": c,
            }
        )
    return sort_by_device_id(pd.DataFrame(rows))


def _prepare_rule_used(rule: pd.DataFrame, future_rule_type: str = "median_interval_major_ratio") -> pd.DataFrame:
    rows = []
    for row in sort_by_device_id(rule).itertuples(index=False):
        rows.append(
            {
                "device_id": row.device_id,
                "maintenance_interval_median": row.maintenance_interval_median,
                "major_ratio": row.major_ratio,
                "I_i": row.maintenance_interval_median,
                "q_major": row.major_ratio,
                "future_rule_type": future_rule_type,
                "last_maintenance_date": row.last_maintenance_date,
                "last_maintenance_type": row.last_maintenance_type,
                "interval_sequence_used": row.interval_sequence,
                "type_sequence_used": row.type_sequence,
            }
        )
    return pd.DataFrame(rows)


def _fit_baseline(train: pd.DataFrame, seasonal: dict[int, float]) -> dict[str, tuple[float, float]]:
    models: dict[str, tuple[float, float]] = {}
    for device_id in sorted_device_ids(train["device_id"].unique()):
        group = train[train["device_id"] == device_id].sort_values("date")
        if group.empty:
            continue
        x = (group["date"] - group["date"].min()).dt.days.to_numpy(dtype=float)
        y = np.array([row.daily_median - _seasonal_value(row.date, seasonal) for row in group.itertuples(index=False)])
        if len(group) < 2 or float(np.var(x)) <= 1e-12:
            models[device_id] = (float(np.nanmean(y)), 0.0)
        else:
            x_mean = float(np.mean(x))
            y_mean = float(np.mean(y))
            denom = float(np.sum((x - x_mean) ** 2))
            slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom) if denom > 1e-12 else 0.0
            intercept = y_mean - slope * x_mean
            models[device_id] = (float(intercept), float(slope))
    return models


def _predict_baseline(daily: pd.DataFrame, train: pd.DataFrame, models: dict[str, tuple[float, float]], seasonal: dict[int, float]) -> pd.DataFrame:
    rows = []
    for row in daily.itertuples(index=False):
        if row.device_id not in models:
            continue
        device_train = train[train["device_id"] == row.device_id]
        origin = device_train["date"].min() if not device_train.empty else daily["date"].min()
        intercept, slope = models[row.device_id]
        t = (row.date - origin).days
        rows.append(
            {
                "device_id": row.device_id,
                "date": row.date,
                "predicted_p": intercept + slope * t + _seasonal_value(row.date, seasonal),
            }
        )
    return pd.DataFrame(rows)


def _actual_validation_frame(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = _usable_daily(daily)
    train_end = pd.Timestamp("2025-10-31")
    val_start = pd.Timestamp("2025-11-01")
    train = usable[usable["date"] <= train_end].copy()
    validation = usable[usable["date"] >= val_start].copy()
    return usable, train, validation


def _maintenance_schedule_from_events(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, device_id: str) -> dict[pd.Timestamp, str]:
    subset = events[
        (events["device_id"] == device_id)
        & (events["event_date"] >= start)
        & (events["event_date"] <= end)
    ]
    return {pd.Timestamp(row.event_date).normalize(): row.maintenance_type for row in subset.itertuples(index=False)}


def _simulate_device(
    *,
    device_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_x: float,
    params: pd.Series,
    seasonal: dict[int, float],
    schedule: dict[pd.Timestamp, str],
    include_decay: bool = True,
    include_gain: bool = True,
    include_hmax: bool = True,
    transition_model: str = "recovery_ratio",
) -> pd.DataFrame:
    rows = []
    x_state = float(initial_x)
    maintenance_index = 0
    current_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    while current_date <= end_date:
        mtype = schedule.get(current_date)
        before = x_state
        h_max = float(params["h_max_initial"]) + float(params["b_h_use"]) * maintenance_index
        if mtype:
            maintenance_index += 1
            if include_gain and transition_model == "recovery_ratio":
                rho = float(params["rho_major_use"] if mtype == "major" else params["rho_medium_use"])
                h_target = h_max if include_hmax else max(float(params["h_max_initial"]), before)
                after = before + rho * (h_target - before)
            else:
                gain = 0.0
                if include_gain:
                    gain = float(params["major_gain_shrink"] if mtype == "major" else params["medium_gain_shrink"])
                after = before + gain
                if include_hmax:
                    after = min(after, h_max)
            x_state = after
        else:
            if include_decay:
                x_state = x_state + float(params["r_i_shrink"])
            after = x_state
        seasonal_effect = _seasonal_value(current_date, seasonal)
        rows.append(
            {
                "device_id": device_id,
                "date": current_date,
                "predicted_p": x_state + seasonal_effect,
                "predicted_x": x_state,
                "seasonal_effect": seasonal_effect,
                "is_future_maintenance": bool(mtype),
                "future_maintenance_type": mtype or "",
                "h_max": h_max,
                "state_before_maintenance": before if mtype else np.nan,
                "state_after_maintenance": after if mtype else np.nan,
            }
        )
        current_date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def _metrics(actual: pd.DataFrame, pred: pd.DataFrame, daily_all: pd.DataFrame, model_name: str) -> dict[str, object]:
    merged = actual[["device_id", "date", "daily_median", "days_since_last_maintenance"]].merge(
        pred[["device_id", "date", "predicted_p"]], on=["device_id", "date"], how="inner"
    )
    if merged.empty:
        return {"model_name": model_name, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "post_maintenance_7d_mae": np.nan, "rolling365_mae": np.nan, "n_validation_samples": 0}
    err = merged["predicted_p"] - merged["daily_median"]
    denom = merged["daily_median"].replace(0, np.nan).abs()
    post = merged[(merged["days_since_last_maintenance"] >= 1) & (merged["days_since_last_maintenance"] <= 7)]

    rolling_errors = []
    for device_id in sorted_device_ids(merged["device_id"].unique()):
        actual_device = daily_all[daily_all["device_id"] == device_id][["date", "daily_median"]].copy()
        pred_device = pred[pred["device_id"] == device_id][["date", "predicted_p"]].copy()
        if actual_device.empty or pred_device.empty:
            continue
        combined = actual_device.merge(pred_device, on="date", how="outer").sort_values("date")
        combined["pred_for_roll"] = combined["predicted_p"].combine_first(combined["daily_median"])
        combined["actual_roll"] = combined["daily_median"].rolling(365, min_periods=30).mean()
        combined["pred_roll"] = combined["pred_for_roll"].rolling(365, min_periods=30).mean()
        validation_dates = set(pred_device["date"])
        roll_subset = combined[combined["date"].isin(validation_dates)].dropna(subset=["actual_roll", "pred_roll"])
        if not roll_subset.empty:
            rolling_errors.extend((roll_subset["pred_roll"] - roll_subset["actual_roll"]).abs().tolist())

    return {
        "model_name": model_name,
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float((err.abs() / denom).dropna().mean()) if denom.notna().any() else np.nan,
        "post_maintenance_7d_mae": float((post["predicted_p"] - post["daily_median"]).abs().mean()) if not post.empty else np.nan,
        "rolling365_mae": float(np.mean(rolling_errors)) if rolling_errors else np.nan,
        "n_validation_samples": int(len(merged)),
    }


def _backtest_predictions(data: dict[str, pd.DataFrame], params: pd.DataFrame, c: int = DEFAULT_SHRINKAGE_C) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = data["daily"]
    events = data["events"]
    seasonal = _seasonal_map(data["seasonal"])
    usable, train, validation = _actual_validation_frame(daily)
    baseline_models = _fit_baseline(train, seasonal)
    baseline_pred = _predict_baseline(validation, train, baseline_models, seasonal)
    rows = [_metrics(validation, baseline_pred, usable, "baseline_season_trend")]
    predictions: dict[str, pd.DataFrame] = {}

    for model_name, transition_model in [
        ("fixed_gain_baseline", "fixed_gain"),
        ("recovery_ratio_main", "recovery_ratio"),
    ]:
        sim_frames = []
        for device_id in sorted_device_ids(validation["device_id"].unique()):
            device_train = train[train["device_id"] == device_id].sort_values("date")
            device_val = validation[validation["device_id"] == device_id].sort_values("date")
            if device_train.empty or device_val.empty:
                continue
            param = params[params["device_id"] == device_id].iloc[0]
            initial_row = device_train.iloc[-1]
            initial_x = float(initial_row["daily_median"] - _seasonal_value(initial_row["date"], seasonal))
            schedule = _maintenance_schedule_from_events(events, device_val["date"].min(), device_val["date"].max(), device_id)
            sim_frames.append(
                _simulate_device(
                    device_id=device_id,
                    start_date=device_val["date"].min(),
                    end_date=device_val["date"].max(),
                    initial_x=initial_x,
                    params=param,
                    seasonal=seasonal,
                    schedule=schedule,
                    transition_model=transition_model,
                )
            )
        pred = pd.concat(sim_frames, ignore_index=True) if sim_frames else pd.DataFrame()
        predictions[model_name] = pred
        rows.append(_metrics(validation, pred, usable, model_name))
    return pd.DataFrame(rows), predictions.get("recovery_ratio_main", pd.DataFrame())


def _ablation_metrics(data: dict[str, pd.DataFrame], params: pd.DataFrame) -> pd.DataFrame:
    daily = data["daily"]
    events = data["events"]
    seasonal = _seasonal_map(data["seasonal"])
    usable, train, validation = _actual_validation_frame(daily)
    baseline_models = _fit_baseline(train, seasonal)
    baseline_pred = _predict_baseline(validation, train, baseline_models, seasonal)
    rows = [
        {
            **_metrics(validation, baseline_pred, usable, "M0_baseline"),
            "use_seasonal": True,
            "use_cycle_decay": False,
            "use_maintenance_gain": False,
            "use_shrinkage": False,
            "use_hmax": False,
        }
    ]
    variants = [
        ("M1_decay", True, False, True, False),
        ("M2_decay_gain", True, True, False, False),
        ("M3_decay_gain_shrinkage", True, True, True, False),
        ("M4_full_hmax", True, True, True, True),
    ]
    for name, use_decay, use_gain, use_shrinkage, use_hmax in variants:
        sim_frames = []
        for device_id in sorted_device_ids(validation["device_id"].unique()):
            device_train = train[train["device_id"] == device_id].sort_values("date")
            device_val = validation[validation["device_id"] == device_id].sort_values("date")
            if device_train.empty or device_val.empty:
                continue
            param = params[params["device_id"] == device_id].iloc[0].copy()
            if not use_shrinkage:
                param["r_i_shrink"] = param["r_i_raw"] if pd.notna(param["r_i_raw"]) else param["r_i_shrink"]
                param["medium_gain_shrink"] = param["medium_gain_raw"] if pd.notna(param["medium_gain_raw"]) else param["medium_gain_shrink"]
                param["major_gain_shrink"] = param["major_gain_raw"] if pd.notna(param["major_gain_raw"]) else param["major_gain_shrink"]
                param["rho_medium_use"] = param["rho_medium_raw"] if pd.notna(param["rho_medium_raw"]) else param["rho_medium_use"]
                param["rho_major_use"] = param["rho_major_raw"] if pd.notna(param["rho_major_raw"]) else param["rho_major_use"]
            initial_row = device_train.iloc[-1]
            initial_x = float(initial_row["daily_median"] - _seasonal_value(initial_row["date"], seasonal))
            schedule = _maintenance_schedule_from_events(events, device_val["date"].min(), device_val["date"].max(), device_id)
            sim_frames.append(
                _simulate_device(
                    device_id=device_id,
                    start_date=device_val["date"].min(),
                    end_date=device_val["date"].max(),
                    initial_x=initial_x,
                    params=param,
                    seasonal=seasonal,
                    schedule=schedule,
                    include_decay=use_decay,
                    include_gain=use_gain,
                    include_hmax=use_hmax,
                )
            )
        pred = pd.concat(sim_frames, ignore_index=True) if sim_frames else pd.DataFrame()
        rows.append(
            {
                **_metrics(validation, pred, usable, name),
                "use_seasonal": True,
                "use_cycle_decay": use_decay,
                "use_maintenance_gain": use_gain,
                "use_shrinkage": use_shrinkage,
                "use_hmax": use_hmax,
            }
        )
    return pd.DataFrame(rows)


def _generate_future_schedule(rule_row: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp, rule_type: str = "median_interval_major_ratio") -> dict[pd.Timestamp, str]:
    schedule: dict[pd.Timestamp, str] = {}
    if rule_type == "historical_sequence_cycle":
        intervals = _parse_sequence(rule_row["interval_sequence"], as_float=True)
        types = _parse_sequence(rule_row["type_sequence"], as_float=False)
        if not intervals:
            intervals = [float(rule_row["maintenance_interval_median"]) if pd.notna(rule_row["maintenance_interval_median"]) else 60.0]
        if not types:
            types = ["medium"]
        current = pd.Timestamp(rule_row["last_maintenance_date"]) if pd.notna(rule_row["last_maintenance_date"]) else start_date
        idx = 0
        while current <= end_date:
            current = current + pd.Timedelta(days=int(round(intervals[idx % len(intervals)])))
            if current >= start_date and current <= end_date:
                schedule[current.normalize()] = types[idx % len(types)]
            idx += 1
        return schedule

    interval = float(rule_row["maintenance_interval_median"]) if pd.notna(rule_row["maintenance_interval_median"]) else 60.0
    interval = max(7.0, interval)
    p_major = float(rule_row["major_ratio"]) if pd.notna(rule_row["major_ratio"]) else 0.0
    current = pd.Timestamp(rule_row["last_maintenance_date"]) if pd.notna(rule_row["last_maintenance_date"]) else start_date
    k = 0
    major_count = 0
    while current <= end_date:
        k += 1
        current = current + pd.Timedelta(days=int(round(interval)))
        target_major = round(k * p_major)
        if target_major > major_count:
            mtype = "major"
            major_count += 1
        else:
            mtype = "medium"
        if current >= start_date and current <= end_date:
            schedule[current.normalize()] = mtype
    return schedule


def _enforce_major_annual_cap(device_id: str, schedule: dict[pd.Timestamp, str]) -> tuple[dict[pd.Timestamp, str], list[dict[str, object]]]:
    capped = dict(schedule)
    rows: list[dict[str, object]] = []
    if not capped:
        return capped, rows
    years = sorted({pd.Timestamp(date).year for date in capped})
    for year in years:
        major_dates = sorted(
            date for date, maintenance_type in capped.items()
            if pd.Timestamp(date).year == year and maintenance_type == "major"
        )
        downgraded = 0
        if len(major_dates) > MAX_MAJOR_PER_DEVICE_YEAR:
            for date in major_dates[MAX_MAJOR_PER_DEVICE_YEAR:]:
                capped[date] = "medium"
                downgraded += 1
        rows.append(
            {
                "device_id": device_id,
                "year": year,
                "n_major_before_cap": int(len(major_dates)),
                "n_major_after_cap": int(min(len(major_dates), MAX_MAJOR_PER_DEVICE_YEAR)),
                "n_major_downgraded_to_medium": int(downgraded),
                "cap_rule": f"0_to_{MAX_MAJOR_PER_DEVICE_YEAR}_major_per_year",
            }
        )
    return capped, rows


def _rolling_with_history(device_daily: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    history = device_daily[["date", "daily_median"]].copy()
    future = future.copy()
    combined = pd.concat(
        [
            history.rename(columns={"daily_median": "value"})[["date", "value"]],
            future.rename(columns={"predicted_p": "value"})[["date", "value"]],
        ],
        ignore_index=True,
    ).sort_values("date")
    combined["rolling365_pred"] = combined["value"].rolling(365, min_periods=365).mean()
    future = future.merge(combined[["date", "rolling365_pred"]], on="date", how="left")
    return future


def _recovery_test(
    device_id: str,
    trigger_row: pd.Series,
    params: pd.Series,
    seasonal: dict[int, float],
    history_and_future: pd.DataFrame,
) -> dict[str, object]:
    trigger_date = pd.Timestamp(trigger_row["date"])
    x_before = float(trigger_row["predicted_x"])
    h_max = float(trigger_row["h_max"])
    x_state = x_before + float(params["rho_major_use"]) * (h_max - x_before)
    test_rows = []
    for offset in range(0, 366):
        date = trigger_date + pd.Timedelta(days=offset)
        if offset > 0:
            x_state += float(params["r_i_shrink"])
        test_rows.append({"date": date, "value": x_state + _seasonal_value(date, seasonal)})
    test_df = pd.DataFrame(test_rows)
    before = history_and_future[history_and_future["date"] < trigger_date][["date", "value"]].tail(364)
    combined = pd.concat([before, test_df], ignore_index=True).sort_values("date")
    combined["rolling"] = combined["value"].rolling(365, min_periods=365).mean()
    max_roll = float(combined["rolling"].dropna().max()) if combined["rolling"].notna().any() else np.nan
    success = bool(pd.notna(max_roll) and max_roll >= FAILURE_THRESHOLD)
    return {
        "device_id": device_id,
        "trigger_date": trigger_date,
        "rolling365_at_trigger": float(trigger_row["rolling365_pred"]),
        "after_major_max_rolling365_next365": max_roll,
        "after_major_recovery_success": success,
        "failure_confirmed": not success,
    }


def _future_simulation(data: dict[str, pd.DataFrame], params: pd.DataFrame, rule_used: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = data["daily"]
    seasonal = _seasonal_map(data["seasonal"])
    usable = _usable_daily(daily)
    start_date = daily["date"].max() + pd.Timedelta(days=1)
    end_date = start_date + pd.Timedelta(days=MAX_SIMULATION_DAYS)
    future_frames = []
    lifetime_rows = []
    recovery_rows = []
    schedule_check_rows = []
    for device_id in sorted_device_ids(params["device_id"].unique()):
        param = params[params["device_id"] == device_id].iloc[0]
        rule = rule_used[rule_used["device_id"] == device_id].iloc[0]
        schedule = _generate_future_schedule(rule, start_date, end_date, rule["future_rule_type"])
        schedule, schedule_rows = _enforce_major_annual_cap(device_id, schedule)
        schedule_check_rows.extend(schedule_rows)
        future = _simulate_device(
            device_id=device_id,
            start_date=start_date,
            end_date=end_date,
            initial_x=float(param["initial_state"]),
            params=param,
            seasonal=seasonal,
            schedule=schedule,
        )
        device_daily = usable[usable["device_id"] == device_id].sort_values("date")
        future = _rolling_with_history(device_daily, future)
        future["is_failure_date"] = False
        history_future_values = pd.concat(
            [
                device_daily[["date", "daily_median"]].rename(columns={"daily_median": "value"}),
                future[["date", "predicted_p"]].rename(columns={"predicted_p": "value"}),
            ],
            ignore_index=True,
        ).sort_values("date")

        failure_row = None
        for _, row in future[future["rolling365_pred"] < FAILURE_THRESHOLD].iterrows():
            test_result = _recovery_test(device_id, row, param, seasonal, history_future_values)
            recovery_rows.append(test_result)
            if test_result["failure_confirmed"]:
                failure_row = row
                break
        if failure_row is None:
            lifetime_rows.append(
                {
                    "device_id": device_id,
                    "predicted_failure_date": "",
                    "remaining_life_days": np.nan,
                    "remaining_life_years": np.nan,
                    "first_date_rolling365_below_37": "",
                    "recovery_test_date": "",
                    "after_major_recovery_success": "",
                    "failure_reason": "not_reached_within_50y",
                }
            )
        else:
            recovery = recovery_rows[-1]
            days = int((pd.Timestamp(failure_row["date"]) - start_date).days)
            lifetime_rows.append(
                {
                    "device_id": device_id,
                    "predicted_failure_date": failure_row["date"],
                    "remaining_life_days": days,
                    "remaining_life_years": days / 365.0,
                    "first_date_rolling365_below_37": failure_row["date"],
                    "recovery_test_date": recovery["trigger_date"],
                    "after_major_recovery_success": recovery["after_major_recovery_success"],
                    "failure_reason": "rolling365_below_37_and_major_recovery_failed",
                }
            )
            future.loc[future["date"] == failure_row["date"], "is_failure_date"] = True
        future_frames.append(future)
    return (
        sort_by_device_id(pd.concat(future_frames, ignore_index=True), "date"),
        sort_by_device_id(pd.DataFrame(lifetime_rows)),
        sort_by_device_id(pd.DataFrame(recovery_rows)) if recovery_rows else pd.DataFrame(columns=["device_id", "trigger_date", "rolling365_at_trigger", "after_major_max_rolling365_next365", "after_major_recovery_success", "failure_confirmed"]),
        sort_by_device_id(pd.DataFrame(schedule_check_rows)) if schedule_check_rows else pd.DataFrame(columns=["device_id", "year", "n_major_before_cap", "n_major_after_cap", "n_major_downgraded_to_medium", "cap_rule"]),
    )


def _robustness_summary(data: dict[str, pd.DataFrame], rule_used: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in SHRINKAGE_GRID:
        params = _build_device_parameters(data, c)
        metrics, _ = _backtest_predictions(data, params, c)
        main = metrics[metrics["model_name"] == "recovery_ratio_main"].iloc[0]
        rows.extend(
            [
                {"test_type": "shrinkage_c", "setting": str(c), "metric": "mae", "value": main["mae"], "notes": "event_driven backtest"},
                {"test_type": "shrinkage_c", "setting": str(c), "metric": "rmse", "value": main["rmse"], "notes": "event_driven backtest"},
                {"test_type": "shrinkage_c", "setting": str(c), "metric": "rolling365_mae", "value": main["rolling365_mae"], "notes": "event_driven backtest"},
            ]
        )
    for rule_type in ["median_interval_major_ratio", "historical_sequence_cycle"]:
        rows.append({"test_type": "future_rule_type", "setting": rule_type, "metric": "defined", "value": 1.0, "notes": "available deterministic future maintenance rule"})
    for days in [15, 30, 60]:
        rows.append({"test_type": "confirmation_window_days", "setting": str(days), "metric": "defined", "value": float(days), "notes": "robustness setting only; main rule uses immediate recovery test trigger"})
    return pd.DataFrame(rows)


def _draw_grouped_bar(metrics: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGBA", (1100, 620), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(30)
    label_font = get_font(18)
    tick_font = get_font(13)
    draw_text(draw, (550, 30), "回测误差对比", title_font, anchor="ma")
    cols = ["mae", "rmse", "post_maintenance_7d_mae", "rolling365_mae"]
    labels = ["MAE", "RMSE", "Post7 MAE", "Rolling365 MAE"]
    left, top, right, bottom = 110, 110, 1020, 540
    draw.rectangle((left, top, right, bottom), outline="#9ca3af")
    vals = metrics[cols].to_numpy(dtype=float)
    ymax = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
    ymax = max(ymax * 1.15, 1.0)
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y = int(bottom - frac * (bottom - top))
        draw.line((left, y, right, y), fill="#e5e7eb")
        draw_text(draw, (left - 8, y), f"{frac * ymax:.1f}", tick_font, anchor="ra")
    colors = ["#9ca3af", "#2563eb", "#f97316", "#10b981"]
    group_w = (right - left) / len(cols)
    for i, col in enumerate(cols):
        x0 = left + i * group_w
        for j, row in enumerate(metrics.itertuples(index=False)):
            value = getattr(row, col)
            if pd.isna(value):
                continue
            bar_w = int(group_w * 0.28)
            x = int(x0 + group_w * 0.28 + j * bar_w)
            y = int(bottom - float(value) / ymax * (bottom - top))
            draw.rectangle((x, y, x + bar_w - 4, bottom), fill=colors[j], outline=colors[j])
        draw_text(draw, (int(x0 + group_w / 2), bottom + 10), labels[i], tick_font, anchor="ma")
    legend_x = 680
    for idx, model_name in enumerate(metrics["model_name"].tolist()):
        draw_text(draw, (legend_x, 70), str(model_name), label_font, fill=colors[idx % len(colors)])
        legend_x += 170
    save_canvas(image, output_path)


def _draw_ablation(ablation: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGBA", (1200, 620), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(30)
    label_font = get_font(18)
    tick_font = get_font(13)
    draw_text(draw, (600, 30), "消融实验对比", title_font, anchor="ma")
    left, top, right, bottom = 100, 110, 1120, 540
    draw.rectangle((left, top, right, bottom), outline="#9ca3af")
    models = ablation["model_name"].tolist()
    vals = ablation[["mae", "rolling365_mae"]].to_numpy(dtype=float)
    ymax = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
    ymax = max(ymax * 1.15, 1.0)
    x_positions = [int(left + (i + 0.5) * (right - left) / len(models)) for i in range(len(models))]
    for metric, color in [("mae", "#2563eb"), ("rolling365_mae", "#f97316")]:
        points = []
        for x, value in zip(x_positions, ablation[metric].astype(float).tolist()):
            if pd.isna(value):
                continue
            y = int(bottom - value / ymax * (bottom - top))
            points.append((x, y))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
    for x, model in zip(x_positions, models):
        draw_text(draw, (x, bottom + 12), model.replace("_", "\n"), tick_font, anchor="ma")
    draw_text(draw, (800, 70), "MAE", label_font, fill="#2563eb")
    draw_text(draw, (900, 70), "Rolling365 MAE", label_font, fill="#f97316")
    save_canvas(image, output_path)


def _draw_lifetime(lifetime: pd.DataFrame, output_path: Path) -> None:
    df = lifetime.copy()
    df["rank_value"] = df["remaining_life_years"].fillna(50.0)
    df = sort_by_device_id(df).sort_values("rank_value")
    image = Image.new("RGBA", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(30)
    tick_font = get_font(15)
    label_font = get_font(18)
    draw_text(draw, (500, 30), "剩余寿命排序", title_font, anchor="ma")
    left, top, right, bottom = 170, 90, 920, 560
    xmax = max(1.0, float(df["rank_value"].max()))
    row_h = (bottom - top) / len(df)
    for idx, row in enumerate(df.itertuples(index=False)):
        y = int(top + idx * row_h + row_h * 0.18)
        h = int(row_h * 0.58)
        w = int(float(row.rank_value) / xmax * (right - left))
        draw_text(draw, (left - 15, y + h // 2), row.device_id, tick_font, anchor="ra")
        draw.rectangle((left, y, left + w, y + h), fill="#2563eb", outline="#2563eb")
        label = ">=50" if pd.isna(row.remaining_life_years) else f"{row.remaining_life_years:.1f}"
        draw_text(draw, (left + w + 8, y + h // 2 - 8), label, tick_font)
    draw_text(draw, ((left + right) // 2, bottom + 35), "剩余寿命（年）", label_font, anchor="ma")
    save_canvas(image, output_path)


def _draw_major_recovery_test(recovery: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGBA", (1200, 680), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(30)
    label_font = get_font(18)
    tick_font = get_font(14)
    draw_text(draw, (600, 30), "强制大维护恢复测试", title_font, anchor="ma")
    if recovery.empty:
        draw_text(draw, (80, 100), "无恢复测试记录", label_font)
        save_canvas(image, output_path)
        return
    df = recovery.sort_values("trigger_date").groupby("device_id", as_index=False).tail(1)
    df = sort_by_device_id(df)
    devices = df["device_id"].tolist()
    values = df["after_major_max_rolling365_next365"].astype(float).tolist()
    left, top, right, bottom = 110, 100, 1120, 590
    draw.rectangle((left, top, right, bottom), outline="#9ca3af")
    ymax = max(FAILURE_THRESHOLD + 10, max([v for v in values if np.isfinite(v)] or [FAILURE_THRESHOLD]))
    ymin = min(0.0, min([v for v in values if np.isfinite(v)] or [0.0]))
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = int(bottom - frac * (bottom - top))
        draw.line((left, y, right, y), fill="#e5e7eb")
        draw_text(draw, (left - 8, y), f"{ymin + frac * (ymax - ymin):.1f}", tick_font, anchor="ra")
    threshold_y = int(bottom - (FAILURE_THRESHOLD - ymin) / (ymax - ymin) * (bottom - top))
    draw.line((left, threshold_y, right, threshold_y), fill="#dc2626", width=2)
    bar_w = max(18, int((right - left) / max(len(devices), 1) * 0.55))
    for idx, (device_id, value) in enumerate(zip(devices, values)):
        x = int(left + (idx + 0.5) * (right - left) / len(devices))
        y = int(bottom - (value - ymin) / (ymax - ymin) * (bottom - top)) if np.isfinite(value) else bottom
        color = "#2563eb" if value >= FAILURE_THRESHOLD else "#dc2626"
        draw.rectangle((x - bar_w // 2, y, x + bar_w // 2, bottom), fill=color, outline=color)
        draw_text(draw, (x, bottom + 10), device_id, tick_font, anchor="ma")
    draw_text(draw, ((left + right) // 2, bottom + 52), "设备编号", label_font, anchor="ma")
    draw_text(draw, (28, (top + bottom) // 2), "测试后未来一年最大滚动年均", label_font)
    save_canvas(image, output_path)


def _draw_rolling365(future: pd.DataFrame, lifetime: pd.DataFrame, output_path: Path) -> None:
    devices = sorted_device_ids(future["device_id"].unique())
    width, height = 1800, 2200
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(34)
    draw_text(draw, (width // 2, 30), "滚动365天平均透水率预测", title_font, anchor="ma")
    y_values = future["rolling365_pred"].dropna().astype(float)
    global_min = min(FAILURE_THRESHOLD - 5, float(y_values.min()) if not y_values.empty else 0)
    global_max = max(FAILURE_THRESHOLD + 20, float(y_values.max()) if not y_values.empty else 100)
    for idx, device_id in enumerate(devices):
        rect = panel_rect(idx, 5, 2, width=width, height=height, top_margin=120)
        subset = future[future["device_id"] == device_id].dropna(subset=["rolling365_pred"]).copy()
        if subset.empty:
            continue
        origin = subset["date"].min()
        x_vals = [(d - origin).days for d in subset["date"]]
        series = [
            {"y_values": [FAILURE_THRESHOLD] * len(x_vals), "color": "#dc2626", "width": 1, "mode": "line"},
            {"y_values": subset["rolling365_pred"].astype(float).tolist(), "color": "#f97316", "width": 2, "mode": "line"},
        ]
        draw_line_chart(image, draw, rect, x_values=x_vals, series=series, title=device_id, x_label="未来天数", y_label="滚动365天均值", y_zero_line=False)
    save_canvas(image, output_path)


def _draw_future_path(device_id: str, daily: pd.DataFrame, future: pd.DataFrame, output_path: Path) -> None:
    history = _usable_daily(daily)
    history = history[history["device_id"] == device_id].tail(730).copy()
    subset = future[future["device_id"] == device_id].copy()
    image = Image.new("RGBA", (1300, 720), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(30)
    draw_text(draw, (650, 30), f"{device_id} 未来路径预测", title_font, anchor="ma")
    origin = history["date"].min() if not history.empty else subset["date"].min()
    x_hist = [(d - origin).days for d in history["date"]]
    x_future = [(d - origin).days for d in subset["date"]]
    x_all = x_hist + x_future
    series = []
    if not history.empty:
        series.append({"y_values": history["daily_median"].astype(float).tolist() + [None] * len(x_future), "color": "#9ca3af", "width": 1, "mode": "line"})
    series.append({"y_values": [None] * len(x_hist) + subset["predicted_p"].astype(float).tolist(), "color": "#2563eb", "width": 2, "mode": "line"})
    rolling_values = [None] * len(x_hist) + [None if pd.isna(v) else float(v) for v in subset["rolling365_pred"]]
    series.append({"y_values": rolling_values, "color": "#f97316", "width": 2, "mode": "line"})
    series.append({"y_values": [FAILURE_THRESHOLD] * len(x_all), "color": "#dc2626", "width": 1, "mode": "line"})
    failure_values = [None] * len(x_hist)
    for is_failure, rolling_value in zip(subset["is_failure_date"], subset["rolling365_pred"]):
        failure_values.append(float(rolling_value) if bool(is_failure) and pd.notna(rolling_value) else None)
    series.append({"y_values": failure_values, "color": "#dc2626", "width": 2, "mode": "markers"})
    rect = PanelGeometry(50, 80, 1250, 670)
    vertical_lines = [(x, "#d1d5db") for x, m in zip(x_future, subset["is_future_maintenance"]) if bool(m)]
    draw_line_chart(image, draw, rect, x_values=x_all, series=series, title="", x_label="日期序号", y_label="透水率", vertical_lines=vertical_lines)
    save_canvas(image, output_path)


def _write_q2_markdown(paths: ProjectPaths, metrics: pd.DataFrame, ablation: pd.DataFrame, lifetime: pd.DataFrame) -> None:
    best = metrics.sort_values("mae").iloc[0] if not metrics.empty else None
    risky = lifetime.sort_values("remaining_life_years", na_position="last").head(3)
    risky_text = risky[
        ["device_id", "predicted_failure_date", "remaining_life_years", "failure_reason"]
    ].to_string(index=False)
    content = f"""# 第二问建模总结

## 1. 模型目标

按照当前固定维护规律预测 10 台过滤器寿命。

## 2. 第一问输入

第二问只读取第一问 CSV，包括日特征、维护事件、维护周期、季节指数、设备核心指标和当前维护规律。

## 3. baseline 模型

baseline 使用设备趋势和月份季节项，不显式处理维护事件。

## 4. 事件驱动退化模型

主模型将透水率拆成设备状态和季节项，在非维护日按周期衰减率更新，在维护日按维护类型恢复。

## 5. 参数收缩方法

默认收缩参数为 `c=5`，并在稳健性分析中比较 `{SHRINKAGE_GRID}`。

## 6. 最大可恢复水平

维护后状态受 `H_i,k^max` 限制，且恢复上限只允许保持或下降。

## 7. 当前维护规律使用方式

主方案使用中位维护间隔和大维护比例，通过确定性累计比例法生成未来维护类型。

## 8. 回测结果

回测指标输出到 `q2_baseline_backtest_metrics.csv`。当前 MAE 最小模型为 `{best['model_name'] if best is not None else 'NA'}`。

## 9. 消融实验结果

消融实验输出到 `q2_ablation_metrics.csv`，用于比较衰减、维护增益、收缩和恢复上限的贡献。

## 10. 稳健性分析

稳健性分析输出到 `q2_robustness_summary.csv`，包括收缩参数、维护规律外推方式和寿命确认窗口设置。

## 11. 未来模拟与寿命判定

未来最多模拟 50 年。首次滚动 365 天均值低于 37 时执行强制大维护恢复测试，未来一年内仍无法回到 37 以上则确认寿命终止。

## 12. 10 台设备寿命预测结论

寿命预测结果输出到 `q2_lifetime_prediction.csv`。剩余寿命较短的设备如下：

```text
{risky_text}
```
"""
    write_markdown(paths.q2_markdown_dir / "q2_modeling_summary.md", content)


def _build_q2_model_parameters(params: pd.DataFrame, rule_used: pd.DataFrame) -> pd.DataFrame:
    merged = params.merge(
        rule_used[["device_id", "maintenance_interval_median", "major_ratio"]],
        on="device_id",
        how="left",
    )
    out = pd.DataFrame(
        {
            "device_id": merged["device_id"],
            "H0_max": merged["H0_max"],
            "b_H_raw": merged["b_h_raw"],
            "b_H_use": merged["b_h_use"],
            "r_net": merged["r_net_use"],
            "rho_medium_raw": merged["rho_medium_raw"],
            "rho_major_raw": merged["rho_major_raw"],
            "rho_medium_use": merged["rho_medium_use"],
            "rho_major_use": merged["rho_major_use"],
            "n_medium_rho_events": merged["n_medium_rho_events"],
            "n_major_rho_events": merged["n_major_rho_events"],
            "I_i": merged["maintenance_interval_median"],
            "q_major": merged["major_ratio"],
            "sigma_i": merged["residual_std"],
            "shrinkage_c": merged["shrinkage_c"],
        }
    )
    return sort_by_device_id(out)


def _build_q2_life_prediction_results(lifetime: pd.DataFrame, recovery: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    current_roll = (
        future.sort_values("date")
        .groupby("device_id", as_index=False)
        .tail(1)[["device_id", "rolling365_pred"]]
        .rename(columns={"rolling365_pred": "current_rolling365"})
    )
    recovery_lookup = recovery.copy()
    if not recovery_lookup.empty:
        recovery_lookup["date_key"] = pd.to_datetime(recovery_lookup["trigger_date"]).dt.strftime("%Y-%m-%d")
    rows: list[dict[str, object]] = []
    for row in lifetime.itertuples(index=False):
        device_id = row.device_id
        predicted_date = getattr(row, "predicted_failure_date")
        date_key = pd.to_datetime(predicted_date).strftime("%Y-%m-%d") if pd.notna(predicted_date) and str(predicted_date) else ""
        rec = pd.DataFrame()
        if date_key and not recovery_lookup.empty:
            rec = recovery_lookup[
                (recovery_lookup["device_id"] == device_id)
                & (recovery_lookup["date_key"] == date_key)
            ]
        m_i = float(rec.iloc[0]["after_major_max_rolling365_next365"]) if not rec.empty else np.nan
        recovery_passed = bool(rec.iloc[0]["after_major_recovery_success"]) if not rec.empty else ""
        rows.append(
            {
                "device_id": device_id,
                "current_rolling365": float(current_roll.loc[current_roll["device_id"] == device_id, "current_rolling365"].iloc[0]) if not current_roll.loc[current_roll["device_id"] == device_id].empty else np.nan,
                "predicted_life_end_date": predicted_date,
                "remaining_life_days": getattr(row, "remaining_life_days"),
                "remaining_life_years": getattr(row, "remaining_life_years"),
                "life_end_reason": getattr(row, "failure_reason"),
                "first_date_rolling365_below_37": getattr(row, "first_date_rolling365_below_37"),
                "major_recovery_test_passed": recovery_passed,
                "M_i_at_life_test": m_i,
                "prediction_interval_low": "",
                "prediction_interval_high": "",
            }
        )
    return sort_by_device_id(pd.DataFrame(rows))


def _write_q2_canonical_markdowns(paths: ProjectPaths, metrics: pd.DataFrame, life_results: pd.DataFrame) -> None:
    best = metrics.sort_values("mae").iloc[0] if not metrics.empty else None
    short_life = life_results.sort_values("remaining_life_years", na_position="last").head(3)
    short_life_text = short_life[
        ["device_id", "predicted_life_end_date", "remaining_life_years", "life_end_reason"]
    ].to_string(index=False)
    model_text = f"""# 第二问模型说明

## 模型目标

第二问只读取第一问输出 CSV，不重读原始 Excel。模型目标是在当前显式维护规律不变的条件下预测 10 台过滤器寿命。

## 第一问输入

输入包括日特征表、维护事件表、维护周期表、月份季节指数、设备核心指标和当前维护规律。小维护没有具体日期，因此作为背景维护处理；中维护和大维护作为显式维护事件。

## 恢复比例主模型

透水率分解为 `P_i,t = X_i,t + S_m(t)`。非维护日按小维护背景下净衰减率 `r_i*` 更新；维护日采用恢复比例状态转移：

```text
X(t+) = X(t-) + rho_use * (Hmax(k) - X(t-))
```

其中 `rho_medium_use` 和 `rho_major_use` 为设备级收缩参数。事件级 `rho_raw/rho_clipped` 只用于第一问统计和诊断，不直接进入第二问模拟。

## 当前维护规律

大维护每台设备每年允许 0 至 4 次，不是年度必做维护。未来日程按历史显式维护规律生成，允许 `q_major=0`；若模拟日程某年大维护超过 4 次，则仅在未来模拟中把超出部分降级为中维护，并记录检查表。

## 寿命判定

当滚动 365 天平均透水率首次低于 37 时，执行一次强制大维护恢复测试。该测试只用于寿命判定，不改变主预测路径；若测试后未来一年滚动年均仍不能回到 37 以上，则确认寿命终止。

## 输出口径

旧表 `q2_device_parameters.csv`、`q2_lifetime_prediction.csv` 保留兼容；论文和后续分析统一以 `q2_model_parameters.csv`、`q2_life_prediction_results.csv`、`q2_backtest_metrics.csv` 为准。
"""
    life_text = f"""# 第二问寿命预测解释

## 判定规则

主判据为 `rolling365_pred < 37` 加强制大维护恢复测试失败。恢复测试统计量为 `M_i_at_life_test`，即强制大维护后未来一年内滚动 365 天均值的最大值。

## 预测结果

canonical 结果表为 `q2_life_prediction_results.csv`。剩余寿命较短的设备如下：

```text
{short_life_text}
```

## 注意事项

50 年为计算截断，不代表对 50 年后状态作可靠预测。`not_reached_within_50y` 表示在截断期内未触发寿命终止条件。
"""
    backtest_text = f"""# 第二问回测总结

## 对照模型

`fixed_gain_baseline` 使用固定增益恢复；`recovery_ratio_main` 使用恢复比例恢复当前可恢复空间的一部分。

## 指标文件

回测结果输出到 `q2_backtest_metrics.csv`。当前 MAE 最小模型为 `{best['model_name'] if best is not None else 'NA'}`。

```text
{metrics[['model_name', 'mae', 'rmse', 'mape', 'n_validation_samples']].to_string(index=False)}
```
"""
    summary_text = f"""# 第二问建模总结

第二问采用恢复比例主模型预测寿命，并保留固定增益模型作为 baseline。小维护作为背景维护；中维护和大维护作为显式维护事件；大维护每年允许 0 至 4 次，不强制年度必做。

canonical 输出包括：

- `q2_model_parameters.csv`
- `q2_life_prediction_results.csv`
- `q2_backtest_metrics.csv`

旧表继续保留兼容，但论文和后续分析以上述 canonical 表为准。

寿命判定采用 rolling365 低于 37 加强制大维护恢复测试。强制大维护只用于判定，不改变主预测路径。
"""
    write_markdown(paths.q2_markdown_dir / "q2_model_description.md", model_text)
    write_markdown(paths.q2_markdown_dir / "q2_life_prediction_explanation.md", life_text)
    write_markdown(paths.q2_markdown_dir / "q2_backtest_summary.md", backtest_text)
    write_markdown(paths.q2_markdown_dir / "q2_modeling_summary.md", summary_text)


def run_q2(paths: ProjectPaths) -> None:
    data = _load_q1(paths)
    params = _build_device_parameters(data, DEFAULT_SHRINKAGE_C)
    rule_used = _prepare_rule_used(data["rule"])
    backtest_metrics, _ = _backtest_predictions(data, params)
    ablation = _ablation_metrics(data, params)
    robustness = _robustness_summary(data, rule_used)
    future, lifetime, recovery, schedule_check = _future_simulation(data, params, rule_used)
    model_parameters = _build_q2_model_parameters(params, rule_used)
    life_prediction_results = _build_q2_life_prediction_results(lifetime, recovery, future)

    write_dataframe(params, paths.q2_tables_dir / "q2_device_parameters.csv")
    write_dataframe(model_parameters, paths.q2_tables_dir / "q2_model_parameters.csv")
    write_dataframe(rule_used, paths.q2_tables_dir / "q2_current_maintenance_rule_used.csv")
    write_dataframe(backtest_metrics, paths.q2_tables_dir / "q2_baseline_backtest_metrics.csv")
    write_dataframe(backtest_metrics.rename(columns={"mae": "MAE", "rmse": "RMSE", "mape": "MAPE", "n_validation_samples": "n_test_days"}), paths.q2_tables_dir / "q2_backtest_metrics.csv")
    write_dataframe(ablation, paths.q2_tables_dir / "q2_ablation_metrics.csv")
    write_dataframe(robustness, paths.q2_tables_dir / "q2_robustness_summary.csv")
    write_dataframe(future, paths.q2_tables_dir / "q2_future_simulation_path.csv")
    write_dataframe(recovery, paths.q2_tables_dir / "q2_recovery_test_results.csv")
    write_dataframe(lifetime, paths.q2_tables_dir / "q2_lifetime_prediction.csv")
    write_dataframe(life_prediction_results, paths.q2_tables_dir / "q2_life_prediction_results.csv")
    write_dataframe(schedule_check, paths.q2_tables_dir / "q2_check_maintenance_schedule_summary.csv")

    _draw_grouped_bar(backtest_metrics, paths.q2_figures_png_dir / "q2_fig_01_backtest_comparison.png")
    _draw_grouped_bar(backtest_metrics, paths.q2_figures_png_dir / "q2_backtest_model_comparison.png")
    _draw_ablation(ablation, paths.q2_figures_png_dir / "q2_fig_02_ablation_comparison.png")
    _draw_lifetime(lifetime, paths.q2_figures_png_dir / "q2_fig_03_remaining_life_ranking.png")
    _draw_rolling365(future, lifetime, paths.q2_figures_png_dir / "q2_fig_04_rolling365_prediction.png")
    _draw_major_recovery_test(recovery, paths.q2_figures_png_dir / "q2_major_recovery_test.png")
    for device_id in sorted_device_ids(params["device_id"].unique()):
        _draw_future_path(device_id, data["daily"], future, paths.q2_figures_png_dir / f"q2_fig_future_path_{device_id}.png")
        _draw_future_path(device_id, data["daily"], future, paths.q2_figures_png_dir / f"q2_prediction_curve_device_{device_id}.png")
    _write_q2_markdown(paths, backtest_metrics, ablation, lifetime)
    _write_q2_canonical_markdowns(paths, backtest_metrics, life_prediction_results)
