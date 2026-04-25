from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .build_schedule import decide_maintenance, schedule_record


MODEL_FIXED = "fixed_gain_baseline"
MODEL_MAIN = "recovery_ratio_main"
MODEL_FIXED_SENS = "fixed_gain_baseline_major_sensitivity"
MODEL_MAIN_SENS = "recovery_ratio_main_major_sensitivity"
MODEL_FIXED_HMAX_OPT = "fixed_gain_baseline_hmax_optimistic"
MODEL_MAIN_HMAX_OPT = "recovery_ratio_main_hmax_optimistic"
MODEL_FIXED_HMAX_PESS = "fixed_gain_baseline_hmax_pessimistic"
MODEL_MAIN_HMAX_PESS = "recovery_ratio_main_hmax_pessimistic"
MODEL_FIXED_HMAX_ENG = "fixed_gain_baseline_hmax_engineering_conservative"
MODEL_MAIN_HMAX_ENG = "recovery_ratio_main_hmax_engineering_conservative"

HORIZON_DAYS = 30 * 365
LIFETIME_TEST_EXTRA_DAYS = 365
THRESHOLD = 37.0
COMMISSION_DATE = pd.Timestamp("2022-04-01")
FAILURE_PRIORITY = {
    "hmax_depleted": 1,
    "unrecoverable_below_37": 2,
    "excessive_maintenance_burden": 3,
    "functional_failure": 4,
    "service_cap_reached": 5,
}


def _history_values_for_rolling(daily: pd.DataFrame, device_id: str) -> list[float]:
    group = daily[daily["device_id"] == device_id].sort_values("date")
    values = group["daily_median"].dropna().astype(float).tail(364).tolist()
    if not values:
        values = [float(group["daily_median"].dropna().astype(float).median())]
    return values


def _x_history(daily: pd.DataFrame, device_id: str, seasonal_level: dict[int, float], initial_x: float) -> list[float]:
    group = daily[(daily["device_id"] == device_id) & daily["daily_median"].notna()].sort_values("date").tail(30)
    values = []
    for _, row in group.iterrows():
        values.append(float(row["daily_median"]) - seasonal_level.get(int(row["month"]), 0.0))
    return values if values else [initial_x]


def _seasonal_kappa(helpers: dict[str, object], maintenance_type: str, month: int) -> tuple[float, float]:
    season = helpers["season_group"](int(month))
    return helpers["seasonal_effect"].get((maintenance_type, season), (1.0, 1.0, False))[:2]


def hmax_trend_for_scenario(row: pd.Series, scenario: str) -> tuple[float, float]:
    if scenario == "optimistic":
        return 0.0, 0.0
    if scenario == "pessimistic":
        trend = float(row.get("hmax_trend_limited", row.get("hmax_trend_used", 0.0)))
        ratio = abs(trend) * 365.0 / max(float(row["h_max_initial"]), 1e-9)
        return trend, ratio
    if scenario == "engineering_conservative":
        hmax_initial = max(float(row["h_max_initial"]), 1e-9)
        raw = min(0.0, float(row.get("hmax_trend_raw", 0.0)))
        floor = -0.30 * hmax_initial / 365.0
        trend = max(raw, floor)
        ratio = abs(trend) * 365.0 / hmax_initial
        return trend, ratio
    trend = float(row.get("hmax_trend_used", 0.0))
    ratio = float(row.get("hmax_annual_drop_ratio_used", 0.0))
    return trend, ratio


def _service_age_years(date: pd.Timestamp) -> float:
    return float((date - COMMISSION_DATE).days / 365.25)


def _hmax_raw(row: pd.Series, hmax_trend: float, day_index: int, medium_count: int, major_count: int) -> float:
    damage = (
        float(row.get("hmax_damage_delta_medium", 0.0)) * medium_count
        + float(row.get("hmax_damage_delta_major", 0.0)) * major_count
    )
    return float(row["h_max_initial"]) + hmax_trend * day_index - damage


def _next_year_max_recoverable(
    row: pd.Series,
    hmax_trend: float,
    day_index: int,
    date: pd.Timestamp,
    medium_count: int,
    major_count: int,
    helpers: dict[str, object],
) -> float:
    damage = (
        float(row.get("hmax_damage_delta_medium", 0.0)) * medium_count
        + float(row.get("hmax_damage_delta_major", 0.0)) * major_count
    )
    values: list[float] = []
    for offset in range(366):
        future_date = date + pd.Timedelta(days=offset)
        hmax_future = float(row["h_max_initial"]) + hmax_trend * (day_index + offset) - damage
        seasonal = helpers["seasonal_level"].get(int(future_date.month), 0.0)
        values.append(hmax_future + seasonal)
    return float(max(values)) if values else -math.inf


def _maintenance_burden_after_add(
    maintenance_dates: list[pd.Timestamp],
    current_date: pd.Timestamp,
) -> tuple[int, float, float, bool]:
    start = current_date - pd.Timedelta(days=364)
    dates = sorted([date for date in maintenance_dates if date >= start] + [current_date])
    count = len(dates)
    if len(dates) >= 2:
        intervals = pd.Series(dates).diff().dt.days.dropna().astype(float)
        avg_interval = float(intervals.mean())
        min_interval = float(intervals.min())
    else:
        avg_interval = math.nan
        min_interval = math.nan
    burden = bool(
        count > 8
        or (np.isfinite(avg_interval) and avg_interval < 45)
        or (np.isfinite(min_interval) and min_interval < 30)
    )
    return count, avg_interval, min_interval, burden


def _maintenance_window_metrics(maintenance_dates: list[pd.Timestamp], current_date: pd.Timestamp) -> tuple[int, float, float]:
    start = current_date - pd.Timedelta(days=364)
    dates = sorted([date for date in maintenance_dates if date >= start])
    count = len(dates)
    if len(dates) >= 2:
        intervals = pd.Series(dates).diff().dt.days.dropna().astype(float)
        return count, float(intervals.mean()), float(intervals.min())
    return count, math.nan, math.nan


def _best_failure(candidates: list[str]) -> str:
    if not candidates:
        return ""
    return sorted(candidates, key=lambda value: FAILURE_PRIORITY[value])[0]


def _functional_failure_date(frame: pd.DataFrame, horizon_days: int) -> pd.Timestamp | pd.NaT:
    if frame.empty or "rolling365_pred" not in frame.columns:
        return pd.NaT
    within = frame.iloc[:horizon_days].copy()
    first_below = within.loc[within["rolling365_pred"].astype(float) < THRESHOLD, "date"]
    for date in first_below:
        lookahead = frame[(frame["date"] >= date) & (frame["date"] <= date + pd.Timedelta(days=365))]
        if len(lookahead) and float(lookahead["rolling365_pred"].max()) < THRESHOLD:
            return pd.Timestamp(date)
    return pd.NaT


def _truncate_at_final_failure(frame: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["is_terminal_day"] = out["is_terminal_day"].astype(bool)
    functional_date = _functional_failure_date(out, horizon_days)
    terminal_rows = out[out["is_terminal_day"]]
    terminal_date = pd.Timestamp(terminal_rows["date"].iloc[0]) if len(terminal_rows) else pd.NaT
    terminal_type = str(terminal_rows["failure_type"].iloc[0]) if len(terminal_rows) else ""
    final_date = terminal_date
    final_type = terminal_type
    if pd.notna(functional_date):
        if pd.isna(final_date) or functional_date < final_date:
            final_date = functional_date
            final_type = "functional_failure"
        elif functional_date == final_date and FAILURE_PRIORITY["functional_failure"] < FAILURE_PRIORITY.get(final_type, 99):
            final_type = "functional_failure"
    if pd.isna(final_date):
        return out
    out = out[out["date"] <= final_date].copy()
    out["is_terminal_day"] = False
    out["failure_type"] = ""
    final_idx = out.index[out["date"] == final_date]
    if len(final_idx):
        out.loc[final_idx[-1], "is_terminal_day"] = True
        out.loc[final_idx[-1], "failure_type"] = final_type
    return out


def apply_fixed_gain(
    row: pd.Series,
    maintenance_type: str,
    month: int,
    x_state: float,
    hmax_t: float,
    helpers: dict[str, object],
) -> float:
    kappa_gain, _ = _seasonal_kappa(helpers, maintenance_type, month)
    gain_col = f"{maintenance_type}_plateau_gain_used"
    gain = float(row[gain_col]) if pd.notna(row.get(gain_col)) else 0.0
    recovered = x_state + gain * kappa_gain
    return max(x_state, min(recovered, hmax_t))


def apply_recovery_ratio(
    row: pd.Series,
    maintenance_type: str,
    month: int,
    x_state: float,
    hmax_t: float,
    helpers: dict[str, object],
) -> tuple[float, str]:
    source_col = f"{maintenance_type}_rho_used_source"
    rho_col = f"{maintenance_type}_recovery_ratio_used"
    source = str(row.get(source_col, "fixed_gain_fallback"))
    if source == "fixed_gain_fallback" or pd.isna(row.get(rho_col)):
        return apply_fixed_gain(row, maintenance_type, month, x_state, hmax_t, helpers), "fixed_gain_fallback"
    _, kappa_rho = _seasonal_kappa(helpers, maintenance_type, month)
    rho = max(0.0, min(1.0, float(row[rho_col]) * kappa_rho))
    recovered = x_state + rho * max(0.0, hmax_t - x_state)
    return max(x_state, min(recovered, hmax_t)), source


def _model_specs() -> list[tuple[str, str, str, bool]]:
    return [
        (MODEL_FIXED, "fixed", "neutral", False),
        (MODEL_MAIN, "main", "neutral", False),
        (MODEL_FIXED_HMAX_OPT, "fixed", "optimistic", False),
        (MODEL_MAIN_HMAX_OPT, "main", "optimistic", False),
        (MODEL_FIXED_HMAX_PESS, "fixed", "pessimistic", False),
        (MODEL_MAIN_HMAX_PESS, "main", "pessimistic", False),
        (MODEL_FIXED_SENS, "fixed", "neutral", True),
        (MODEL_MAIN_SENS, "main", "neutral", True),
    ]


def simulate_future_paths(
    params: pd.DataFrame,
    daily: pd.DataFrame,
    helpers: dict[str, object],
    horizon_years: int = 30,
    include_lifetime_test_extra: bool = True,
    model_specs: list[tuple[str, str, str, bool]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_start: pd.Timestamp = helpers["prediction_start"]
    extra_days = LIFETIME_TEST_EXTRA_DAYS if include_lifetime_test_extra else 0
    dates = pd.date_range(prediction_start, periods=horizon_years * 365 + extra_days, freq="D")
    specs = model_specs if model_specs is not None else _model_specs()
    path_rows: list[dict[str, object]] = []
    schedule_rows: list[dict[str, object]] = []

    for _, param in params.iterrows():
        device_id = str(param["device_id"])
        for model_name, model_family, hmax_scenario, use_global_major_fallback in specs:
            hmax_trend, hmax_annual_ratio = hmax_trend_for_scenario(param, hmax_scenario)
            x_state = float(param["initial_x_state"])
            x_history = _x_history(daily, device_id, helpers["seasonal_level"], x_state)
            rolling_values = _history_values_for_rolling(daily, device_id)
            last_maintenance_date = pd.to_datetime(param["last_maintenance_date"], errors="coerce")
            last_major_date = pd.to_datetime(param["last_major_maintenance_date"], errors="coerce")
            medium_since_last_major = int(param["medium_since_last_major_at_end"])
            if (
                use_global_major_fallback
                and int(param.get("n_major", 0)) == 0
                and pd.isna(last_major_date)
            ):
                last_major_date = prediction_start
                medium_since_last_major = 0
            event_index = 0
            medium_count = 0
            major_count = 0
            maintenance_dates: list[pd.Timestamp] = []
            local_path_rows: list[dict[str, object]] = []
            local_schedule_rows: list[dict[str, object]] = []
            service_cap_scenario = str(param.get("service_cap_scenario", "neutral"))
            service_cap_years = float(param.get("service_cap_years", 15.0))

            for day_index, date in enumerate(dates):
                month = int(date.month)
                raw_hmax_t = _hmax_raw(param, hmax_trend, day_index, medium_count, major_count)
                hmax_t = max(0.0, raw_hmax_t)
                hmax_damage_cumulative = (
                    float(param.get("hmax_damage_delta_medium", 0.0)) * medium_count
                    + float(param.get("hmax_damage_delta_major", 0.0)) * major_count
                )
                maintenance_count_365d, avg_interval_365d, min_interval_365d = _maintenance_window_metrics(
                    maintenance_dates, date
                )
                service_age = _service_age_years(date)
                candidates: list[str] = []
                if raw_hmax_t <= 0:
                    candidates.append("hmax_depleted")
                if _next_year_max_recoverable(
                    param, hmax_trend, day_index, date, medium_count, major_count, helpers
                ) < THRESHOLD:
                    candidates.append("unrecoverable_below_37")
                if service_age >= service_cap_years:
                    candidates.append("service_cap_reached")
                maintenance_type, rule_type = decide_maintenance(
                    param,
                    date,
                    x_history,
                    last_maintenance_date,
                    last_major_date,
                    medium_since_last_major,
                    use_global_major_fallback=use_global_major_fallback,
                )
                rho_used_source = ""
                if maintenance_type and not any(
                    item in candidates for item in ["hmax_depleted", "unrecoverable_below_37"]
                ):
                    projected_count, projected_avg, projected_min, projected_burden = _maintenance_burden_after_add(
                        maintenance_dates, date
                    )
                    if projected_burden:
                        candidates.append("excessive_maintenance_burden")
                        maintenance_count_365d = max(projected_count - 1, 0)
                        avg_interval_365d = projected_avg
                        min_interval_365d = projected_min

                failure_type = _best_failure(candidates)
                is_terminal_day = bool(failure_type)
                if maintenance_type and not is_terminal_day:
                    if model_family == "fixed":
                        x_state = apply_fixed_gain(param, maintenance_type, month, x_state, hmax_t, helpers)
                        rho_used_source = "fixed_gain_model"
                    else:
                        x_state, rho_used_source = apply_recovery_ratio(
                            param, maintenance_type, month, x_state, hmax_t, helpers
                        )
                        if rho_used_source == "fixed_gain_fallback":
                            rule_type = f"{rule_type}_rho_fallback_fixed_gain"
                    last_maintenance_date = date
                    if maintenance_type == "major":
                        last_major_date = date
                        medium_since_last_major = 0
                        major_count += 1
                    else:
                        medium_since_last_major += 1
                        medium_count += 1
                    maintenance_dates.append(date)
                    event_index += 1
                    record = schedule_record(device_id, event_index, date, maintenance_type, rule_type)
                    record["model_name"] = model_name
                    record["hmax_scenario"] = hmax_scenario
                    record["service_cap_scenario"] = service_cap_scenario
                    record["rho_used_source"] = rho_used_source
                    local_schedule_rows.append(record)
                    raw_hmax_t = _hmax_raw(param, hmax_trend, day_index, medium_count, major_count)
                    hmax_t = max(0.0, raw_hmax_t)
                    hmax_damage_cumulative = (
                        float(param.get("hmax_damage_delta_medium", 0.0)) * medium_count
                        + float(param.get("hmax_damage_delta_major", 0.0)) * major_count
                    )
                    maintenance_count_365d, avg_interval_365d, min_interval_365d = _maintenance_window_metrics(
                        maintenance_dates, date
                    )
                elif not is_terminal_day:
                    decay_lambda = helpers["decay_lambda"].get(month, 1.0)
                    x_state = x_state + float(param["cycle_decay_rate_used"]) * decay_lambda

                seasonal = helpers["seasonal_level"].get(month, 0.0)
                predicted = x_state + seasonal
                rolling_values.append(predicted)
                if len(rolling_values) > 365:
                    rolling_values = rolling_values[-365:]
                rolling365 = float(np.mean(rolling_values)) if len(rolling_values) >= 365 else math.nan
                x_history.append(x_state)
                if len(x_history) > 60:
                    x_history = x_history[-60:]
                local_path_rows.append(
                    {
                        "model_name": model_name,
                        "hmax_scenario": hmax_scenario,
                        "service_cap_scenario": service_cap_scenario,
                        "device_id": device_id,
                        "date": date,
                        "x_state": x_state,
                        "seasonal_level": seasonal,
                        "predicted_permeability": predicted,
                        "rolling365_pred": rolling365,
                        "is_maintenance_day": bool(maintenance_type),
                        "maintenance_type": maintenance_type,
                        "rho_used_source": rho_used_source,
                        "hmax_t": hmax_t,
                        "hmax_trend_used": hmax_trend,
                        "hmax_annual_drop_ratio_used": hmax_annual_ratio,
                        "hmax_damage_cumulative": hmax_damage_cumulative,
                        "service_age_years": service_age,
                        "maintenance_count_365d": maintenance_count_365d,
                        "avg_maintenance_interval_365d": avg_interval_365d,
                        "min_maintenance_interval_365d": min_interval_365d,
                        "is_terminal_day": is_terminal_day,
                        "failure_type": failure_type,
                    }
                )
                if is_terminal_day:
                    break
            local_path = _truncate_at_final_failure(pd.DataFrame(local_path_rows), horizon_years * 365)
            if not local_path.empty:
                terminal_dates = local_path.loc[local_path["is_terminal_day"].astype(bool), "date"]
                if len(terminal_dates):
                    cutoff = pd.Timestamp(terminal_dates.iloc[0])
                    local_schedule_rows = [
                        row for row in local_schedule_rows if pd.Timestamp(row["date"]) <= cutoff
                    ]
                path_rows.extend(local_path.to_dict("records"))
                schedule_rows.extend(local_schedule_rows)
    return pd.DataFrame(path_rows), pd.DataFrame(schedule_rows)
