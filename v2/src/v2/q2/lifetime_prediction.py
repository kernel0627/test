from __future__ import annotations

import math

import numpy as np
import pandas as pd


THRESHOLD = 37.0
LIFETIME_TEST_EXTRA_DAYS = 365
COMMISSION_DATE = pd.Timestamp("2022-04-01")
FAILURE_PRIORITY = {
    "hmax_depleted": 1,
    "unrecoverable_below_37": 2,
    "functional_failure": 4,
    "service_cap_reached": 5,
}


def current_real_rolling365(daily: pd.DataFrame, device_id: str) -> float:
    group = daily[(daily["device_id"] == device_id) & daily["daily_median"].notna()].sort_values("date")
    values = group["daily_median"].astype(float).tail(365)
    return float(values.mean()) if len(values) else math.nan


def _functional_failure_date(group: pd.DataFrame, horizon_days: int) -> tuple[pd.Timestamp | pd.NaT, float]:
    within = group.iloc[:horizon_days].copy()
    first_below = within.loc[within["rolling365_pred"].astype(float) < THRESHOLD, "date"]
    for date in first_below:
        lookahead = group[(group["date"] >= date) & (group["date"] <= date + pd.Timedelta(days=365))]
        m_value = float(lookahead["rolling365_pred"].max()) if len(lookahead) else np.nan
        if np.isfinite(m_value) and m_value < THRESHOLD:
            return pd.Timestamp(date), m_value
    return pd.NaT, np.nan


def _earliest_failure(candidates: dict[str, pd.Timestamp | pd.NaT]) -> tuple[pd.Timestamp | pd.NaT, str]:
    valid = [(name, pd.Timestamp(date)) for name, date in candidates.items() if pd.notna(date)]
    if not valid:
        return pd.NaT, ""
    valid.sort(key=lambda item: (item[1], FAILURE_PRIORITY[item[0]]))
    return valid[0][1], valid[0][0]


def _format_censored_total(age_start: float, horizon_years: int) -> str:
    return f">{age_start + horizon_years:.2f}"


def build_lifetime_predictions(
    paths_df: pd.DataFrame,
    params: pd.DataFrame,
    daily: pd.DataFrame,
    prediction_start: pd.Timestamp,
    horizon_years: int = 30,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, device_id), group in paths_df.groupby(["model_name", "device_id"], sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        hmax_scenario = group["hmax_scenario"].iloc[0] if "hmax_scenario" in group.columns and len(group) else "neutral"
        service_cap_scenario = (
            group["service_cap_scenario"].iloc[0]
            if "service_cap_scenario" in group.columns and len(group)
            else "neutral"
        )
        horizon_days = int(horizon_years * 365)
        param = params[params["device_id"] == device_id].iloc[0]
        commission = COMMISSION_DATE
        age_start = (prediction_start - commission).days / 365.25
        functional_date, m_at_test = _functional_failure_date(group, horizon_days)
        terminal = group[group.get("is_terminal_day", pd.Series(False, index=group.index)).astype(bool)]
        terminal_date = pd.Timestamp(terminal["date"].iloc[0]) if len(terminal) else pd.NaT
        terminal_type = str(terminal["failure_type"].iloc[0]) if len(terminal) else ""
        recovery_date = terminal_date if terminal_type in {"hmax_depleted", "unrecoverable_below_37"} else pd.NaT
        if "maintenance_burden_flag" in group.columns:
            burden_rows = group[group["maintenance_burden_flag"].fillna(False).astype(bool)]
        else:
            burden_rows = pd.DataFrame()
        if len(burden_rows):
            burden_warning_date = pd.to_datetime(
                burden_rows["maintenance_burden_warning_date"].dropna().iloc[0],
                errors="coerce",
            )
            burden_warning_flag = True
            burden_reason = str(burden_rows["maintenance_burden_reason"].dropna().iloc[0])
            burden_trigger_annual_count = burden_rows["burden_trigger_annual_count"].dropna().iloc[0]
            burden_trigger_avg_interval = burden_rows["burden_trigger_avg_interval"].dropna().iloc[0]
            burden_trigger_min_interval = burden_rows["burden_trigger_min_interval"].dropna().iloc[0]
        else:
            burden_warning_date = pd.NaT
            burden_warning_flag = False
            burden_reason = ""
            burden_trigger_annual_count = np.nan
            burden_trigger_avg_interval = np.nan
            burden_trigger_min_interval = np.nan
        service_cap_date = commission + pd.to_timedelta(float(param.get("service_cap_years", 15.0)) * 365.25, unit="D")
        if service_cap_date > prediction_start + pd.Timedelta(days=horizon_days):
            service_cap_date = pd.NaT
        candidates = {
            "functional_failure": functional_date,
            "hmax_depleted": terminal_date if terminal_type == "hmax_depleted" else pd.NaT,
            "unrecoverable_below_37": terminal_date if terminal_type == "unrecoverable_below_37" else pd.NaT,
            "service_cap_reached": service_cap_date,
        }
        final_date, final_type = _earliest_failure(candidates)
        status = "lifetime_end" if pd.notna(final_date) else f"not_reached_within_{horizon_years}y"
        if pd.notna(final_date):
            remaining = (final_date - prediction_start).days / 365.25
            total = age_start + remaining
            is_censored_30y = False if horizon_years == 30 else np.nan
        else:
            remaining = f">{horizon_years}"
            total = _format_censored_total(age_start, horizon_years)
            is_censored_30y = True if horizon_years == 30 else np.nan
        final_output_date = final_date if pd.notna(final_date) else f">{horizon_years}年"
        rows.append(
            {
                "model_name": model_name,
                "hmax_scenario": hmax_scenario,
                "service_cap_scenario": service_cap_scenario,
                "device_id": device_id,
                "commission_date": commission,
                "prediction_start_date": prediction_start,
                "age_at_prediction_start_years": age_start,
                "current_real_rolling365": current_real_rolling365(daily, device_id),
                "first_date_rolling365_below_37": functional_date,
                "functional_failure_date": functional_date,
                "recovery_failure_date": recovery_date,
                "maintenance_burden_failure_date": pd.NaT,
                "maintenance_burden_warning_date": burden_warning_date,
                "maintenance_burden_flag": burden_warning_flag,
                "maintenance_burden_reason": burden_reason,
                "burden_trigger_annual_count": burden_trigger_annual_count,
                "burden_trigger_avg_interval": burden_trigger_avg_interval,
                "burden_trigger_min_interval": burden_trigger_min_interval,
                "service_cap_date": service_cap_date,
                "final_lifetime_end_date": final_output_date,
                "final_remaining_life_years": remaining,
                "final_failure_type": final_type if final_type else "",
                "lifetime_end_date": final_output_date,
                "is_censored_30y": is_censored_30y,
                "remaining_life_years": remaining,
                "predicted_total_life_years": total,
                "status": status,
                "M_at_lifetime_test": m_at_test,
            }
        )
    return pd.DataFrame(rows)


def build_extended_lifetime_predictions(
    paths_df: pd.DataFrame,
    params: pd.DataFrame,
    daily: pd.DataFrame,
    prediction_start: pd.Timestamp,
    extended_horizon_years: int = 100,
) -> pd.DataFrame:
    result = build_lifetime_predictions(
        paths_df,
        params,
        daily,
        prediction_start,
        horizon_years=extended_horizon_years,
    )
    if result.empty:
        return result
    result = result.rename(
        columns={
            "first_date_rolling365_below_37": "extended_first_date_rolling365_below_37",
            "lifetime_end_date": "extended_lifetime_end_date",
            "remaining_life_years": "extended_remaining_life_years",
            "predicted_total_life_years": "extended_predicted_total_life_years",
            "status": "extended_status",
            "M_at_lifetime_test": "extended_M_at_lifetime_test",
            "functional_failure_date": "extended_functional_failure_date",
            "recovery_failure_date": "extended_recovery_failure_date",
            "maintenance_burden_failure_date": "extended_maintenance_burden_failure_date",
            "maintenance_burden_warning_date": "extended_maintenance_burden_warning_date",
            "maintenance_burden_flag": "extended_maintenance_burden_flag",
            "maintenance_burden_reason": "extended_maintenance_burden_reason",
            "burden_trigger_annual_count": "extended_burden_trigger_annual_count",
            "burden_trigger_avg_interval": "extended_burden_trigger_avg_interval",
            "burden_trigger_min_interval": "extended_burden_trigger_min_interval",
            "service_cap_date": "extended_service_cap_date",
            "final_lifetime_end_date": "extended_final_lifetime_end_date",
            "final_remaining_life_years": "extended_final_remaining_life_years",
            "final_failure_type": "extended_final_failure_type",
        }
    )
    result.insert(5, "extended_horizon_years", extended_horizon_years)
    drop_columns = [column for column in ["is_censored_30y"] if column in result.columns]
    return result.drop(columns=drop_columns)
