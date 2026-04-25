from __future__ import annotations

import math

import numpy as np
import pandas as pd


THRESHOLD = 37.0
LIFETIME_TEST_EXTRA_DAYS = 365


def current_real_rolling365(daily: pd.DataFrame, device_id: str) -> float:
    group = daily[(daily["device_id"] == device_id) & daily["daily_median"].notna()].sort_values("date")
    values = group["daily_median"].astype(float).tail(365)
    return float(values.mean()) if len(values) else math.nan


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
        horizon_days = int(horizon_years * 365)
        within = group.iloc[:horizon_days].copy()
        first_below = within.loc[within["rolling365_pred"] < THRESHOLD, "date"]
        lifetime_date = pd.NaT
        first_below_date = pd.NaT
        m_at_test = np.nan
        status = f"not_reached_within_{horizon_years}y"
        if len(first_below):
            for date in first_below:
                first_below_date = date if pd.isna(first_below_date) else first_below_date
                lookahead = group[(group["date"] >= date) & (group["date"] <= date + pd.Timedelta(days=365))]
                m_value = float(lookahead["rolling365_pred"].max()) if len(lookahead) else np.nan
                if np.isfinite(m_value) and m_value < THRESHOLD:
                    lifetime_date = date
                    m_at_test = m_value
                    status = "lifetime_end"
                    break
        param = params[params["device_id"] == device_id].iloc[0]
        commission = pd.Timestamp("2022-04-01")
        age_start = (prediction_start - commission).days / 365.25
        if pd.notna(lifetime_date):
            remaining = (lifetime_date - prediction_start).days / 365.25
            total = age_start + remaining
            is_censored_30y = False if horizon_years == 30 else np.nan
        else:
            remaining = f">{horizon_years}"
            total = f">{age_start + horizon_years:.2f}"
            lifetime_date = f">{horizon_years}年"
            is_censored_30y = True if horizon_years == 30 else np.nan
        rows.append(
            {
                "model_name": model_name,
                "hmax_scenario": hmax_scenario,
                "device_id": device_id,
                "commission_date": commission,
                "prediction_start_date": prediction_start,
                "age_at_prediction_start_years": age_start,
                "current_real_rolling365": current_real_rolling365(daily, device_id),
                "first_date_rolling365_below_37": first_below_date,
                "lifetime_end_date": lifetime_date,
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
        }
    )
    result.insert(5, "extended_horizon_years", extended_horizon_years)
    drop_columns = [column for column in ["is_censored_30y"] if column in result.columns]
    return result.drop(columns=drop_columns)
