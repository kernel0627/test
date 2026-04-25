from __future__ import annotations

import math

import numpy as np
import pandas as pd


THRESHOLD = 37.0


def current_real_rolling365(daily: pd.DataFrame, device_id: str) -> float:
    group = daily[(daily["device_id"] == device_id) & daily["daily_median"].notna()].sort_values("date")
    values = group["daily_median"].astype(float).tail(365)
    return float(values.mean()) if len(values) else math.nan


def build_lifetime_predictions(
    paths_df: pd.DataFrame,
    params: pd.DataFrame,
    daily: pd.DataFrame,
    prediction_start: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, device_id), group in paths_df.groupby(["model_name", "device_id"], sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        hmax_scenario = group["hmax_scenario"].iloc[0] if "hmax_scenario" in group.columns and len(group) else "neutral"
        within = group.iloc[: 30 * 365].copy()
        first_below = within.loc[within["rolling365_pred"] < THRESHOLD, "date"]
        lifetime_date = pd.NaT
        first_below_date = pd.NaT
        m_at_test = np.nan
        status = "not_reached_within_horizon"
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
        else:
            remaining = np.nan
            total = np.nan
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
                "remaining_life_years": remaining,
                "predicted_total_life_years": total,
                "status": status,
                "M_at_lifetime_test": m_at_test,
            }
        )
    return pd.DataFrame(rows)
