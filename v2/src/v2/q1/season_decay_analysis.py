from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .build_daily_features import COMMISSION_DATE, usable_daily
from .load_inputs import sort_by_device, sorted_device_ids
from .season_time_analysis import _device_dummies, _month_dummies, _ols


def _safe_quantile(series: pd.Series, q: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.quantile(q)) if len(clean) else math.nan


def build_data_overview_v2(daily: pd.DataFrame) -> pd.DataFrame:
    usable = usable_daily(daily)
    rows: list[dict[str, object]] = []
    for device_id, group in daily.groupby("device_id", sort=False):
        usable_group = usable[usable["device_id"] == device_id].sort_values("date")
        values = usable_group["daily_median"].astype(float)
        recent = usable_group.tail(14)["daily_median"].astype(float)
        start_date = group["date"].min()
        end_date = group["date"].max()
        rows.append(
            {
                "device_id": device_id,
                "start_date": start_date,
                "end_date": end_date,
                "commission_date": COMMISSION_DATE,
                "age_days_at_start": int((start_date - COMMISSION_DATE).days),
                "age_days_at_end": int((end_date - COMMISSION_DATE).days),
                "n_valid_days": int(values.count()),
                "n_maintenance_gap_days": int((group["daily_quality"] == "maintenance_gap").sum()),
                "n_random_gap_days": int((group["daily_quality"] == "random_gap").sum()),
                "alpha_median": float(values.median()) if len(values) else math.nan,
                "alpha_mean": float(values.mean()) if len(values) else math.nan,
                "daily_median_std": float(values.std(ddof=0)) if len(values) >= 2 else math.nan,
                "daily_median_min": float(values.min()) if len(values) else math.nan,
                "daily_median_max": float(values.max()) if len(values) else math.nan,
                "current_state_level": float(recent.median()) if len(recent) else math.nan,
                "n_days_below_37": int((values < 37).sum()),
            }
        )
    return sort_by_device(pd.DataFrame(rows), "device_id")


def build_season_decay_table(
    daily: pd.DataFrame,
    pure_segments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = usable_daily(daily)
    usable = usable.dropna(subset=["daily_median", "days_since_last_maintenance", "days_from_observation_start"]).copy()
    if usable.empty:
        raise ValueError("No usable daily rows for season/time analysis.")

    alpha = usable.groupby("device_id")["daily_median"].median().rename("alpha_median")
    usable = usable.merge(alpha, on="device_id", how="left")
    usable["centered_per"] = usable["daily_median"] - usable["alpha_median"]

    raw = (
        usable.groupby("month")
        .agg(
            seasonal_index_raw=("centered_per", "median"),
            n_samples=("daily_median", "size"),
            median_centered_per=("centered_per", "median"),
            q25_centered_per=("centered_per", lambda s: _safe_quantile(s, 0.25)),
            q75_centered_per=("centered_per", lambda s: _safe_quantile(s, 0.75)),
        )
        .reindex(range(1, 13))
        .reset_index()
    )

    design_no_month = usable[["days_since_last_maintenance", "days_from_observation_start"]].copy()
    design_no_month = pd.concat([design_no_month, _device_dummies(usable["device_id"])], axis=1)
    design_no_month.insert(0, "const", 1.0)
    _, residual_no_month, _ = _ols(usable["daily_median"].to_numpy(dtype=float), design_no_month)
    usable["residual_no_month"] = residual_no_month
    adjusted = (
        usable.groupby("month")
        .agg(seasonal_index_adjusted=("residual_no_month", "median"))
        .reindex(range(1, 13))
        .reset_index()
    )
    seasonal = raw.merge(adjusted, on="month", how="left")
    seasonal["seasonal_level_used"] = seasonal["seasonal_index_adjusted"]
    center = seasonal["seasonal_level_used"].dropna().mean()
    seasonal["seasonal_level_used"] = seasonal["seasonal_level_used"] - center

    design = usable[["days_since_last_maintenance", "days_from_observation_start"]].copy()
    design = pd.concat([design, _device_dummies(usable["device_id"]), _month_dummies(usable["month"])], axis=1)
    design.insert(0, "const", 1.0)
    regression, _, _ = _ols(usable["daily_median"].to_numpy(dtype=float), design)

    if pure_segments.empty:
        monthly_decay = pd.DataFrame({"month": range(1, 13), "value": np.nan, "n_samples": 0})
    else:
        eligible = pure_segments[pure_segments["eligible_pure_decay"].astype(bool)].copy()
        eligible["decay_intensity"] = -eligible["decay_rate"].astype(float)
        monthly_decay = (
            eligible.groupby("month")
            .agg(value=("decay_intensity", "median"), n_samples=("decay_intensity", "size"))
            .reindex(range(1, 13))
            .reset_index()
        )

    seasonal_rows = pd.DataFrame(
        {
            "section": "seasonal_level",
            "term": "seasonal_level_used",
            "month": seasonal["month"],
            "value": seasonal["seasonal_level_used"],
            "coef": np.nan,
            "std_err": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "n_samples": seasonal["n_samples"],
            "r_squared": np.nan,
            "model_name": "median_residual_centered_month_level",
        }
    )
    monthly_decay_rows = pd.DataFrame(
        {
            "section": "monthly_decay_intensity",
            "term": "monthly_decay_intensity",
            "month": monthly_decay["month"],
            "value": monthly_decay["value"],
            "coef": np.nan,
            "std_err": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "n_samples": monthly_decay["n_samples"],
            "r_squared": np.nan,
            "model_name": "pure_decay_segment_month_median",
        }
    )
    regression_rows = regression.assign(section="regression", month=np.nan, value=np.nan)[
        ["section", "term", "month", "value", "coef", "std_err", "t_value", "p_value", "n_samples", "r_squared", "model_name"]
    ]
    combined = pd.concat(
        [
            seasonal_rows,
            monthly_decay_rows,
            regression_rows,
        ],
        ignore_index=True,
    )
    return combined, seasonal
