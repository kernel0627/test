from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .load_inputs import sort_by_device


UPPER_PHYSICAL_BOUND = 250.0
HIGH_QUALITY_MIN_RECORDS = 8
HIGH_QUALITY_MIN_COVERAGE = 8.0
LOW_QUALITY_MIN_RECORDS = 4
COMMISSION_DATE = pd.Timestamp("2022-04-01")


def _quality(n_valid: int, coverage: float, is_maintenance_day: bool) -> str:
    if n_valid == 0:
        return "maintenance_gap" if is_maintenance_day else "random_gap"
    if n_valid >= HIGH_QUALITY_MIN_RECORDS and coverage >= HIGH_QUALITY_MIN_COVERAGE:
        return "high_quality"
    if n_valid >= LOW_QUALITY_MIN_RECORDS:
        return "low_quality"
    return "insufficient"


def _gap_type(daily_quality: str) -> str:
    if daily_quality == "maintenance_gap":
        return "maintenance_gap"
    if daily_quality == "random_gap":
        return "random_gap"
    return "none"


def _linear_slope(y_values: pd.Series) -> float:
    y = pd.to_numeric(y_values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(y) < 2:
        return math.nan
    x = np.arange(len(y), dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0:
        return math.nan
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def clean_hourly(hourly_raw: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly_raw.copy()
    hourly["is_missing_raw"] = hourly["per_raw"].isna()
    hourly["is_invalid_physical"] = hourly["per_raw"].notna() & (
        (hourly["per_raw"] < 0) | (hourly["per_raw"] > UPPER_PHYSICAL_BOUND)
    )
    hourly["is_candidate_anomaly"] = hourly["is_invalid_physical"]
    hourly["is_excluded_from_analysis"] = hourly["is_invalid_physical"]
    hourly["anomaly_reason"] = np.where(hourly["is_invalid_physical"], "invalid_physical", "")
    hourly["per_analysis"] = hourly["per_raw"].where(~hourly["is_excluded_from_analysis"])

    # Mark isolated local spikes with a robust same-device rolling-MAD rule. The rule is
    # deliberately conservative so true low permeability states are retained.
    frames: list[pd.DataFrame] = []
    maintenance_days = set(
        zip(
            maintenance["device_id"].astype(str),
            pd.to_datetime(maintenance["event_date"]).dt.normalize(),
        )
    )
    for device_id, group in hourly.sort_values(["device_id", "time"]).groupby("device_id", sort=False):
        group = group.copy()
        values = group["per_analysis"].astype(float)
        rolling_median = values.rolling(25, center=True, min_periods=12).median()
        mad = (values - rolling_median).abs().rolling(25, center=True, min_periods=12).median()
        robust_scale = 1.4826 * mad
        z = (values - rolling_median).abs() / robust_scale.replace(0, np.nan)
        near_maintenance = group["date"].map(lambda date: (device_id, date) in maintenance_days)
        isolated = (z > 8.0) & (~near_maintenance) & values.notna()
        group.loc[isolated, "is_candidate_anomaly"] = True
        group.loc[isolated, "is_excluded_from_analysis"] = True
        group.loc[isolated, "anomaly_reason"] = "confirmed_isolated_anomaly"
        group.loc[isolated, "per_analysis"] = np.nan
        frames.append(group)

    hourly = pd.concat(frames, ignore_index=True)

    day_types = (
        maintenance.groupby(["device_id", "event_date"])["maintenance_type"]
        .agg(lambda s: "|".join(sorted(set(s))))
        .reset_index()
        .rename(columns={"event_date": "date", "maintenance_type": "maintenance_type_on_day"})
    )
    hourly = hourly.merge(day_types, on=["device_id", "date"], how="left")
    hourly["is_maintenance_day"] = hourly["maintenance_type_on_day"].notna()
    hourly["maintenance_type_on_day"] = hourly["maintenance_type_on_day"].fillna("")
    return sort_by_device(hourly, "time")


def _summarize_day(day_df: pd.DataFrame) -> dict[str, object]:
    valid = day_df.dropna(subset=["per_analysis"]).sort_values("time")
    if valid.empty:
        coverage = 0.0
    elif len(valid) == 1:
        coverage = 0.0
    else:
        coverage = float((valid["time"].iloc[-1] - valid["time"].iloc[0]).total_seconds() / 3600.0)

    values = valid["per_analysis"].astype(float)
    return {
        "daily_mean": float(values.mean()) if len(values) else math.nan,
        "daily_median": float(values.median()) if len(values) else math.nan,
        "daily_max": float(values.max()) if len(values) else math.nan,
        "daily_min": float(values.min()) if len(values) else math.nan,
        "daily_range": float(values.max() - values.min()) if len(values) else math.nan,
        "daily_std": float(values.std(ddof=0)) if len(values) >= 2 else math.nan,
        "n_total_records": int(len(day_df)),
        "n_valid_records": int(values.count()),
        "daily_coverage_hours": coverage,
    }


def build_daily_features(hourly: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    day_types = (
        maintenance.groupby(["device_id", "event_date"])["maintenance_type"]
        .agg(lambda s: "|".join(sorted(set(s))))
        .to_dict()
    )

    for device_id, group in hourly.groupby("device_id", sort=False):
        start = group["date"].min()
        end = group["date"].max()
        all_dates = pd.date_range(start, end, freq="D")
        by_date = {date: day for date, day in group.groupby("date")}
        for date in all_dates:
            day_df = by_date.get(date, group.iloc[0:0])
            summary = _summarize_day(day_df)
            maintenance_type = day_types.get((device_id, date), "")
            is_maintenance_day = bool(maintenance_type)
            quality = _quality(
                int(summary["n_valid_records"]),
                float(summary["daily_coverage_hours"]),
                is_maintenance_day,
            )
            rows.append(
                {
                    "device_id": device_id,
                    "date": date,
                    "month": int(date.month),
                    **summary,
                    "daily_quality": quality,
                    "gap_type": _gap_type(quality),
                    "is_maintenance_day": is_maintenance_day,
                    "maintenance_type_on_day": maintenance_type,
                }
            )

    daily = sort_by_device(pd.DataFrame(rows), "date")
    frames: list[pd.DataFrame] = []
    for device_id, group in daily.groupby("device_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True).copy()
        event_dates = maintenance.loc[maintenance["device_id"] == device_id, "event_date"].sort_values().to_numpy(dtype="datetime64[D]")
        dates = group["date"].to_numpy(dtype="datetime64[D]")
        if len(event_dates):
            last_idx = np.searchsorted(event_dates, dates, side="right") - 1
            next_idx = np.searchsorted(event_dates, dates, side="left")
            days_since = np.full(len(group), np.nan)
            days_to = np.full(len(group), np.nan)
            has_last = last_idx >= 0
            has_next = next_idx < len(event_dates)
            days_since[has_last] = (dates[has_last] - event_dates[last_idx[has_last]]).astype("timedelta64[D]").astype(float)
            days_to[has_next] = (event_dates[next_idx[has_next]] - dates[has_next]).astype("timedelta64[D]").astype(float)
            group["days_since_last_maintenance"] = days_since
            group["days_to_next_maintenance"] = days_to
        else:
            group["days_since_last_maintenance"] = np.nan
            group["days_to_next_maintenance"] = np.nan
        group["days_from_observation_start"] = (group["date"] - group["date"].min()).dt.days.astype(int)
        group["commission_date"] = COMMISSION_DATE
        group["age_days_from_commission"] = (group["date"] - COMMISSION_DATE).dt.days.astype(int)
        frames.append(group)

    columns = [
        "device_id",
        "date",
        "month",
        "daily_mean",
        "daily_median",
        "daily_max",
        "daily_min",
        "daily_range",
        "daily_std",
        "n_total_records",
        "n_valid_records",
        "daily_coverage_hours",
        "daily_quality",
        "gap_type",
        "is_maintenance_day",
        "maintenance_type_on_day",
        "days_since_last_maintenance",
        "days_to_next_maintenance",
        "days_from_observation_start",
        "commission_date",
        "age_days_from_commission",
    ]
    return sort_by_device(pd.concat(frames, ignore_index=True)[columns], "date")


def usable_daily(daily: pd.DataFrame) -> pd.DataFrame:
    return daily[
        daily["daily_quality"].isin(["high_quality", "low_quality"])
        & daily["daily_median"].notna()
        & (~daily["is_maintenance_day"].astype(bool))
    ].copy()
