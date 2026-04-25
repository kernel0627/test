from __future__ import annotations

import math

import numpy as np
import pandas as pd


def recent_decline_speed(x_history: list[float], window_days: int = 14) -> float:
    if len(x_history) <= window_days:
        return 0.0
    return float(-(x_history[-1] - x_history[-1 - window_days]) / window_days)


def decide_maintenance(
    row: pd.Series,
    current_date: pd.Timestamp,
    x_history: list[float],
    last_maintenance_date: pd.Timestamp,
    last_major_date: pd.Timestamp | pd.NaT,
    medium_since_last_major: int,
    *,
    use_global_major_fallback: bool = False,
) -> tuple[str, str]:
    tau = int((current_date - last_maintenance_date).days) if pd.notna(last_maintenance_date) else 10**9
    tau_major = int((current_date - last_major_date).days) if pd.notna(last_major_date) else 10**9

    major_due_time = False
    major_due_count = False
    major_source = str(row["major_rule_source"])
    n_major = int(row.get("n_major", 0)) if pd.notna(row.get("n_major", np.nan)) else 0
    source_suffix = major_source
    count_disabled_reason = ""
    if use_global_major_fallback and n_major < 2:
        major_interval = row.get("major_global_fallback_interval_days", np.nan)
        major_count = row.get("major_global_fallback_event_count", np.nan)
        source_suffix = "global_fallback_sensitivity"
    elif n_major >= 2 and major_source != "no_major_in_device_history":
        major_interval = row["major_interval_median"]
        major_count = row["medium_count_between_major_median"]
        source_suffix = major_source
    else:
        major_interval = np.nan
        major_count = np.nan
        if n_major == 1:
            count_disabled_reason = "single_major_no_device_interval"

    if np.isfinite(major_interval):
        major_due_time = tau_major >= int(major_interval)
    if np.isfinite(major_count):
        rounded_count = int(round(float(major_count)))
        if rounded_count <= 0:
            count_disabled_reason = "major_count_disabled_zero_k"
        else:
            major_due_count = medium_since_last_major >= rounded_count
    if major_due_time or major_due_count:
        trigger = "triggered_major_time" if major_due_time else "triggered_major_medium_count"
        if count_disabled_reason and major_due_time:
            return "major", f"{trigger}_{source_suffix}_{count_disabled_reason}"
        return "major", f"{trigger}_{source_suffix}"

    speed = recent_decline_speed(x_history)
    medium_due_time = tau >= int(row["maintenance_interval_median"])
    medium_due_speed = (
        speed >= float(row["medium_pre_decay_speed_median"])
        and tau >= int(row["medium_min_interval"])
    )
    if medium_due_time or medium_due_speed:
        return "medium", "triggered_medium_time" if medium_due_time else "triggered_medium_decline_speed"
    return "", "normal_operation"


def schedule_record(
    device_id: str,
    future_event_index: int,
    date: pd.Timestamp,
    maintenance_type: str,
    rule_type: str,
) -> dict[str, object]:
    return {
        "device_id": device_id,
        "future_event_index": future_event_index,
        "date": date,
        "maintenance_type": maintenance_type,
        "rule_type": rule_type,
        "year": int(date.year),
    }
