from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .build_daily_features import usable_daily
from .load_inputs import sort_by_device, sorted_device_ids


PRE_DECAY_WINDOW_DAYS = 14


def _fmt_sequence(values) -> str:
    return "|".join("" if pd.isna(value) else str(int(value)) for value in values)


def _pre_decay_speed(daily: pd.DataFrame, event_date: pd.Timestamp, window_days: int = PRE_DECAY_WINDOW_DAYS) -> float:
    start = event_date - pd.Timedelta(days=window_days)
    rows = daily[(daily["date"] >= start) & (daily["date"] < event_date)].sort_values("date")
    rows = rows[rows["daily_median"].notna()]
    if len(rows) < 2:
        return math.nan
    first = rows.iloc[0]
    last = rows.iloc[-1]
    days = int((last["date"] - first["date"]).days)
    if days <= 0:
        return math.nan
    return float(-(float(last["daily_median"]) - float(first["daily_median"])) / days)


def _medium_counts_between_major(events: pd.DataFrame) -> list[int]:
    major_positions = events.index[events["maintenance_type"] == "major"].tolist()
    counts: list[int] = []
    for left, right in zip(major_positions[:-1], major_positions[1:]):
        between = events.loc[left + 1 : right - 1]
        counts.append(int((between["maintenance_type"] == "medium").sum()))
    return counts


def build_current_maintenance_rule(
    maintenance: pd.DataFrame,
    device_ids: list[str],
    daily: pd.DataFrame,
) -> pd.DataFrame:
    global_maintenance_intervals: list[float] = []
    global_major_count = int((maintenance["maintenance_type"] == "major").sum())
    global_total_count = int(len(maintenance))
    for _, group in maintenance.groupby("device_id", sort=False):
        events = group.sort_values("event_date").reset_index(drop=True)
        global_maintenance_intervals.extend(events["event_date"].diff().dt.days.dropna().astype(float).tolist())

    global_maintenance_interval_median = (
        float(pd.Series(global_maintenance_intervals).median())
        if global_maintenance_intervals
        else math.nan
    )
    global_major_event_count = global_total_count / global_major_count if global_major_count else math.nan
    global_major_interval_median = (
        global_major_event_count * global_maintenance_interval_median
        if np.isfinite(global_major_event_count) and np.isfinite(global_maintenance_interval_median)
        else math.nan
    )
    global_medium_count_between_major = global_major_event_count if np.isfinite(global_major_event_count) else math.nan

    usable = usable_daily(daily)
    rows: list[dict[str, object]] = []
    for device_id in sorted_device_ids(device_ids):
        events = maintenance[maintenance["device_id"] == device_id].sort_values("event_date").reset_index(drop=True)
        device_daily = usable[usable["device_id"] == device_id].copy()
        medium_events = events[events["maintenance_type"] == "medium"].copy()
        major_events = events[events["maintenance_type"] == "major"].copy()

        medium_intervals = medium_events["event_date"].diff().dt.days.dropna().astype(float)
        medium_pre_speeds = [
            _pre_decay_speed(device_daily, event_date)
            for event_date in medium_events["event_date"]
        ]
        medium_pre_speeds = pd.Series(medium_pre_speeds, dtype=float).dropna()
        major_intervals = major_events["event_date"].diff().dt.days.dropna().astype(float)
        counts_between_major = pd.Series(_medium_counts_between_major(events), dtype=float)

        n_medium = int(len(medium_events))
        n_major = int(len(major_events))
        n_total = int(len(events))
        last = events.iloc[-1] if n_total else None
        last_major = major_events.iloc[-1] if n_major else None
        if last_major is not None:
            medium_since_last_major = int(
                (
                    (events["maintenance_type"] == "medium")
                    & (events["event_date"] > last_major["event_date"])
                ).sum()
            )
        else:
            medium_since_last_major = int(n_medium)

        if n_major >= 2:
            major_rule_source = "device_history"
            major_interval_used = float(major_intervals.median())
            medium_count_between_major_used = float(counts_between_major.median()) if len(counts_between_major) else math.nan
        elif n_major == 1 and np.isfinite(global_major_interval_median):
            major_rule_source = "device_single_major_with_global_fallback"
            major_interval_used = global_major_interval_median
            medium_count_between_major_used = global_medium_count_between_major
        else:
            major_rule_source = "no_major_in_device_history"
            major_interval_used = math.nan
            medium_count_between_major_used = math.nan

        medium_interval_median = float(medium_intervals.median()) if len(medium_intervals) else math.nan
        medium_interval_q25 = float(medium_intervals.quantile(0.25)) if len(medium_intervals) else math.nan
        medium_interval_q75 = float(medium_intervals.quantile(0.75)) if len(medium_intervals) else math.nan
        medium_min_interval = float(max(7.0, medium_interval_q25)) if np.isfinite(medium_interval_q25) else 14.0
        rule_reliable = bool(
            n_medium >= 4
            and (
                n_major >= 2
                or major_rule_source in {"device_single_major_with_global_fallback", "no_major_in_device_history"}
            )
        )

        rows.append(
            {
                "device_id": device_id,
                "n_medium": n_medium,
                "n_major": n_major,
                "medium_interval_median": medium_interval_median,
                "medium_interval_q25": medium_interval_q25,
                "medium_interval_q75": medium_interval_q75,
                "medium_min_interval": medium_min_interval,
                "medium_pre_decay_speed_median": float(medium_pre_speeds.median()) if len(medium_pre_speeds) else math.nan,
                "medium_pre_decay_speed_q25": float(medium_pre_speeds.quantile(0.25)) if len(medium_pre_speeds) else math.nan,
                "medium_pre_decay_speed_q75": float(medium_pre_speeds.quantile(0.75)) if len(medium_pre_speeds) else math.nan,
                "major_interval_median": major_interval_used,
                "major_interval_q25": float(major_intervals.quantile(0.25)) if len(major_intervals) else math.nan,
                "major_interval_q75": float(major_intervals.quantile(0.75)) if len(major_intervals) else math.nan,
                "medium_count_between_major_median": medium_count_between_major_used,
                "major_ratio": n_major / n_total if n_total else 0.0,
                "major_rule_source": major_rule_source,
                "major_global_fallback_interval_days": global_major_interval_median,
                "major_global_fallback_event_count": global_medium_count_between_major,
                "rule_reliable": rule_reliable,
                "last_maintenance_date": last["event_date"] if last is not None else pd.NaT,
                "last_maintenance_type": last["maintenance_type"] if last is not None else "",
                "last_major_maintenance_date": last_major["event_date"] if last_major is not None else pd.NaT,
                "medium_since_last_major_at_end": medium_since_last_major,
                "interval_sequence": _fmt_sequence(events["event_date"].diff().dt.days.dropna().astype(float)),
                "type_sequence": "|".join(events["maintenance_type"].astype(str).tolist()),
            }
        )
    return sort_by_device(pd.DataFrame(rows), "device_id")
