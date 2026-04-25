from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .build_daily_features import usable_daily
from .load_inputs import sort_by_device, sorted_device_ids


EXCLUDE_EVENT_WINDOW_DAYS = 3
MIN_SEGMENT_DAYS = 5
MAX_DAILY_UP_JUMP = 8.0
EPS = 1e-6


def _period_stage(date: pd.Timestamp, device_dates: pd.Series) -> str:
    midpoint = pd.to_datetime(device_dates.min()) + (pd.to_datetime(device_dates.max()) - pd.to_datetime(device_dates.min())) / 2
    return "early" if date <= midpoint else "late"


def build_pure_decay_segments(daily: pd.DataFrame, maintenance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = usable_daily(daily)
    rows: list[dict[str, object]] = []

    for device_id, device_daily in daily.groupby("device_id", sort=False):
        device_usable = usable[usable["device_id"] == device_id].sort_values("date").reset_index(drop=True)
        if device_usable.empty:
            continue

        event_dates = maintenance.loc[maintenance["device_id"] == device_id, "event_date"].sort_values().tolist()
        excluded_dates: set[pd.Timestamp] = set()
        for event_date in event_dates:
            for offset in range(-EXCLUDE_EVENT_WINDOW_DAYS, EXCLUDE_EVENT_WINDOW_DAYS + 1):
                excluded_dates.add(event_date + pd.Timedelta(days=offset))

        segment: list[pd.Series] = []
        segment_id = 1

        def flush_segment(current: list[pd.Series], reason: str = "") -> None:
            nonlocal segment_id
            if len(current) < MIN_SEGMENT_DAYS:
                return
            frame = pd.DataFrame(current)
            start = frame.iloc[0]
            end = frame.iloc[-1]
            length = int((end["date"] - start["date"]).days)
            if length <= 0:
                return
            start_level = float(start["daily_median"])
            end_level = float(end["daily_median"])
            drop = end_level - start_level
            decay_rate = drop / length
            eligible = decay_rate <= 0
            rows.append(
                {
                    "device_id": device_id,
                    "segment_id": segment_id,
                    "start_date": start["date"],
                    "end_date": end["date"],
                    "length_days": length,
                    "start_level": start_level,
                    "end_level": end_level,
                    "drop": drop,
                    "decay_rate": decay_rate if eligible else np.nan,
                    "month": int(start["date"].month),
                    "period_stage": _period_stage(start["date"], device_daily["date"]),
                    "eligible_pure_decay": bool(eligible),
                    "ineligible_reason": "" if eligible else (reason or "non_declining_segment"),
                }
            )
            segment_id += 1

        previous: pd.Series | None = None
        for _, row in device_usable.iterrows():
            date = row["date"]
            value = float(row["daily_median"])
            break_segment = False
            reason = ""

            if date in excluded_dates:
                break_segment = True
                reason = "near_medium_or_major_maintenance"
            elif previous is not None:
                day_gap = int((date - previous["date"]).days)
                jump = value - float(previous["daily_median"])
                if day_gap != 1:
                    break_segment = True
                    reason = "non_continuous_dates"
                elif jump > MAX_DAILY_UP_JUMP:
                    break_segment = True
                    reason = "large_positive_jump_check_only"

            if break_segment:
                flush_segment(segment, reason)
                segment = []
                previous = None
                if date not in excluded_dates:
                    segment.append(row)
                    previous = row
                continue

            segment.append(row)
            previous = row

        flush_segment(segment)

    segments = sort_by_device(pd.DataFrame(rows), "start_date") if rows else pd.DataFrame()
    summary_rows: list[dict[str, object]] = []
    for device_id in sorted_device_ids(daily["device_id"].drop_duplicates()):
        group = segments[(segments["device_id"] == device_id) & (segments["eligible_pure_decay"].astype(bool))] if not segments.empty else pd.DataFrame()
        rates = group["decay_rate"].dropna().astype(float) if len(group) else pd.Series(dtype=float)
        early = group.loc[group["period_stage"] == "early", "decay_rate"].dropna().astype(float) if len(group) else pd.Series(dtype=float)
        late = group.loc[group["period_stage"] == "late", "decay_rate"].dropna().astype(float) if len(group) else pd.Series(dtype=float)
        early_med = float(early.median()) if len(early) else np.nan
        late_med = float(late.median()) if len(late) else np.nan
        if np.isfinite(early_med) and np.isfinite(late_med):
            acceleration = (abs(late_med) - abs(early_med)) / (abs(early_med) + EPS)
        else:
            acceleration = np.nan
        monthly_counts = group.groupby("month").size() if len(group) else pd.Series(dtype=int)
        summary_rows.append(
            {
                "device_id": device_id,
                "n_pure_decay_segments": int(len(group)),
                "pure_decay_rate_median": float(rates.median()) if len(rates) else np.nan,
                "pure_decay_rate_q25": float(rates.quantile(0.25)) if len(rates) else np.nan,
                "pure_decay_rate_q75": float(rates.quantile(0.75)) if len(rates) else np.nan,
                "early_pure_decay_rate_median": early_med,
                "late_pure_decay_rate_median": late_med,
                "aging_acceleration_ratio": acceleration,
                "monthly_decay_sensitive": bool((monthly_counts >= 2).sum() >= 3),
            }
        )
    return segments, sort_by_device(pd.DataFrame(summary_rows), "device_id")
