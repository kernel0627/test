from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .load_inputs import sort_by_device


VALID_QUALITIES = {"high_quality", "low_quality"}


def _window_values(daily: pd.DataFrame, event_date: pd.Timestamp, offsets: range) -> pd.Series:
    dates = {event_date + pd.Timedelta(days=offset) for offset in offsets}
    rows = daily[
        daily["date"].isin(dates)
        & daily["daily_quality"].isin(VALID_QUALITIES)
        & daily["daily_median"].notna()
        & (~daily["is_maintenance_day"].astype(bool))
    ]
    return rows["daily_median"].astype(float)


def build_cycles(daily: pd.DataFrame, maintenance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for device_id, events in maintenance.groupby("device_id", sort=False):
        events = events.sort_values("event_date").reset_index(drop=True)
        device_daily = daily[daily["device_id"] == device_id].copy()
        for idx in range(len(events) - 1):
            start = events.iloc[idx]
            nxt = events.iloc[idx + 1]
            length = int((nxt["event_date"] - start["event_date"]).days)
            start_values = _window_values(device_daily, start["event_date"], range(1, 4))
            end_values = _window_values(device_daily, nxt["event_date"], range(-3, 0))
            start_level = float(start_values.median()) if len(start_values) else math.nan
            end_level = float(end_values.median()) if len(end_values) else math.nan
            reasons: list[str] = []
            if length < 7:
                reasons.append("cycle_length_lt_7")
            if len(start_values) < 2:
                reasons.append("post_window_valid_days_lt_2")
            if len(end_values) < 2:
                reasons.append("pre_next_window_valid_days_lt_2")
            eligible = len(reasons) == 0
            cycle_drop = end_level - start_level if eligible else math.nan
            decay_rate = cycle_drop / length if eligible and length > 0 else math.nan
            rows.append(
                {
                    "device_id": device_id,
                    "cycle_id": idx + 1,
                    "start_event_date": start["event_date"],
                    "start_maintenance_type": start["maintenance_type"],
                    "next_event_date": nxt["event_date"],
                    "next_maintenance_type": nxt["maintenance_type"],
                    "cycle_length_days": length,
                    "cycle_start_level": start_level,
                    "cycle_end_level": end_level,
                    "cycle_drop": cycle_drop,
                    "cycle_decay_rate": decay_rate,
                    "eligible_for_cycle_analysis": eligible,
                    "ineligible_reason": "|".join(reasons),
                }
            )

    cycles = sort_by_device(pd.DataFrame(rows), "start_event_date") if rows else pd.DataFrame()
    summary_rows: list[dict[str, object]] = []
    for device_id in daily["device_id"].drop_duplicates():
        group = cycles[(cycles["device_id"] == device_id) & (cycles["eligible_for_cycle_analysis"].astype(bool))] if not cycles.empty else pd.DataFrame()
        rates = group["cycle_decay_rate"].astype(float) if len(group) else pd.Series(dtype=float)
        drops = group["cycle_drop"].astype(float) if len(group) else pd.Series(dtype=float)
        lengths = group["cycle_length_days"].astype(float) if len(group) else pd.Series(dtype=float)
        summary_rows.append(
            {
                "device_id": device_id,
                "n_valid_cycles": int(len(group)),
                "cycle_length_median": float(lengths.median()) if len(lengths) else np.nan,
                "cycle_decay_rate_mean": float(rates.mean()) if len(rates) else np.nan,
                "cycle_decay_rate_median": float(rates.median()) if len(rates) else np.nan,
                "cycle_decay_rate_q25": float(rates.quantile(0.25)) if len(rates) else np.nan,
                "cycle_decay_rate_q75": float(rates.quantile(0.75)) if len(rates) else np.nan,
                "cycle_drop_median": float(drops.median()) if len(drops) else np.nan,
            }
        )
    return cycles, sort_by_device(pd.DataFrame(summary_rows), "device_id")
