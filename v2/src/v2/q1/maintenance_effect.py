from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .cycle_decay import VALID_QUALITIES
from .load_inputs import sort_by_device


def _valid_rows(daily: pd.DataFrame, event_date: pd.Timestamp, offsets: range | list[int]) -> pd.DataFrame:
    dates = {event_date + pd.Timedelta(days=offset) for offset in offsets}
    return daily[
        daily["date"].isin(dates)
        & daily["daily_quality"].isin(VALID_QUALITIES)
        & daily["daily_median"].notna()
        & (~daily["is_maintenance_day"].astype(bool))
    ].sort_values("date")


def _single_day_level(daily: pd.DataFrame, event_date: pd.Timestamp, offset: int) -> float:
    rows = _valid_rows(daily, event_date, [offset])
    if rows.empty:
        return math.nan
    return float(rows["daily_median"].iloc[0])


def build_event_effects(daily: pd.DataFrame, maintenance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for _, event in maintenance.sort_values(["device_id", "event_date"]).iterrows():
        device_daily = daily[daily["device_id"] == event["device_id"]]
        event_date = event["event_date"]
        pre_rows = _valid_rows(device_daily, event_date, range(-3, 0))
        post3_rows = _valid_rows(device_daily, event_date, range(1, 4))
        post7_rows = _valid_rows(device_daily, event_date, range(1, 8))
        pre = float(pre_rows["daily_median"].median()) if len(pre_rows) else math.nan
        post_day1 = _single_day_level(device_daily, event_date, 1)
        post3 = float(post3_rows["daily_median"].median()) if len(post3_rows) else math.nan
        post_day7 = _single_day_level(device_daily, event_date, 7)

        eligible_jump = np.isfinite(pre) and np.isfinite(post_day1) and pre > 0
        eligible_plateau = np.isfinite(pre) and len(post3_rows) >= 2 and pre > 0
        eligible_post_decay = np.isfinite(post_day7) and np.isfinite(pre) and np.isfinite(post3) and abs(post3 - pre) > 1e-9
        reasons: list[str] = []
        if len(pre_rows) < 2:
            reasons.append("pre_window_valid_days_lt_2")
        if not np.isfinite(post_day1):
            reasons.append("post_day1_missing")
        if len(post3_rows) < 2:
            reasons.append("post_1_3_valid_days_lt_2")
        if len(post7_rows) < 3 or not np.isfinite(post_day7):
            reasons.append("post_1_7_valid_days_lt_3")

        jump_gain = post_day1 - pre if eligible_jump else math.nan
        plateau_gain = post3 - pre if eligible_plateau else math.nan
        hold_ratio = (post_day7 - pre) / (post3 - pre) if eligible_post_decay else math.nan
        rows.append(
            {
                "device_id": event["device_id"],
                "event_date": event_date,
                "maintenance_type": event["maintenance_type"],
                "pre_level_median_3d": pre,
                "post_day1_level": post_day1,
                "post_level_median_3d": post3,
                "jump_gain": jump_gain,
                "plateau_gain": plateau_gain,
                "jump_gain_ratio": jump_gain / pre if eligible_jump else math.nan,
                "plateau_gain_ratio": plateau_gain / pre if eligible_plateau else math.nan,
                "post_day7_level": post_day7,
                "post_7d_change": post_day7 - post_day1 if np.isfinite(post_day7) and np.isfinite(post_day1) else math.nan,
                "hold_ratio_7d": hold_ratio,
                "upper_recovery_ratio": math.nan,
                "eligible_jump": bool(eligible_jump),
                "eligible_plateau": bool(eligible_plateau),
                "eligible_post_decay": bool(eligible_post_decay),
                "ineligible_reason": "|".join(reasons),
            }
        )

    events = sort_by_device(pd.DataFrame(rows), "event_date")
    h_ref = (
        events.loc[events["eligible_plateau"].astype(bool)]
        .groupby("device_id")["post_level_median_3d"]
        .quantile(0.90)
        .rename("h_ref_q90")
    )
    events = events.merge(h_ref, on="device_id", how="left")
    events["upper_recovery_ratio"] = events["post_level_median_3d"].astype(float) / events["h_ref_q90"].astype(float)
    events = events.drop(columns=["h_ref_q90"])

    summary_rows: list[dict[str, object]] = []
    for maintenance_type in ["medium", "major"]:
        group = events[events["maintenance_type"] == maintenance_type]
        jump = group.loc[group["eligible_jump"], "jump_gain"].dropna().astype(float)
        jump_ratio = group.loc[group["eligible_jump"], "jump_gain_ratio"].dropna().astype(float)
        plateau = group.loc[group["eligible_plateau"], "plateau_gain"].dropna().astype(float)
        plateau_ratio = group.loc[group["eligible_plateau"], "plateau_gain_ratio"].dropna().astype(float)
        upper = group.loc[group["eligible_plateau"], "upper_recovery_ratio"].dropna().astype(float)
        hold = group.loc[group["eligible_post_decay"], "hold_ratio_7d"].dropna().astype(float)
        post_change = group.loc[group["eligible_post_decay"], "post_7d_change"].dropna().astype(float)
        summary_rows.append(
            {
                "maintenance_type": maintenance_type,
                "n_events": int(len(group)),
                "jump_gain_median": float(jump.median()) if len(jump) else np.nan,
                "jump_gain_ratio_median": float(jump_ratio.median()) if len(jump_ratio) else np.nan,
                "plateau_gain_median": float(plateau.median()) if len(plateau) else np.nan,
                "plateau_gain_ratio_median": float(plateau_ratio.median()) if len(plateau_ratio) else np.nan,
                "upper_recovery_ratio_median": float(upper.median()) if len(upper) else np.nan,
                "hold_ratio_7d_median": float(hold.median()) if len(hold) else np.nan,
                "post_7d_change_median": float(post_change.median()) if len(post_change) else np.nan,
            }
        )
    return events, pd.DataFrame(summary_rows)
