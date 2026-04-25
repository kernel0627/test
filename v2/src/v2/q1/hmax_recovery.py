from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .load_inputs import sort_by_device, sorted_device_ids


RECENT_K = 3
DELTA_H = 5.0


def _line_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0:
        return math.nan
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _rho_stats(events: pd.DataFrame, maintenance_type: str) -> dict[str, object]:
    group = events[(events["maintenance_type"] == maintenance_type) & events["eligible_rho"].astype(bool)]
    raw = group["rho_raw"].dropna().astype(float)
    clipped = group["rho_clipped"].dropna().astype(float)
    clip_count = int(((raw < 0) | (raw > 1)).sum()) if len(raw) else 0
    clip_ratio = clip_count / len(raw) if len(raw) else np.nan
    return {
        f"{maintenance_type}_rho_median": float(clipped.median()) if len(clipped) else np.nan,
        f"{maintenance_type}_rho_clip_ratio": clip_ratio,
        f"{maintenance_type}_rho_reliable": bool(len(clipped) >= 2 and (not np.isfinite(clip_ratio) or clip_ratio <= 0.3)),
        f"n_{maintenance_type}_rho_events": int(len(clipped)),
    }


def build_hmax_recovery(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    overview: pd.DataFrame,
    device_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    h_events = events[
        events["post_level_median_3d"].notna() & events["eligible_plateau"].astype(bool)
    ].copy()
    h_events = sort_by_device(h_events, "event_date")

    summary_rows: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    for device_id in sorted_device_ids(device_ids):
        obs_start = daily.loc[daily["device_id"] == device_id, "date"].min()
        current_state = float(overview.loc[overview["device_id"] == device_id, "current_state_level"].iloc[0])
        group = h_events[h_events["device_id"] == device_id].sort_values("event_date").reset_index(drop=True).copy()
        n_points = int(len(group))
        forced_to_current = False

        if n_points:
            group["event_order"] = np.arange(1, n_points + 1)
            group["days_from_observation_start"] = (group["event_date"] - obs_start).dt.days.astype(float)
            post = group["post_level_median_3d"].astype(float)
            h_ref_q90 = float(post.quantile(0.90))
            recent = post.tail(RECENT_K)
            recent_q75 = float(recent.quantile(0.75)) if len(recent) else np.nan
            last_post = float(post.iloc[-1])
            slope_day = _line_slope(
                group["days_from_observation_start"].to_numpy(dtype=float),
                post.to_numpy(dtype=float),
            )
        else:
            group["event_order"] = []
            group["days_from_observation_start"] = []
            h_ref_q90 = np.nan
            recent_q75 = np.nan
            last_post = np.nan
            slope_day = np.nan

        candidates = [current_state]
        candidates.extend(v for v in [recent_q75, last_post] if np.isfinite(v))
        h_initial = float(max(candidates)) if candidates else current_state
        if h_initial < current_state:
            h_initial = current_state
            forced_to_current = True
        hmax_gap = h_initial - current_state
        hmax_trend_used = min(0.0, float(slope_day)) if np.isfinite(slope_day) else 0.0
        hmax_reliable = bool(n_points >= 3 and not forced_to_current and np.isfinite(h_initial))

        device_events = events[events["device_id"] == device_id].copy()
        device_events["h_ref_q90"] = h_ref_q90
        device_events["h_max_initial"] = h_initial
        denom = h_initial - device_events["pre_level_median_3d"].astype(float)
        valid = (
            device_events["eligible_plateau"].astype(bool)
            & device_events["plateau_gain"].notna()
            & denom.gt(DELTA_H)
        )
        device_events["rho_raw"] = np.where(valid, device_events["plateau_gain"].astype(float) / denom, np.nan)
        device_events["rho_clipped"] = device_events["rho_raw"].clip(lower=0, upper=1)
        device_events["eligible_rho"] = valid
        device_events["delta_H"] = DELTA_H
        event_frames.append(device_events)

        row: dict[str, object] = {
            "device_id": device_id,
            "h_ref_q90": h_ref_q90,
            "h_max_initial": h_initial,
            "current_state_level": current_state,
            "hmax_gap_to_current": hmax_gap,
            "hmax_reliable": hmax_reliable,
            "hmax_trend_slope_per_day": slope_day,
            "hmax_trend_used": hmax_trend_used,
        }
        row.update(_rho_stats(device_events, "medium"))
        row.update(_rho_stats(device_events, "major"))
        summary_rows.append(row)

    events_with_rho = sort_by_device(pd.concat(event_frames, ignore_index=True), "event_date") if event_frames else events.copy()
    recovery_summary = sort_by_device(pd.DataFrame(summary_rows), "device_id")
    return recovery_summary, events_with_rho
