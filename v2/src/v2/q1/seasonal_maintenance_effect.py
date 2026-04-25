from __future__ import annotations

import numpy as np
import pandas as pd


SEASON_GROUPS = {
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
    1: "winter",
    2: "winter",
}


def season_group_from_month(month: int) -> str:
    return SEASON_GROUPS[int(month)]


def build_seasonal_maintenance_effect(events_with_rho: pd.DataFrame) -> pd.DataFrame:
    events = events_with_rho.copy()
    events["month"] = pd.to_datetime(events["event_date"]).dt.month
    events["season_group"] = events["month"].map(SEASON_GROUPS)

    global_gain = events.groupby("maintenance_type")["plateau_gain"].median().to_dict()
    global_rho = events.loc[events["eligible_rho"].astype(bool)].groupby("maintenance_type")["rho_clipped"].median().to_dict()

    rows: list[dict[str, object]] = []
    for maintenance_type in ["medium", "major"]:
        for season_group in ["spring", "summer", "autumn", "winter"]:
            group = events[(events["maintenance_type"] == maintenance_type) & (events["season_group"] == season_group)]
            eligible_rho = group[group["eligible_rho"].astype(bool)]
            raw = eligible_rho["rho_raw"].dropna().astype(float)
            rho = eligible_rho["rho_clipped"].dropna().astype(float)
            clip_count = int(((raw < 0) | (raw > 1)).sum()) if len(raw) else 0
            clip_ratio = clip_count / len(raw) if len(raw) else np.nan
            gain = group["plateau_gain"].dropna().astype(float)
            gain_ratio = group["plateau_gain_ratio"].dropna().astype(float)
            gain_med = float(gain.median()) if len(gain) else np.nan
            rho_med = float(rho.median()) if len(rho) else np.nan
            reliable = bool(len(group) >= 3 and (not np.isfinite(clip_ratio) or clip_ratio <= 0.3))
            base_gain = global_gain.get(maintenance_type, np.nan)
            base_rho = global_rho.get(maintenance_type, np.nan)
            kappa_gain = gain_med / base_gain if reliable and np.isfinite(gain_med) and np.isfinite(base_gain) and abs(base_gain) > 1e-9 else 1.0
            kappa_rho = rho_med / base_rho if reliable and np.isfinite(rho_med) and np.isfinite(base_rho) and abs(base_rho) > 1e-9 else 1.0
            rows.append(
                {
                    "maintenance_type": maintenance_type,
                    "season_group": season_group,
                    "n_events": int(len(group)),
                    "plateau_gain_median": gain_med,
                    "plateau_gain_ratio_median": float(gain_ratio.median()) if len(gain_ratio) else np.nan,
                    "rho_median": rho_med,
                    "rho_clip_ratio": clip_ratio,
                    "kappa_gain": float(kappa_gain),
                    "kappa_rho": float(kappa_rho),
                    "season_maintenance_effect_reliable": reliable,
                }
            )
    return pd.DataFrame(rows)
