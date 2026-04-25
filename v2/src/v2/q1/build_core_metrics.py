from __future__ import annotations

import numpy as np
import pandas as pd

from .load_inputs import sort_by_device, sorted_device_ids


def _global_rho(recovery: pd.DataFrame, maintenance_type: str) -> float:
    values = recovery.loc[
        recovery[f"{maintenance_type}_rho_reliable"].astype(bool),
        f"{maintenance_type}_rho_median",
    ].dropna().astype(float)
    return float(values.median()) if len(values) else np.nan


def build_core_metrics(
    device_ids: list[str],
    overview: pd.DataFrame,
    rule: pd.DataFrame,
    decay_summary: pd.DataFrame,
    effect_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
) -> pd.DataFrame:
    medium_global_rho = _global_rho(recovery_summary, "medium")
    major_global_rho = _global_rho(recovery_summary, "major")
    effect_by_type = effect_summary.set_index("maintenance_type")
    rows: list[dict[str, object]] = []

    for device_id in sorted_device_ids(device_ids):
        overview_row = overview[overview["device_id"] == device_id].iloc[0]
        rule_row = rule[rule["device_id"] == device_id].iloc[0]
        decay_row = decay_summary[decay_summary["device_id"] == device_id].iloc[0]
        recovery_row = recovery_summary[recovery_summary["device_id"] == device_id].iloc[0]

        medium_rho_used = recovery_row["medium_rho_median"] if bool(recovery_row["medium_rho_reliable"]) else medium_global_rho
        major_rho_used = recovery_row["major_rho_median"] if bool(recovery_row["major_rho_reliable"]) else major_global_rho
        rho_reliable_flag = bool(
            recovery_row["medium_rho_reliable"]
            and (recovery_row["major_rho_reliable"] or float(rule_row["major_ratio"]) == 0)
        )

        rows.append(
            {
                "device_id": device_id,
                "alpha_median": overview_row["alpha_median"],
                "current_state_level": overview_row["current_state_level"],
                "maintenance_interval_median": rule_row["medium_interval_median"],
                "medium_interval_median": rule_row["medium_interval_median"],
                "medium_pre_decay_speed_median": rule_row["medium_pre_decay_speed_median"],
                "medium_min_interval": rule_row["medium_min_interval"],
                "major_interval_median": rule_row["major_interval_median"],
                "medium_count_between_major_median": rule_row["medium_count_between_major_median"],
                "major_ratio": rule_row["major_ratio"],
                "cycle_decay_rate_median": decay_row["cycle_decay_rate_median"],
                "pure_decay_rate_median": decay_row["pure_decay_rate_median"],
                "monthly_decay_sensitive": decay_row["monthly_decay_sensitive"],
                "medium_plateau_gain_median": effect_by_type.loc["medium", "plateau_gain_median"] if "medium" in effect_by_type.index else np.nan,
                "major_plateau_gain_median": effect_by_type.loc["major", "plateau_gain_median"] if "major" in effect_by_type.index else np.nan,
                "medium_plateau_gain_ratio_median": effect_by_type.loc["medium", "plateau_gain_ratio_median"] if "medium" in effect_by_type.index else np.nan,
                "major_plateau_gain_ratio_median": effect_by_type.loc["major", "plateau_gain_ratio_median"] if "major" in effect_by_type.index else np.nan,
                "medium_recovery_ratio_used": medium_rho_used,
                "major_recovery_ratio_used": major_rho_used,
                "h_max_initial": recovery_row["h_max_initial"],
                "hmax_trend_used": recovery_row["hmax_trend_used"],
                "rho_reliable_flag": rho_reliable_flag,
                "n_valid_days": overview_row["n_valid_days"],
                "n_valid_cycles": decay_row["n_valid_cycles"],
            }
        )
    return sort_by_device(pd.DataFrame(rows), "device_id")
