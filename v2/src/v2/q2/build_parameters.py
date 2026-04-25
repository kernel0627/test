from __future__ import annotations

import math

import numpy as np
import pandas as pd


HMAX_MAX_ANNUAL_DROP_RATIO = 0.20


def _seasonal_level_map(season_decay: pd.DataFrame) -> dict[int, float]:
    rows = season_decay[season_decay["section"] == "seasonal_level"]
    return {int(row["month"]): float(row["value"]) for _, row in rows.dropna(subset=["month", "value"]).iterrows()}


def _decay_lambda_map(season_decay: pd.DataFrame) -> dict[int, float]:
    rows = season_decay[season_decay["section"] == "monthly_decay_intensity"].copy()
    valid = rows[rows["value"].notna() & (rows["n_samples"].fillna(0).astype(float) >= 2)]
    mean_value = float(valid["value"].mean()) if len(valid) else math.nan
    result: dict[int, float] = {}
    for month in range(1, 13):
        row = valid[valid["month"].astype(float) == float(month)]
        if row.empty or not np.isfinite(mean_value) or abs(mean_value) <= 1e-9:
            result[month] = 1.0
        else:
            result[month] = max(0.25, min(3.0, float(row["value"].iloc[0]) / mean_value))
    return result


def _season_group(month: int) -> str:
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    if month in [9, 10, 11]:
        return "autumn"
    return "winter"


def _seasonal_effect_maps(seasonal_effect: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float, bool]]:
    maps: dict[tuple[str, str], tuple[float, float, bool]] = {}
    for _, row in seasonal_effect.iterrows():
        reliable = bool(row.get("season_maintenance_effect_reliable", False))
        kg = float(row.get("kappa_gain", 1.0)) if reliable and pd.notna(row.get("kappa_gain")) else 1.0
        kr = float(row.get("kappa_rho", 1.0)) if reliable and pd.notna(row.get("kappa_rho")) else 1.0
        maps[(str(row["maintenance_type"]), str(row["season_group"]))] = (kg, kr, reliable)
    return maps


def _global_reliable_rho(hmax: pd.DataFrame, maintenance_type: str) -> float:
    reliable_col = f"{maintenance_type}_rho_reliable"
    value_col = f"{maintenance_type}_rho_median"
    values = hmax.loc[hmax[reliable_col].astype(bool), value_col].dropna().astype(float)
    return float(values.median()) if len(values) else math.nan


def _rho_value_and_source(
    hmax_row: pd.Series,
    maintenance_type: str,
    global_rho: float,
) -> tuple[float, str, bool]:
    reliable_col = f"{maintenance_type}_rho_reliable"
    value_col = f"{maintenance_type}_rho_median"
    if bool(hmax_row.get(reliable_col, False)) and pd.notna(hmax_row.get(value_col)):
        return float(hmax_row[value_col]), "device_rho", True
    if np.isfinite(global_rho):
        return float(global_rho), "global_rho", False
    return math.nan, "fixed_gain_fallback", False


def _limited_hmax_trend(hmax_initial: float, hmax_trend_raw: float) -> tuple[float, float, float]:
    raw_clipped = min(0.0, float(hmax_trend_raw) if np.isfinite(hmax_trend_raw) else 0.0)
    floor = -HMAX_MAX_ANNUAL_DROP_RATIO * float(hmax_initial) / 365.0
    limited = max(raw_clipped, floor)
    annual_drop_ratio = abs(limited) * 365.0 / max(float(hmax_initial), 1e-9)
    return limited, 0.5 * limited, annual_drop_ratio


def build_model_parameters(outputs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, object]]:
    core = outputs["core"]
    rule = outputs["rule"]
    hmax = outputs["hmax_quality"]
    daily = outputs["daily"]
    season_decay = outputs["season_decay"]
    seasonal_level = _seasonal_level_map(season_decay)
    decay_lambda = _decay_lambda_map(season_decay)
    seasonal_effect = _seasonal_effect_maps(outputs["seasonal_effect"])
    medium_global_rho = _global_reliable_rho(hmax, "medium")
    major_global_rho = _global_reliable_rho(hmax, "major")

    prediction_start = daily["date"].max() + pd.Timedelta(days=1)
    rows: list[dict[str, object]] = []
    global_cycle = core["cycle_decay_rate_median"].dropna().astype(float).median()
    for _, core_row in core.iterrows():
        device_id = core_row["device_id"]
        rule_row = rule[rule["device_id"] == device_id].iloc[0]
        hmax_row = hmax[hmax["device_id"] == device_id].iloc[0]
        season0 = seasonal_level.get(int(prediction_start.month), 0.0)
        cycle = core_row["cycle_decay_rate_median"]
        if pd.isna(cycle):
            cycle = core_row["pure_decay_rate_median"]
        if pd.isna(cycle):
            cycle = global_cycle
        if pd.isna(cycle):
            cycle = -0.1
        cycle = min(float(cycle), -1e-4)
        medium_rho, medium_source, medium_device_reliable = _rho_value_and_source(hmax_row, "medium", medium_global_rho)
        major_rho, major_source, major_device_reliable = _rho_value_and_source(hmax_row, "major", major_global_rho)
        hmax_initial = float(core_row["h_max_initial"])
        raw_trend = hmax_row.get("hmax_trend_slope_per_day", hmax_row.get("hmax_trend_used", 0.0))
        raw_trend = float(raw_trend) if pd.notna(raw_trend) else 0.0
        hmax_trend_limited, hmax_trend_neutral, hmax_annual_drop_ratio = _limited_hmax_trend(hmax_initial, raw_trend)
        rho_reliable_flag = bool(
            medium_source in {"device_rho", "global_rho"}
            and (float(rule_row["major_ratio"]) == 0 or major_source in {"device_rho", "global_rho"})
        )
        rows.append(
            {
                "device_id": device_id,
                "current_state_level": float(core_row["current_state_level"]),
                "initial_x_state": float(core_row["current_state_level"]) - season0,
                "maintenance_interval_median": float(rule_row["medium_interval_median"]),
                "major_ratio": float(rule_row["major_ratio"]),
                "n_major": int(rule_row["n_major"]),
                "cycle_decay_rate_used": cycle,
                "h_max_initial": hmax_initial,
                "hmax_trend_raw": min(0.0, raw_trend),
                "hmax_trend_limited": hmax_trend_limited,
                "hmax_trend_used": hmax_trend_neutral,
                "hmax_annual_drop_ratio_used": hmax_annual_drop_ratio / 2.0,
                "hmax_scenario": "neutral",
                "hmax_main_scenario": "neutral",
                "medium_plateau_gain_used": float(core_row["medium_plateau_gain_median"]),
                "major_plateau_gain_used": float(core_row["major_plateau_gain_median"]) if pd.notna(core_row["major_plateau_gain_median"]) else float(core_row["medium_plateau_gain_median"]),
                "medium_recovery_ratio_used": medium_rho,
                "major_recovery_ratio_used": major_rho,
                "medium_rho_used_source": medium_source,
                "major_rho_used_source": major_source,
                "rho_reliable_flag": rho_reliable_flag,
                "medium_rho_reliable": medium_source in {"device_rho", "global_rho"},
                "major_rho_reliable": major_source in {"device_rho", "global_rho"},
                "medium_device_rho_reliable": medium_device_reliable,
                "major_device_rho_reliable": major_device_reliable,
                "medium_min_interval": float(rule_row["medium_min_interval"]),
                "medium_pre_decay_speed_median": float(rule_row["medium_pre_decay_speed_median"]) if pd.notna(rule_row["medium_pre_decay_speed_median"]) else 0.2,
                "major_interval_median": float(rule_row["major_interval_median"]) if pd.notna(rule_row["major_interval_median"]) else np.nan,
                "medium_count_between_major_median": float(rule_row["medium_count_between_major_median"]) if pd.notna(rule_row["medium_count_between_major_median"]) else np.nan,
                "major_global_fallback_interval_days": float(rule_row["major_global_fallback_interval_days"]) if pd.notna(rule_row["major_global_fallback_interval_days"]) else np.nan,
                "major_global_fallback_event_count": float(rule_row["major_global_fallback_event_count"]) if pd.notna(rule_row["major_global_fallback_event_count"]) else np.nan,
                "major_rule_source": rule_row["major_rule_source"],
                "last_maintenance_date": rule_row["last_maintenance_date"],
                "last_major_maintenance_date": rule_row["last_major_maintenance_date"],
                "last_maintenance_type": rule_row["last_maintenance_type"],
                "medium_since_last_major_at_end": int(rule_row["medium_since_last_major_at_end"]),
            }
        )
    params = pd.DataFrame(rows)
    helpers = {
        "seasonal_level": seasonal_level,
        "decay_lambda": decay_lambda,
        "seasonal_effect": seasonal_effect,
        "season_group": _season_group,
        "prediction_start": prediction_start,
    }
    return params, helpers
