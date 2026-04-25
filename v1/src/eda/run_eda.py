from __future__ import annotations

import os
from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from cleaning.build_cleaned_datasets import (
    CleanedDatasetBundle,
    QUALITY_HIGH,
    QUALITY_INSUFFICIENT,
    QUALITY_LOW,
    QUALITY_MAINTENANCE_GAP,
    QUALITY_RANDOM_GAP,
    _count_quality_days,
    _long_gap_count,
)
from common.device_order import sort_by_device_id, sorted_device_ids
from common.io_utils import write_dataframe, write_markdown
from common.labels_zh import MAINTENANCE_TYPE_LABELS
from common.pillow_plotting import PanelGeometry, draw_bar_chart, draw_line_chart, draw_text, get_font, panel_rect, save_canvas
from common.paths import ProjectPaths

MAINTENANCE_COLORS = {
    "medium": "#4C78A8",
    "major": "#E45756",
}


def _gauss_jordan_inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    aug = np.hstack([matrix.copy(), np.eye(n, dtype=float)])

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row, col]))
        pivot_value = aug[pivot, col]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Regression matrix is singular or ill-conditioned.")
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]

        aug[col] = aug[col] / aug[col, col]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row, col]
            if factor != 0.0:
                aug[row] = aug[row] - factor * aug[col]

    return aug[:, n:]


def _safe_line_fit(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if len(x_values) < 2:
        return np.nan, np.nan
    x_mean = float(np.mean(x_values))
    y_mean = float(np.mean(y_values))
    denom = float(np.sum((x_values - x_mean) ** 2))
    if denom <= 1e-12:
        return np.nan, np.nan
    slope = float(np.sum((x_values - x_mean) * (y_values - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _usable_daily(daily: pd.DataFrame) -> pd.DataFrame:
    return daily[
        daily["daily_quality"].isin([QUALITY_HIGH, QUALITY_LOW])
        & daily["daily_median"].notna()
        & (~daily["is_maintenance_day"])
    ].copy()


def _device_dummies(device_series: pd.Series) -> pd.DataFrame:
    categories = sorted_device_ids(device_series.dropna().unique())
    categorical = pd.Categorical(device_series, categories=categories, ordered=True)
    dummies = pd.get_dummies(categorical, prefix="device", drop_first=True, dtype=float)
    dummies.index = device_series.index
    return dummies


def _build_data_overview(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device_id, device_hourly in hourly.groupby("device_id", sort=True):
        device_daily = daily.loc[daily["device_id"] == device_id].sort_values("date")
        total_days = len(device_daily)
        low_quality_ratio = (
            float((device_daily["daily_quality"] == QUALITY_LOW).mean()) if total_days else np.nan
        )
        rows.append(
            {
                "device_id": device_id,
                "hourly_records": int(len(device_hourly)),
                "observation_start": device_hourly["time"].min(),
                "observation_end": device_hourly["time"].max(),
                "usable_days": int(device_daily["daily_quality"].isin([QUALITY_HIGH, QUALITY_LOW]).sum()),
                "candidate_anomaly_count": int(device_hourly["is_candidate_anomaly"].sum()),
                "excluded_anomaly_count": int(device_hourly["is_excluded_from_analysis"].sum()),
                "maintenance_gap_days": _count_quality_days(device_daily["daily_quality"], QUALITY_MAINTENANCE_GAP),
                "random_gap_days": _count_quality_days(device_daily["daily_quality"], QUALITY_RANDOM_GAP),
                "random_long_gap_count": _long_gap_count(device_daily["gap_type"], [QUALITY_RANDOM_GAP]),
                "low_quality_day_ratio": low_quality_ratio,
            }
        )
    return sort_by_device_id(pd.DataFrame(rows))


def _build_monthly_seasonal_index(daily: pd.DataFrame) -> pd.DataFrame:
    usable = _usable_daily(daily)
    usable["device_mean_daily_median"] = usable.groupby("device_id")["daily_median"].transform("mean")
    usable["centered_daily_median"] = usable["daily_median"] - usable["device_mean_daily_median"]
    seasonal_raw = (
        usable.groupby("month", as_index=False)
        .agg(
            seasonal_index_raw=("centered_daily_median", "mean"),
            n_days=("date", "size"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    seasonal_adj = _build_adjusted_seasonal_index(daily)
    seasonal = seasonal_raw.merge(seasonal_adj, on="month", how="left")
    return seasonal


def _build_maintenance_effect_summary(events: pd.DataFrame) -> pd.DataFrame:
    eligible = events[events["eligible_for_effect_analysis"]].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "maintenance_type",
                "n_events",
                "gain_abs_mean",
                "gain_abs_median",
                "gain_abs_q25",
                "gain_abs_q75",
                "gain_abs_std",
                "gain_abs_min",
                "gain_abs_max",
                "gain_rel_mean",
                "gain_rel_median",
                "gain_rel_q25",
                "gain_rel_q75",
            ]
        )

    rows: list[dict[str, object]] = []
    for maintenance_type, group in eligible.groupby("maintenance_type", sort=True):
        rows.append(
            {
                "maintenance_type": maintenance_type,
                "n_events": int(len(group)),
                "gain_abs_mean": float(group["gain_abs"].mean()),
                "gain_abs_median": float(group["gain_abs"].median()),
                "gain_abs_q25": float(group["gain_abs"].quantile(0.25)),
                "gain_abs_q75": float(group["gain_abs"].quantile(0.75)),
                "gain_abs_std": float(group["gain_abs"].std(ddof=0)),
                "gain_abs_min": float(group["gain_abs"].min()),
                "gain_abs_max": float(group["gain_abs"].max()),
                "gain_rel_mean": float(group["gain_rel"].mean()),
                "gain_rel_median": float(group["gain_rel"].median()),
                "gain_rel_q25": float(group["gain_rel"].quantile(0.25)),
                "gain_rel_q75": float(group["gain_rel"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows).sort_values("maintenance_type").reset_index(drop=True)


def _build_maintenance_recovery_ratio_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (device_id, maintenance_type), group in events.groupby(["device_id", "maintenance_type"], dropna=False, sort=True):
        eligible = group[group["eligible_for_rho_estimation"].astype(bool)].copy()
        rho = eligible["rho_clipped"].dropna().astype(float)
        raw = eligible["rho_raw"].dropna().astype(float)
        clipped_count = int(((raw < 0) | (raw > 1)).sum()) if len(raw) else 0
        rows.append(
            {
                "device_id": device_id,
                "maintenance_type": maintenance_type,
                "n_eligible_events": int(len(eligible)),
                "rho_mean": float(rho.mean()) if not rho.empty else np.nan,
                "rho_median": float(rho.median()) if not rho.empty else np.nan,
                "rho_q25": float(rho.quantile(0.25)) if not rho.empty else np.nan,
                "rho_q75": float(rho.quantile(0.75)) if not rho.empty else np.nan,
                "rho_std": float(rho.std(ddof=0)) if len(rho) >= 2 else 0.0 if len(rho) == 1 else np.nan,
                "rho_clipped_ratio": clipped_count / len(raw) if len(raw) else np.nan,
            }
        )
    return sort_by_device_id(pd.DataFrame(rows))


def _run_regression(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, object]]:
    regression_df = _usable_daily(daily).copy()
    regression_df = regression_df.dropna(
        subset=["daily_median", "days_since_last_maintenance", "days_from_observation_start"]
    ).copy()
    regression_df["post_medium_window_1_3"] = regression_df["post_medium_window_1_3"].astype(int)
    regression_df["post_major_window_1_3"] = regression_df["post_major_window_1_3"].astype(int)

    device_levels = sorted_device_ids(regression_df["device_id"].unique())
    base_device = device_levels[0]

    design_df = regression_df[
        [
            "days_since_last_maintenance",
            "days_from_observation_start",
            "post_medium_window_1_3",
            "post_major_window_1_3",
        ]
    ].copy()
    device_dummies = _device_dummies(regression_df["device_id"])
    month_dummies = pd.get_dummies(regression_df["month"].astype(int), prefix="month", drop_first=True, dtype=float)
    design_df = pd.concat([design_df, device_dummies, month_dummies], axis=1)
    design_df.insert(0, "const", 1.0)

    X = design_df.astype(float).to_numpy()
    y = regression_df["daily_median"].astype(float).to_numpy()

    xtx = np.einsum("ni,nj->ij", X, X)
    xty = np.einsum("ni,n->i", X, y)
    xtx_inv = _gauss_jordan_inverse(xtx)
    beta = np.einsum("ij,j->i", xtx_inv, xty)
    fitted = np.einsum("ij,j->i", X, beta)
    residuals = y - fitted
    leverage = np.clip(np.einsum("ij,jk,ik->i", X, xtx_inv, X), 0.0, 0.999999)
    hc3_scale = (residuals / (1.0 - leverage)) ** 2
    middle = np.einsum("ni,nj,n->ij", X, X, hc3_scale)
    cov_hc3 = np.einsum("ij,jk,kl->il", xtx_inv, middle, xtx_inv)
    std_err = np.sqrt(np.clip(np.diag(cov_hc3), a_min=0.0, a_max=None))
    z_scores = np.divide(beta, std_err, out=np.full_like(beta, np.nan), where=std_err > 0)
    p_values = np.array([math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else np.nan for z in z_scores])
    conf_low = beta - 1.96 * std_err
    conf_high = beta + 1.96 * std_err
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_res = float(np.sum(residuals**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    nobs = len(y)
    n_params = X.shape[1]
    adj_r_squared = 1.0 - (1.0 - r_squared) * (nobs - 1) / (nobs - n_params) if nobs > n_params else np.nan

    terms = design_df.columns.tolist()
    summary = pd.DataFrame(
        {
            "term": terms,
            "estimate": beta,
            "std_err_hc3": std_err,
            "p_value": p_values,
            "conf_low": conf_low,
            "conf_high": conf_high,
            "nobs": nobs,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
        }
    )

    raw_effects = {device_id: 0.0 for device_id in device_levels}
    for device_id in device_levels:
        if device_id == base_device:
            raw_effects[device_id] = 0.0
        else:
            raw_effects[device_id] = float(summary.loc[summary["term"] == f"device_{device_id}", "estimate"].iloc[0])
    mean_effect = float(np.mean(list(raw_effects.values())))
    device_effects = {device_id: raw_effects[device_id] - mean_effect for device_id in device_levels}
    model_info = {
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "regression_df": regression_df[["device_id", "date", "month"]].reset_index(drop=True),
        "fitted": fitted,
        "residuals": residuals,
    }
    return summary, device_effects, model_info


def _build_adjusted_seasonal_index(daily: pd.DataFrame) -> pd.DataFrame:
    usable = _usable_daily(daily).copy()
    usable = usable.dropna(
        subset=["daily_median", "days_since_last_maintenance", "days_from_observation_start"]
    ).copy()
    if usable.empty:
        return pd.DataFrame(columns=["month", "seasonal_index_adjusted", "adjusted_n_days"])

    design_df = usable[["days_since_last_maintenance", "days_from_observation_start"]].copy()
    device_dummies = _device_dummies(usable["device_id"])
    design_df = pd.concat([design_df, device_dummies], axis=1)
    design_df.insert(0, "const", 1.0)

    X = design_df.astype(float).to_numpy()
    y = usable["daily_median"].astype(float).to_numpy()
    xtx = np.einsum("ni,nj->ij", X, X)
    xty = np.einsum("ni,n->i", X, y)
    xtx_inv = _gauss_jordan_inverse(xtx)
    beta = np.einsum("ij,j->i", xtx_inv, xty)
    fitted = np.einsum("ij,j->i", X, beta)
    usable["residual_no_month"] = y - fitted

    adjusted = (
        usable.groupby("month", as_index=False)
        .agg(
            seasonal_index_adjusted=("residual_no_month", "mean"),
            adjusted_n_days=("date", "size"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    return adjusted


def _build_post_recovery_trend_table(events: pd.DataFrame) -> pd.DataFrame:
    eligible = events[events["eligible_for_effect_analysis"]].copy()
    eligible = sort_by_device_id(eligible.dropna(subset=["post_level_median_3d"]), "event_date")
    if eligible.empty:
        return pd.DataFrame(
            columns=["device_id", "event_sequence", "event_date", "post_level_median_3d", "fitted_post_level"]
        )

    rows: list[dict[str, object]] = []
    for device_id, group in eligible.groupby("device_id", sort=False):
        group = group.copy().reset_index(drop=True)
        group["event_sequence"] = np.arange(1, len(group) + 1)
        group["days_from_observation_start"] = (
            pd.to_datetime(group["event_date"]) - pd.to_datetime(group["event_date"]).min()
        ).dt.days.astype(int)
        if len(group) >= 2:
            slope, intercept = _safe_line_fit(
                group["days_from_observation_start"].to_numpy(dtype=float),
                group["post_level_median_3d"].to_numpy(dtype=float),
            )
            group["fitted_post_level_time"] = intercept + slope * group["days_from_observation_start"]
        else:
            group["fitted_post_level_time"] = np.nan
        rows.extend(
            group[
                [
                    "device_id",
                    "event_sequence",
                    "event_date",
                    "days_from_observation_start",
                    "post_level_median_3d",
                    "fitted_post_level_time",
                ]
            ].to_dict("records")
        )
    return sort_by_device_id(pd.DataFrame(rows), "event_date")


def _build_device_core_metrics(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    cycles: pd.DataFrame,
    device_effects: dict[str, float],
) -> pd.DataFrame:
    usable = _usable_daily(daily)
    rows: list[dict[str, object]] = []
    for device_id in sorted_device_ids(daily["device_id"].unique()):
        device_usable = usable[usable["device_id"] == device_id]
        device_events_all = events[events["device_id"] == device_id].copy()
        device_events = (
            device_events_all[device_events_all["eligible_for_effect_analysis"]]
            .dropna(subset=["post_level_median_3d"])
            .sort_values("event_date")
            .copy()
        )
        device_cycles = cycles[(cycles["device_id"] == device_id) & (cycles["eligible_for_cycle_analysis"])].copy()
        medium_rho = device_events_all[
            (device_events_all["maintenance_type"] == "medium")
            & device_events_all["eligible_for_rho_estimation"].astype(bool)
        ]["rho_clipped"].dropna().astype(float)
        major_rho = device_events_all[
            (device_events_all["maintenance_type"] == "major")
            & device_events_all["eligible_for_rho_estimation"].astype(bool)
        ]["rho_clipped"].dropna().astype(float)

        post_recovery_trend_n_events = int(len(device_events))
        if post_recovery_trend_n_events >= 2:
            event_time = (
                pd.to_datetime(device_events["event_date"]) - pd.to_datetime(device_events["event_date"]).min()
            ).dt.days.to_numpy(dtype=float)
            slope, _ = _safe_line_fit(
                event_time,
                device_events["post_level_median_3d"].to_numpy(dtype=float),
            )
        else:
            slope = np.nan

        rows.append(
            {
                "device_id": device_id,
                "alpha_raw": float(device_usable["daily_median"].mean()) if not device_usable.empty else np.nan,
                "alpha_reg_fe": device_effects.get(device_id, np.nan),
                "cycle_decay_rate_median": float(device_cycles["cycle_decay_rate"].median()) if not device_cycles.empty else np.nan,
                "cycle_decay_rate_mean": float(device_cycles["cycle_decay_rate"].mean()) if not device_cycles.empty else np.nan,
                "r_net_small_maintenance_background": float(device_cycles["cycle_decay_rate"].median()) if not device_cycles.empty else np.nan,
                "medium_gain_median": float(
                    device_events.loc[device_events["maintenance_type"] == "medium", "gain_abs"].median()
                )
                if not device_events.loc[device_events["maintenance_type"] == "medium"].empty
                else np.nan,
                "major_gain_median": float(
                    device_events.loc[device_events["maintenance_type"] == "major", "gain_abs"].median()
                )
                if not device_events.loc[device_events["maintenance_type"] == "major"].empty
                else np.nan,
                "medium_recovery_ratio_median": float(medium_rho.median()) if not medium_rho.empty else np.nan,
                "major_recovery_ratio_median": float(major_rho.median()) if not major_rho.empty else np.nan,
                "post_recovery_trend_slope": slope,
                "post_recovery_trend_slope_per_event": slope,
                "post_recovery_trend_n_events": post_recovery_trend_n_events,
                "post_recovery_trend_reliable": post_recovery_trend_n_events >= 4,
                "n_medium_events": int((device_events_all["maintenance_type"] == "medium").sum()),
                "n_major_events": int((device_events_all["maintenance_type"] == "major").sum()),
                "n_medium_rho_events": int(len(medium_rho)),
                "n_major_rho_events": int(len(major_rho)),
                "n_valid_cycles": int(len(device_cycles)),
            }
        )
    return sort_by_device_id(pd.DataFrame(rows))


def _run_maintenance_gain_regression(
    events: pd.DataFrame,
    cycles: pd.DataFrame,
    seasonal: pd.DataFrame,
) -> pd.DataFrame:
    eligible = events[events["eligible_for_effect_analysis"]].copy()
    eligible = eligible.dropna(subset=["gain_abs", "pre_level_median_3d"]).copy()
    if eligible.empty:
        return pd.DataFrame()

    intervals = (
        cycles[["device_id", "current_event_date", "cycle_length_days"]]
        .rename(columns={"current_event_date": "event_date", "cycle_length_days": "maintenance_interval_days"})
        .copy()
    )
    eligible = eligible.merge(intervals, on=["device_id", "event_date"], how="left")
    eligible["month"] = pd.to_datetime(eligible["event_date"]).dt.month
    eligible = eligible.merge(
        seasonal[["month", "seasonal_index_adjusted"]],
        on="month",
        how="left",
    )
    eligible["is_major"] = (eligible["maintenance_type"] == "major").astype(int)

    device_levels = sorted_device_ids(eligible["device_id"].unique())
    base_device = device_levels[0]

    design_df = eligible[["is_major", "pre_level_median_3d", "maintenance_interval_days", "seasonal_index_adjusted"]].copy()
    design_df["maintenance_interval_days"] = design_df["maintenance_interval_days"].fillna(
        design_df["maintenance_interval_days"].median()
    )
    design_df["seasonal_index_adjusted"] = design_df["seasonal_index_adjusted"].fillna(0.0)
    device_dummies = _device_dummies(eligible["device_id"])
    design_df = pd.concat([design_df, device_dummies], axis=1)
    design_df.insert(0, "const", 1.0)

    X = design_df.astype(float).to_numpy()
    y = eligible["gain_abs"].astype(float).to_numpy()
    xtx = np.einsum("ni,nj->ij", X, X)
    xty = np.einsum("ni,n->i", X, y)
    ridge = 1e-8
    ridge_matrix = np.eye(xtx.shape[0], dtype=float) * ridge
    ridge_matrix[0, 0] = 0.0
    xtx_inv = _gauss_jordan_inverse(xtx + ridge_matrix)
    beta = np.einsum("ij,j->i", xtx_inv, xty)
    fitted = np.einsum("ij,j->i", X, beta)
    residuals = y - fitted
    leverage = np.clip(np.einsum("ij,jk,ik->i", X, xtx_inv, X), 0.0, 0.999999)
    hc3_scale = (residuals / (1.0 - leverage)) ** 2
    middle = np.einsum("ni,nj,n->ij", X, X, hc3_scale)
    cov_hc3 = np.einsum("ij,jk,kl->il", xtx_inv, middle, xtx_inv)
    std_err = np.sqrt(np.clip(np.diag(cov_hc3), a_min=0.0, a_max=None))
    z_scores = np.divide(beta, std_err, out=np.full_like(beta, np.nan), where=std_err > 0)
    p_values = np.array([math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else np.nan for z in z_scores])
    conf_low = beta - 1.96 * std_err
    conf_high = beta + 1.96 * std_err

    summary = pd.DataFrame(
        {
            "term": design_df.columns.tolist(),
            "estimate": beta,
            "std_err_hc3": std_err,
            "p_value": p_values,
            "conf_low": conf_low,
            "conf_high": conf_high,
            "nobs": len(y),
        }
    )
    summary["base_device"] = base_device
    return summary


def _build_event_alignment_table(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    usable = _usable_daily(daily)
    rows: list[dict[str, object]] = []
    eligible = events[events["eligible_for_effect_analysis"]].copy()

    for event in eligible.itertuples(index=False):
        device_daily = usable[usable["device_id"] == event.device_id]
        baseline = event.pre_level_median_3d
        for relative_day in range(-3, 11):
            current_date = event.event_date + pd.Timedelta(days=relative_day)
            match = device_daily.loc[device_daily["date"] == current_date]
            if match.empty:
                continue
            rows.append(
                {
                    "device_id": event.device_id,
                    "maintenance_type": event.maintenance_type,
                    "event_date": event.event_date,
                    "relative_day": relative_day,
                    "daily_median": float(match.iloc[0]["daily_median"]),
                    "centered_value": float(match.iloc[0]["daily_median"] - baseline),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "maintenance_type",
                "relative_day",
                "median_centered_value",
                "mean_centered_value",
                "n_events",
            ]
        )

    alignment = pd.DataFrame(rows)
    return (
        alignment.groupby(["maintenance_type", "relative_day"], as_index=False)
        .agg(
            median_centered_value=("centered_value", "median"),
            mean_centered_value=("centered_value", "mean"),
            n_events=("device_id", "size"),
        )
        .sort_values(["maintenance_type", "relative_day"])
        .reset_index(drop=True)
    )


def _build_cycle_decay_tables(daily: pd.DataFrame, cycles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = _usable_daily(daily).copy()
    device_means = usable.groupby("device_id")["daily_median"].mean().to_dict()
    rows: list[dict[str, object]] = []

    eligible_cycles = cycles[cycles["eligible_for_cycle_analysis"]].copy()
    for cycle in eligible_cycles.itertuples(index=False):
        cycle_days = usable[
            (usable["device_id"] == cycle.device_id)
            & (usable["date"] >= cycle.cycle_start_date)
            & (usable["date"] <= cycle.cycle_end_date)
            & (usable["days_since_last_maintenance"].notna())
        ].copy()
        if cycle_days.empty:
            continue
        cycle_days["centered_daily_median"] = cycle_days["daily_median"] - device_means[cycle.device_id]
        for row in cycle_days.itertuples(index=False):
            rows.append(
                {
                    "device_id": cycle.device_id,
                    "cycle_id": cycle.cycle_id,
                    "days_since_last_maintenance": int(row.days_since_last_maintenance),
                    "daily_median": float(row.daily_median),
                    "centered_daily_median": float(row.centered_daily_median),
                }
            )

    if not rows:
        empty = pd.DataFrame(columns=["days_since_last_maintenance", "median_daily_median", "mean_daily_median", "n_points"])
        return empty.copy(), empty.copy()

    curve_df = pd.DataFrame(rows)
    curve_df = curve_df[curve_df["days_since_last_maintenance"] <= 90].copy()

    raw_curve = (
        curve_df.groupby("days_since_last_maintenance", as_index=False)
        .agg(
            median_daily_median=("daily_median", "median"),
            mean_daily_median=("daily_median", "mean"),
            n_points=("device_id", "size"),
        )
        .sort_values("days_since_last_maintenance")
        .reset_index(drop=True)
    )
    centered_curve = (
        curve_df.groupby("days_since_last_maintenance", as_index=False)
        .agg(
            median_centered_daily_median=("centered_daily_median", "median"),
            mean_centered_daily_median=("centered_daily_median", "mean"),
            n_points=("device_id", "size"),
        )
        .sort_values("days_since_last_maintenance")
        .reset_index(drop=True)
    )
    return raw_curve, centered_curve


def _plot_all_devices_trend(daily: pd.DataFrame, maintenance: pd.DataFrame, output_path: Path) -> None:
    width, height = 1800, 2200
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(34)
    legend_font = get_font(18)

    draw_text(draw, (width // 2, 30), "10台设备透水率趋势及维护标记", title_font, anchor="ma")

    legend_y = 80
    legend_items = [
        ("绘图用日中位数", "#1f77b4"),
        ("绘图插值点", "#6b7280"),
        (MAINTENANCE_TYPE_LABELS["medium"], MAINTENANCE_COLORS["medium"]),
        (MAINTENANCE_TYPE_LABELS["major"], MAINTENANCE_COLORS["major"]),
        ("维护性缺失", "#9ca3af"),
    ]
    legend_x = 80
    for label, color in legend_items:
        draw.line((legend_x, legend_y, legend_x + 24, legend_y), fill=color, width=3)
        draw_text(draw, (legend_x + 34, legend_y - 10), label, legend_font, fill="black")
        legend_x += 260

    devices = sorted_device_ids(daily["device_id"].unique())
    for idx, device_id in enumerate(devices):
        rect = panel_rect(idx, 5, 2, width=width, height=height, top_margin=130)
        device_daily = daily[daily["device_id"] == device_id].sort_values("date")
        device_maintenance = maintenance[maintenance["device_id"] == device_id]

        x_numeric = [(ts - device_daily["date"].min()).days for ts in device_daily["date"]]
        x_tick_idx = np.linspace(0, len(device_daily) - 1, num=min(5, len(device_daily)), dtype=int)
        x_ticks = [(x_numeric[i], device_daily.iloc[i]["date"].strftime("%Y-%m")) for i in x_tick_idx]

        shaded_ranges = []
        for gap_date in device_daily.loc[device_daily["is_maintenance_gap"], "date"]:
            day_num = (gap_date - device_daily["date"].min()).days
            shaded_ranges.append((day_num - 0.5, day_num + 0.5, "#e5e7eb"))

        vertical_lines = []
        for maintenance_type, color in MAINTENANCE_COLORS.items():
            for event_date in device_maintenance.loc[device_maintenance["maintenance_type"] == maintenance_type, "event_date"]:
                vertical_lines.append(((event_date - device_daily["date"].min()).days, color))

        y_values = [None if pd.isna(v) else float(v) for v in device_daily["daily_median_plot"]]
        draw_line_chart(
            image,
            draw,
            rect,
            x_values=x_numeric,
            series=[{"y_values": y_values, "color": "#1f77b4", "width": 2, "mode": "line"}],
            title=device_id,
            x_label="日期",
            y_label="日中位透水率",
            x_tick_labels=x_ticks,
            vertical_lines=vertical_lines,
            shaded_ranges=shaded_ranges,
        )

        interpolated = device_daily[device_daily["is_interpolated_plot"]]
        if not interpolated.empty:
            inner_draw = ImageDraw.Draw(image)
            inner_left = rect.left + 72
            inner_top = rect.top + 36
            inner_right = rect.right - 18
            inner_bottom = rect.bottom - 42
            valid_values = [v for v in y_values if v is not None]
            ymin = min(valid_values) if valid_values else 0.0
            ymax = max(valid_values) if valid_values else 1.0
            if ymin == ymax:
                ymin -= 1.0
                ymax += 1.0
            y_pad = 0.08 * (ymax - ymin)
            ymin -= y_pad
            ymax += y_pad
            for _, row in interpolated.iterrows():
                x = (row["date"] - device_daily["date"].min()).days
                y = float(row["daily_median_plot"])
                px = int(inner_left + (x - min(x_numeric)) / (max(x_numeric) - min(x_numeric)) * (inner_right - inner_left)) if max(x_numeric) > min(x_numeric) else inner_left
                py = int(inner_bottom - (y - ymin) / (ymax - ymin) * (inner_bottom - inner_top))
                inner_draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="#6b7280", outline="#6b7280")

    save_canvas(image, output_path)


def _plot_monthly_seasonal_index(seasonal: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGBA", (1000, 520), "white")
    draw = ImageDraw.Draw(image)
    rect = PanelGeometry(40, 40, 960, 480)
    draw_bar_chart(
        draw,
        rect,
        categories=seasonal["month"].astype(str).tolist(),
        values=seasonal["seasonal_index_adjusted"].astype(float).tolist(),
        title="月份季节指数",
        x_label="月份",
        y_label="季节指数",
        image=image,
    )
    save_canvas(image, output_path)


def _plot_event_alignment(alignment: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGBA", (1100, 560), "white")
    draw = ImageDraw.Draw(image)
    series = []
    for maintenance_type in ["medium", "major"]:
        subset = alignment[alignment["maintenance_type"] == maintenance_type]
        if subset.empty:
            continue
        series.append(
            {
                "label": MAINTENANCE_TYPE_LABELS[maintenance_type],
                "y_values": subset["median_centered_value"].astype(float).tolist(),
                "color": MAINTENANCE_COLORS[maintenance_type],
                "width": 3,
                "mode": "line+markers",
            }
        )
    rect = PanelGeometry(60, 60, 1060, 520)
    x_vals = sorted(alignment["relative_day"].unique().tolist()) if not alignment.empty else []
    x_ticks = [(x, str(int(x))) for x in x_vals]
    draw_line_chart(
        image,
        draw,
        rect,
        x_values=x_vals,
        series=series,
        title="维护事件对齐图（以前3天为基准）",
        x_label="相对维护日",
        y_label="相对维护前的日中位透水率变化",
        x_tick_labels=x_ticks,
        y_zero_line=True,
        vertical_lines=[(0, "#111827")],
    )
    save_canvas(image, output_path)


def _plot_cycle_decay_raw(raw_curve: pd.DataFrame, output_path: Path) -> None:
    plot_df = raw_curve[raw_curve["n_points"] >= 5].copy()
    if plot_df.empty:
        plot_df = raw_curve.copy()
    image = Image.new("RGBA", (1100, 560), "white")
    draw = ImageDraw.Draw(image)
    rect = PanelGeometry(60, 60, 1060, 520)
    x_vals = plot_df["days_since_last_maintenance"].astype(float).tolist()
    x_ticks = [(float(v), str(int(v))) for v in plot_df["days_since_last_maintenance"][:: max(1, len(plot_df)//6)]]
    draw_line_chart(
        image,
        draw,
        rect,
        x_values=x_vals,
        series=[{"y_values": plot_df["median_daily_median"].astype(float).tolist(), "color": "#0f766e", "width": 3, "mode": "line+markers"}],
        title="维护周期衰减曲线（原始水平）",
        x_label="距上次维护天数",
        y_label="日中位透水率",
        x_tick_labels=x_ticks,
    )
    save_canvas(image, output_path)


def _plot_cycle_decay_centered(centered_curve: pd.DataFrame, output_path: Path) -> None:
    plot_df = centered_curve[centered_curve["n_points"] >= 5].copy()
    if plot_df.empty:
        plot_df = centered_curve.copy()
    image = Image.new("RGBA", (1100, 560), "white")
    draw = ImageDraw.Draw(image)
    rect = PanelGeometry(60, 60, 1060, 520)
    x_vals = plot_df["days_since_last_maintenance"].astype(float).tolist()
    x_ticks = [(float(v), str(int(v))) for v in plot_df["days_since_last_maintenance"][:: max(1, len(plot_df)//6)]]
    draw_line_chart(
        image,
        draw,
        rect,
        x_values=x_vals,
        series=[{"y_values": plot_df["median_centered_daily_median"].astype(float).tolist(), "color": "#7c3aed", "width": 3, "mode": "line+markers"}],
        title="维护周期衰减曲线（设备中心化）",
        x_label="距上次维护天数",
        y_label="中心化日中位透水率",
        x_tick_labels=x_ticks,
        y_zero_line=True,
    )
    save_canvas(image, output_path)


def _plot_post_recovery_trend(recovery_trend: pd.DataFrame, output_path: Path) -> None:
    devices = sorted_device_ids(recovery_trend["device_id"].unique()) if not recovery_trend.empty else []
    width, height = 1800, 2200
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(34)
    legend_font = get_font(18)
    draw_text(draw, (width // 2, 30), "维护后恢复高点趋势", title_font, anchor="ma")
    draw.line((80, 82, 104, 82), fill="#dc2626", width=2)
    draw_text(draw, (112, 70), "线性趋势", legend_font)
    draw.ellipse((300, 76, 308, 84), fill="#2563eb", outline="#2563eb")
    draw_text(draw, (320, 70), "恢复高点", legend_font)

    for idx, device_id in enumerate(devices):
        device_df = recovery_trend[recovery_trend["device_id"] == device_id].sort_values("event_sequence")
        rect = panel_rect(idx, 5, 2, width=width, height=height, top_margin=130)
        x_vals = device_df["event_sequence"].astype(float).tolist()
        x_ticks = [(float(v), str(int(v))) for v in device_df["event_sequence"].tolist()]
        series = [
            {"y_values": device_df["post_level_median_3d"].astype(float).tolist(), "color": "#2563eb", "width": 2, "mode": "markers"}
        ]
        if device_df["fitted_post_level_time"].notna().any():
            series.append(
                {"y_values": [None if pd.isna(v) else float(v) for v in device_df["fitted_post_level_time"]], "color": "#dc2626", "width": 2, "mode": "line"}
            )
        draw_line_chart(
            image,
            draw,
            rect,
            x_values=x_vals,
            series=series,
            title=device_id,
            x_label="维护事件序号",
            y_label="维护后恢复高点",
            x_tick_labels=x_ticks,
        )

    save_canvas(image, output_path)


def _plot_missingness_structure(daily: pd.DataFrame, output_path: Path) -> None:
    width, height = 1800, 900
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(34)
    label_font = get_font(18)
    tick_font = get_font(14)

    draw_text(draw, (width // 2, 30), "缺失结构图（设备 × 日期）", title_font, anchor="ma")

    legend_items = [
        ("高质量", "#4C78A8"),
        ("低质量", "#F2B447"),
        ("记录不足", "#8B6FD8"),
        ("维护性缺失", "#6C757D"),
        ("随机缺失", "#E45756"),
    ]
    legend_x = 80
    legend_y = 85
    for label, color in legend_items:
        draw.rectangle((legend_x, legend_y, legend_x + 20, legend_y + 20), fill=color, outline=color)
        draw_text(draw, (legend_x + 30, legend_y - 3), label, label_font)
        legend_x += 220

    devices = sorted_device_ids(daily["device_id"].unique())
    all_dates = pd.to_datetime(sorted(daily["date"].unique()))
    left, top, right, bottom = 140, 140, 1720, 820
    draw.rectangle((left, top, right, bottom), outline="#9ca3af", width=1)

    quality_colors = {
        QUALITY_HIGH: "#4C78A8",
        QUALITY_LOW: "#F2B447",
        QUALITY_INSUFFICIENT: "#8B6FD8",
        QUALITY_MAINTENANCE_GAP: "#6C757D",
        QUALITY_RANDOM_GAP: "#E45756",
    }

    day_width = max((right - left) / max(len(all_dates), 1), 1.0)
    row_height = max((bottom - top) / max(len(devices), 1), 1.0)

    for row_idx, device_id in enumerate(devices):
        y0 = int(top + row_idx * row_height)
        y1 = int(top + (row_idx + 1) * row_height)
        draw_text(draw, (left - 12, (y0 + y1) // 2), device_id, tick_font, anchor="ra")
        device_daily = daily[daily["device_id"] == device_id].sort_values("date")
        quality_map = {pd.Timestamp(d).normalize(): q for d, q in zip(device_daily["date"], device_daily["daily_quality"])}
        for col_idx, current_date in enumerate(all_dates):
            x0 = int(left + col_idx * day_width)
            x1 = int(left + (col_idx + 1) * day_width)
            color = quality_colors.get(quality_map.get(current_date.normalize(), QUALITY_RANDOM_GAP), "#ef4444")
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=None)

    tick_positions = np.linspace(0, len(all_dates) - 1, num=min(8, len(all_dates)), dtype=int)
    for idx in tick_positions:
        x = int(left + (idx + 0.5) * day_width)
        draw.line((x, bottom, x, bottom + 6), fill="#374151", width=1)
        draw_text(draw, (x, bottom + 10), all_dates[idx].strftime("%Y-%m"), tick_font, anchor="ma")

    draw_text(draw, ((left + right) // 2, bottom + 38), "日期", label_font, anchor="ma")
    draw_text(draw, (50, (top + bottom) // 2), "设备编号", label_font)
    save_canvas(image, output_path)


def _plot_maintenance_gain_comparison(events: pd.DataFrame, output_path: Path) -> None:
    eligible = events[events["eligible_for_effect_analysis"]].copy()
    groups = [
        ("中维护", eligible.loc[eligible["maintenance_type"] == "medium", "gain_abs"].dropna().tolist(), "#d97706"),
        ("大维护", eligible.loc[eligible["maintenance_type"] == "major", "gain_abs"].dropna().tolist(), "#b91c1c"),
    ]

    width, height = 1000, 620
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(34)
    label_font = get_font(18)
    tick_font = get_font(14)

    draw_text(draw, (width // 2, 30), "维护增益比较图", title_font, anchor="ma")
    left, top, right, bottom = 110, 110, 920, 540
    draw.rectangle((left, top, right, bottom), outline="#9ca3af", width=1)

    values = [v for _, data, _ in groups for v in data]
    if not values:
        draw_text(draw, (width // 2, height // 2), "无可用维护增益数据", label_font, anchor="ma")
        save_canvas(image, output_path)
        return

    ymin = min(0.0, min(values))
    ymax = max(values)
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    ypad = 0.08 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(bottom - frac * (bottom - top))
        draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        value = ymin + frac * (ymax - ymin)
        draw_text(draw, (left - 8, y), f"{value:.1f}", tick_font, fill="#374151", anchor="ra")

    zero_y = int(bottom - (0 - ymin) / (ymax - ymin) * (bottom - top))
    draw.line((left, zero_y, right, zero_y), fill="#111827", width=2)

    centers = [int(left + (i + 0.5) * (right - left) / len(groups)) for i in range(len(groups))]
    box_width = 120
    for center_x, (label, data, color) in zip(centers, groups):
        if not data:
            continue
        data_sorted = sorted(float(v) for v in data)
        q1 = float(np.quantile(data_sorted, 0.25))
        q2 = float(np.quantile(data_sorted, 0.50))
        q3 = float(np.quantile(data_sorted, 0.75))
        vmin = float(min(data_sorted))
        vmax = float(max(data_sorted))

        def ymap(v: float) -> int:
            return int(bottom - (v - ymin) / (ymax - ymin) * (bottom - top))

        x0, x1 = center_x - box_width // 2, center_x + box_width // 2
        draw.rectangle((x0, ymap(q3), x1, ymap(q1)), outline=color, width=2, fill="#ffffff")
        draw.line((x0, ymap(q2), x1, ymap(q2)), fill=color, width=3)
        draw.line((center_x, ymap(vmax), center_x, ymap(q3)), fill=color, width=2)
        draw.line((center_x, ymap(q1), center_x, ymap(vmin)), fill=color, width=2)
        draw.line((center_x - 22, ymap(vmax), center_x + 22, ymap(vmax)), fill=color, width=2)
        draw.line((center_x - 22, ymap(vmin), center_x + 22, ymap(vmin)), fill=color, width=2)

        for idx, value in enumerate(data_sorted):
            jitter = ((idx % 7) - 3) * 6
            px = center_x + jitter
            py = ymap(float(value))
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color, outline=color)

        draw_text(draw, (center_x, bottom + 12), label, label_font, anchor="ma")

    draw_text(draw, ((left + right) // 2, bottom + 46), "维护类型", label_font, anchor="ma")
    draw_text(draw, (30, (top + bottom) // 2), "维护增益（绝对提升量）", label_font)
    save_canvas(image, output_path)


def _build_current_maintenance_rule(maintenance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device_id in sorted_device_ids(maintenance["device_id"].unique()):
        group = maintenance[maintenance["device_id"] == device_id].sort_values("event_date").copy()
        intervals = group["days_since_previous_maintenance"].dropna().astype(float)
        n_total = int(len(group))
        n_medium = int((group["maintenance_type"] == "medium").sum())
        n_major = int((group["maintenance_type"] == "major").sum())
        rows.append(
            {
                "device_id": device_id,
                "n_maintenance_total": n_total,
                "n_medium": n_medium,
                "n_major": n_major,
                "medium_ratio": n_medium / n_total if n_total else np.nan,
                "major_ratio": n_major / n_total if n_total else np.nan,
                "maintenance_interval_mean": float(intervals.mean()) if not intervals.empty else np.nan,
                "maintenance_interval_median": float(intervals.median()) if not intervals.empty else np.nan,
                "maintenance_interval_q25": float(intervals.quantile(0.25)) if not intervals.empty else np.nan,
                "maintenance_interval_q75": float(intervals.quantile(0.75)) if not intervals.empty else np.nan,
                "maintenance_interval_std": float(intervals.std(ddof=0)) if len(intervals) >= 2 else 0.0,
                "last_maintenance_date": group["event_date"].max(),
                "last_maintenance_type": group.sort_values("event_date").iloc[-1]["maintenance_type"] if n_total else "",
                "interval_sequence": ";".join(str(int(round(v))) for v in intervals.tolist()),
                "type_sequence": ";".join(group["maintenance_type"].astype(str).tolist()),
            }
        )
    return sort_by_device_id(pd.DataFrame(rows))


def _build_correlation_matrix(daily: pd.DataFrame, seasonal: pd.DataFrame) -> pd.DataFrame:
    usable = daily[
        daily["daily_quality"].isin([QUALITY_HIGH, QUALITY_LOW])
        & daily["daily_median"].notna()
        & (~daily["is_maintenance_day"])
    ].copy()
    usable["device_mean"] = usable.groupby("device_id")["daily_median"].transform("mean")
    seasonal_map = seasonal.set_index("month")["seasonal_index_adjusted"].to_dict()
    usable["device_centered_per"] = usable["daily_median"] - usable["device_mean"]
    usable["season_adjusted_per"] = usable["device_centered_per"] - usable["month"].map(seasonal_map).fillna(0.0)
    usable["post_medium_window_1_3"] = usable["post_medium_window_1_3"].astype(int)
    usable["post_major_window_1_3"] = usable["post_major_window_1_3"].astype(int)
    variables = [
        "daily_median",
        "device_centered_per",
        "season_adjusted_per",
        "days_since_last_maintenance",
        "days_from_observation_start",
        "post_medium_window_1_3",
        "post_major_window_1_3",
    ]
    matrix = usable[variables].corr(method="spearman")
    return matrix.reset_index().rename(columns={"index": "variable"})


def _build_window_sensitivity_gain(daily: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    usable = daily[
        daily["daily_quality"].isin([QUALITY_HIGH, QUALITY_LOW])
        & daily["daily_median"].notna()
        & (~daily["is_maintenance_day"])
    ].copy()
    rows: list[dict[str, object]] = []
    for window_days in [1, 3, 5, 7]:
        min_days = max(1, math.ceil(window_days * 0.6))
        event_rows: list[dict[str, object]] = []
        for event in maintenance.itertuples(index=False):
            device_daily = usable[usable["device_id"] == event.device_id]
            event_date = pd.Timestamp(event.event_date)
            pre = device_daily[
                (device_daily["date"] >= event_date - pd.Timedelta(days=window_days))
                & (device_daily["date"] <= event_date - pd.Timedelta(days=1))
            ]
            post = device_daily[
                (device_daily["date"] >= event_date + pd.Timedelta(days=1))
                & (device_daily["date"] <= event_date + pd.Timedelta(days=window_days))
            ]
            if len(pre) < min_days or len(post) < min_days:
                continue
            pre_level = float(pre["daily_median"].median())
            post_level = float(post["daily_median"].median())
            gain_abs = post_level - pre_level
            gain_rel = gain_abs / pre_level if pre_level != 0 else np.nan
            event_rows.append(
                {
                    "maintenance_type": event.maintenance_type,
                    "gain_abs": gain_abs,
                    "gain_rel": gain_rel,
                }
            )
        event_df = pd.DataFrame(event_rows)
        for maintenance_type in ["medium", "major"]:
            subset = event_df[event_df["maintenance_type"] == maintenance_type] if not event_df.empty else pd.DataFrame()
            rows.append(
                {
                    "window_days": window_days,
                    "maintenance_type": maintenance_type,
                    "n_eligible_events": int(len(subset)),
                    "gain_abs_mean": float(subset["gain_abs"].mean()) if not subset.empty else np.nan,
                    "gain_abs_median": float(subset["gain_abs"].median()) if not subset.empty else np.nan,
                    "gain_abs_q25": float(subset["gain_abs"].quantile(0.25)) if not subset.empty else np.nan,
                    "gain_abs_q75": float(subset["gain_abs"].quantile(0.75)) if not subset.empty else np.nan,
                    "gain_rel_median": float(subset["gain_rel"].median()) if not subset.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _plot_correlation_heatmap(correlation: pd.DataFrame, output_path: Path) -> None:
    variables = correlation["variable"].tolist()
    matrix = correlation.set_index("variable").loc[variables, variables]
    cell = 88
    left, top = 290, 130
    width = left + cell * len(variables) + 80
    height = top + cell * len(variables) + 80
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(30)
    label_font = get_font(13)
    value_font = get_font(16)
    draw_text(draw, (width // 2, 35), "Spearman Correlation", title_font, anchor="ma")

    def color_for(value: float) -> str:
        value = max(-1.0, min(1.0, value))
        if value >= 0:
            intensity = int(255 * (1 - value))
            return f"#{intensity:02x}{intensity:02x}ff"
        intensity = int(255 * (1 + value))
        return f"#ff{intensity:02x}{intensity:02x}"

    for r, row_name in enumerate(variables):
        y = top + r * cell
        draw_text(draw, (left - 10, y + cell // 2), row_name, label_font, anchor="ra")
        for c, col_name in enumerate(variables):
            x = left + c * cell
            value = float(matrix.loc[row_name, col_name])
            draw.rectangle((x, y, x + cell, y + cell), fill=color_for(value), outline="white")
            draw_text(draw, (x + cell // 2, y + cell // 2 - 8), f"{value:.2f}", value_font, anchor="ma")
    for c, name in enumerate(variables):
        x = left + c * cell
        draw_text(draw, (x + cell // 2, top - 10), name, label_font, anchor="ms")
    save_canvas(image, output_path)


def _plot_window_sensitivity(window_df: pd.DataFrame, output_path: Path) -> None:
    image = Image.new("RGBA", (1100, 780), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(32)
    label_font = get_font(18)
    tick_font = get_font(14)
    draw_text(draw, (550, 30), "维护窗口敏感性分析", title_font, anchor="ma")
    colors = {"medium": "#2563eb", "major": "#dc2626"}
    labels = {"medium": "中维护", "major": "大维护"}
    windows = [1, 3, 5, 7]

    def draw_panel(top: int, bottom: int, metric: str, y_label: str) -> None:
        left, right = 100, 1020
        draw.rectangle((left, top, right, bottom), outline="#9ca3af", width=1)
        vals = window_df[metric].dropna().astype(float).tolist()
        ymin = min(0.0, min(vals)) if vals else 0.0
        ymax = max(vals) if vals else 1.0
        if ymin == ymax:
            ymax += 1.0
        ypad = 0.08 * (ymax - ymin)
        ymin -= ypad
        ymax += ypad
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            y = int(bottom - frac * (bottom - top))
            draw.line((left, y, right, y), fill="#e5e7eb", width=1)
            draw_text(draw, (left - 8, y), f"{ymin + frac * (ymax - ymin):.1f}", tick_font, anchor="ra")
        x_positions = {w: int(left + (idx + 0.5) * (right - left) / len(windows)) for idx, w in enumerate(windows)}
        for maintenance_type in ["medium", "major"]:
            subset = window_df[window_df["maintenance_type"] == maintenance_type].set_index("window_days")
            points: list[tuple[int, int]] = []
            for w in windows:
                if w not in subset.index or pd.isna(subset.loc[w, metric]):
                    continue
                value = float(subset.loc[w, metric])
                x = x_positions[w]
                y = int(bottom - (value - ymin) / (ymax - ymin) * (bottom - top))
                points.append((x, y))
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors[maintenance_type], outline=colors[maintenance_type])
                if metric == "gain_abs_median":
                    n = int(subset.loc[w, "n_eligible_events"])
                    draw_text(draw, (x, y - 24), f"n={n}", tick_font, fill=colors[maintenance_type], anchor="ma")
            if len(points) >= 2:
                draw.line(points, fill=colors[maintenance_type], width=3)
        for w, x in x_positions.items():
            draw.line((x, bottom, x, bottom + 5), fill="#374151", width=1)
            draw_text(draw, (x, bottom + 10), str(w), tick_font, anchor="ma")
        draw_text(draw, ((left + right) // 2, bottom + 42), "窗口长度 w（天）", label_font, anchor="ma")
        draw_text(draw, (30, (top + bottom) // 2), y_label, label_font)

    draw_panel(110, 390, "gain_abs_median", "绝对增益中位数")
    draw_panel(500, 700, "n_eligible_events", "有效事件数")
    legend_x = 760
    for maintenance_type in ["medium", "major"]:
        draw.line((legend_x, 78, legend_x + 30, 78), fill=colors[maintenance_type], width=4)
        draw_text(draw, (legend_x + 40, 68), labels[maintenance_type], label_font)
        legend_x += 150
    save_canvas(image, output_path)


def _plot_current_maintenance_rule(rule_df: pd.DataFrame, maintenance: pd.DataFrame, output_path: Path) -> None:
    width, height = 1200, 820
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(32)
    label_font = get_font(18)
    tick_font = get_font(14)
    devices = sorted_device_ids(rule_df["device_id"].tolist())
    draw_text(draw, (width // 2, 30), "当前固定维护规律", title_font, anchor="ma")

    left, top, right, bottom = 90, 110, 1120, 430
    draw.rectangle((left, top, right, bottom), outline="#9ca3af", width=1)
    intervals = maintenance.dropna(subset=["days_since_previous_maintenance"]).copy()
    vals = intervals["days_since_previous_maintenance"].astype(float).tolist()
    ymin, ymax = (0.0, max(vals) * 1.1) if vals else (0.0, 1.0)
    x_positions = {d: int(left + (idx + 0.5) * (right - left) / len(devices)) for idx, d in enumerate(devices)}
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = int(bottom - frac * (bottom - top))
        draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        draw_text(draw, (left - 8, y), f"{ymin + frac * (ymax - ymin):.0f}", tick_font, anchor="ra")
    for device_id in devices:
        x = x_positions[device_id]
        subset = intervals[intervals["device_id"] == device_id]
        for idx, value in enumerate(subset["days_since_previous_maintenance"].astype(float).tolist()):
            y = int(bottom - (value - ymin) / (ymax - ymin) * (bottom - top))
            jitter = ((idx % 5) - 2) * 5
            draw.ellipse((x + jitter - 3, y - 3, x + jitter + 3, y + 3), fill="#2563eb", outline="#2563eb")
        med = rule_df.loc[rule_df["device_id"] == device_id, "maintenance_interval_median"].iloc[0]
        if pd.notna(med):
            y = int(bottom - (float(med) - ymin) / (ymax - ymin) * (bottom - top))
            draw.line((x - 18, y, x + 18, y), fill="#dc2626", width=3)
        draw_text(draw, (x, bottom + 8), device_id, tick_font, anchor="ma")
    draw_text(draw, ((left + right) // 2, top - 28), "Panel A：维护间隔分布与中位数", label_font, anchor="ma")
    draw_text(draw, (30, (top + bottom) // 2), "维护间隔（天）", label_font)

    left, top, right, bottom = 90, 535, 1120, 760
    draw.rectangle((left, top, right, bottom), outline="#9ca3af", width=1)
    bar_w = int((right - left) / len(devices) * 0.62)
    for device_id in devices:
        x = x_positions[device_id]
        row = rule_df[rule_df["device_id"] == device_id].iloc[0]
        medium_ratio = float(row["medium_ratio"]) if pd.notna(row["medium_ratio"]) else 0.0
        major_ratio = float(row["major_ratio"]) if pd.notna(row["major_ratio"]) else 0.0
        y_medium = int(bottom - medium_ratio * (bottom - top))
        y_major = int(y_medium - major_ratio * (bottom - top))
        draw.rectangle((x - bar_w // 2, y_medium, x + bar_w // 2, bottom), fill="#2563eb", outline="#2563eb")
        draw.rectangle((x - bar_w // 2, y_major, x + bar_w // 2, y_medium), fill="#dc2626", outline="#dc2626")
        draw_text(draw, (x, bottom + 8), device_id, tick_font, anchor="ma")
    for frac in [0, 0.5, 1.0]:
        y = int(bottom - frac * (bottom - top))
        draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        draw_text(draw, (left - 8, y), f"{frac:.1f}", tick_font, anchor="ra")
    draw_text(draw, ((left + right) // 2, top - 28), "Panel B：维护类型比例", label_font, anchor="ma")
    draw_text(draw, (30, (top + bottom) // 2), "比例", label_font)
    draw.rectangle((850, 486, 870, 506), fill="#2563eb")
    draw_text(draw, (880, 482), "中维护", label_font)
    draw.rectangle((980, 486, 1000, 506), fill="#dc2626")
    draw_text(draw, (1010, 482), "大维护", label_font)
    save_canvas(image, output_path)


def _write_q1_aliases(paths: ProjectPaths, cleaned: CleanedDatasetBundle) -> None:
    write_dataframe(cleaned.hourly, paths.q1_cleaned_csv_dir / "q1_cleaned_filter_hourly_long.csv")
    write_dataframe(cleaned.maintenance, paths.q1_cleaned_csv_dir / "q1_cleaned_maintenance_records.csv")
    write_dataframe(cleaned.daily, paths.q1_cleaned_csv_dir / "q1_cleaned_filter_daily_features.csv")
    write_dataframe(cleaned.maintenance_events, paths.q1_cleaned_csv_dir / "q1_cleaned_maintenance_event_effects.csv")
    write_dataframe(cleaned.cycles, paths.q1_cleaned_csv_dir / "q1_cleaned_maintenance_cycles.csv")


def _write_q1_eda_summary(
    paths: ProjectPaths,
    correlation: pd.DataFrame,
    window_sensitivity: pd.DataFrame,
    maintenance_rule: pd.DataFrame,
) -> None:
    content = f"""# 第一问 EDA 总结

## 1. 数据清洗口径

小时级原值保留在 `per_raw`，分析值保留在 `per_analysis`，第一问主分析使用日中位透水率。

小维护约每 3 至 5 天进行一次，但附件 2 未提供小维护日期，因此本文不将小维护作为显式维护事件，也不计算 `G_small` 或 `rho_small`。小维护被视为高频、低成本、未记录的背景维护机制，其影响包含在日常波动和两次显式维护之间的净衰减率 `r_i*` 中。

## 2. 缺失类型说明

维护日整天无数据定义为 `maintenance_gap`，非维护日整天无数据定义为 `random_gap`。

## 3. 异常值处理

仅剔除物理异常和确认孤立异常，维护日前后跳升、连续下降和真实低值不因数值低而删除。

## 4. 设备差异

设备差异由 `q1_table_05_device_core_metrics.csv` 中的 `alpha_raw` 和 `alpha_reg_fe` 描述。

## 5. 相关性分析

Spearman 相关矩阵输出到 `q1_table_correlation_matrix.csv`，热力图输出到 `q1_fig_09_correlation_heatmap.png`。相关性只说明变量关系方向，不作为因果结论。

## 6. 季节周期

月份季节指数输出到 `q1_table_02_monthly_seasonal_index.csv`，后续建模优先使用 `seasonal_index_adjusted`。

## 7. 当前维护规律

当前维护规律输出到 `q1_current_maintenance_rule.csv`，包括维护间隔中位数、大维护比例和历史维护序列，是第二问生成未来维护日程的直接输入。

## 8. 维护周期衰减

周期衰减由 `q1_table_07_cycle_decay_raw.csv` 和 `q1_table_08_cycle_decay_centered.csv` 描述，中心化版本更适合论文解释。这里的维护周期定义为相邻有记录的中维护或大维护之间的周期，不是大维护年度周期；衰减率解释为小维护背景下的净衰减率。

## 9. 窗口敏感性

窗口敏感性输出到 `q1_table_window_sensitivity_gain.csv` 和 `q1_fig_10_window_sensitivity_gain.png`。主窗口仍采用 3 天，1 天窗口易受波动影响，5 天和 7 天窗口可能混入自然衰减并减少有效事件数。

## 10. 中/大维护影响

维护即时效果以维护事件表中的 `gain_abs` 和 `gain_rel` 为主，不用日尺度回归窗口变量替代维护效果主结论。

恢复比例 `rho` 输出到 `q1_maintenance_recovery_ratio.csv`。事件级 `rho_raw` 允许小于 0 或大于 1，模型使用 `rho_clipped` 及第二问收缩后的设备级 `rho_use`。大维护每年允许 0 至 4 次，并非必然年度固定维护，因此 `G_major` 和 `rho_major` 的解释需要强调样本有限和参数收缩。

## 11. 恢复高点趋势

恢复高点趋势输出到 `q1_table_09_post_recovery_trend.csv`，并进入第二问恢复上限模型。

## 12. 回归验证

回归结果输出到 `q1_table_04_regression_summary.csv` 和 `q1_table_10_maintenance_gain_regression.csv`，只作为解释性验证。

## 13. 第一问核心指标

核心影响指标为 `alpha_i, S_m, r_i, G_medium, G_major, b_H`；承接第二问的维护规律为 `R_i=(I_i,p_i_major,interval_sequence,type_sequence)`。

## 14. 第二问输入文件说明

第二问读取 `q1_cleaned_filter_daily_features.csv`、`q1_cleaned_maintenance_event_effects.csv`、`q1_cleaned_maintenance_cycles.csv`、`q1_table_02_monthly_seasonal_index.csv`、`q1_table_05_device_core_metrics.csv`、`q1_current_maintenance_rule.csv`。

第二问主模型读取设备级收缩参数 `rho_medium_use` 和 `rho_major_use`，不直接使用事件级 `rho_raw`。

## 附：新增结果规模

- 相关矩阵变量数：{max(len(correlation) - 1, 0)}
- 窗口敏感性记录数：{len(window_sensitivity)}
- 当前维护规律设备数：{len(maintenance_rule)}
"""
    write_markdown(paths.q1_markdown_dir / "q1_eda_summary.md", content)


def _write_eda_markdown(
    paths: ProjectPaths,
    seasonal: pd.DataFrame,
    maintenance_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
    maintenance_gain_regression: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    peak_month = seasonal.sort_values("seasonal_index_adjusted", ascending=False).iloc[0]
    trough_month = seasonal.sort_values("seasonal_index_adjusted", ascending=True).iloc[0]
    medium_gain = maintenance_summary.loc[maintenance_summary["maintenance_type"] == "medium", "gain_abs_median"]
    major_gain = maintenance_summary.loc[maintenance_summary["maintenance_type"] == "major", "gain_abs_median"]
    medium_gain_rel = maintenance_summary.loc[maintenance_summary["maintenance_type"] == "medium", "gain_rel_median"]
    major_gain_rel = maintenance_summary.loc[maintenance_summary["maintenance_type"] == "major", "gain_rel_median"]
    days_since_coef = regression_summary.loc[
        regression_summary["term"] == "days_since_last_maintenance", "estimate"
    ]
    time_trend_coef = regression_summary.loc[
        regression_summary["term"] == "days_from_observation_start", "estimate"
    ]
    major_gain_coef = maintenance_gain_regression.loc[
        maintenance_gain_regression["term"] == "is_major", "estimate"
    ] if not maintenance_gain_regression.empty else pd.Series(dtype=float)
    maintenance_gap_count = int((daily["daily_quality"] == QUALITY_MAINTENANCE_GAP).sum())
    random_gap_count = int((daily["daily_quality"] == QUALITY_RANDOM_GAP).sum())

    content = f"""# 第一问结论摘要

## 1. 数据处理结论

- 已经将缺失值区分为 `maintenance_gap` 和 `random_gap`
- 小时级原值保留在 `cleaned/csv/cleaned_filter_hourly_long.csv`
- 主分析不做插值，只有 `daily_median_plot` 用于绘图连贯性
- 第一问来源型清洗数据位于：
  - `cleaned/csv/cleaned_filter_hourly_long.csv`
  - `cleaned/csv/cleaned_maintenance_records.csv`
- 第一问分析型派生数据位于：
  - `cleaned/csv/cleaned_filter_daily_features.csv`
  - `cleaned/csv/cleaned_maintenance_event_effects.csv`
  - `cleaned/csv/cleaned_maintenance_cycles.csv`

## 2. 主分析口径

- 主分析不是回归
- 主分析使用 3 张分析型派生数据表：
  - `cleaned_filter_daily_features.csv`
  - `cleaned_maintenance_event_effects.csv`
  - `cleaned_maintenance_cycles.csv`
- 日特征表用于设备差异、季节性、主趋势和回归样本
- 维护事件表用于中维护/大维护即时增益
- 维护周期表用于周期衰减率
- 恢复高点趋势来自维护事件表中的维护后水平
- 回归只负责验证，不负责定义核心指标

## 3. 周期性结论

- 季节指数最高月份：`{int(peak_month['month'])}`，修正指数为 `{peak_month['seasonal_index_adjusted']:.3f}`
- 季节指数最低月份：`{int(trough_month['month'])}`，修正指数为 `{trough_month['seasonal_index_adjusted']:.3f}`
- 月份季节指数对应表：`tables/q1_table_02_monthly_seasonal_index.csv`
- 其中 `seasonal_index_raw` 用于直观展示，`seasonal_index_adjusted` 用于后续分析

## 4. 下降趋势结论

- 维护周期内下降由 `r_i` 描述，对应：
  - `tables/q1_table_07_cycle_decay_raw.csv`
  - `tables/q1_table_08_cycle_decay_centered.csv`
- 长期恢复能力变化由 `b_H` 描述，对应：
  - `tables/q1_table_09_post_recovery_trend.csv`
  - `figures/png/q1_fig_06_post_recovery_trend.png`

## 5. 维护影响结论

- 中维护即时增益中位数：`{float(medium_gain.iloc[0]) if not medium_gain.empty else float('nan'):.3f}`
- 大维护即时增益中位数：`{float(major_gain.iloc[0]) if not major_gain.empty else float('nan'):.3f}`
- 中维护相对增益中位数：`{float(medium_gain_rel.iloc[0]) if not medium_gain_rel.empty else float('nan'):.3f}`
- 大维护相对增益中位数：`{float(major_gain_rel.iloc[0]) if not major_gain_rel.empty else float('nan'):.3f}`
- 维护影响统计表：`tables/q1_table_03_maintenance_effect_summary.csv`
- 控制回归中的 `is_major` 系数：`{float(major_gain_coef.iloc[0]) if not major_gain_coef.empty else float('nan'):.5f}`
- 维护增益控制回归表：`tables/q1_table_10_maintenance_gain_regression.csv`

## 6. 回归验证

- `days_since_last_maintenance` 系数：`{float(days_since_coef.iloc[0]) if not days_since_coef.empty else float('nan'):.5f}`
- `days_from_observation_start` 系数：`{float(time_trend_coef.iloc[0]) if not time_trend_coef.empty else float('nan'):.5f}`
- 回归样本只使用 `high_quality` 和 `low_quality` 的非维护日样本
- 维护日缺失不进入回归
- 回归仅用于验证影响因素，不作为寿命预测模型

## 7. 第一问交给第二问的核心指标

- 设备基准水平 `alpha_i`
- 月份季节指数 `S_m`
- 维护周期衰减率 `r_i`
- 中维护即时增益 `G_medium`
- 大维护即时增益 `G_major`
- 维护后恢复高点趋势 `b_H`
"""
    write_markdown(paths.q1_markdown_dir / "q1_findings_summary.md", content)

    regression_note = """# 回归解释说明

## 模型口径

- 因变量：`daily_median`
- 样本：`daily_quality` 为 `high_quality` 或 `low_quality`，且 `daily_median` 非空，并排除维护日
- 协变量：
  - 设备固定效应
  - 月份效应
  - 距上次维护天数
  - 观测窗口内时间趋势
  - 中维护后 1 到 3 天窗口
  - 大维护后 1 到 3 天窗口

## 解释边界

- `days_from_observation_start` 仅表示观测窗口内长期变化
- 不将其直接解释为完整寿命阶段的全部老化过程
- 标准误采用 HC3 稳健标准误
- 该回归用于解释性验证，不用于严格因果识别
"""
    write_markdown(paths.q1_markdown_dir / "q1_regression_notes.md", regression_note)


def _write_q1_output_index(paths: ProjectPaths) -> None:
    content = """# 第一问论文交付索引

## 1. 最终清洗数据

### 1.1 来源型清洗数据

- `cleaned/csv/cleaned_filter_hourly_long.csv`
  - 对应附件 1 的小时级透水率清洗长表
- `cleaned/csv/cleaned_maintenance_records.csv`
  - 对应附件 2 的维护记录清洗表
  - 保留维护类型，并补充维护顺序与前后维护间隔

### 1.2 分析型派生数据

- `cleaned/csv/cleaned_filter_daily_features.csv`
  - 第一问主分析日特征表
- `cleaned/csv/cleaned_maintenance_event_effects.csv`
  - 维护前后窗口效果表，用于 `G_medium`、`G_major`
- `cleaned/csv/cleaned_maintenance_cycles.csv`
  - 维护周期表，用于 `r_i`

对应 Parquet 版本位于：`cleaned/parquet/`

## 2. 检查表

- `tables/q1_check_duplicate_timestamps.csv`
  - 小时级重复时间戳检查
- `tables/q1_check_per_value_range.csv`
  - 透水率取值范围检查
- `tables/q1_check_row_counts.csv`
  - 清洗前后行数与质量分层计数
- `tables/q1_check_maintenance_gap_summary.csv`
  - 维护性缺失统计
- `tables/q1_check_random_gap_summary.csv`
  - 随机缺失统计
- `tables/q1_check_random_long_gap_summary.csv`
  - 随机长缺口统计
- `tables/q1_check_event_eligibility_summary.csv`
  - 维护事件可分析性统计
- `tables/q1_check_cycle_eligibility_summary.csv`
  - 维护周期可分析性统计

## 3. 第一问核心表

- `tables/q1_table_01_data_overview.csv`
  - 数据概况与设备差异
- `tables/q1_table_02_monthly_seasonal_index.csv`
  - 原始季节指数与修正季节指数
- `tables/q1_table_03_maintenance_effect_summary.csv`
  - 中维护/大维护增益统计
- `tables/q1_table_04_regression_summary.csv`
  - 回归验证结果
- `tables/q1_table_05_device_core_metrics.csv`
  - `alpha_i`、`r_i`、`G_medium`、`G_major`、`b_H`
- `tables/q1_table_06_maintenance_event_alignment.csv`
  - 维护事件对齐数据
- `tables/q1_table_07_cycle_decay_raw.csv`
  - 周期衰减原始水平
- `tables/q1_table_08_cycle_decay_centered.csv`
  - 周期衰减中心化结果
- `tables/q1_table_09_post_recovery_trend.csv`
  - 维护后恢复高点趋势
- `tables/q1_table_10_maintenance_gain_regression.csv`
  - 维护增益控制回归
- `tables/q1_maintenance_recovery_ratio.csv`
  - 中维护和大维护恢复比例 rho 统计
- `tables/q1_table_correlation_matrix.csv`
  - Spearman 相关矩阵
- `tables/q1_table_window_sensitivity_gain.csv`
  - 维护窗口敏感性分析
- `tables/q1_current_maintenance_rule.csv`
  - 当前固定维护规律

## 4. 第一问最终 PNG 图

- `figures/png/q1_fig_01_all_devices_trend_with_maintenance.png`
  - 图 1：设备差异与维护标记
- `figures/png/q1_fig_02_monthly_seasonal_index.png`
  - 图 2：季节指数
- `figures/png/q1_fig_03_maintenance_event_alignment.png`
  - 图 3：维护事件对齐
- `figures/png/q1_fig_04_cycle_decay_raw.png`
  - 图 4：周期衰减原始水平
- `figures/png/q1_fig_05_cycle_decay_centered.png`
  - 图 5：周期衰减中心化
- `figures/png/q1_fig_06_post_recovery_trend.png`
  - 图 6：恢复高点趋势
- `figures/png/q1_fig_07_missingness_structure.png`
  - 图 7：缺失结构图
- `figures/png/q1_fig_08_maintenance_gain_comparison.png`
  - 图 8：维护增益比较
- `figures/png/q1_fig_09_correlation_heatmap.png`
  - 图 9：相关性热力图
- `figures/png/q1_fig_10_window_sensitivity_gain.png`
  - 图 10：窗口敏感性
- `figures/png/q1_fig_11_current_maintenance_rule.png`
  - 图 11：当前维护规律

## 5. 主分析口径

- 主分析不是回归
- 主分析使用：
  - `cleaned_filter_daily_features.csv`
  - `cleaned_maintenance_event_effects.csv`
  - `cleaned_maintenance_cycles.csv`
- 回归仅用于解释性验证

## 6. 文字说明

- `markdown/q1_cleaning_notes.md`
- `markdown/q1_findings_summary.md`
- `markdown/q1_regression_notes.md`
- `markdown/q1_outputs_index.md`
- `markdown/q1_eda_summary.md`

## 7. 说明

- 第一问最终图像只保留 `png`
- `html/svg` 不作为最终交付物保留
"""
    write_markdown(paths.q1_markdown_dir / "q1_outputs_index.md", content)


def run_eda(paths: ProjectPaths, cleaned: CleanedDatasetBundle) -> None:
    hourly = cleaned.hourly
    daily = cleaned.daily
    maintenance = cleaned.maintenance
    maintenance_events = cleaned.maintenance_events
    cycles = cleaned.cycles

    overview = _build_data_overview(hourly, daily)
    seasonal = _build_monthly_seasonal_index(daily)
    maintenance_summary = _build_maintenance_effect_summary(maintenance_events)
    recovery_ratio_summary = _build_maintenance_recovery_ratio_summary(maintenance_events)
    regression_summary, device_effects, regression_info = _run_regression(daily)
    device_core_metrics = _build_device_core_metrics(daily, maintenance_events, cycles, device_effects)
    alignment_table = _build_event_alignment_table(daily, maintenance_events)
    cycle_raw_table, cycle_centered_table = _build_cycle_decay_tables(daily, cycles)
    recovery_trend_table = _build_post_recovery_trend_table(maintenance_events)
    maintenance_gain_regression = _run_maintenance_gain_regression(
        maintenance_events,
        cycles,
        seasonal,
    )
    correlation_matrix = _build_correlation_matrix(daily, seasonal)
    window_sensitivity = _build_window_sensitivity_gain(daily, maintenance)
    current_maintenance_rule = _build_current_maintenance_rule(maintenance)

    write_dataframe(overview, paths.q1_tables_dir / "q1_table_01_data_overview.csv")
    write_dataframe(seasonal, paths.q1_tables_dir / "q1_table_02_monthly_seasonal_index.csv")
    write_dataframe(maintenance_summary, paths.q1_tables_dir / "q1_table_03_maintenance_effect_summary.csv")
    write_dataframe(regression_summary, paths.q1_tables_dir / "q1_table_04_regression_summary.csv")
    write_dataframe(device_core_metrics, paths.q1_tables_dir / "q1_table_05_device_core_metrics.csv")
    write_dataframe(alignment_table, paths.q1_tables_dir / "q1_table_06_maintenance_event_alignment.csv")
    write_dataframe(cycle_raw_table, paths.q1_tables_dir / "q1_table_07_cycle_decay_raw.csv")
    write_dataframe(cycle_centered_table, paths.q1_tables_dir / "q1_table_08_cycle_decay_centered.csv")
    write_dataframe(recovery_trend_table, paths.q1_tables_dir / "q1_table_09_post_recovery_trend.csv")
    write_dataframe(maintenance_gain_regression, paths.q1_tables_dir / "q1_table_10_maintenance_gain_regression.csv")
    write_dataframe(recovery_ratio_summary, paths.q1_tables_dir / "q1_maintenance_recovery_ratio.csv")
    write_dataframe(correlation_matrix, paths.q1_tables_dir / "q1_table_correlation_matrix.csv")
    write_dataframe(window_sensitivity, paths.q1_tables_dir / "q1_table_window_sensitivity_gain.csv")
    write_dataframe(current_maintenance_rule, paths.q1_tables_dir / "q1_current_maintenance_rule.csv")
    _write_q1_aliases(paths, cleaned)

    _plot_all_devices_trend(daily, maintenance, paths.q1_figures_png_dir / "q1_fig_01_all_devices_trend_with_maintenance.png")
    _plot_monthly_seasonal_index(seasonal, paths.q1_figures_png_dir / "q1_fig_02_monthly_seasonal_index.png")
    _plot_event_alignment(alignment_table, paths.q1_figures_png_dir / "q1_fig_03_maintenance_event_alignment.png")
    _plot_cycle_decay_raw(cycle_raw_table, paths.q1_figures_png_dir / "q1_fig_04_cycle_decay_raw.png")
    _plot_cycle_decay_centered(cycle_centered_table, paths.q1_figures_png_dir / "q1_fig_05_cycle_decay_centered.png")
    _plot_post_recovery_trend(recovery_trend_table, paths.q1_figures_png_dir / "q1_fig_06_post_recovery_trend.png")
    _plot_missingness_structure(daily, paths.q1_figures_png_dir / "q1_fig_07_missingness_structure.png")
    _plot_maintenance_gain_comparison(maintenance_events, paths.q1_figures_png_dir / "q1_fig_08_maintenance_gain_comparison.png")
    _plot_correlation_heatmap(correlation_matrix, paths.q1_figures_png_dir / "q1_fig_09_correlation_heatmap.png")
    _plot_window_sensitivity(window_sensitivity, paths.q1_figures_png_dir / "q1_fig_10_window_sensitivity_gain.png")
    _plot_current_maintenance_rule(current_maintenance_rule, maintenance, paths.q1_figures_png_dir / "q1_fig_11_current_maintenance_rule.png")
    _write_q1_output_index(paths)
    _write_q1_eda_summary(paths, correlation_matrix, window_sensitivity, current_maintenance_rule)

    _write_eda_markdown(
        paths,
        seasonal,
        maintenance_summary,
        regression_summary,
        maintenance_gain_regression,
        daily,
    )
