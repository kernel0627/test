from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd

from common.io_utils import artifact_paths, write_dataframe, write_markdown
from common.device_order import sort_by_device_id
from common.paths import ProjectPaths


UPPER_PHYSICAL_BOUND = 250.0
UPPER_BOUND_REVIEW_THRESHOLD = 200.0
MAX_SHORT_PLOT_GAP_DAYS = 2
MIN_VALID_DAYS_FOR_TREND = 4
LONG_GAP_DAYS = 3
ANOMALY_WINDOW_HOURS = 24
ANOMALY_MIN_POINTS = 12
ANOMALY_Z_THRESHOLD = 6.0
ANOMALY_RECOVERY_SCALE = 3.0

HIGH_QUALITY_MIN_RECORDS = 8
HIGH_QUALITY_MIN_COVERAGE = 8.0
LOW_QUALITY_MIN_RECORDS = 4
RHO_RECOVERABLE_SPACE_THRESHOLD = 5.0

MAINTENANCE_TYPE_MAP = {
    "中维护": "medium",
    "大维护": "major",
    "medium": "medium",
    "major": "major",
}

QUALITY_HIGH = "high_quality"
QUALITY_LOW = "low_quality"
QUALITY_INSUFFICIENT = "insufficient"
QUALITY_MAINTENANCE_GAP = "maintenance_gap"
QUALITY_RANDOM_GAP = "random_gap"

RHO_REASON_EFFECT_INELIGIBLE = "effect_ineligible"
RHO_REASON_MISSING_PRE_OR_POST = "missing_pre_or_post"
RHO_REASON_INVALID_HMAX = "invalid_hmax"
RHO_REASON_SMALL_RECOVERABLE_SPACE = "small_recoverable_space"


@dataclass
class CleanedDatasetBundle:
    hourly: pd.DataFrame
    daily: pd.DataFrame
    maintenance: pd.DataFrame
    maintenance_events: pd.DataFrame
    cycles: pd.DataFrame
    duplicate_summary: pd.DataFrame
    row_count_summary: pd.DataFrame
    per_value_range: pd.DataFrame
    parquet_written: dict[str, bool]


def _standardize_device_id(raw_value: str) -> str:
    digits = re.findall(r"\d+", str(raw_value))
    if not digits:
        raise ValueError(f"Cannot parse device id from {raw_value!r}")
    return f"a{int(digits[0])}"


def _safe_quantile(series: pd.Series, q: float) -> float:
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean[np.isfinite(clean)]
    if clean.empty:
        return math.nan
    return float(clean.quantile(q))


def _gap_run_lengths(qualities: pd.Series, target_labels: list[str]) -> list[int]:
    is_target = qualities.isin(target_labels).to_numpy()
    run_lengths: list[int] = []
    run_length = 0
    for flag in is_target:
        if flag:
            run_length += 1
        else:
            if run_length > 0:
                run_lengths.append(run_length)
            run_length = 0
    if run_length > 0:
        run_lengths.append(run_length)
    return run_lengths


def _long_gap_count(qualities: pd.Series, target_labels: list[str]) -> int:
    run_lengths = _gap_run_lengths(qualities, target_labels)
    return int(sum(length >= LONG_GAP_DAYS for length in run_lengths))


def _max_gap_run(qualities: pd.Series, target_labels: list[str]) -> int:
    run_lengths = _gap_run_lengths(qualities, target_labels)
    return max(run_lengths) if run_lengths else 0


def _has_random_long_gap(window_df: pd.DataFrame) -> bool:
    gap_source = window_df["gap_type"] if "gap_type" in window_df.columns else window_df["daily_quality"]
    return _long_gap_count(gap_source, [QUALITY_RANDOM_GAP]) > 0


def _count_quality_days(qualities: pd.Series, target_label: str) -> int:
    return int((qualities == target_label).sum())


def _quality_label_from_counts(
    n_valid_records: int,
    daily_coverage_hours: float,
    is_maintenance_day: bool,
) -> str:
    if n_valid_records == 0:
        return QUALITY_MAINTENANCE_GAP if is_maintenance_day else QUALITY_RANDOM_GAP
    if n_valid_records >= HIGH_QUALITY_MIN_RECORDS and daily_coverage_hours >= HIGH_QUALITY_MIN_COVERAGE:
        return QUALITY_HIGH
    if n_valid_records >= LOW_QUALITY_MIN_RECORDS:
        return QUALITY_LOW
    return QUALITY_INSUFFICIENT


def _gap_type_from_quality(daily_quality: str) -> str:
    if daily_quality == QUALITY_MAINTENANCE_GAP:
        return QUALITY_MAINTENANCE_GAP
    if daily_quality == QUALITY_RANDOM_GAP:
        return QUALITY_RANDOM_GAP
    return "none"


def _long_gap_summary(
    daily: pd.DataFrame,
    *,
    target_label: str,
    value_column_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device_id, device_daily in daily.groupby("device_id", sort=True):
        gap_source = device_daily["gap_type"] if "gap_type" in device_daily.columns else device_daily["daily_quality"]
        rows.append(
            {
                "device_id": device_id,
                value_column_name: _long_gap_count(gap_source, [target_label]),
                "max_run_days": _max_gap_run(gap_source, [target_label]),
            }
        )
    rows.append(
        {
            "device_id": "all",
            value_column_name: int(sum(row[value_column_name] for row in rows)),
            "max_run_days": max((row["max_run_days"] for row in rows), default=0),
        }
    )
    return sort_by_device_id(pd.DataFrame(rows))


def _linear_slope(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> float:
    if len(x_values) < 2:
        return math.nan
    x_mean = x_values.mean()
    y_mean = y_values.mean()
    denom = np.sum((x_values - x_mean) ** 2)
    if denom <= 0:
        return math.nan
    return float(np.sum((x_values - x_mean) * (y_values - y_mean)) / denom)


def _linear_fit(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[float, float]:
    if len(x_values) < 2:
        return math.nan, math.nan
    slope = _linear_slope(x_values, y_values)
    if not np.isfinite(slope):
        return math.nan, math.nan
    intercept = float(y_values.mean() - slope * x_values.mean())
    return slope, intercept


def _build_duplicate_summary(hourly_raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device_id, device_df in hourly_raw.groupby("device_id", sort=True):
        duplicate_counts = device_df.groupby("time").size()
        duplicate_timestamps = int((duplicate_counts > 1).sum())
        duplicate_rows = int(duplicate_counts[duplicate_counts > 1].sum() - duplicate_timestamps)
        rows.append(
            {
                "device_id": device_id,
                "raw_rows": int(len(device_df)),
                "duplicate_timestamps": duplicate_timestamps,
                "duplicate_extra_rows": duplicate_rows,
            }
        )
    summary = sort_by_device_id(pd.DataFrame(rows))
    overall = pd.DataFrame(
        [
            {
                "device_id": "all",
                "raw_rows": int(summary["raw_rows"].sum()),
                "duplicate_timestamps": int(summary["duplicate_timestamps"].sum()),
                "duplicate_extra_rows": int(summary["duplicate_extra_rows"].sum()),
            }
        ]
    )
    return pd.concat([summary, overall], ignore_index=True)


def _load_observations(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    excel_file = pd.ExcelFile(paths.observations_xlsx)
    frames: list[pd.DataFrame] = []

    for sheet_name in excel_file.sheet_names:
        sheet_df = excel_file.parse(sheet_name)
        if sheet_df.shape[1] < 2:
            continue
        renamed = sheet_df.iloc[:, :2].copy()
        renamed.columns = ["time", "per_raw"]
        renamed["device_sheet"] = sheet_name
        renamed["device_id"] = _standardize_device_id(sheet_name)
        renamed["time"] = pd.to_datetime(renamed["time"], errors="coerce")
        renamed["per_raw"] = pd.to_numeric(renamed["per_raw"], errors="coerce")
        frames.append(renamed)

    hourly_raw = pd.concat(frames, ignore_index=True)
    hourly_raw = hourly_raw.dropna(subset=["time"]).copy()
    hourly_raw["date"] = hourly_raw["time"].dt.normalize()

    duplicate_summary = _build_duplicate_summary(hourly_raw)

    grouped = (
        sort_by_device_id(hourly_raw, "time")
        .groupby(["device_id", "device_sheet", "time"], as_index=False)
        .agg(per_raw=("per_raw", "mean"))
    )
    grouped["date"] = grouped["time"].dt.normalize()
    grouped["per_raw"] = grouped["per_raw"].astype(float)
    return sort_by_device_id(grouped, "time"), duplicate_summary


def _load_maintenance(paths: ProjectPaths) -> pd.DataFrame:
    maintenance_raw = pd.read_excel(paths.maintenance_xlsx)
    maintenance = maintenance_raw.iloc[:, :3].copy()
    maintenance.columns = ["device_id", "event_date", "maintenance_type"]
    maintenance["device_id"] = maintenance["device_id"].map(_standardize_device_id)
    maintenance["event_date"] = pd.to_datetime(maintenance["event_date"], errors="coerce").dt.normalize()
    maintenance["maintenance_type"] = (
        maintenance["maintenance_type"].astype(str).str.strip().map(MAINTENANCE_TYPE_MAP)
    )
    maintenance = maintenance.dropna(subset=["device_id", "event_date", "maintenance_type"]).copy()
    maintenance = sort_by_device_id(maintenance, "event_date")

    frames: list[pd.DataFrame] = []
    for device_id, device_df in maintenance.groupby("device_id", sort=True):
        device_df = device_df.copy().reset_index(drop=True)
        device_df["event_order"] = np.arange(1, len(device_df) + 1, dtype=int)
        device_df["days_since_previous_maintenance"] = (
            device_df["event_date"].diff().dt.days.astype("float")
        )
        device_df["days_to_next_maintenance"] = (
            (device_df["event_date"].shift(-1) - device_df["event_date"]).dt.days.astype("float")
        )
        frames.append(device_df)

    maintenance = pd.concat(frames, ignore_index=True)
    ordered_columns = [
        "device_id",
        "event_date",
        "maintenance_type",
        "event_order",
        "days_since_previous_maintenance",
        "days_to_next_maintenance",
    ]
    return sort_by_device_id(maintenance[ordered_columns], "event_date")


def _build_per_value_range(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device_id, device_df in hourly.groupby("device_id", sort=True):
        rows.append(
            {
                "device_id": device_id,
                "min": float(device_df["per_raw"].min(skipna=True)),
                "max": float(device_df["per_raw"].max(skipna=True)),
                "p99": _safe_quantile(device_df["per_raw"], 0.99),
                "p99_9": _safe_quantile(device_df["per_raw"], 0.999),
            }
        )
    per_device = pd.DataFrame(rows)
    overall = pd.DataFrame(
        [
            {
                "device_id": "all",
                "min": float(hourly["per_raw"].min(skipna=True)),
                "max": float(hourly["per_raw"].max(skipna=True)),
                "p99": _safe_quantile(hourly["per_raw"], 0.99),
                "p99_9": _safe_quantile(hourly["per_raw"], 0.999),
            }
        ]
    )
    combined = pd.concat([per_device, overall], ignore_index=True)
    combined["upper_bound_review_needed"] = (
        (combined["p99_9"] >= UPPER_BOUND_REVIEW_THRESHOLD)
        | (combined["max"] >= UPPER_PHYSICAL_BOUND * 0.92)
    )
    combined["upper_bound_used"] = UPPER_PHYSICAL_BOUND
    return sort_by_device_id(combined)


def _maintenance_day_maps(maintenance: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[pd.Timestamp]]]:
    daily_map = (
        maintenance.groupby(["device_id", "event_date"])["maintenance_type"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .reset_index()
        .rename(columns={"event_date": "date", "maintenance_type": "maintenance_type_on_day"})
    )

    buffer_dates: dict[str, set[pd.Timestamp]] = {}
    for device_id, device_df in maintenance.groupby("device_id"):
        dates: set[pd.Timestamp] = set()
        for event_date in device_df["event_date"]:
            for offset in (-1, 0, 1):
                dates.add(event_date + pd.Timedelta(days=offset))
        buffer_dates[device_id] = dates
    return daily_map, buffer_dates


def _detect_device_anomalies(
    device_df: pd.DataFrame,
    maintenance_buffer_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    df = device_df.copy()
    n_rows = len(df)
    candidate = np.zeros(n_rows, dtype=bool)
    excluded = np.zeros(n_rows, dtype=bool)
    reason = np.full(n_rows, "", dtype=object)

    per_analysis = df["per_raw"].to_numpy(dtype=float, copy=True)
    is_missing_raw = np.isnan(per_analysis)
    is_invalid_physical = (~is_missing_raw) & ((per_analysis < 0) | (per_analysis > UPPER_PHYSICAL_BOUND))

    candidate[is_invalid_physical] = True
    excluded[is_invalid_physical] = True
    reason[is_invalid_physical] = "invalid_physical"
    per_analysis[is_invalid_physical] = np.nan

    times_ns = df["time"].astype("int64").to_numpy()
    dates = df["date"].to_list()
    values = per_analysis.copy()
    valid_positions = np.flatnonzero(~np.isnan(values))

    day_ns = pd.Timedelta(hours=ANOMALY_WINDOW_HOURS).value

    for idx in valid_positions:
        current_date = dates[idx]
        if current_date in maintenance_buffer_dates:
            continue

        left = np.searchsorted(times_ns, times_ns[idx] - day_ns, side="left")
        right = np.searchsorted(times_ns, times_ns[idx] + day_ns, side="right")
        window = values[left:right]
        window = window[~np.isnan(window)]
        if len(window) < ANOMALY_MIN_POINTS:
            continue

        median = float(np.median(window))
        mad = float(np.median(np.abs(window - median)))
        if mad <= 1e-9:
            continue

        robust_scale = 1.4826 * mad
        z_score = abs(values[idx] - median) / robust_scale
        if z_score <= ANOMALY_Z_THRESHOLD:
            continue

        candidate[idx] = True
        reason[idx] = "candidate_local_mad"

        valid_pos_idx = int(np.searchsorted(valid_positions, idx))
        prev_idxs = valid_positions[max(0, valid_pos_idx - 3) : valid_pos_idx]
        next_idxs = valid_positions[valid_pos_idx + 1 : valid_pos_idx + 4]
        if len(prev_idxs) < 3 or len(next_idxs) < 3:
            continue

        prev_vals = values[prev_idxs]
        next_vals = values[next_idxs]
        recovery_threshold = ANOMALY_RECOVERY_SCALE * robust_scale
        recovered = bool(
            np.all(np.abs(prev_vals - median) <= recovery_threshold)
            and np.all(np.abs(next_vals - median) <= recovery_threshold)
        )

        current_sign = np.sign(values[idx] - median)
        same_direction_adjacent = False
        for neighbor_idx in [prev_idxs[-1], next_idxs[0]]:
            if np.sign(values[neighbor_idx] - median) == current_sign and abs(values[neighbor_idx] - median) > recovery_threshold:
                same_direction_adjacent = True
                break

        if recovered and not same_direction_adjacent:
            excluded[idx] = True
            reason[idx] = "confirmed_isolated_anomaly"
            per_analysis[idx] = np.nan

    df["is_missing_raw"] = is_missing_raw
    df["is_invalid_physical"] = is_invalid_physical
    df["is_candidate_anomaly"] = candidate
    df["is_excluded_from_analysis"] = excluded
    df["anomaly_reason"] = reason
    df["per_analysis"] = per_analysis
    return df


def _apply_anomaly_logic(hourly: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    maintenance_day_map, maintenance_buffer_dates = _maintenance_day_maps(maintenance)

    merged = hourly.merge(maintenance_day_map, on=["device_id", "date"], how="left")
    merged["is_maintenance_day"] = merged["maintenance_type_on_day"].notna()
    merged["maintenance_type_on_day"] = merged["maintenance_type_on_day"].fillna("")
    merged["is_structural_missing_context"] = merged["is_maintenance_day"]

    device_frames: list[pd.DataFrame] = []
    for device_id, device_df in merged.groupby("device_id", sort=True):
        device_frames.append(
            _detect_device_anomalies(
                device_df.sort_values("time").reset_index(drop=True),
                maintenance_buffer_dates.get(device_id, set()),
            )
        )

    cleaned = pd.concat(device_frames, ignore_index=True)
    return sort_by_device_id(cleaned, "time")


def _summarize_day(day_df: pd.DataFrame) -> pd.Series:
    valid = day_df.dropna(subset=["per_analysis"]).sort_values("time")
    daily_mean = float(valid["per_analysis"].mean()) if not valid.empty else math.nan
    daily_median = float(valid["per_analysis"].median()) if not valid.empty else math.nan
    daily_max = float(valid["per_analysis"].max()) if not valid.empty else math.nan
    daily_min = float(valid["per_analysis"].min()) if not valid.empty else math.nan
    daily_range = daily_max - daily_min if not valid.empty else math.nan
    daily_std = float(valid["per_analysis"].std(ddof=0)) if len(valid) >= 2 else math.nan
    daily_first = float(valid.iloc[0]["per_analysis"]) if not valid.empty else math.nan
    daily_last = float(valid.iloc[-1]["per_analysis"]) if not valid.empty else math.nan
    daily_first_last_diff = daily_last - daily_first if not valid.empty else math.nan

    if len(valid) >= 2:
        hours = (
            (valid["time"] - valid["time"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        )
        intraday_slope = _linear_slope(hours.astype(float), valid["per_analysis"].to_numpy(dtype=float))
        daily_coverage_hours = float(
            (valid["time"].iloc[-1] - valid["time"].iloc[0]).total_seconds() / 3600.0
        )
    else:
        intraday_slope = math.nan
        daily_coverage_hours = 0.0

    n_total_records = int(len(day_df))
    n_valid_records = int(valid["per_analysis"].count())
    n_anomaly_records = int(day_df["is_candidate_anomaly"].sum())

    return pd.Series(
        {
            "daily_mean": daily_mean,
            "daily_median": daily_median,
            "daily_max": daily_max,
            "daily_min": daily_min,
            "daily_range": daily_range,
            "daily_std": daily_std,
            "daily_first": daily_first,
            "daily_last": daily_last,
            "daily_first_last_diff": daily_first_last_diff,
            "intraday_slope": intraday_slope,
            "n_total_records": n_total_records,
            "n_valid_records": n_valid_records,
            "n_anomaly_records": n_anomaly_records,
            "daily_coverage_hours": daily_coverage_hours,
        }
    )


def _assign_maintenance_distances(daily: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for device_id, device_daily in daily.groupby("device_id", sort=True):
        device_df = device_daily.sort_values("date").copy()
        device_dates = device_df["date"].to_numpy(dtype="datetime64[D]")
        events = maintenance.loc[maintenance["device_id"] == device_id].sort_values("event_date")
        event_dates = events["event_date"].to_numpy(dtype="datetime64[D]")
        event_types = events["maintenance_type"].to_numpy(dtype=object)

        if len(event_dates) == 0:
            device_df["days_since_last_maintenance"] = math.nan
            device_df["days_to_next_maintenance"] = math.nan
            device_df["last_maintenance_type"] = ""
            device_df["next_maintenance_type"] = ""
            device_df["post_medium_window_1_3"] = False
            device_df["post_major_window_1_3"] = False
            frames.append(device_df)
            continue

        last_idx = np.searchsorted(event_dates, device_dates, side="right") - 1
        next_idx = np.searchsorted(event_dates, device_dates, side="left")

        days_since = np.full(len(device_df), np.nan, dtype=float)
        days_to = np.full(len(device_df), np.nan, dtype=float)
        last_type = np.full(len(device_df), "", dtype=object)
        next_type = np.full(len(device_df), "", dtype=object)

        valid_last = last_idx >= 0
        valid_next = next_idx < len(event_dates)

        days_since[valid_last] = (
            device_dates[valid_last] - event_dates[last_idx[valid_last]]
        ).astype("timedelta64[D]").astype(float)
        last_type[valid_last] = event_types[last_idx[valid_last]]

        days_to[valid_next] = (
            event_dates[next_idx[valid_next]] - device_dates[valid_next]
        ).astype("timedelta64[D]").astype(float)
        next_type[valid_next] = event_types[next_idx[valid_next]]

        device_df["days_since_last_maintenance"] = days_since
        device_df["days_to_next_maintenance"] = days_to
        device_df["last_maintenance_type"] = last_type
        device_df["next_maintenance_type"] = next_type
        device_df["post_medium_window_1_3"] = (
            (device_df["days_since_last_maintenance"] >= 1)
            & (device_df["days_since_last_maintenance"] <= 3)
            & (device_df["last_maintenance_type"] == "medium")
        )
        device_df["post_major_window_1_3"] = (
            (device_df["days_since_last_maintenance"] >= 1)
            & (device_df["days_since_last_maintenance"] <= 3)
            & (device_df["last_maintenance_type"] == "major")
        )
        frames.append(device_df)
    return pd.concat(frames, ignore_index=True)


def _build_daily_features(hourly: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    maintenance_day_map, _ = _maintenance_day_maps(maintenance)
    day_stats = (
        hourly.groupby(["device_id", "date"], sort=True)[
            ["time", "per_analysis", "is_candidate_anomaly"]
        ]
        .apply(_summarize_day)
        .reset_index()
    )

    all_days: list[pd.DataFrame] = []
    for device_id, device_hourly in hourly.groupby("device_id", sort=True):
        start_date = device_hourly["date"].min()
        end_date = device_hourly["date"].max()
        calendar = pd.DataFrame(
            {
                "device_id": device_id,
                "date": pd.date_range(start=start_date, end=end_date, freq="D"),
            }
        )
        all_days.append(calendar)
    calendar = pd.concat(all_days, ignore_index=True)

    daily = calendar.merge(day_stats, on=["device_id", "date"], how="left")
    daily = daily.merge(maintenance_day_map, on=["device_id", "date"], how="left")
    daily["is_maintenance_day"] = daily["maintenance_type_on_day"].notna()
    daily["maintenance_type_on_day"] = daily["maintenance_type_on_day"].fillna("")
    daily["month"] = daily["date"].dt.month
    for column in ["n_total_records", "n_valid_records", "n_anomaly_records"]:
        daily[column] = daily[column].fillna(0).astype(int)
    daily["daily_coverage_hours"] = daily["daily_coverage_hours"].fillna(0.0)

    daily["daily_quality"] = daily.apply(
        lambda row: _quality_label_from_counts(
            int(row["n_valid_records"]),
            float(row["daily_coverage_hours"]),
            bool(row["is_maintenance_day"]),
        ),
        axis=1,
    )
    daily["gap_type"] = daily["daily_quality"].map(_gap_type_from_quality)
    daily["is_maintenance_gap"] = daily["daily_quality"] == QUALITY_MAINTENANCE_GAP
    daily["is_random_gap"] = daily["daily_quality"] == QUALITY_RANDOM_GAP
    daily["is_gap_day"] = daily["daily_quality"].isin([QUALITY_MAINTENANCE_GAP, QUALITY_RANDOM_GAP])

    daily = _assign_maintenance_distances(daily, maintenance)

    frames: list[pd.DataFrame] = []
    for device_id, device_daily in daily.groupby("device_id", sort=True):
        device_df = device_daily.sort_values("date").copy()
        device_df["days_from_observation_start"] = (
            device_df["date"] - device_df["date"].min()
        ).dt.days.astype(int)

        plot_series = device_df["daily_median"].copy()
        segment_id = device_df["is_maintenance_gap"].cumsum()
        non_maintenance_mask = ~device_df["is_maintenance_gap"]
        for current_segment, segment_index in device_df[non_maintenance_mask].groupby(segment_id[non_maintenance_mask]).groups.items():
            _ = current_segment
            segment_values = plot_series.loc[segment_index].interpolate(
                method="linear",
                limit=MAX_SHORT_PLOT_GAP_DAYS,
                limit_area="inside",
            )
            plot_series.loc[segment_index] = segment_values
        plot_series.loc[device_df["is_maintenance_gap"]] = np.nan
        device_df["daily_median_plot"] = plot_series
        device_df["is_interpolated_plot"] = (
            device_df["is_random_gap"] & device_df["daily_median_plot"].notna()
        )
        frames.append(device_df)
    daily = pd.concat(frames, ignore_index=True)

    ordered_columns = [
        "device_id",
        "date",
        "month",
        "daily_mean",
        "daily_median",
        "daily_max",
        "daily_min",
        "daily_range",
        "daily_std",
        "daily_first",
        "daily_last",
        "daily_first_last_diff",
        "intraday_slope",
        "n_total_records",
        "n_valid_records",
        "n_anomaly_records",
        "daily_coverage_hours",
        "daily_quality",
        "gap_type",
        "is_maintenance_gap",
        "is_random_gap",
        "is_gap_day",
        "is_maintenance_day",
        "maintenance_type_on_day",
        "days_since_last_maintenance",
        "days_to_next_maintenance",
        "post_medium_window_1_3",
        "post_major_window_1_3",
        "days_from_observation_start",
        "daily_median_plot",
        "is_interpolated_plot",
        "last_maintenance_type",
        "next_maintenance_type",
    ]
    return sort_by_device_id(daily[ordered_columns], "date")


def _build_maintenance_event_effects(
    maintenance: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for device_id, events in maintenance.groupby("device_id", sort=True):
        device_daily = daily.loc[daily["device_id"] == device_id].sort_values("date").copy()
        device_daily["usable_day"] = (
            device_daily["daily_quality"].isin([QUALITY_HIGH, QUALITY_LOW])
            & device_daily["daily_median"].notna()
            & ~device_daily["is_maintenance_day"]
        )
        event_dates = list(events["event_date"])
        event_types = list(events["maintenance_type"])

        for idx, (event_date, maintenance_type) in enumerate(zip(event_dates, event_types)):
            pre_start = event_date - pd.Timedelta(days=3)
            pre_end = event_date - pd.Timedelta(days=1)
            post_start = event_date + pd.Timedelta(days=1)
            post_end = event_date + pd.Timedelta(days=3)

            pre_window = device_daily[(device_daily["date"] >= pre_start) & (device_daily["date"] <= pre_end)].copy()
            post_window = device_daily[(device_daily["date"] >= post_start) & (device_daily["date"] <= post_end)].copy()
            pre_window_usable = pre_window[pre_window["usable_day"]]
            post_window_usable = post_window[post_window["usable_day"]]

            other_events = [d for j, d in enumerate(event_dates) if j != idx]
            conflicting_events = [
                d
                for d in other_events
                if (event_date - pd.Timedelta(days=3)) <= d <= (event_date + pd.Timedelta(days=3))
            ]

            ineligible_reasons: list[str] = []
            if len(pre_window_usable) < 2:
                ineligible_reasons.append("insufficient_pre_days")
            if len(post_window_usable) < 2:
                ineligible_reasons.append("insufficient_post_days")
            if conflicting_events:
                ineligible_reasons.append("conflicting_maintenance")
            if _has_random_long_gap(pre_window) or _has_random_long_gap(post_window):
                ineligible_reasons.append("random_long_gap_in_window")
            if maintenance_type not in {"medium", "major"}:
                ineligible_reasons.append("unknown_maintenance_type")

            eligible = not ineligible_reasons

            pre_level = float(pre_window_usable["daily_median"].mean()) if len(pre_window_usable) >= 2 else math.nan
            post_level = float(post_window_usable["daily_median"].mean()) if len(post_window_usable) >= 2 else math.nan
            gain_abs = post_level - pre_level if pd.notna(pre_level) and pd.notna(post_level) else math.nan
            gain_rel = gain_abs / pre_level if pd.notna(gain_abs) and pd.notna(pre_level) and abs(pre_level) > 1e-9 else math.nan

            pre_7d = device_daily[
                (device_daily["date"] >= event_date - pd.Timedelta(days=7))
                & (device_daily["date"] <= event_date - pd.Timedelta(days=1))
                & device_daily["usable_day"]
            ]
            post_7d = device_daily[
                (device_daily["date"] >= event_date + pd.Timedelta(days=1))
                & (device_daily["date"] <= event_date + pd.Timedelta(days=7))
                & device_daily["usable_day"]
            ]

            pre_slope = _linear_slope(
                (pre_7d["date"] - pre_7d["date"].min()).dt.days.to_numpy(dtype=float),
                pre_7d["daily_median"].to_numpy(dtype=float),
            ) if len(pre_7d) >= 2 else math.nan
            post_slope = _linear_slope(
                (post_7d["date"] - post_7d["date"].min()).dt.days.to_numpy(dtype=float),
                post_7d["daily_median"].to_numpy(dtype=float),
            ) if len(post_7d) >= 2 else math.nan

            post_intraday_std_mean_7d = float(post_7d["daily_std"].dropna().mean()) if not post_7d["daily_std"].dropna().empty else math.nan

            recovery_stable = False
            if pd.notna(gain_abs) and gain_abs > 0 and len(post_window_usable) == 3:
                threshold = pre_level + 0.7 * gain_abs
                recovery_stable = bool((post_window_usable["daily_median"] >= threshold).all())

            next_event_date = event_dates[idx + 1] if idx + 1 < len(event_dates) else device_daily["date"].max() + pd.Timedelta(days=1)
            effect_duration = math.nan
            effect_duration_right_censored = False
            if pd.notna(gain_abs):
                if gain_abs <= 0:
                    effect_duration = 0.0
                else:
                    decay_threshold = pre_level + 0.3 * gain_abs
                    after_event = device_daily[
                        (device_daily["date"] >= event_date + pd.Timedelta(days=1))
                        & (device_daily["date"] < next_event_date)
                        & device_daily["usable_day"]
                    ].copy()
                    hit = after_event[after_event["daily_median"] <= decay_threshold]
                    if not hit.empty:
                        effect_duration = float((hit.iloc[0]["date"] - event_date).days)
                    else:
                        effect_duration_right_censored = True
                        effect_duration = float(max((next_event_date - event_date).days - 1, 0))

            rows.append(
                {
                    "device_id": device_id,
                    "event_date": event_date,
                    "maintenance_type": maintenance_type,
                    "pre_level_median_3d": pre_level,
                    "post_level_median_3d": post_level,
                    "gain_abs": gain_abs,
                    "gain_rel": gain_rel,
                    "pre_slope_7d": pre_slope,
                    "post_slope_7d": post_slope,
                    "post_intraday_std_mean_7d": post_intraday_std_mean_7d,
                    "recovery_stable": recovery_stable,
                    "effect_duration_days": effect_duration,
                    "effect_duration_right_censored": effect_duration_right_censored,
                    "eligible_for_effect_analysis": eligible,
                    "ineligible_reason": "|".join(ineligible_reasons),
                    "pre_available_days": int(len(pre_window_usable)),
                    "post_available_days": int(len(post_window_usable)),
                    "post_recovery_level": post_level,
                }
            )

    return _add_recovery_ratio_fields(sort_by_device_id(pd.DataFrame(rows), "event_date"))


def _add_recovery_ratio_fields(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        for column in [
            "H_max_est",
            "recoverable_space",
            "rho_raw",
            "rho_clipped",
            "eligible_for_rho_estimation",
            "rho_ineligible_reason",
        ]:
            events[column] = []
        return events

    events = events.copy()
    events["H_max_est"] = np.nan
    events["recoverable_space"] = np.nan
    events["rho_raw"] = np.nan
    events["rho_clipped"] = np.nan
    events["eligible_for_rho_estimation"] = False
    events["rho_ineligible_reason"] = ""

    eligible_posts = events.loc[
        events["eligible_for_effect_analysis"].astype(bool)
        & events["post_level_median_3d"].notna(),
        "post_level_median_3d",
    ].astype(float)
    all_posts = events["post_level_median_3d"].dropna().astype(float)
    if not eligible_posts.empty:
        global_h = float(eligible_posts.quantile(0.9))
    elif not all_posts.empty:
        global_h = float(all_posts.quantile(0.9))
    else:
        global_h = math.nan

    for device_id, device_events in events.groupby("device_id", sort=False):
        device_idx = device_events.sort_values("event_date").index.tolist()
        device_sorted = events.loc[device_idx].copy()
        valid = device_sorted[
            device_sorted["eligible_for_effect_analysis"].astype(bool)
            & device_sorted["post_level_median_3d"].notna()
        ].copy()
        device_posts = device_sorted["post_level_median_3d"].dropna().astype(float)
        if not valid.empty:
            device_h_fallback = float(valid["post_level_median_3d"].astype(float).quantile(0.9))
        elif not device_posts.empty:
            device_h_fallback = float(device_posts.quantile(0.9))
        else:
            device_h_fallback = global_h

        if len(valid) >= 2:
            x = np.arange(len(valid), dtype=float)
            y = valid["post_level_median_3d"].astype(float).to_numpy()
            slope, intercept = _linear_fit(x, y)
        else:
            slope, intercept = math.nan, math.nan

        valid_seen = 0
        for row_index in device_idx:
            row = events.loc[row_index]
            if bool(row["eligible_for_effect_analysis"]) and pd.notna(row["post_level_median_3d"]):
                current_sequence = valid_seen
                valid_seen += 1
            else:
                current_sequence = max(valid_seen - 1, 0)

            h_candidates: list[float] = []
            if np.isfinite(device_h_fallback):
                h_candidates.append(float(device_h_fallback))
            if np.isfinite(global_h):
                h_candidates.append(float(global_h))
            if np.isfinite(slope) and np.isfinite(intercept):
                h_candidates.append(float(intercept + slope * current_sequence))
            if pd.notna(row["post_level_median_3d"]):
                h_candidates.append(float(row["post_level_median_3d"]))

            h_max = max(h_candidates) if h_candidates else math.nan
            events.at[row_index, "H_max_est"] = h_max

            reasons: list[str] = []
            if not bool(row["eligible_for_effect_analysis"]):
                reasons.append(RHO_REASON_EFFECT_INELIGIBLE)
            if pd.isna(row["pre_level_median_3d"]) or pd.isna(row["post_level_median_3d"]) or pd.isna(row["gain_abs"]):
                reasons.append(RHO_REASON_MISSING_PRE_OR_POST)
            if not np.isfinite(h_max):
                reasons.append(RHO_REASON_INVALID_HMAX)

            if np.isfinite(h_max) and pd.notna(row["pre_level_median_3d"]):
                recoverable_space = float(h_max - row["pre_level_median_3d"])
                events.at[row_index, "recoverable_space"] = recoverable_space
                if recoverable_space <= RHO_RECOVERABLE_SPACE_THRESHOLD:
                    reasons.append(RHO_REASON_SMALL_RECOVERABLE_SPACE)

            if reasons:
                events.at[row_index, "rho_ineligible_reason"] = ";".join(dict.fromkeys(reasons))
                continue

            recoverable_space = float(events.at[row_index, "recoverable_space"])
            rho_raw = float(row["gain_abs"]) / recoverable_space
            events.at[row_index, "rho_raw"] = rho_raw
            events.at[row_index, "rho_clipped"] = min(1.0, max(0.0, rho_raw))
            events.at[row_index, "eligible_for_rho_estimation"] = True

    return sort_by_device_id(events, "event_date")


def _build_maintenance_cycles(
    maintenance: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device_id, events in maintenance.groupby("device_id", sort=True):
        device_daily = daily.loc[daily["device_id"] == device_id].sort_values("date").copy()
        device_daily["usable_day"] = (
            device_daily["daily_quality"].isin([QUALITY_HIGH, QUALITY_LOW])
            & device_daily["daily_median"].notna()
            & ~device_daily["is_maintenance_day"]
        )
        event_rows = list(events.itertuples(index=False))

        for idx in range(len(event_rows) - 1):
            current = event_rows[idx]
            nxt = event_rows[idx + 1]
            cycle_start_date = current.event_date + pd.Timedelta(days=1)
            cycle_end_date = nxt.event_date - pd.Timedelta(days=1)
            cycle_length_days = int((nxt.event_date - current.event_date).days - 1)
            is_right_censored_cycle = False

            start_window = device_daily[
                (device_daily["date"] >= current.event_date + pd.Timedelta(days=1))
                & (device_daily["date"] <= current.event_date + pd.Timedelta(days=3))
            ]
            end_window = device_daily[
                (device_daily["date"] >= nxt.event_date - pd.Timedelta(days=3))
                & (device_daily["date"] <= nxt.event_date - pd.Timedelta(days=1))
            ]
            start_window_usable = start_window[start_window["usable_day"]]
            end_window_usable = end_window[end_window["usable_day"]]

            cycle_window = device_daily[
                (device_daily["date"] >= cycle_start_date) & (device_daily["date"] <= cycle_end_date)
            ].copy()
            usable_ratio = (
                float(cycle_window["usable_day"].mean()) if not cycle_window.empty else 0.0
            )

            ineligible_reasons: list[str] = []
            if cycle_length_days < 7:
                ineligible_reasons.append("cycle_too_short")
            if len(start_window_usable) < 2:
                ineligible_reasons.append("insufficient_start_window")
            if len(end_window_usable) < 2:
                ineligible_reasons.append("insufficient_end_window")
            if _has_random_long_gap(cycle_window):
                ineligible_reasons.append("random_long_gap_in_cycle")
            if usable_ratio < 0.6:
                ineligible_reasons.append("low_usable_ratio")

            eligible = not ineligible_reasons
            cycle_start_level = float(start_window_usable["daily_median"].mean()) if len(start_window_usable) >= 2 else math.nan
            cycle_end_level = float(end_window_usable["daily_median"].mean()) if len(end_window_usable) >= 2 else math.nan
            cycle_decay_amount = cycle_end_level - cycle_start_level if pd.notna(cycle_start_level) and pd.notna(cycle_end_level) else math.nan
            cycle_decay_rate = cycle_decay_amount / cycle_length_days if eligible and cycle_length_days > 0 and pd.notna(cycle_decay_amount) else math.nan

            rows.append(
                {
                    "device_id": device_id,
                    "cycle_id": f"{device_id}_cycle_{idx + 1}",
                    "current_event_date": current.event_date,
                    "current_maintenance_type": current.maintenance_type,
                    "next_event_date": nxt.event_date,
                    "next_maintenance_type": nxt.maintenance_type,
                    "cycle_start_date": cycle_start_date,
                    "cycle_end_date": cycle_end_date,
                    "cycle_length_days": cycle_length_days,
                    "is_right_censored_cycle": is_right_censored_cycle,
                    "cycle_start_level": cycle_start_level,
                    "cycle_end_level": cycle_end_level,
                    "cycle_decay_amount": cycle_decay_amount,
                    "cycle_decay_rate": cycle_decay_rate,
                    "usable_day_ratio": usable_ratio,
                    "eligible_for_cycle_analysis": eligible,
                    "ineligible_reason": "|".join(ineligible_reasons),
                    "n_start_days": int(len(start_window_usable)),
                    "n_end_days": int(len(end_window_usable)),
                }
            )
    return sort_by_device_id(pd.DataFrame(rows), "current_event_date")


def _build_gap_quality_summary(
    daily: pd.DataFrame,
    *,
    target_label: str,
    value_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device_id, device_daily in daily.groupby("device_id", sort=True):
        rows.append(
            {
                "device_id": device_id,
                value_column: _count_quality_days(device_daily["daily_quality"], target_label),
            }
        )
    rows.append(
        {
            "device_id": "all",
            value_column: int(sum(row[value_column] for row in rows)),
        }
    )
    return sort_by_device_id(pd.DataFrame(rows))


def _build_gap_type_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device_id, device_daily in daily.groupby("device_id", sort=True):
        rows.append(
            {
                "device_id": device_id,
                "n_maintenance_gap_days": _count_quality_days(device_daily["daily_quality"], QUALITY_MAINTENANCE_GAP),
                "n_random_gap_days": _count_quality_days(device_daily["daily_quality"], QUALITY_RANDOM_GAP),
                "n_insufficient_days": _count_quality_days(device_daily["daily_quality"], QUALITY_INSUFFICIENT),
                "n_low_quality_days": _count_quality_days(device_daily["daily_quality"], QUALITY_LOW),
                "n_high_quality_days": _count_quality_days(device_daily["daily_quality"], QUALITY_HIGH),
            }
        )
    rows.append(
        {
            "device_id": "all",
            "n_maintenance_gap_days": int(sum(row["n_maintenance_gap_days"] for row in rows)),
            "n_random_gap_days": int(sum(row["n_random_gap_days"] for row in rows)),
            "n_insufficient_days": int(sum(row["n_insufficient_days"] for row in rows)),
            "n_low_quality_days": int(sum(row["n_low_quality_days"] for row in rows)),
            "n_high_quality_days": int(sum(row["n_high_quality_days"] for row in rows)),
        }
    )
    return sort_by_device_id(pd.DataFrame(rows))


def _build_rho_estimation_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = list(events.groupby(["device_id", "maintenance_type"], dropna=False, sort=True))
    for (device_id, maintenance_type), group in groups:
        eligible = group[group["eligible_for_rho_estimation"].astype(bool)]
        rho_raw = eligible["rho_raw"].dropna().astype(float)
        rho_clipped = eligible["rho_clipped"].dropna().astype(float)
        clipped_count = int(((rho_raw < 0) | (rho_raw > 1)).sum()) if not rho_raw.empty else 0
        rows.append(
            {
                "device_id": device_id,
                "maintenance_type": maintenance_type,
                "total_events": int(len(group)),
                "n_rho_eligible_events": int(len(eligible)),
                "n_rho_ineligible_events": int(len(group) - len(eligible)),
                "n_rho_raw_negative": int((rho_raw < 0).sum()) if not rho_raw.empty else 0,
                "n_rho_raw_above_1": int((rho_raw > 1).sum()) if not rho_raw.empty else 0,
                "rho_clipped_ratio": clipped_count / len(rho_raw) if len(rho_raw) else np.nan,
                "rho_clipped_min": float(rho_clipped.min()) if not rho_clipped.empty else np.nan,
                "rho_clipped_max": float(rho_clipped.max()) if not rho_clipped.empty else np.nan,
            }
        )

    if rows:
        total = pd.DataFrame(rows)
        all_rho_raw = events.loc[events["eligible_for_rho_estimation"].astype(bool), "rho_raw"].dropna().astype(float)
        all_rho_clipped = events.loc[events["eligible_for_rho_estimation"].astype(bool), "rho_clipped"].dropna().astype(float)
        all_clipped_count = int(((all_rho_raw < 0) | (all_rho_raw > 1)).sum()) if not all_rho_raw.empty else 0
        rows.append(
            {
                "device_id": "all",
                "maintenance_type": "all",
                "total_events": int(len(events)),
                "n_rho_eligible_events": int(events["eligible_for_rho_estimation"].sum()),
                "n_rho_ineligible_events": int(len(events) - events["eligible_for_rho_estimation"].sum()),
                "n_rho_raw_negative": int((all_rho_raw < 0).sum()) if not all_rho_raw.empty else 0,
                "n_rho_raw_above_1": int((all_rho_raw > 1).sum()) if not all_rho_raw.empty else 0,
                "rho_clipped_ratio": all_clipped_count / len(all_rho_raw) if len(all_rho_raw) else np.nan,
                "rho_clipped_min": float(all_rho_clipped.min()) if not all_rho_clipped.empty else np.nan,
                "rho_clipped_max": float(all_rho_clipped.max()) if not all_rho_clipped.empty else np.nan,
            }
        )
    return sort_by_device_id(pd.DataFrame(rows))


def _build_random_long_gap_summary(daily: pd.DataFrame) -> pd.DataFrame:
    return _long_gap_summary(
        daily,
        target_label=QUALITY_RANDOM_GAP,
        value_column_name="random_long_gap_count",
    )


def _build_event_eligibility_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for maintenance_type, group in events.groupby("maintenance_type", dropna=False, sort=True):
        eligible_count = int(group["eligible_for_effect_analysis"].sum())
        rows.append(
            {
                "maintenance_type": maintenance_type,
                "total_events": int(len(group)),
                "eligible_events": eligible_count,
                "ineligible_events": int(len(group) - eligible_count),
                "eligible_ratio": eligible_count / len(group) if len(group) else np.nan,
            }
        )
    return sort_by_device_id(pd.DataFrame(rows))


def _build_cycle_eligibility_summary(cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device_id, group in cycles.groupby("device_id", sort=True):
        eligible_count = int(group["eligible_for_cycle_analysis"].sum())
        rows.append(
            {
                "device_id": device_id,
                "total_cycles": int(len(group)),
                "eligible_cycles": eligible_count,
                "ineligible_cycles": int(len(group) - eligible_count),
                "right_censored_cycles": int(group["is_right_censored_cycle"].sum()) if "is_right_censored_cycle" in group.columns else 0,
                "eligible_ratio": eligible_count / len(group) if len(group) else np.nan,
            }
        )
    rows.append(
        {
            "device_id": "all",
            "total_cycles": int(len(cycles)),
            "eligible_cycles": int(cycles["eligible_for_cycle_analysis"].sum()),
            "ineligible_cycles": int(len(cycles) - cycles["eligible_for_cycle_analysis"].sum()),
            "right_censored_cycles": int(cycles["is_right_censored_cycle"].sum()) if "is_right_censored_cycle" in cycles.columns else 0,
            "eligible_ratio": float(cycles["eligible_for_cycle_analysis"].mean()) if len(cycles) else np.nan,
        }
    )
    return sort_by_device_id(pd.DataFrame(rows))


def _build_row_count_summary(
    hourly_raw: pd.DataFrame,
    hourly_cleaned: pd.DataFrame,
    daily: pd.DataFrame,
    maintenance_events: pd.DataFrame,
    cycles: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "hourly_rows_after_dedup", "value": int(len(hourly_raw))},
            {"metric": "hourly_rows_cleaned", "value": int(len(hourly_cleaned))},
            {"metric": "hourly_missing_raw", "value": int(hourly_cleaned["is_missing_raw"].sum())},
            {"metric": "hourly_invalid_physical", "value": int(hourly_cleaned["is_invalid_physical"].sum())},
            {"metric": "hourly_candidate_anomaly", "value": int(hourly_cleaned["is_candidate_anomaly"].sum())},
            {"metric": "hourly_excluded_from_analysis", "value": int(hourly_cleaned["is_excluded_from_analysis"].sum())},
            {"metric": "daily_rows", "value": int(len(daily))},
            {"metric": "daily_high_quality_rows", "value": int((daily["daily_quality"] == QUALITY_HIGH).sum())},
            {"metric": "daily_low_quality_rows", "value": int((daily["daily_quality"] == QUALITY_LOW).sum())},
            {"metric": "daily_insufficient_rows", "value": int((daily["daily_quality"] == QUALITY_INSUFFICIENT).sum())},
            {"metric": "daily_maintenance_gap_rows", "value": int((daily["daily_quality"] == QUALITY_MAINTENANCE_GAP).sum())},
            {"metric": "daily_random_gap_rows", "value": int((daily["daily_quality"] == QUALITY_RANDOM_GAP).sum())},
            {"metric": "event_rows", "value": int(len(maintenance_events))},
            {"metric": "eligible_event_rows", "value": int(maintenance_events["eligible_for_effect_analysis"].sum())},
            {"metric": "cycle_rows", "value": int(len(cycles))},
            {"metric": "eligible_cycle_rows", "value": int(cycles["eligible_for_cycle_analysis"].sum())},
        ]
    )


def _write_cleaning_markdown(
    paths: ProjectPaths,
    per_value_range: pd.DataFrame,
    parquet_written: dict[str, bool],
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    maintenance_events: pd.DataFrame,
    cycles: pd.DataFrame,
) -> None:
    overall_row = per_value_range.loc[per_value_range["device_id"] == "all"].iloc[0]
    quality_counts = daily["daily_quality"].value_counts().to_dict()
    content = f"""# 数据清洗说明

## 输入识别

- 观测数据工作簿：`{paths.observations_xlsx.name}`
- 维护记录工作簿：`{paths.maintenance_xlsx.name}`

## 缺失值处理口径

- 维护日整天无数据定义为 `maintenance_gap`
- 非维护日整天无数据定义为 `random_gap`
- 维护性缺失属于结构性缺失，不做插值，不作为普通质量问题
- 随机缺失只允许在绘图列中对短缺口做插值

## 异常处理口径

- 小时级原值保留在 `per_raw`
- 分析值保留在 `per_analysis`
- 物理异常和确认孤立异常从 `per_analysis` 中剔除
- 候选异常但未确认的点保留在 `per_analysis`
- 维护日及其前后 1 天不做孤立异常剔除
- 连续低值、连续下降、低于 37 的点不因数值低而删除

## 上界阈值检查

- 工程上界阈值：`{UPPER_PHYSICAL_BOUND:.0f}`
- 全体数据最大值：`{overall_row['max']:.3f}`
- 全体数据 P99：`{overall_row['p99']:.3f}`
- 全体数据 P99.9：`{overall_row['p99_9']:.3f}`
- 是否需要复核上界阈值：`{bool(overall_row['upper_bound_review_needed'])}`

## 日质量分类

- `high_quality`：有效记录数不少于 8 且覆盖时长不少于 8 小时
- `low_quality`：有效记录数 4 到 7，或覆盖时长不足
- `insufficient`：有效记录数少于 4
- `maintenance_gap`：维护日整天无数据
- `random_gap`：非维护日整天无数据

当前各类样本数量：

- `high_quality`：`{quality_counts.get(QUALITY_HIGH, 0)}`
- `low_quality`：`{quality_counts.get(QUALITY_LOW, 0)}`
- `insufficient`：`{quality_counts.get(QUALITY_INSUFFICIENT, 0)}`
- `maintenance_gap`：`{quality_counts.get(QUALITY_MAINTENANCE_GAP, 0)}`
- `random_gap`：`{quality_counts.get(QUALITY_RANDOM_GAP, 0)}`

## 导出状态

- 小时级 Parquet：`{parquet_written['hourly']}`
- 日特征 Parquet：`{parquet_written['daily']}`
- 维护事件 Parquet：`{parquet_written['maintenance_events']}`
- 维护周期 Parquet：`{parquet_written['cycles']}`

## 数据规模

- 小时级记录数：`{len(hourly)}`
- 日特征记录数：`{len(daily)}`
- 维护事件数：`{len(maintenance_events)}`
- 维护周期数：`{len(cycles)}`
"""
    write_markdown(paths.q1_markdown_dir / "q1_cleaning_notes.md", content)


def build_cleaned_datasets(paths: ProjectPaths) -> CleanedDatasetBundle:
    hourly_raw, duplicate_summary = _load_observations(paths)
    maintenance = _load_maintenance(paths)
    hourly = _apply_anomaly_logic(hourly_raw, maintenance)
    per_value_range = _build_per_value_range(hourly)
    daily = _build_daily_features(hourly, maintenance)
    maintenance_events = _build_maintenance_event_effects(maintenance, daily)
    cycles = _build_maintenance_cycles(maintenance, daily)
    maintenance_gap_summary = _build_gap_quality_summary(
        daily,
        target_label=QUALITY_MAINTENANCE_GAP,
        value_column="maintenance_gap_days",
    )
    gap_type_summary = _build_gap_type_summary(daily)
    random_gap_summary = _build_gap_quality_summary(
        daily,
        target_label=QUALITY_RANDOM_GAP,
        value_column="random_gap_days",
    )
    random_long_gap_summary = _build_random_long_gap_summary(daily)
    event_eligibility_summary = _build_event_eligibility_summary(maintenance_events)
    rho_estimation_summary = _build_rho_estimation_summary(maintenance_events)
    cycle_eligibility_summary = _build_cycle_eligibility_summary(cycles)
    row_count_summary = _build_row_count_summary(
        hourly_raw,
        hourly,
        daily,
        maintenance_events,
        cycles,
    )

    hourly_csv_path, hourly_parquet_path = artifact_paths(
        paths.q1_cleaned_csv_dir, paths.q1_cleaned_parquet_dir, "cleaned_filter_hourly_long"
    )
    daily_csv_path, daily_parquet_path = artifact_paths(
        paths.q1_cleaned_csv_dir, paths.q1_cleaned_parquet_dir, "cleaned_filter_daily_features"
    )
    events_csv_path, events_parquet_path = artifact_paths(
        paths.q1_cleaned_csv_dir, paths.q1_cleaned_parquet_dir, "cleaned_maintenance_event_effects"
    )
    cycles_csv_path, cycles_parquet_path = artifact_paths(
        paths.q1_cleaned_csv_dir, paths.q1_cleaned_parquet_dir, "cleaned_maintenance_cycles"
    )
    maintenance_csv_path, maintenance_parquet_path = artifact_paths(
        paths.q1_cleaned_csv_dir, paths.q1_cleaned_parquet_dir, "cleaned_maintenance_records"
    )

    parquet_written = {
        "hourly": write_dataframe(
            hourly,
            hourly_csv_path,
            hourly_parquet_path,
        ),
        "daily": write_dataframe(
            daily,
            daily_csv_path,
            daily_parquet_path,
        ),
        "maintenance_events": write_dataframe(
            maintenance_events,
            events_csv_path,
            events_parquet_path,
        ),
        "cycles": write_dataframe(
            cycles,
            cycles_csv_path,
            cycles_parquet_path,
        ),
    }
    write_dataframe(
        maintenance,
        maintenance_csv_path,
        maintenance_parquet_path,
    )

    write_dataframe(
        duplicate_summary,
        paths.q1_tables_dir / "q1_check_duplicate_timestamps.csv",
    )
    write_dataframe(
        row_count_summary,
        paths.q1_tables_dir / "q1_check_row_counts.csv",
    )
    write_dataframe(
        per_value_range,
        paths.q1_tables_dir / "q1_check_per_value_range.csv",
    )
    write_dataframe(
        maintenance_gap_summary,
        paths.q1_tables_dir / "q1_check_maintenance_gap_summary.csv",
    )
    write_dataframe(
        gap_type_summary,
        paths.q1_tables_dir / "q1_check_gap_type_summary.csv",
    )
    write_dataframe(
        random_gap_summary,
        paths.q1_tables_dir / "q1_check_random_gap_summary.csv",
    )
    write_dataframe(
        random_long_gap_summary,
        paths.q1_tables_dir / "q1_check_random_long_gap_summary.csv",
    )
    write_dataframe(
        event_eligibility_summary,
        paths.q1_tables_dir / "q1_check_event_eligibility_summary.csv",
    )
    write_dataframe(
        rho_estimation_summary,
        paths.q1_tables_dir / "q1_check_rho_estimation_summary.csv",
    )
    write_dataframe(
        cycle_eligibility_summary,
        paths.q1_tables_dir / "q1_check_cycle_eligibility_summary.csv",
    )

    _write_cleaning_markdown(
        paths,
        per_value_range,
        parquet_written,
        hourly,
        daily,
        maintenance_events,
        cycles,
    )

    return CleanedDatasetBundle(
        hourly=hourly,
        daily=daily,
        maintenance=maintenance,
        maintenance_events=maintenance_events,
        cycles=cycles,
        duplicate_summary=duplicate_summary,
        row_count_summary=row_count_summary,
        per_value_range=per_value_range,
        parquet_written=parquet_written,
    )
