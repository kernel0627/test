from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .simulate_models import MODEL_FIXED, MODEL_MAIN, apply_fixed_gain, apply_recovery_ratio


VALIDATION_DAYS = 180
RHO_DELTA_H = 5.0


def _clean_maintenance_type(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text if text in {"medium", "major"} else ""


def _window_values(group: pd.DataFrame, event_date: pd.Timestamp, start_offset: int, end_offset: int) -> pd.Series:
    start = event_date + pd.Timedelta(days=start_offset)
    end = event_date + pd.Timedelta(days=end_offset)
    values = group.loc[
        (group["date"] >= start) & (group["date"] <= end) & group["daily_median"].notna(),
        "daily_median",
    ].astype(float)
    return values


def _window_median(group: pd.DataFrame, event_date: pd.Timestamp, start_offset: int, end_offset: int) -> float:
    values = _window_values(group, event_date, start_offset, end_offset)
    return float(values.median()) if len(values) else math.nan


def _training_events(training: pd.DataFrame) -> pd.DataFrame:
    events = training.copy()
    events["maintenance_type_on_day"] = events["maintenance_type_on_day"].map(_clean_maintenance_type)
    return events[events["maintenance_type_on_day"].isin(["medium", "major"])].sort_values("date")


def _estimate_cycle_decay(group: pd.DataFrame, events: pd.DataFrame, fallback: float) -> float:
    rates: list[float] = []
    event_rows = events.reset_index(drop=True)
    for idx in range(len(event_rows) - 1):
        start_event = event_rows.loc[idx]
        next_event = event_rows.loc[idx + 1]
        start_date = pd.Timestamp(start_event["date"])
        next_date = pd.Timestamp(next_event["date"])
        length = int((next_date - start_date).days)
        if length < 7:
            continue
        start_values = _window_values(group, start_date, 1, 3)
        end_values = _window_values(group, next_date, -3, -1)
        if len(start_values) < 2 or len(end_values) < 2:
            continue
        rates.append((float(end_values.median()) - float(start_values.median())) / length)
    return float(np.median(rates)) if rates else float(fallback)


def _estimate_plateau_gain(
    group: pd.DataFrame,
    events: pd.DataFrame,
    maintenance_type: str,
    fallback: float,
) -> float:
    gains: list[float] = []
    for _, event in events[events["maintenance_type_on_day"] == maintenance_type].iterrows():
        event_date = pd.Timestamp(event["date"])
        pre_values = _window_values(group, event_date, -3, -1)
        post_values = _window_values(group, event_date, 1, 3)
        if len(pre_values) < 2 or len(post_values) < 2:
            continue
        gains.append(float(post_values.median()) - float(pre_values.median()))
    return float(np.median(gains)) if gains else float(fallback)


def _estimate_hmax(group: pd.DataFrame, events: pd.DataFrame, fallback: float, validation_initial_level: float) -> float:
    post_levels: list[float] = []
    for _, event in events.iterrows():
        level = _window_median(group, pd.Timestamp(event["date"]), 1, 3)
        if np.isfinite(level):
            post_levels.append(level)
    if post_levels:
        recent = post_levels[-3:]
        estimate = float(np.median(recent))
    else:
        estimate = float(fallback)
    return max(estimate, float(validation_initial_level))


def _estimate_rho(
    group: pd.DataFrame,
    events: pd.DataFrame,
    maintenance_type: str,
    hmax_initial: float,
    fallback: float,
    fallback_source: str,
) -> tuple[float, str]:
    values: list[float] = []
    for _, event in events[events["maintenance_type_on_day"] == maintenance_type].iterrows():
        event_date = pd.Timestamp(event["date"])
        pre = _window_median(group, event_date, -3, -1)
        post = _window_median(group, event_date, 1, 3)
        if not np.isfinite(pre) or not np.isfinite(post):
            continue
        denom = float(hmax_initial) - pre
        if denom <= RHO_DELTA_H:
            continue
        values.append(max(0.0, min(1.0, (post - pre) / denom)))
    if values:
        return float(np.median(values)), "device_rho"
    if fallback_source in {"device_rho", "global_rho"} and pd.notna(fallback):
        return float(fallback), fallback_source
    return math.nan, "fixed_gain_fallback"


def _prepare_backtest_param(
    group: pd.DataFrame,
    training: pd.DataFrame,
    validation_valid: pd.DataFrame,
    base_param: pd.Series,
    helpers: dict[str, object],
) -> pd.Series:
    events = _training_events(training)
    validation_start = pd.Timestamp(validation_valid["date"].min())
    last_train_valid = training[training["daily_median"].notna()].sort_values("date").tail(1)
    if len(last_train_valid):
        initial_level = float(last_train_valid["daily_median"].iloc[0])
        initial_month = int(last_train_valid["month"].iloc[0])
    else:
        initial_level = float(validation_valid["daily_median"].iloc[0])
        initial_month = int(validation_valid["month"].iloc[0])
    initial_x = initial_level - helpers["seasonal_level"].get(initial_month, 0.0)
    param = base_param.copy()
    param["initial_x_state"] = initial_x
    param["cycle_decay_rate_used"] = _estimate_cycle_decay(
        training,
        events,
        float(base_param["cycle_decay_rate_used"]),
    )
    param["medium_plateau_gain_used"] = _estimate_plateau_gain(
        training,
        events,
        "medium",
        float(base_param["medium_plateau_gain_used"]),
    )
    param["major_plateau_gain_used"] = _estimate_plateau_gain(
        training,
        events,
        "major",
        float(base_param["major_plateau_gain_used"]),
    )
    param["h_max_initial"] = _estimate_hmax(
        training,
        events,
        float(base_param["h_max_initial"]),
        initial_level,
    )
    for maintenance_type in ["medium", "major"]:
        rho, source = _estimate_rho(
            training,
            events,
            maintenance_type,
            float(param["h_max_initial"]),
            base_param.get(f"{maintenance_type}_recovery_ratio_used", math.nan),
            str(base_param.get(f"{maintenance_type}_rho_used_source", "fixed_gain_fallback")),
        )
        param[f"{maintenance_type}_recovery_ratio_used"] = rho
        param[f"{maintenance_type}_rho_used_source"] = source
        param[f"{maintenance_type}_rho_reliable"] = source in {"device_rho", "global_rho"}
    param["hmax_scenario"] = "neutral"
    param["hmax_trend_used"] = float(base_param.get("hmax_trend_used", 0.0))
    param["hmax_annual_drop_ratio_used"] = float(base_param.get("hmax_annual_drop_ratio_used", 0.0))
    param["validation_start"] = validation_start
    return param


def _simulate_known_schedule(
    group: pd.DataFrame,
    validation_valid: pd.DataFrame,
    param: pd.Series,
    helpers: dict[str, object],
    model_name: str,
) -> pd.DataFrame:
    validation_start = pd.Timestamp(validation_valid["date"].min())
    validation_end = pd.Timestamp(validation_valid["date"].max())
    calendar = pd.DataFrame({"date": pd.date_range(validation_start, validation_end, freq="D")})
    merged = calendar.merge(
        group[["date", "month", "daily_median", "maintenance_type_on_day"]],
        on="date",
        how="left",
    )
    x_state = float(param["initial_x_state"])
    rolling_pred_values = group[
        (group["date"] < validation_start) & group["daily_median"].notna()
    ].sort_values("date")["daily_median"].astype(float).tail(364).tolist()
    rolling_true_values = rolling_pred_values.copy()
    pred_by_date: dict[pd.Timestamp, float] = {}
    rows: list[dict[str, object]] = []

    for day_index, row in merged.iterrows():
        date = pd.Timestamp(row["date"])
        month = int(row["month"]) if pd.notna(row.get("month")) else int(date.month)
        hmax_t = max(0.0, float(param["h_max_initial"]) + float(param["hmax_trend_used"]) * day_index)
        maintenance_type = _clean_maintenance_type(row.get("maintenance_type_on_day"))
        rho_used_source = ""
        if maintenance_type:
            if model_name == MODEL_FIXED:
                x_state = apply_fixed_gain(param, maintenance_type, month, x_state, hmax_t, helpers)
                rho_used_source = "fixed_gain_model"
            else:
                x_state, rho_used_source = apply_recovery_ratio(param, maintenance_type, month, x_state, hmax_t, helpers)
        else:
            decay_lambda = helpers["decay_lambda"].get(month, 1.0)
            x_state = x_state + float(param["cycle_decay_rate_used"]) * decay_lambda
        seasonal = helpers["seasonal_level"].get(month, 0.0)
        pred = x_state + seasonal
        pred_by_date[date] = pred
        if pd.notna(row.get("daily_median")):
            rolling_pred_values.append(pred)
            rolling_true_values.append(float(row["daily_median"]))
            if len(rolling_pred_values) > 365:
                rolling_pred_values = rolling_pred_values[-365:]
            if len(rolling_true_values) > 365:
                rolling_true_values = rolling_true_values[-365:]
            rolling_pred = float(np.mean(rolling_pred_values)) if len(rolling_pred_values) >= 365 else math.nan
            rolling_true = float(np.mean(rolling_true_values)) if len(rolling_true_values) >= 365 else math.nan
            rows.append(
                {
                    "device_id": param["device_id"],
                    "model_name": model_name,
                    "date": date,
                    "daily_median": float(row["daily_median"]),
                    "predicted_permeability": pred,
                    "error": pred - float(row["daily_median"]),
                    "rolling365_pred": rolling_pred,
                    "rolling365_true": rolling_true,
                    "maintenance_type": maintenance_type,
                    "rho_used_source": rho_used_source,
                }
            )
    return pd.DataFrame(rows)


def _post_maintenance_dates(group: pd.DataFrame, validation_dates: set[pd.Timestamp]) -> set[pd.Timestamp]:
    result: set[pd.Timestamp] = set()
    events = group.copy()
    events["maintenance_type_on_day"] = events["maintenance_type_on_day"].map(_clean_maintenance_type)
    for event_date in events.loc[events["maintenance_type_on_day"].isin(["medium", "major"]), "date"]:
        event_date = pd.Timestamp(event_date)
        for offset in range(1, 8):
            date = event_date + pd.Timedelta(days=offset)
            if date in validation_dates:
                result.add(date)
    return result


def _metrics(frame: pd.DataFrame, post_dates: set[pd.Timestamp], device_id: str, model_name: str) -> dict[str, object]:
    if frame.empty:
        err = pd.Series(dtype=float)
        rolling_err = pd.Series(dtype=float)
        post_err = pd.Series(dtype=float)
    else:
        err = frame["error"].astype(float)
        rolling_frame = frame[frame["rolling365_pred"].notna() & frame["rolling365_true"].notna()]
        rolling_err = rolling_frame["rolling365_pred"].astype(float) - rolling_frame["rolling365_true"].astype(float)
        if "is_post_maintenance_7d" in frame.columns:
            post_frame = frame[frame["is_post_maintenance_7d"].astype(bool)]
        else:
            post_frame = frame[frame["date"].isin(post_dates)]
        post_err = post_frame["error"].astype(float)
    return {
        "device_id": device_id,
        "model_name": model_name,
        "MAE": float(err.abs().mean()) if len(err) else math.nan,
        "RMSE": float(np.sqrt(np.mean(err**2))) if len(err) else math.nan,
        "post_maintenance_7d_MAE": float(post_err.abs().mean()) if len(post_err) else math.nan,
        "rolling365_MAE": float(rolling_err.abs().mean()) if len(rolling_err) else math.nan,
        "n_validation_days": int(len(err)),
        "n_post_maintenance_7d_days": int(len(post_err)),
        "n_rolling365_days": int(len(rolling_err)),
    }


def build_backtest_metrics(daily: pd.DataFrame, params: pd.DataFrame, helpers: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    post_date_by_device: dict[str, set[pd.Timestamp]] = {}

    for _, base_param in params.iterrows():
        device_id = str(base_param["device_id"])
        group = daily[daily["device_id"] == device_id].sort_values("date").copy()
        valid = group[group["daily_median"].notna()].sort_values("date")
        if len(valid) < 30:
            continue
        validation_valid = valid.tail(min(VALIDATION_DAYS, len(valid))).copy()
        validation_start = pd.Timestamp(validation_valid["date"].min())
        training = group[group["date"] < validation_start].copy()
        if training[training["daily_median"].notna()].empty:
            training = group[group["date"] <= validation_start].copy()
        backtest_param = _prepare_backtest_param(group, training, validation_valid, base_param, helpers)
        validation_dates = set(pd.to_datetime(validation_valid["date"]))
        post_dates = _post_maintenance_dates(
            group[(group["date"] >= validation_start) & (group["date"] <= pd.Timestamp(validation_valid["date"].max()))],
            validation_dates,
        )
        post_date_by_device[device_id] = post_dates
        for model_name in [MODEL_FIXED, MODEL_MAIN]:
            pred = _simulate_known_schedule(group, validation_valid, backtest_param, helpers, model_name)
            pred["is_post_maintenance_7d"] = pred["date"].isin(post_dates)
            prediction_frames.append(pred)
            rows.append(_metrics(pred, post_dates, device_id, model_name))

    validation = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    for model_name in [MODEL_FIXED, MODEL_MAIN]:
        frame = validation[validation["model_name"] == model_name] if len(validation) else pd.DataFrame()
        rows.append(_metrics(frame, set(), "all", model_name))

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["model_rank_key"] = result["device_id"].map(lambda x: -1 if x == "all" else int(str(x).replace("a", "")))
    result = result.sort_values(["model_rank_key", "model_name"]).drop(columns=["model_rank_key"]).reset_index(drop=True)
    return result
