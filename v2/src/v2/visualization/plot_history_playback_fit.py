from __future__ import annotations

"""
本脚本仅用于绘制第二问模型在历史真实维护日程下的回放拟合效果。
该图用于检查 fixed gain baseline 和 recovery ratio main 对历史透水率变化的解释能力。
本脚本不改变第二问寿命预测结果，也不参与维护策略优化。
"""

import argparse
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..q2.build_parameters import build_model_parameters
from ..q2.load_q1_outputs import load_q1_outputs
from ..q2.paths import get_paths
from ..q2.simulate_models import MODEL_FIXED, MODEL_MAIN, apply_fixed_gain, apply_recovery_ratio


DEFAULT_VALIDATION_DAYS = 180
THRESHOLD = 37.0


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _clean_maintenance_type(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text if text in {"medium", "major"} else ""


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _select_playback_window(group: pd.DataFrame, full_history: bool) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    valid = group[group["daily_median"].notna()].sort_values("date").copy()
    if valid.empty:
        raise ValueError("no valid daily_median records")
    window_valid = valid if full_history else valid.tail(min(DEFAULT_VALIDATION_DAYS, len(valid))).copy()
    return pd.Timestamp(window_valid["date"].min()), pd.Timestamp(window_valid["date"].max()), window_valid


def _simulate_known_history(
    group: pd.DataFrame,
    param: pd.Series,
    helpers: dict[str, object],
    model_name: str,
    full_history: bool,
) -> pd.DataFrame:
    start_date, end_date, window_valid = _select_playback_window(group, full_history)
    first_valid = window_valid.iloc[0]
    initial_month = int(first_valid["month"])
    x_state = float(first_valid["daily_median"]) - helpers["seasonal_level"].get(initial_month, 0.0)
    calendar = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    merged = calendar.merge(
        group[["date", "month", "daily_median", "maintenance_type_on_day"]],
        on="date",
        how="left",
    )

    rows: list[dict[str, object]] = []
    for day_index, row in merged.iterrows():
        date = pd.Timestamp(row["date"])
        month = int(row["month"]) if pd.notna(row.get("month")) else int(date.month)
        hmax_t = max(0.0, float(param["h_max_initial"]) + float(param["hmax_trend_used"]) * day_index)
        maintenance_type = _clean_maintenance_type(row.get("maintenance_type_on_day"))
        rho_used_source = ""

        # The first valid day anchors the playback state to the real observed level.
        if date != start_date:
            if maintenance_type:
                if model_name == MODEL_FIXED:
                    x_state = apply_fixed_gain(param, maintenance_type, month, x_state, hmax_t, helpers)
                    rho_used_source = "fixed_gain_model"
                else:
                    x_state, rho_used_source = apply_recovery_ratio(
                        param,
                        maintenance_type,
                        month,
                        x_state,
                        hmax_t,
                        helpers,
                    )
            else:
                decay_lambda = helpers["decay_lambda"].get(month, 1.0)
                x_state += float(param["cycle_decay_rate_used"]) * decay_lambda

        seasonal = helpers["seasonal_level"].get(month, 0.0)
        predicted = x_state + seasonal
        rows.append(
            {
                "device_id": param["device_id"],
                "model_name": model_name,
                "date": date,
                "daily_median": float(row["daily_median"]) if pd.notna(row.get("daily_median")) else math.nan,
                "predicted_permeability": predicted,
                "maintenance_type": maintenance_type,
                "rho_used_source": rho_used_source,
                "hmax_t": hmax_t,
            }
        )
    return pd.DataFrame(rows)


def _post_maintenance_dates(group: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    events = group.copy()
    events["maintenance_type_on_day"] = events["maintenance_type_on_day"].map(_clean_maintenance_type)
    events = events[
        (events["date"] >= start)
        & (events["date"] <= end)
        & events["maintenance_type_on_day"].isin(["medium", "major"])
    ]
    valid_dates = set(pd.to_datetime(group.loc[group["daily_median"].notna(), "date"]))
    for event_date in pd.to_datetime(events["date"]):
        for offset in range(1, 8):
            date = event_date + pd.Timedelta(days=offset)
            if start <= date <= end and date in valid_dates:
                dates.add(date)
    return dates


def _metric_row(
    frame: pd.DataFrame,
    post_dates: set[pd.Timestamp],
    device_id: str,
    model_name: str,
    note: str,
) -> dict[str, object]:
    valid = frame[frame["daily_median"].notna()].copy()
    if valid.empty:
        err = pd.Series(dtype=float)
        post_err = pd.Series(dtype=float)
    else:
        valid["error"] = valid["predicted_permeability"].astype(float) - valid["daily_median"].astype(float)
        err = valid["error"].astype(float)
        post_err = valid.loc[valid["date"].isin(post_dates), "error"].astype(float)
    return {
        "device_id": device_id,
        "model_name": model_name,
        "fit_period_start": frame["date"].min().strftime("%Y-%m-%d") if len(frame) else "",
        "fit_period_end": frame["date"].max().strftime("%Y-%m-%d") if len(frame) else "",
        "n_days": int(len(err)),
        "mae": float(err.abs().mean()) if len(err) else math.nan,
        "rmse": float(np.sqrt(np.mean(err**2))) if len(err) else math.nan,
        "post_maintenance_7d_mae": float(post_err.abs().mean()) if len(post_err) else math.nan,
        "note": note,
    }


def _plot_device(
    device_id: str,
    group: pd.DataFrame,
    fixed: pd.DataFrame,
    recovery: pd.DataFrame,
    output_dir: Path,
) -> None:
    actual = group[(group["date"] >= fixed["date"].min()) & (group["date"] <= fixed["date"].max())].copy()
    actual = actual[actual["daily_median"].notna()]
    events = group[
        (group["date"] >= fixed["date"].min())
        & (group["date"] <= fixed["date"].max())
        & group["maintenance_type_on_day"].map(_clean_maintenance_type).isin(["medium", "major"])
    ].copy()
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=180)
    ax.plot(actual["date"], actual["daily_median"], color="#111827", linewidth=1.4, label="真实日中位透水率")
    ax.plot(fixed["date"], fixed["predicted_permeability"], color="#2563eb", linewidth=1.2, label="fixed gain baseline 回放")
    ax.plot(recovery["date"], recovery["predicted_permeability"], color="#dc2626", linewidth=1.2, label="recovery ratio main 回放")

    shown_medium = False
    shown_major = False
    for _, event in events.iterrows():
        event_type = _clean_maintenance_type(event["maintenance_type_on_day"])
        if event_type == "medium":
            ax.axvline(
                event["date"],
                color="#059669",
                linestyle="--",
                linewidth=0.75,
                alpha=0.45,
                label="中维护日期" if not shown_medium else None,
            )
            shown_medium = True
        elif event_type == "major":
            ax.axvline(
                event["date"],
                color="#7c3aed",
                linestyle="-.",
                linewidth=0.85,
                alpha=0.55,
                label="大维护日期" if not shown_major else None,
            )
            shown_major = True
    ax.axhline(THRESHOLD, color="#f97316", linestyle=":", linewidth=1.2, label="寿命阈值 37")
    ax.set_title(f"设备 {device_id} 历史回放拟合图：真实透水率与两类退化模型对比")
    ax.set_xlabel("日期")
    ax.set_ylabel("透水率")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / f"{device_id}_history_playback_fit.png")
    plt.close(fig)


def run(full_history: bool = False) -> None:
    _configure_matplotlib()
    paths = get_paths()
    q1_outputs = load_q1_outputs(paths)
    _, helpers = build_model_parameters(q1_outputs)
    params_path = paths.q2_tables_dir / "表01_模型参数表.csv"
    params = _read_csv(params_path)
    daily = q1_outputs["daily"].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

    output_dir = paths.root / "v2" / "figures" / "history_fit"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []

    for device_id in sorted(daily["device_id"].dropna().unique(), key=lambda x: int(str(x).replace("a", ""))):
        try:
            group = daily[daily["device_id"] == device_id].sort_values("date").copy()
            param = params[params["device_id"] == device_id]
            if param.empty:
                raise ValueError("missing q2 model parameters")
            param_row = param.iloc[0].copy()
            fixed = _simulate_known_history(group, param_row, helpers, MODEL_FIXED, full_history)
            recovery = _simulate_known_history(group, param_row, helpers, MODEL_MAIN, full_history)
            post_dates = _post_maintenance_dates(group, fixed["date"].min(), fixed["date"].max())
            note = "full_history" if full_history else "default_last_180_valid_days"
            metrics.append(_metric_row(fixed, post_dates, device_id, MODEL_FIXED, note))
            metrics.append(_metric_row(recovery, post_dates, device_id, MODEL_MAIN, note))
            _plot_device(device_id, group, fixed, recovery, output_dir)
        except Exception as exc:  # pragma: no cover - defensive batch behavior
            warnings.warn(f"{device_id} history playback failed: {exc}")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / "history_fit_metrics_summary.csv", index=False, encoding="utf-8-sig")
    print(f"history playback fit outputs written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Q2 history playback fit for each device.")
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Use the full historical period instead of the default last 180 valid days.",
    )
    args = parser.parse_args()
    run(full_history=bool(args.full_history))


if __name__ == "__main__":
    main()
