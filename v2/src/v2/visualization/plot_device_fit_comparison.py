from __future__ import annotations

"""Plot one device's historical permeability and Q2 playback-fit curves.

The script is deterministic: it reads existing Q1/Q2 CSV outputs and reuses the
Q2 history playback transition functions. No random sampling or model training
is performed.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..q2.build_parameters import build_model_parameters
from ..q2.load_q1_outputs import load_q1_outputs
from ..q2.paths import get_paths
from ..q2.simulate_models import MODEL_FIXED, MODEL_MAIN
from .plot_history_playback_fit import _configure_matplotlib, _simulate_known_history


def _read_params(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _output_path(root: Path, output: str | None, device_id: str, start_year: str, end_year: str) -> Path:
    if output:
        path = Path(output)
        return path if path.is_absolute() else root / path
    return root / f"{device_id}_{start_year}_{end_year}_fit_vs_true.png"


def plot_device_fit(
    device_id: str,
    start_date: str,
    end_date: str,
    output: str | None = None,
) -> Path:
    _configure_matplotlib()
    paths = get_paths()
    q1_outputs = load_q1_outputs(paths)
    _, helpers = build_model_parameters(q1_outputs)
    params = _read_params(paths.q2_tables_dir / "表01_模型参数表.csv")
    daily = q1_outputs["daily"].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

    group = daily[daily["device_id"].astype(str) == device_id].sort_values("date").copy()
    if group.empty:
        raise ValueError(f"No daily records found for device {device_id}.")
    param = params[params["device_id"].astype(str) == device_id]
    if param.empty:
        raise ValueError(f"No Q2 parameters found for device {device_id}.")
    param_row = param.iloc[0].copy()

    fixed = _simulate_known_history(group, param_row, helpers, MODEL_FIXED, full_history=True)
    recovery = _simulate_known_history(group, param_row, helpers, MODEL_MAIN, full_history=True)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    actual = group[(group["date"] >= start) & (group["date"] <= end) & group["daily_median"].notna()].copy()
    fixed = fixed[(fixed["date"] >= start) & (fixed["date"] <= end)].copy()
    recovery = recovery[(recovery["date"] >= start) & (recovery["date"] <= end)].copy()
    if actual.empty:
        raise ValueError(f"No valid actual daily_median values for {device_id} in {start_date} to {end_date}.")

    fig, ax = plt.subplots(figsize=(13, 6), dpi=200)
    ax.plot(
        actual["date"],
        actual["daily_median"],
        color="#111827",
        linewidth=1.5,
        label="真实日中位透水率",
    )
    ax.plot(
        fixed["date"],
        fixed["predicted_permeability"],
        color="#2563eb",
        linewidth=1.25,
        label="拟合透水率：fixed gain baseline",
    )
    ax.plot(
        recovery["date"],
        recovery["predicted_permeability"],
        color="#dc2626",
        linewidth=1.25,
        label="拟合透水率：recovery ratio main",
    )
    ax.axhline(37.0, color="#f97316", linestyle=":", linewidth=1.2, label="寿命阈值 37")
    ax.set_title(f"设备 {device_id} 2024-2026 年透水率拟合与真实数据对比")
    ax.set_xlabel("日期")
    ax.set_ylabel("透水率")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = _output_path(paths.root, output, device_id, str(start.year), str(end.year))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one device's true vs fitted permeability curves.")
    parser.add_argument("--device", default="a1", help="Device id, for example a1.")
    parser.add_argument("--start", default="2024-01-01", help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default="2026-12-31", help="End date, YYYY-MM-DD.")
    parser.add_argument("--output", default=None, help="Output PNG path. Relative paths are under project root.")
    args = parser.parse_args()
    output_path = plot_device_fit(args.device, args.start, args.end, args.output)
    print(f"device fit comparison plot written to: {output_path}")


if __name__ == "__main__":
    main()
