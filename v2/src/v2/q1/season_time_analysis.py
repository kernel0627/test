from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .build_daily_features import usable_daily
from .load_inputs import sort_by_device, sorted_device_ids


def _safe_quantile(series: pd.Series, q: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.quantile(q)) if len(clean) else math.nan


def _inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    eye = np.eye(n, dtype=float)
    aug = np.hstack([matrix.copy(), eye])
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row, col]))
        if abs(aug[pivot, col]) < 1e-10:
            aug[col, col] += 1e-6
            pivot = col
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        pivot_value = aug[col, col]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Regression matrix is singular.")
        aug[col] = aug[col] / pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = aug[row, col]
            if factor != 0:
                aug[row] = aug[row] - factor * aug[col]
    return aug[:, n:]


def _ols(y: np.ndarray, design: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float]:
    x = design.astype(float).to_numpy()
    y = np.asarray(y, dtype=float)
    xtx = np.einsum("ni,nj->ij", x, x)
    xtx_inv = _inverse(xtx)
    xty = np.einsum("ni,n->i", x, y)
    beta = np.einsum("ij,j->i", xtx_inv, xty)
    fitted = np.einsum("ni,i->n", x, beta)
    resid = y - fitted
    n = len(y)
    p = x.shape[1]
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum(resid**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    sigma2 = ss_res / max(n - p, 1)
    cov = sigma2 * xtx_inv
    std_err = np.sqrt(np.clip(np.diag(cov), 0, None))
    t_value = np.divide(beta, std_err, out=np.full_like(beta, np.nan), where=std_err > 0)
    # Normal approximation is sufficient for this diagnostic table.
    p_value = np.array([math.erfc(abs(t) / math.sqrt(2.0)) if np.isfinite(t) else math.nan for t in t_value])
    summary = pd.DataFrame(
        {
            "term": design.columns,
            "coef": beta,
            "std_err": std_err,
            "t_value": t_value,
            "p_value": p_value,
            "n_samples": n,
            "r_squared": r_squared,
            "model_name": "daily_median_device_month_tau_time_ols",
        }
    )
    return summary, resid, r_squared


def _device_dummies(series: pd.Series) -> pd.DataFrame:
    devices = sorted_device_ids(series.dropna().unique())
    categorical = pd.Categorical(series, categories=devices, ordered=True)
    dummies = pd.get_dummies(categorical, prefix="device", drop_first=True, dtype=float)
    dummies.index = series.index
    return dummies


def _month_dummies(series: pd.Series) -> pd.DataFrame:
    categorical = pd.Categorical(series.astype(int), categories=list(range(1, 13)), ordered=True)
    dummies = pd.get_dummies(categorical, prefix="month", drop_first=True, dtype=float)
    dummies.index = series.index
    return dummies


def build_season_time_table(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = usable_daily(daily)
    usable = usable.dropna(subset=["daily_median", "days_since_last_maintenance", "days_from_observation_start"]).copy()
    if usable.empty:
        raise ValueError("No usable daily rows for season/time analysis.")

    alpha = usable.groupby("device_id")["daily_median"].median().rename("alpha_median")
    usable = usable.merge(alpha, on="device_id", how="left")
    usable["centered_per"] = usable["daily_median"] - usable["alpha_median"]

    raw = (
        usable.groupby("month")
        .agg(
            seasonal_index_raw=("centered_per", "median"),
            n_samples=("daily_median", "size"),
            median_centered_per=("centered_per", "median"),
            q25_centered_per=("centered_per", lambda s: _safe_quantile(s, 0.25)),
            q75_centered_per=("centered_per", lambda s: _safe_quantile(s, 0.75)),
        )
        .reindex(range(1, 13))
        .reset_index()
    )

    design_no_month = usable[["days_since_last_maintenance", "days_from_observation_start"]].copy()
    design_no_month = pd.concat([design_no_month, _device_dummies(usable["device_id"])], axis=1)
    design_no_month.insert(0, "const", 1.0)
    _, residual_no_month, _ = _ols(usable["daily_median"].to_numpy(dtype=float), design_no_month)
    usable["residual_no_month"] = residual_no_month

    adjusted = (
        usable.groupby("month")
        .agg(seasonal_index_adjusted=("residual_no_month", "median"))
        .reindex(range(1, 13))
        .reset_index()
    )
    seasonal = raw.merge(adjusted, on="month", how="left")
    seasonal["seasonal_index_used"] = seasonal["seasonal_index_adjusted"]
    center = seasonal["seasonal_index_used"].dropna().mean()
    seasonal["seasonal_index_used"] = seasonal["seasonal_index_used"] - center

    design = usable[["days_since_last_maintenance", "days_from_observation_start"]].copy()
    design = pd.concat([design, _device_dummies(usable["device_id"]), _month_dummies(usable["month"])], axis=1)
    design.insert(0, "const", 1.0)
    regression, _, _ = _ols(usable["daily_median"].to_numpy(dtype=float), design)

    seasonal_rows = pd.DataFrame(
        {
            "section": "seasonal_index",
            "term": "seasonal_index_used",
            "month": seasonal["month"],
            "value": seasonal["seasonal_index_used"],
            "coef": np.nan,
            "std_err": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "n_samples": seasonal["n_samples"],
            "r_squared": np.nan,
            "model_name": "median_residual_centered_month_index",
        }
    )
    regression_rows = regression.assign(section="regression", month=np.nan, value=np.nan)[
        ["section", "term", "month", "value", "coef", "std_err", "t_value", "p_value", "n_samples", "r_squared", "model_name"]
    ]
    combined = pd.concat(
        [
            seasonal_rows[
                ["section", "term", "month", "value", "coef", "std_err", "t_value", "p_value", "n_samples", "r_squared", "model_name"]
            ],
            regression_rows,
        ],
        ignore_index=True,
    )
    return combined, seasonal


def build_data_overview(daily: pd.DataFrame) -> pd.DataFrame:
    usable = usable_daily(daily)
    rows: list[dict[str, object]] = []
    for device_id, group in daily.groupby("device_id", sort=False):
        usable_group = usable[usable["device_id"] == device_id]
        values = usable_group["daily_median"].astype(float)
        rows.append(
            {
                "device_id": device_id,
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
                "n_valid_days": int(values.count()),
                "n_maintenance_gap_days": int((group["daily_quality"] == "maintenance_gap").sum()),
                "n_random_gap_days": int((group["daily_quality"] == "random_gap").sum()),
                "alpha_median": float(values.median()) if len(values) else math.nan,
                "alpha_mean": float(values.mean()) if len(values) else math.nan,
                "daily_median_std": float(values.std(ddof=0)) if len(values) >= 2 else math.nan,
                "daily_median_min": float(values.min()) if len(values) else math.nan,
                "daily_median_max": float(values.max()) if len(values) else math.nan,
                "n_days_below_37": int((values < 37).sum()),
            }
        )
    return sort_by_device(pd.DataFrame(rows), "device_id")
