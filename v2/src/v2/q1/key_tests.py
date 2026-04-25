from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SPEARMAN_VARIABLES = [
    "daily_median",
    "device_centered_permeability",
    "season_adjusted_permeability",
    "days_since_last_maintenance",
    "days_from_observation_start",
    "month",
    "is_post_medium_window",
    "is_post_major_window",
]


def _seasonal_level_map(season_decay: pd.DataFrame) -> dict[int, float]:
    rows = season_decay[season_decay["section"].astype(str) == "seasonal_level"].copy()
    return {
        int(row["month"]): float(row["value"])
        for _, row in rows.dropna(subset=["month", "value"]).iterrows()
    }


def _with_test_features(daily: pd.DataFrame, events: pd.DataFrame, season_decay: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["daily_median"] = pd.to_numeric(out["daily_median"], errors="coerce")
    device_median = out.groupby("device_id")["daily_median"].transform("median")
    seasonal_level = _seasonal_level_map(season_decay)
    out["device_centered_permeability"] = out["daily_median"] - device_median
    out["season_adjusted_permeability"] = (
        out["device_centered_permeability"] - out["month"].map(seasonal_level).fillna(0.0)
    )
    out["is_post_medium_window"] = 0
    out["is_post_major_window"] = 0

    event_rows = events.dropna(subset=["device_id", "event_date", "maintenance_type"]).copy()
    event_rows["event_date"] = pd.to_datetime(event_rows["event_date"], errors="coerce")
    for _, event in event_rows.iterrows():
        flag = f"is_post_{event['maintenance_type']}_window"
        if flag not in out.columns:
            continue
        start = event["event_date"] + pd.Timedelta(days=1)
        end = event["event_date"] + pd.Timedelta(days=7)
        mask = (out["device_id"] == event["device_id"]) & (out["date"] >= start) & (out["date"] <= end)
        out.loc[mask, flag] = 1
    return out


def _interpret_corr(x: str, y: str, corr: float, p_value: float) -> str:
    if not np.isfinite(corr):
        return "样本不足，不能解释相关方向。"
    direction = "正相关" if corr > 0 else "负相关" if corr < 0 else "近似无相关"
    strength = "较强" if abs(corr) >= 0.5 else "中等" if abs(corr) >= 0.3 else "较弱"
    sig = "显著" if np.isfinite(p_value) and p_value < 0.05 else "不显著"
    return f"{x} 与 {y} 呈{strength}{direction}，{sig}；该结果只说明变量关系方向，不代表因果。"


def _spearman_pair(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    pair = pair[np.isfinite(pair["left"].astype(float)) & np.isfinite(pair["right"].astype(float))]
    if len(pair) < 3 or pair["left"].nunique() <= 1 or pair["right"].nunique() <= 1:
        return np.nan, np.nan
    x = pair["left"].astype(float).rank(method="average").to_numpy(dtype=float)
    y = pair["right"].astype(float).rank(method="average").to_numpy(dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    x_values = [float(value) for value in x]
    y_values = [float(value) for value in y]
    xx = sum(value * value for value in x_values)
    yy = sum(value * value for value in y_values)
    xy = sum(left_value * right_value for left_value, right_value in zip(x_values, y_values))
    denom = math.sqrt(xx * yy)
    if denom <= 0:
        return np.nan, np.nan
    corr = float(xy / denom)
    corr = max(-1.0, min(1.0, corr))
    if abs(corr) >= 1.0:
        p_value = 0.0
    else:
        t_stat = corr * math.sqrt((len(pair) - 2) / max(1e-12, 1.0 - corr**2))
        p_value = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    return corr, p_value


def build_spearman_table(daily: pd.DataFrame, events: pd.DataFrame, season_decay: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _with_test_features(daily, events, season_decay)
    rows: list[dict[str, object]] = []
    corr_matrix = pd.DataFrame(index=SPEARMAN_VARIABLES, columns=SPEARMAN_VARIABLES, dtype=float)
    for var_x in SPEARMAN_VARIABLES:
        for var_y in SPEARMAN_VARIABLES:
            if var_x == var_y:
                corr_matrix.loc[var_x, var_y] = 1.0
                continue
            pair = features[[var_x, var_y]].dropna()
            pair = pair[np.isfinite(pair[var_x].astype(float)) & np.isfinite(pair[var_y].astype(float))]
            corr, p_value = _spearman_pair(pair[var_x], pair[var_y])
            corr_matrix.loc[var_x, var_y] = corr
            if var_x < var_y:
                rows.append(
                    {
                        "var_x": var_x,
                        "var_y": var_y,
                        "spearman_corr": corr,
                        "p_value": p_value,
                        "n_samples": int(len(pair)),
                        "interpretation": _interpret_corr(var_x, var_y, corr, p_value),
                    }
                )
    return pd.DataFrame(rows), corr_matrix


def _safe_wilcoxon(values: pd.Series, alternative: str) -> tuple[float, float]:
    clean = values.dropna().astype(float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 3 or np.allclose(clean, 0):
        return np.nan, np.nan
    stat, p_value = stats.wilcoxon(clean, alternative=alternative, zero_method="wilcox")
    return float(stat), float(p_value)


def _safe_mannwhitneyu(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    lvals = left.dropna().astype(float)
    rvals = right.dropna().astype(float)
    lvals = lvals[np.isfinite(lvals)]
    rvals = rvals[np.isfinite(rvals)]
    if len(lvals) < 3 or len(rvals) < 3:
        return np.nan, np.nan
    stat, p_value = stats.mannwhitneyu(lvals, rvals, alternative="two-sided")
    return float(stat), float(p_value)


def _conclusion(is_significant: bool, positive_text: str, negative_text: str) -> str:
    return positive_text if is_significant else negative_text


def build_key_test_tables(
    daily: pd.DataFrame,
    season_decay: pd.DataFrame,
    events: pd.DataFrame,
    cycles: pd.DataFrame,
    figure_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spearman, corr_matrix = build_spearman_table(daily, events, season_decay)
    summary_rows: list[dict[str, object]] = []

    target_pairs = [
        ("daily_median", "days_since_last_maintenance"),
        ("season_adjusted_permeability", "days_since_last_maintenance"),
        ("daily_median", "days_from_observation_start"),
        ("device_centered_permeability", "month"),
        ("daily_median", "is_post_medium_window"),
        ("daily_median", "is_post_major_window"),
    ]
    for var_x, var_y in target_pairs:
        row = spearman[(spearman["var_x"] == min(var_x, var_y)) & (spearman["var_y"] == max(var_x, var_y))]
        if row.empty:
            continue
        item = row.iloc[0]
        summary_rows.append(
            {
                "section": "Spearman相关性",
                "test_name": f"{var_x} vs {var_y}",
                "method": "Spearman rank correlation",
                "target": f"{var_x}; {var_y}",
                "statistic": item["spearman_corr"],
                "p_value": item["p_value"],
                "conclusion": item["interpretation"],
                "used_in_paper": "用于说明变量关系方向，不作为因果结论。",
            }
        )

    eligible_events = events.copy()
    eligible_events["plateau_gain"] = pd.to_numeric(eligible_events["plateau_gain"], errors="coerce")
    eligible_events["plateau_gain_ratio"] = pd.to_numeric(eligible_events["plateau_gain_ratio"], errors="coerce")
    eligible_events["hold_ratio_7d"] = pd.to_numeric(eligible_events["hold_ratio_7d"], errors="coerce")
    if "eligible_plateau" in eligible_events.columns:
        eligible_events = eligible_events[eligible_events["eligible_plateau"].astype(bool)]
    for maintenance_type in ["medium", "major"]:
        group = eligible_events[eligible_events["maintenance_type"] == maintenance_type]
        stat, p_value = _safe_wilcoxon(group["plateau_gain"], alternative="greater")
        significant = bool(np.isfinite(p_value) and p_value < 0.05)
        summary_rows.append(
            {
                "section": "维护前后显著性",
                "test_name": f"{maintenance_type}_pre_post_wilcoxon",
                "method": "Wilcoxon signed-rank test, alternative: gain > 0",
                "target": f"{maintenance_type} plateau_gain",
                "statistic": stat,
                "p_value": p_value,
                "conclusion": _conclusion(
                    significant,
                    f"{maintenance_type}维护后透水率平台水平相较维护前显著提升。",
                    f"{maintenance_type}维护后提升未达到显著水平，可能与样本量或维护发生工况差异有关，不能直接判定维护无效。",
                ),
                "used_in_paper": "用于验证维护确实具有恢复作用。",
            }
        )

    medium = eligible_events[eligible_events["maintenance_type"] == "medium"]
    major = eligible_events[eligible_events["maintenance_type"] == "major"]
    for metric in ["plateau_gain_ratio", "hold_ratio_7d"]:
        stat, p_value = _safe_mannwhitneyu(medium[metric], major[metric])
        significant = bool(np.isfinite(p_value) and p_value < 0.05)
        summary_rows.append(
            {
                "section": "中大维护差异",
                "test_name": f"medium_vs_major_{metric}",
                "method": "Mann-Whitney U test",
                "target": metric,
                "statistic": stat,
                "p_value": p_value,
                "conclusion": _conclusion(
                    significant,
                    f"中维护与大维护在 {metric} 上存在显著差异。",
                    f"中维护与大维护在 {metric} 上未表现出显著差异，不能简单假设大维护一定优于中维护。",
                ),
                "used_in_paper": "用于说明中/大维护不能只按短期提升大小排序。",
            }
        )

    cycle_rates = pd.to_numeric(cycles.get("cycle_decay_rate", pd.Series(dtype=float)), errors="coerce")
    if "eligible_for_cycle_analysis" in cycles.columns:
        cycle_rates = pd.to_numeric(cycles.loc[cycles["eligible_for_cycle_analysis"].astype(bool), "cycle_decay_rate"], errors="coerce")
    stat, p_value = _safe_wilcoxon(cycle_rates, alternative="less")
    significant = bool(np.isfinite(p_value) and p_value < 0.05)
    summary_rows.append(
        {
            "section": "维护周期衰减",
            "test_name": "cycle_decay_rate_less_than_zero",
            "method": "One-sample Wilcoxon signed-rank test, alternative: median < 0",
            "target": "cycle_decay_rate",
            "statistic": stat,
            "p_value": p_value,
            "conclusion": _conclusion(
                significant,
                "维护周期内透水率存在显著下降趋势，支持第二问非维护日退化方程。",
                "维护周期衰减未达到显著水平，仍作为工程退化趋势进行描述。",
            ),
            "used_in_paper": "用于验证非维护日退化项的必要性。",
        }
    )

    _write_heatmap(corr_matrix, figure_path)
    return spearman, pd.DataFrame(summary_rows)


def _write_heatmap(corr_matrix: pd.DataFrame, figure_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        for candidate in [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _color(value: float) -> tuple[int, int, int]:
        if not np.isfinite(value):
            return (235, 235, 235)
        value = max(-1.0, min(1.0, float(value)))
        if value < 0:
            ratio = value + 1.0
            return (
                int(49 + (247 - 49) * ratio),
                int(130 + (247 - 130) * ratio),
                int(189 + (247 - 189) * ratio),
            )
        ratio = value
        return (
            int(247 + (202 - 247) * ratio),
            int(247 + (0 - 247) * ratio),
            int(247 + (32 - 247) * ratio),
        )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    variables = list(corr_matrix.index)
    cell = 76
    left = 260
    top = 100
    width = left + cell * len(variables) + 40
    height = top + cell * len(variables) + 260
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(24)
    label_font = _font(13)
    value_font = _font(15)
    draw.text((40, 28), "透水率关键变量 Spearman 相关性热力图", fill=(20, 20, 20), font=title_font)
    for row_idx, row_name in enumerate(variables):
        y = top + row_idx * cell
        draw.text((20, y + 25), row_name, fill=(30, 30, 30), font=label_font)
        for col_idx, col_name in enumerate(variables):
            x = left + col_idx * cell
            value = pd.to_numeric(pd.Series([corr_matrix.loc[row_name, col_name]]), errors="coerce").iloc[0]
            draw.rectangle([x, y, x + cell, y + cell], fill=_color(value), outline=(255, 255, 255))
            text = "NA" if not np.isfinite(value) else f"{value:.2f}"
            text_color = (255, 255, 255) if np.isfinite(value) and abs(float(value)) > 0.55 else (30, 30, 30)
            box = draw.textbbox((0, 0), text, font=value_font)
            draw.text(
                (x + (cell - (box[2] - box[0])) / 2, y + (cell - (box[3] - box[1])) / 2),
                text,
                fill=text_color,
                font=value_font,
            )
    for col_idx, col_name in enumerate(variables):
        x = left + col_idx * cell
        draw.text((x + 4, top + cell * len(variables) + 12), col_name, fill=(30, 30, 30), font=label_font)
    draw.text((40, height - 60), "说明：颜色越红表示正相关越强，越蓝表示负相关越强；该图只用于观察变量关系方向。", fill=(50, 50, 50), font=label_font)
    image.save(figure_path)


def write_key_test_markdown(path: Path) -> None:
    content = """# 第一问检验补充说明

## 1. 检验目的

第一问补充检验不是为了堆叠统计方法，而是为建模链条提供必要证据：透水率与维护间隔、时间、季节等变量存在关系；维护前后确实存在恢复作用；维护周期内存在退化趋势。

## 2. 相关性分析

Spearman 相关性分析用于观察变量关系方向，变量包括透水率、设备中心化透水率、季节调整透水率、距上次维护天数、观测起点以来天数、月份和维护后窗口标记。

相关性分析只说明变量关系方向，不代表因果关系。论文中可用热力图说明变量选择依据，但不能把相关性直接解释为因果效应。

## 3. 维护前后检验

中维护和大维护分别使用 Wilcoxon signed-rank test 检验维护后短期平台水平是否高于维护前基准。若 p < 0.05，说明该类维护后透水率平台水平相较维护前显著提升。

若某类维护不显著，也不应直接写成“维护无效”。更稳妥的解释是样本量有限、维护发生时设备状态和季节条件不同，导致观测提升不一定显著。

## 4. 中/大维护差异

中维护和大维护只比较 `plateau_gain_ratio` 与 `hold_ratio_7d` 两个指标。若差异不显著，说明不能简单假设大维护一定优于中维护，第三问需要综合恢复效果、成本和可能损伤来决策。

## 5. 维护周期衰减

维护周期衰减检验使用单样本 Wilcoxon 检验 `cycle_decay_rate < 0`。若显著，说明两次维护之间透水率存在净下降趋势，支持第二问非维护日退化方程。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
