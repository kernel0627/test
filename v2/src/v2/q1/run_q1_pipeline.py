from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .build_core_metrics import build_core_metrics
from .build_daily_features import build_daily_features, clean_hourly
from .cycle_decay import build_cycles
from .export_chinese_headers import write_chinese_header_tables
from .hmax_recovery import build_hmax_recovery
from .load_inputs import load_maintenance, load_observations, sorted_device_ids
from .maintenance_effect import build_event_effects
from .maintenance_rule import build_current_maintenance_rule
from .paths import ensure_output_dirs, get_paths
from .pure_decay import build_pure_decay_segments
from .season_decay_analysis import build_data_overview_v2, build_season_decay_table
from .seasonal_maintenance_effect import build_seasonal_maintenance_effect
from .write_summary import write_markdown_summary


DAILY_FEATURES_FILE = "日尺度特征表.csv"
CYCLE_DETAIL_FILE = "维护周期明细表.csv"
EVENT_DETAIL_FILE = "维护事件明细表.csv"
PURE_DECAY_FILE = "纯净衰减片段表.csv"

TABLE_01_FILE = "表01_数据质量与设备基础概览.csv"
TABLE_02_FILE = "表02_季节水平项与月度衰减强度.csv"
TABLE_03_FILE = "表03_当前固定维护触发规律.csv"
TABLE_04_FILE = "表04_衰减指标汇总.csv"
TABLE_05_FILE = "表05_维护事件效应汇总.csv"
TABLE_06_FILE = "表06_Hmax与恢复比例质量评估.csv"
TABLE_07_FILE = "表07_设备核心指标汇总.csv"
TABLE_08_FILE = "表08_季节对维护效应影响.csv"

LEGACY_TABLE_FILES = [
    "表02_季节项与长期时间衰减.csv",
    "表03_当前固定维护规律.csv",
    "表04_维护周期衰减汇总.csv",
    "表06_Hmax与恢复比例汇总.csv",
]

DATE_COLUMNS = {
    "date",
    "event_date",
    "start_date",
    "end_date",
    "start_event_date",
    "next_event_date",
    "last_maintenance_date",
    "commission_date",
}


def _format_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in DATE_COLUMNS.intersection(out.columns):
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
        out[column] = out[column].fillna("")
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _format_for_csv(df).to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        if path.exists():
            print("Skipped one locked existing CSV file.")
            return
        raise


def _archive_legacy_tables(paths) -> None:
    archive_dir = paths.tables_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for filename in LEGACY_TABLE_FILES:
        source = paths.tables_dir / filename
        if not source.exists():
            continue
        target = archive_dir / filename
        if target.exists():
            stem = target.stem
            suffix = target.suffix
            counter = 1
            while True:
                candidate = archive_dir / f"{stem}_archived_{counter}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                counter += 1
        try:
            source.replace(target)
        except PermissionError:
            print(f"Skipped locked legacy table: {source.name}")


def _build_decay_summary(cycle_summary: pd.DataFrame, pure_summary: pd.DataFrame) -> pd.DataFrame:
    merged = cycle_summary.merge(pure_summary, on="device_id", how="outer")
    columns = [
        "device_id",
        "n_valid_cycles",
        "cycle_length_median",
        "cycle_decay_rate_median",
        "cycle_decay_rate_q25",
        "cycle_decay_rate_q75",
        "n_pure_decay_segments",
        "pure_decay_rate_median",
        "pure_decay_rate_q25",
        "pure_decay_rate_q75",
        "early_pure_decay_rate_median",
        "late_pure_decay_rate_median",
        "aging_acceleration_ratio",
        "monthly_decay_sensitive",
    ]
    for column in columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged[columns]


def _assert_outputs(paths, maintenance: pd.DataFrame) -> None:
    expected = [
        paths.cleaned_dir / DAILY_FEATURES_FILE,
        paths.cleaned_dir / CYCLE_DETAIL_FILE,
        paths.cleaned_dir / EVENT_DETAIL_FILE,
        paths.cleaned_dir / PURE_DECAY_FILE,
        paths.tables_dir / TABLE_01_FILE,
        paths.tables_dir / TABLE_02_FILE,
        paths.tables_dir / TABLE_03_FILE,
        paths.tables_dir / TABLE_04_FILE,
        paths.tables_dir / TABLE_05_FILE,
        paths.tables_dir / TABLE_06_FILE,
        paths.tables_dir / TABLE_07_FILE,
        paths.tables_dir / TABLE_08_FILE,
        paths.markdown_dir / "第一问分析总结.md",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise AssertionError("Missing expected q1 outputs: " + ", ".join(missing))

    rule = pd.read_csv(paths.tables_dir / TABLE_03_FILE)
    core = pd.read_csv(paths.tables_dir / TABLE_07_FILE)
    hmax = pd.read_csv(paths.tables_dir / TABLE_06_FILE)
    season = pd.read_csv(paths.tables_dir / TABLE_02_FILE)
    effect = pd.read_csv(paths.tables_dir / TABLE_05_FILE)
    events = pd.read_csv(paths.cleaned_dir / EVENT_DETAIL_FILE)

    if len(rule) != 10:
        raise AssertionError("Q1 table03 must contain 10 rows.")
    if len(core) != 10:
        raise AssertionError("Q1 table07 must contain 10 rows.")
    if not (hmax["h_max_initial"].astype(float) + 1e-9 >= hmax["current_state_level"].astype(float)).all():
        raise AssertionError("Q1 table06 has h_max_initial below current_state_level.")
    for maintenance_type in ["medium", "major"]:
        ratio_col = f"{maintenance_type}_rho_clip_ratio"
        reliable_col = f"{maintenance_type}_rho_reliable"
        mask = hmax[ratio_col].fillna(0).astype(float) > 0.3
        if mask.any() and hmax.loc[mask, reliable_col].astype(str).str.lower().isin(["true", "1"]).any():
            raise AssertionError("Q1 table06 rho reliability violates clip-ratio rule.")
    sections = set(season["section"].astype(str))
    if not {"seasonal_level", "monthly_decay_intensity"}.issubset(sections):
        raise AssertionError("Q1 table02 must contain seasonal_level and monthly_decay_intensity.")
    required_effect_columns = {"jump_gain_median", "jump_gain_ratio_median", "plateau_gain_median", "plateau_gain_ratio_median"}
    if not required_effect_columns.issubset(set(effect.columns)):
        raise AssertionError("Q1 table05 is missing absolute or ratio gain columns.")
    if len(events) != len(maintenance):
        raise AssertionError("Q1 event detail does not contain all maintenance events.")
    for pattern in ["*.png", "*.html"]:
        found = list(paths.q1_dir.rglob(pattern))
        if found:
            raise AssertionError(f"Unexpected visual outputs found: {found[:3]}")
    if (paths.q1_dir / "figures").exists():
        raise AssertionError("Unexpected figures directory under v2/q_1.")


def main() -> None:
    paths = get_paths()
    ensure_output_dirs(paths)
    _archive_legacy_tables(paths)

    hourly_raw = load_observations(paths)
    maintenance = load_maintenance(paths)
    hourly = clean_hourly(hourly_raw, maintenance)
    daily = build_daily_features(hourly, maintenance)
    device_ids = sorted_device_ids(daily["device_id"].unique())

    overview = build_data_overview_v2(daily)
    cycles, cycle_summary = build_cycles(daily, maintenance)
    pure_segments, pure_summary = build_pure_decay_segments(daily, maintenance)
    decay_summary = _build_decay_summary(cycle_summary, pure_summary)
    season_decay, _seasonal = build_season_decay_table(daily, pure_segments)
    rule = build_current_maintenance_rule(maintenance, device_ids, daily)
    events, effect_summary = build_event_effects(daily, maintenance)
    recovery_summary, events_with_rho = build_hmax_recovery(daily, events, overview, device_ids)
    seasonal_effect = build_seasonal_maintenance_effect(events_with_rho)
    core = build_core_metrics(device_ids, overview, rule, decay_summary, effect_summary, recovery_summary)

    table_outputs = {
        TABLE_01_FILE: overview,
        TABLE_02_FILE: season_decay,
        TABLE_03_FILE: rule,
        TABLE_04_FILE: decay_summary,
        TABLE_05_FILE: effect_summary,
        TABLE_06_FILE: recovery_summary,
        TABLE_07_FILE: core,
        TABLE_08_FILE: seasonal_effect,
    }

    _write_csv(daily, paths.cleaned_dir / DAILY_FEATURES_FILE)
    _write_csv(cycles, paths.cleaned_dir / CYCLE_DETAIL_FILE)
    _write_csv(events, paths.cleaned_dir / EVENT_DETAIL_FILE)
    _write_csv(pure_segments, paths.cleaned_dir / PURE_DECAY_FILE)
    for filename, df in table_outputs.items():
        _write_csv(df, paths.tables_dir / filename)
    write_chinese_header_tables(paths, {_name: _format_for_csv(_df) for _name, _df in table_outputs.items()})

    write_markdown_summary(paths, overview, season_decay, rule, decay_summary, effect_summary, recovery_summary, seasonal_effect, core)
    _assert_outputs(paths, maintenance)
    print("v2 q1 pipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"v2 q1 pipeline failed: {exc}", file=sys.stderr)
        raise
