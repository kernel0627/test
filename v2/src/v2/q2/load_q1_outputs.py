from __future__ import annotations

import pandas as pd

from .paths import Q2Paths


def _read_csv(path):
    return pd.read_csv(path)


def load_q1_outputs(paths: Q2Paths) -> dict[str, pd.DataFrame]:
    outputs = {
        "rule": _read_csv(paths.q1_tables_dir / "表03_当前固定维护触发规律.csv"),
        "core": _read_csv(paths.q1_tables_dir / "表07_设备核心指标汇总.csv"),
        "season_decay": _read_csv(paths.q1_tables_dir / "表02_季节水平项与月度衰减强度.csv"),
        "hmax_quality": _read_csv(paths.q1_tables_dir / "表06_Hmax与恢复比例质量评估.csv"),
        "seasonal_effect": _read_csv(paths.q1_tables_dir / "表08_季节对维护效应影响.csv"),
        "daily": _read_csv(paths.q1_cleaned_dir / "日尺度特征表.csv"),
        "events": _read_csv(paths.q1_cleaned_dir / "维护事件明细表.csv"),
    }
    for key in ["daily"]:
        outputs[key]["date"] = pd.to_datetime(outputs[key]["date"], errors="coerce")
    if "event_date" in outputs["events"].columns:
        outputs["events"]["event_date"] = pd.to_datetime(outputs["events"]["event_date"], errors="coerce")
    for key in ["rule"]:
        for column in ["last_maintenance_date", "last_major_maintenance_date"]:
            if column in outputs[key].columns:
                outputs[key][column] = pd.to_datetime(outputs[key][column], errors="coerce")
    return outputs
