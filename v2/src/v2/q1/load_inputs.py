from __future__ import annotations

import re

import pandas as pd

from .paths import Q1Paths


def standardize_device_id(value: object) -> str:
    digits = re.findall(r"\d+", str(value))
    if not digits:
        raise ValueError(f"Cannot parse device id from {value!r}")
    return f"a{int(digits[0])}"


def device_sort_key(value: object) -> tuple[int, str]:
    match = re.fullmatch(r"a(\d+)", str(value).strip().lower())
    if match:
        return int(match.group(1)), str(value)
    return 10**9, str(value)


def sorted_device_ids(values) -> list[str]:
    return sorted([str(value) for value in values], key=device_sort_key)


def sort_by_device(df: pd.DataFrame, *extra_columns: str) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    if "device_id" not in df.columns:
        return df.sort_values(list(extra_columns)).reset_index(drop=True) if extra_columns else df.reset_index(drop=True)
    ordered = df.assign(_device_sort=df["device_id"].map(lambda x: device_sort_key(x)[0]))
    return ordered.sort_values(["_device_sort", *extra_columns]).drop(columns="_device_sort").reset_index(drop=True)


def load_observations(paths: Q1Paths) -> pd.DataFrame:
    excel = pd.ExcelFile(paths.observations_xlsx)
    frames: list[pd.DataFrame] = []
    for sheet_name in excel.sheet_names:
        sheet = excel.parse(sheet_name)
        if sheet.shape[1] < 2:
            continue
        frame = sheet.iloc[:, :2].copy()
        frame.columns = ["time", "per_raw"]
        frame["device_sheet"] = sheet_name
        frame["device_id"] = standardize_device_id(sheet_name)
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        frame["per_raw"] = pd.to_numeric(frame["per_raw"], errors="coerce")
        frames.append(frame)

    if not frames:
        raise ValueError("附件1 does not contain usable observation sheets.")

    hourly = pd.concat(frames, ignore_index=True).dropna(subset=["time"]).copy()
    hourly["date"] = hourly["time"].dt.normalize()
    hourly = (
        hourly.groupby(["device_id", "device_sheet", "time"], as_index=False)
        .agg(per_raw=("per_raw", "mean"))
    )
    hourly["date"] = hourly["time"].dt.normalize()
    return sort_by_device(hourly, "time")


def normalize_maintenance_type(value: object) -> str | None:
    text = str(value).strip().lower()
    if text in {"medium", "middle"} or "\u4e2d" in text:
        return "medium"
    if text == "major" or "\u5927" in text:
        return "major"
    return None


def load_maintenance(paths: Q1Paths) -> pd.DataFrame:
    raw = pd.read_excel(paths.maintenance_xlsx)
    maintenance = raw.iloc[:, :3].copy()
    maintenance.columns = ["device_id", "event_date", "maintenance_type"]
    maintenance["device_id"] = maintenance["device_id"].map(standardize_device_id)
    maintenance["event_date"] = pd.to_datetime(maintenance["event_date"], errors="coerce").dt.normalize()
    maintenance["maintenance_type"] = maintenance["maintenance_type"].map(normalize_maintenance_type)
    maintenance = maintenance.dropna(subset=["device_id", "event_date", "maintenance_type"]).copy()
    maintenance = sort_by_device(maintenance, "event_date")

    frames: list[pd.DataFrame] = []
    for device_id, group in maintenance.groupby("device_id", sort=False):
        group = group.sort_values("event_date").reset_index(drop=True).copy()
        group["event_order"] = range(1, len(group) + 1)
        group["days_since_previous_maintenance"] = group["event_date"].diff().dt.days
        group["days_to_next_maintenance"] = (group["event_date"].shift(-1) - group["event_date"]).dt.days
        frames.append(group)
    return sort_by_device(pd.concat(frames, ignore_index=True), "event_date")
