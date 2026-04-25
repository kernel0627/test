from __future__ import annotations

import re


def device_sort_value(device_id: object) -> int:
    match = re.fullmatch(r"a(\d+)", str(device_id).strip().lower())
    if match:
        return int(match.group(1))
    return 10**9


def sorted_device_ids(device_ids) -> list[str]:
    return sorted(device_ids, key=lambda value: (device_sort_value(value), str(value)))


def sort_by_device_id(df, *extra_columns):
    if "device_id" not in df.columns:
        return df.sort_values(list(extra_columns)).reset_index(drop=True) if extra_columns else df.reset_index(drop=True)
    sort_columns = ["_device_sort_value", *extra_columns]
    return (
        df.assign(_device_sort_value=df["device_id"].map(device_sort_value))
        .sort_values(sort_columns)
        .drop(columns=["_device_sort_value"])
        .reset_index(drop=True)
    )
