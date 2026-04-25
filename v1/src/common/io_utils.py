from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_dataframe(
    df: pd.DataFrame,
    csv_path: Path,
    parquet_path: Path | None = None,
    *,
    index: bool = False,
) -> bool:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=index, encoding="utf-8")

    if parquet_path is None:
        return False

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=index)
        return True
    except Exception:
        return False


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_timestamp(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, errors="coerce")


def artifact_paths(base_csv_dir: Path, base_parquet_dir: Path, stem: str) -> tuple[Path, Path]:
    return base_csv_dir / f"{stem}.csv", base_parquet_dir / f"{stem}.parquet"

