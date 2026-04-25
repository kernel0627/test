from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    src_dir: Path
    task_dir: Path
    data_dir: Path
    q1_dir: Path
    q1_cleaned_csv_dir: Path
    q1_cleaned_parquet_dir: Path
    q1_tables_dir: Path
    q1_figures_png_dir: Path
    q1_figures_html_dir: Path
    q1_markdown_dir: Path
    q2_dir: Path
    q2_tables_dir: Path
    q2_figures_png_dir: Path
    q2_markdown_dir: Path
    csv_dir: Path
    csv_cleaned_dir: Path
    csv_checks_dir: Path
    csv_eda_dir: Path
    parquet_dir: Path
    parquet_cleaned_dir: Path
    cleaned_dir: Path
    eda_dir: Path
    figures_dir: Path
    figures_static_dir: Path
    figures_html_dir: Path
    figures_converted_dir: Path
    tables_dir: Path
    markdown_dir: Path
    meta_dir: Path
    checks_dir: Path
    logs_dir: Path
    temp_dir: Path
    observations_xlsx: Path
    maintenance_xlsx: Path


def _resolve_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _detect_workbooks(task_dir: Path) -> tuple[Path, Path]:
    xlsx_paths = sorted(task_dir.glob("*.xlsx"))
    if not xlsx_paths:
        raise FileNotFoundError(f"No xlsx files found under {task_dir}")

    observations_xlsx: Path | None = None
    maintenance_xlsx: Path | None = None

    for xlsx_path in xlsx_paths:
        excel_file = pd.ExcelFile(xlsx_path)
        sheet_names = excel_file.sheet_names
        if len(sheet_names) >= 10 and all(name.startswith("A_") for name in sheet_names[:10]):
            observations_xlsx = xlsx_path
            continue
        sample = excel_file.parse(sheet_names[0], nrows=5)
        if sample.shape[1] >= 3:
            maintenance_xlsx = xlsx_path

    if observations_xlsx is None or maintenance_xlsx is None:
        raise FileNotFoundError(
            "Failed to identify observation workbook and maintenance workbook."
        )

    return observations_xlsx, maintenance_xlsx


def get_project_paths() -> ProjectPaths:
    root = _resolve_root()
    task_dir = root / "task_and_requirements"
    data_dir = root / "data"
    observations_xlsx, maintenance_xlsx = _detect_workbooks(task_dir)

    return ProjectPaths(
        root=root,
        src_dir=root / "src",
        task_dir=task_dir,
        data_dir=data_dir,
        q1_dir=data_dir / "q_1",
        q1_cleaned_csv_dir=data_dir / "q_1" / "cleaned" / "csv",
        q1_cleaned_parquet_dir=data_dir / "q_1" / "cleaned" / "parquet",
        q1_tables_dir=data_dir / "q_1" / "tables",
        q1_figures_png_dir=data_dir / "q_1" / "figures" / "png",
        q1_figures_html_dir=data_dir / "q_1" / "figures" / "html",
        q1_markdown_dir=data_dir / "q_1" / "markdown",
        q2_dir=data_dir / "q_2",
        q2_tables_dir=data_dir / "q_2" / "tables",
        q2_figures_png_dir=data_dir / "q_2" / "figures" / "png",
        q2_markdown_dir=data_dir / "q_2" / "markdown",
        csv_dir=data_dir / "csv",
        csv_cleaned_dir=data_dir / "csv" / "cleaned",
        csv_checks_dir=data_dir / "csv" / "checks",
        csv_eda_dir=data_dir / "csv" / "eda",
        parquet_dir=data_dir / "parquet",
        parquet_cleaned_dir=data_dir / "parquet" / "cleaned",
        cleaned_dir=data_dir / "01_cleaned",
        eda_dir=data_dir / "02_eda",
        figures_dir=data_dir / "02_eda" / "figures",
        figures_static_dir=data_dir / "02_eda" / "figures" / "static",
        figures_html_dir=data_dir / "02_eda" / "figures" / "html",
        figures_converted_dir=data_dir / "02_eda" / "figures" / "converted_png",
        tables_dir=data_dir / "02_eda" / "tables",
        markdown_dir=data_dir / "02_eda" / "markdown",
        meta_dir=data_dir / "90_meta",
        checks_dir=data_dir / "90_meta" / "checks",
        logs_dir=data_dir / "90_meta" / "logs",
        temp_dir=data_dir / "90_meta" / "tmp",
        observations_xlsx=observations_xlsx,
        maintenance_xlsx=maintenance_xlsx,
    )


def ensure_output_dirs(paths: ProjectPaths) -> None:
    for directory in [
        paths.q1_dir,
        paths.q1_cleaned_csv_dir,
        paths.q1_cleaned_parquet_dir,
        paths.q1_tables_dir,
        paths.q1_figures_png_dir,
        paths.q1_figures_html_dir,
        paths.q1_markdown_dir,
        paths.q2_dir,
        paths.q2_tables_dir,
        paths.q2_figures_png_dir,
        paths.q2_markdown_dir,
        paths.csv_dir,
        paths.csv_cleaned_dir,
        paths.csv_checks_dir,
        paths.csv_eda_dir,
        paths.parquet_dir,
        paths.parquet_cleaned_dir,
        paths.cleaned_dir,
        paths.eda_dir,
        paths.figures_dir,
        paths.figures_static_dir,
        paths.figures_html_dir,
        paths.figures_converted_dir,
        paths.tables_dir,
        paths.markdown_dir,
        paths.meta_dir,
        paths.checks_dir,
        paths.logs_dir,
        paths.temp_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
