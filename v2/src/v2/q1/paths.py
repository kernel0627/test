from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Q1Paths:
    root: Path
    task_dir: Path
    q1_dir: Path
    cleaned_dir: Path
    tables_dir: Path
    tables_cn_dir: Path
    markdown_dir: Path
    observations_xlsx: Path
    maintenance_xlsx: Path


def project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "task_and_requirements").exists():
            return path
    raise FileNotFoundError("Cannot locate project root containing task_and_requirements.")


def _detect_workbooks(task_dir: Path) -> tuple[Path, Path]:
    observations: Path | None = None
    maintenance: Path | None = None

    for path in sorted(task_dir.glob("*.xlsx")):
        excel = pd.ExcelFile(path)
        sheet_names = excel.sheet_names
        if len(sheet_names) >= 10 and all(str(name).startswith("A_") for name in sheet_names[:10]):
            observations = path
            continue

        sample = excel.parse(sheet_names[0], nrows=5)
        if sample.shape[1] >= 3:
            maintenance = path

    if observations is None or maintenance is None:
        raise FileNotFoundError(f"Cannot identify 附件1/附件2 under {task_dir}")
    return observations, maintenance


def get_paths() -> Q1Paths:
    root = project_root()
    task_dir = root / "task_and_requirements"
    observations, maintenance = _detect_workbooks(task_dir)
    q1_dir = root / "v2" / "q_1"
    return Q1Paths(
        root=root,
        task_dir=task_dir,
        q1_dir=q1_dir,
        cleaned_dir=q1_dir / "cleaned",
        tables_dir=q1_dir / "tables",
        tables_cn_dir=q1_dir / "tables_中文表头",
        markdown_dir=q1_dir / "markdown",
        observations_xlsx=observations,
        maintenance_xlsx=maintenance,
    )


def ensure_output_dirs(paths: Q1Paths) -> None:
    for directory in [paths.cleaned_dir, paths.tables_dir, paths.tables_cn_dir, paths.markdown_dir]:
        directory.mkdir(parents=True, exist_ok=True)
