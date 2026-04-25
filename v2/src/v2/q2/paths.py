from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Q2Paths:
    root: Path
    q1_dir: Path
    q1_cleaned_dir: Path
    q1_tables_dir: Path
    q2_dir: Path
    q2_tables_dir: Path
    q2_markdown_dir: Path


def project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "task_and_requirements").exists():
            return path
    raise FileNotFoundError("Cannot locate project root containing task_and_requirements.")


def get_paths() -> Q2Paths:
    root = project_root()
    q1_dir = root / "v2" / "q_1"
    q2_dir = root / "v2" / "q_2"
    return Q2Paths(
        root=root,
        q1_dir=q1_dir,
        q1_cleaned_dir=q1_dir / "cleaned",
        q1_tables_dir=q1_dir / "tables",
        q2_dir=q2_dir,
        q2_tables_dir=q2_dir / "tables",
        q2_markdown_dir=q2_dir / "markdown",
    )


def ensure_output_dirs(paths: Q2Paths) -> None:
    paths.q2_tables_dir.mkdir(parents=True, exist_ok=True)
    paths.q2_markdown_dir.mkdir(parents=True, exist_ok=True)
