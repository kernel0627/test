from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Q3Paths:
    root: Path
    q2_tables_dir: Path
    q3_dir: Path
    q3_tables_dir: Path
    q3_markdown_dir: Path


def project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "task_and_requirements").exists():
            return path
    raise FileNotFoundError("Cannot locate project root containing task_and_requirements.")


def get_paths() -> Q3Paths:
    root = project_root()
    q3_dir = root / "v2" / "q_3"
    return Q3Paths(
        root=root,
        q2_tables_dir=root / "v2" / "q_2" / "tables",
        q3_dir=q3_dir,
        q3_tables_dir=q3_dir / "tables",
        q3_markdown_dir=q3_dir / "markdown",
    )


def ensure_output_dirs(paths: Q3Paths) -> None:
    paths.q3_tables_dir.mkdir(parents=True, exist_ok=True)
    paths.q3_markdown_dir.mkdir(parents=True, exist_ok=True)
