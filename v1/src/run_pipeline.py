from __future__ import annotations

from cleaning.build_cleaned_datasets import build_cleaned_datasets
from common.paths import ensure_output_dirs, get_project_paths
from eda.run_eda import run_eda
from modeling.run_q2 import run_q2


def main() -> None:
    paths = get_project_paths()
    ensure_output_dirs(paths)
    cleaned_outputs = build_cleaned_datasets(paths)
    run_eda(paths, cleaned_outputs)
    run_q2(paths)


if __name__ == "__main__":
    main()
