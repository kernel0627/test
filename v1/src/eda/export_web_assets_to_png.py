from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import ensure_output_dirs, get_project_paths


BROWSER_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


def detect_browser() -> Path:
    for candidate in BROWSER_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No supported Chrome/Edge browser executable found.")


def to_file_url(path: Path) -> str:
    return Path(path).resolve().as_uri()


def convert_asset(browser_path: Path, source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(browser_path),
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--hide-scrollbars",
        "--window-size=1800,1400",
        f"--screenshot={str(target_path)}",
        to_file_url(source_path),
    ]
    completed = subprocess.run(args, capture_output=True)
    if completed.returncode != 0:
        stdout_text = completed.stdout.decode("utf-8", errors="ignore") if completed.stdout else ""
        stderr_text = completed.stderr.decode("utf-8", errors="ignore") if completed.stderr else ""
        raise RuntimeError(
            f"Failed to convert {source_path.name} -> {target_path.name}\n"
            f"stdout:\n{stdout_text}\n"
            f"stderr:\n{stderr_text}"
        )


def target_png_path(paths, source_path: Path) -> Path:
    if source_path.suffix.lower() == ".html":
        return paths.q1_figures_png_dir / f"{source_path.stem}.png"
    stem = source_path.stem
    target = paths.q1_figures_png_dir / f"{stem}.png"
    if target.exists():
        target = paths.q1_figures_png_dir / f"{stem}_from_svg.png"
    return target


def main() -> None:
    paths = get_project_paths()
    ensure_output_dirs(paths)
    browser_path = detect_browser()

    source_paths = sorted(paths.q1_figures_html_dir.glob("*.html")) + sorted(paths.q1_figures_png_dir.glob("*.svg"))
    if not source_paths:
        print("No html/svg assets found to convert.")
        return

    for source_path in source_paths:
        target_path = target_png_path(paths, source_path)
        convert_asset(browser_path, source_path, target_path)
        print(f"converted: {source_path.name} -> {target_path.name}")
        if source_path.suffix.lower() in {".html", ".svg"}:
            source_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
