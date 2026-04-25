from __future__ import annotations

from pathlib import Path

_v2_src = Path(__file__).resolve().parents[1] / "v2" / "src"
if _v2_src.exists():
    __path__.append(str(_v2_src))
