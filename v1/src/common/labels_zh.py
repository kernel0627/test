from __future__ import annotations

import matplotlib as mpl
import seaborn as sns


FONT_FAMILY_FALLBACKS = [
    "Microsoft YaHei",
    "SimHei",
    "KaiTi",
    "FangSong",
]

MAINTENANCE_TYPE_LABELS = {
    "medium": "中维护",
    "major": "大维护",
}


def configure_chinese_plotting() -> None:
    sns.set_theme(style="whitegrid")
    mpl.rcParams["font.sans-serif"] = FONT_FAMILY_FALLBACKS
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.dpi"] = 120
