
from __future__ import annotations

import os
import urllib.request

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

PAL20 = sns.color_palette("tab20", 20)
PAL10 = sns.color_palette("tab10", 10)

_BENGALI_FONT_URL = (
    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
    "NotoSansBengali/NotoSansBengali-Regular.ttf"
)


def apply_style() -> None:
    """Apply the publication-style rcParams used across all figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
        }
    )


def ensure_bengali_font(font_path: str | None = None) -> tuple[FontProperties, FontProperties]:

    font_path = font_path or os.path.expanduser(
        "~/.cache/dialect_fusion/NotoSansBengali-Regular.ttf"
    )
    if not os.path.exists(font_path):
        os.makedirs(os.path.dirname(font_path), exist_ok=True)
        urllib.request.urlretrieve(_BENGALI_FONT_URL, font_path)
        fm.fontManager.addfont(font_path)
        matplotlib.font_manager._load_fontmanager(try_read_cache=False)

    return FontProperties(fname=font_path, size=10), FontProperties(fname=font_path, size=12)
