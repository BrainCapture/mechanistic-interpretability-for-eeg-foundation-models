"""Nature-portfolio figure style for the SAE4EEG project.

Usage
-----
    from mecheeg.nature_fig_style import (
        set_nature_style, mm_to_inches, NATURE_WIDTHS,
        nature_palette, style_axes, save_nature_figure, nature_figure,
    )

    set_nature_style()                       # call once at top of script/notebook
    fig, ax = nature_figure(width="single", height_mm=60)
    ax.scatter(x, y, color=nature_palette()[0])
    save_nature_figure(fig, "fig1.pdf")

Design choices
--------------
- Arial is bundled in ``mecheeg/fonts/`` so every collaborator gets the same
  rendering without depending on a system path. Import emits a UserWarning if
  matplotlib does not resolve ``font.family = "Arial"`` to one of the bundled
  files (e.g., the font cache is stale or the bundled files are missing).
- ``pdf.fonttype = 42`` (TrueType) keeps text editable in Illustrator.
- Default palette is Wong's 8-color colorblind-safe set (Nature Methods, 2011),
  which matches what most Nature journals recommend.
- All widths use Nature's published column sizes in millimeters; helpers
  convert to inches for matplotlib. NATURE_WIDTHS["single"] = 88 mm matches
  Nature's published figure guide exactly.
- Layout uses ``constrained_layout=True`` by default in ``nature_figure``;
  ``savefig.bbox`` is deliberately NOT "tight" so the saved PDF is exactly
  the size you asked for. Tight bbox silently rescales the canvas to wrap
  labels, which then propagates as font-size drift when the PDF is included
  in LaTeX with ``\\includegraphics[width=\\linewidth]``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

__version__ = "1.1.0"


# --------------------------------------------------------------------------- #
# Font registration                                                           #
# --------------------------------------------------------------------------- #

_FONT_DIR = Path(__file__).resolve().parent / "fonts"
_FONT_FILES = ("ARIAL.TTF", "ARIALBD.TTF", "ARIALI.TTF", "ARIALBI.TTF")


def _register_bundled_fonts() -> None:
    for name in _FONT_FILES:
        path = _FONT_DIR / name
        if path.exists():
            fm.fontManager.addfont(str(path))


def _verify_arial_resolution() -> None:
    bundled_paths = {str((_FONT_DIR / name).resolve()) for name in _FONT_FILES}
    try:
        resolved = Path(fm.findfont("Arial", fallback_to_default=False)).resolve()
    except Exception as e:
        warnings.warn(
            f"nature_fig_style: could not resolve 'Arial' ({e}). Bundled fonts at "
            f"{_FONT_DIR} did not register or are missing. Figures will fall back "
            f"to the matplotlib default font (DejaVu Sans), which is NOT Nature-"
            f"compliant. Confirm the fonts/ directory exists and clear the "
            f"matplotlib cache (~/.cache/matplotlib) if needed.",
            UserWarning,
            stacklevel=2,
        )
        return
    if str(resolved) not in bundled_paths:
        warnings.warn(
            f"nature_fig_style: matplotlib resolved 'Arial' to {resolved}, not "
            f"the bundled font at {_FONT_DIR}. Saved PDFs may use a system Arial "
            f"that differs subtly from the bundled file. Delete "
            f"~/.cache/matplotlib and reimport to force a cache rebuild.",
            UserWarning,
            stacklevel=2,
        )


_register_bundled_fonts()
_verify_arial_resolution()


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

NATURE_WIDTHS = {
    "single": 88.0,      # one column — Nature spec
    "onehalf": 120.0,    # 1.5 columns
    "double": 180.0,     # full page width — Nature spec
}

WONG_PALETTE: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

OKABE_ITO: tuple[str, ...] = WONG_PALETTE[1:]


# --------------------------------------------------------------------------- #
# Core API                                                                    #
# --------------------------------------------------------------------------- #

def mm_to_inches(mm: float) -> float:
    return mm / 25.4


def set_nature_style(base_font_pt: float = 7, linewidth: float = 0.6) -> None:
    """Apply Nature-portfolio rcParams globally.

    Parameters
    ----------
    base_font_pt:
        Body font size in points. Nature's caption text is typically 7 pt;
        use 8 for slightly larger panels, 6 only for very dense figures.
    linewidth:
        Spine and tick line width in points. 0.6 pt matches Nature's house style.
    """
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "font.family": "Arial",
        "font.size": base_font_pt,
        "axes.titlesize": base_font_pt,
        "axes.labelsize": base_font_pt,
        "xtick.labelsize": base_font_pt,
        "ytick.labelsize": base_font_pt,
        "legend.fontsize": base_font_pt,
        "figure.titlesize": base_font_pt,

        "axes.linewidth": linewidth,
        "xtick.major.width": linewidth,
        "ytick.major.width": linewidth,
        "xtick.minor.width": linewidth * 0.75,
        "ytick.minor.width": linewidth * 0.75,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 1.0,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.titlepad": 4.0,
        "axes.labelpad": 2.0,

        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,

        # NOT bbox="tight": tight bbox recomputes the canvas at save time,
        # breaking column-width accuracy. Use constrained_layout instead.
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "standard",
        "savefig.transparent": False,

        "axes.prop_cycle": mpl.cycler(color=list(WONG_PALETTE)),
    })


def nature_palette(n: int | None = None, include_black: bool = False) -> list[str]:
    """Return ``n`` colorblind-safe hex colors from the Wong palette."""
    base = WONG_PALETTE if include_black else OKABE_ITO
    if n is None:
        return list(base)
    if n <= len(base):
        return list(base[:n])
    return [base[i % len(base)] for i in range(n)]


def category_color_map(categories: Sequence[str], include_black: bool = False) -> dict[str, str]:
    """Map an ordered sequence of category labels to stable palette colors."""
    colors = nature_palette(len(categories), include_black=include_black)
    return dict(zip(categories, colors))


def style_axes(ax: plt.Axes, *, hide: Iterable[str] = ("top", "right")) -> plt.Axes:
    """Apply per-axis Nature defaults (hide spines, clear grid, tight ticks)."""
    for side in hide:
        ax.spines[side].set_visible(False)
    for side in set(["top", "right", "left", "bottom"]) - set(hide):
        ax.spines[side].set_linewidth(mpl.rcParams["axes.linewidth"])
    ax.grid(False)
    ax.tick_params(
        axis="both",
        width=mpl.rcParams["xtick.major.width"],
        length=mpl.rcParams["xtick.major.size"],
    )
    return ax


def nature_figure(
    width: float | str = "single",
    height_mm: float = 60.0,
    nrows: int = 1,
    ncols: int = 1,
    **subplots_kwargs,
) -> tuple[plt.Figure, plt.Axes | list[plt.Axes]]:
    """Create a figure sized to a Nature column width.

    Parameters
    ----------
    width:
        Key from ``NATURE_WIDTHS`` ("single", "onehalf", "double") or an
        explicit width in millimeters.
    height_mm:
        Figure height in millimeters.
    """
    width_mm = NATURE_WIDTHS[width] if isinstance(width, str) else float(width)
    figsize = (mm_to_inches(width_mm), mm_to_inches(height_mm))
    subplots_kwargs.setdefault("constrained_layout", True)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **subplots_kwargs)
    return fig, axes


def save_nature_figure(fig: plt.Figure, path: str | Path, *, dpi: int = 300) -> None:
    """Save a figure at exactly the size set by ``nature_figure``.

    Does NOT pass ``bbox_inches="tight"`` — see module docstring for rationale.
    Prefers PDF/SVG for vector output; falls back to raster at the requested DPI.
    """
    fig.savefig(path, dpi=dpi)
