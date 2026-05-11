"""Figure 7 (appendix): SAE dictionary size across encoders and expansion rates.

Each cell shows |D| = E * d, where d is the encoder's embedding dimension.
Rendered as a 3 x 7 heatmap with a light-brown gradient. Sits next to the
taxonomy grid (Figure 3) for context on what expansion ratio corresponds
to in dictionary size.

This figure is fully self-contained — all values are hardcoded below.

Output::

    Figure 7/figure.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = Path(__file__).resolve().parent

# Match the styling used in plot_paper_taxonomy_grid.py
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral", "Computer Modern Roman"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

ENCODERS = [
    ("SleepFM", 128),
    ("LaBraM",  200),
    ("REVE",    512),
]
EXPANSIONS = [1, 2, 4, 8, 16, 32, 64]

# Refined two-stop brown gradient — pale cream → soft taupe (low contrast,
# subtle so the cell numbers stay the visual focus).
BROWN_CMAP = LinearSegmentedColormap.from_list(
    "dict_brown", ["#faf3e7", "#bda079"]
)


def main() -> None:
    n_enc = len(ENCODERS)
    n_exp = len(EXPANSIONS)
    mat = np.zeros((n_enc, n_exp), dtype=int)
    for i, (_, d) in enumerate(ENCODERS):
        for j, E in enumerate(EXPANSIONS):
            mat[i, j] = E * d

    fig_w, fig_h = 9.5, 3.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")

    im = ax.imshow(
        mat, aspect="auto", cmap=BROWN_CMAP,
        norm=LogNorm(vmin=mat.min(), vmax=mat.max()),
        origin="upper",
    )

    log_vmin, log_vmax = np.log(mat.min()), np.log(mat.max())
    for i in range(n_enc):
        for j in range(n_exp):
            v = mat[i, j]
            norm_v = (np.log(v) - log_vmin) / (log_vmax - log_vmin)
            color = "white" if norm_v > 0.72 else "#3a2820"
            ax.text(
                j, i, f"{int(v):,}",
                ha="center", va="center",
                fontsize=13, color=color,
            )

    ax.set_xticks(range(n_exp))
    ax.set_xticklabels([str(E) for E in EXPANSIONS])
    ax.set_yticks(range(n_enc))
    ax.set_yticklabels([name for name, _ in ENCODERS])
    ax.set_xlabel("SAE expansion rate $E$")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")

    ax.set_title(
        "Number of SAE features",
        fontsize=15, color="#222222", pad=10,
    )

    plt.tight_layout()
    for suffix in (".png", ".pdf"):
        out = HERE / f"figure{suffix}"
        plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
        print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
