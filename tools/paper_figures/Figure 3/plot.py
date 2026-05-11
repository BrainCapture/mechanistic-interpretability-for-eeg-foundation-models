"""Figure 3: 3 encoders x 3 taxonomy metrics (sep / ent / dead) heatmaps.

Layout (one block per encoder, stacked vertically):

    SleepFM   |  Separable  |  Entangled  |   Dead    |
    LaBraM    |  Separable  |  Entangled  |   Dead    |
    REVE      |  Separable  |  Entangled  |   Dead    |

Each row uses an independent subgridspec so column widths within an encoder
match its number of expansion ratios; row heights match the number of layers.

Source data: ``data.json`` (next to this script). Each entry maps the
experiment name (e.g. ``sleepfm_finetuned_exp4_layer1``) to a
``{separable, entangled, dead}`` dict — pre-classified percentages
extracted from the development repo's per-experiment ``taxonomy_cache.pt``
files (see README.md for provenance).

Output::

    Figure 3/figure.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data.json"


# ── Paper-ready styling (mathtext serif — LaTeX-look without TeX dep) ────────-
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral", "Computer Modern Roman"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "hatch.linewidth": 1.2,
})


# Unified expansion grid across encoders so the heatmaps line up vertically
# in the transposed view. Cells without a trained SAE (REVE/LaBraM at E=32, 64)
# render as hatched "missing".
_E_GRID = [1, 2, 4, 8, 16, 32, 64]

ENCODER_GRIDS = [
    {
        "key":    "sleepfm",
        "name":   "SleepFM",
        "layers": [0, 1, 2],
        "Es":     _E_GRID,
        "exp_for": (
            lambda L, E: (
                f"sleepfm_finetuned_layer{L}" if E == 1
                else f"sleepfm_finetuned_exp{E}_layer{L}"
            )
        ),
    },
    {
        "key":    "labram",
        "name":   "LaBraM",
        "layers": list(range(12)),
        "Es":     _E_GRID,
        "exp_for": (
            lambda L, E: (
                f"labram_layer{L}" if E == 1
                else f"labram_layer{L}_exp{E}"
            )
        ),
    },
    {
        "key":    "reve",
        "name":   "REVE",
        "layers": list(range(22)),
        "Es":     _E_GRID,
        "exp_for": (
            lambda L, E: (
                f"reve_qjbe08_layer{L}" if E == 1
                else f"reve_qjbe08_exp{E}_layer{L}"
            )
        ),
    },
]


_TAXONOMY_TABLE: Optional[dict] = None


def _load_taxonomy(exp_name: str) -> Optional[dict]:
    """Return pre-classified taxonomy fractions for `exp_name`, or None if absent.

    Data is read once from ``data.json``; on subsequent calls a cached lookup
    table is used. See README.md for the JSON schema and provenance.
    """
    global _TAXONOMY_TABLE
    if _TAXONOMY_TABLE is None:
        _TAXONOMY_TABLE = json.loads(DATA_PATH.read_text())
    return _TAXONOMY_TABLE.get(exp_name)


def _build_matrices(grid):
    Ls = grid["layers"]; Es = grid["Es"]
    nL, nE = len(Ls), len(Es)
    sep = np.full((nL, nE), np.nan)
    ent = np.full((nL, nE), np.nan)
    dead = np.full((nL, nE), np.nan)
    miss = np.zeros((nL, nE), dtype=bool)
    for i, L in enumerate(Ls):
        for j, E in enumerate(Es):
            t = _load_taxonomy(grid["exp_for"](L, E))
            if t is None:
                miss[i, j] = True
                continue
            sep[i, j]  = t["separable"]
            ent[i, j]  = t["entangled"]
            dead[i, j] = t["dead"]
    return sep, ent, dead, miss


def _draw_heatmap(ax, mat, miss, layers, Es, cmap, vmin, vmax,
                  font_cell, font_tick, best_E_idx=None,
                  global_best=None):
    """Transposed view: x-axis = layer, y-axis = expansion."""
    nL, nE = len(layers), len(Es)
    mat_disp = np.where(miss, np.nan, mat) * 100  # show as percent
    vmin_p = vmin * 100; vmax_p = vmax * 100
    # Transpose so rows = Es (y-axis), cols = layers (x-axis).
    mat_T = mat_disp.T
    miss_T = miss.T
    im = ax.imshow(mat_T, aspect="auto", cmap=cmap, vmin=vmin_p, vmax=vmax_p,
                   origin="lower")
    rng = vmax_p - vmin_p
    for j in range(nE):
        for i in range(nL):
            if miss_T[j, i]:
                ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1,
                                           hatch="//", fill=False,
                                           edgecolor="#888888", lw=0.5))
                continue
            v = mat_T[j, i]
            norm_v = (v - vmin_p) / rng if rng > 0 else 0.5
            color = "white" if norm_v > 0.65 else "black"
            ax.text(i, j, f"{v:.0f}", ha="center", va="center",
                    fontsize=font_cell, color=color)
    ax.set_xticks(range(nL))
    ax.set_xticklabels([f"L{L+1}" for L in layers], fontsize=font_tick)
    ax.set_yticks(range(nE))
    ax.set_yticklabels([str(E) for E in Es], fontsize=font_tick)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    if best_E_idx is not None:
        for i in range(nL):
            j = int(best_E_idx[i])
            if j < 0 or miss_T[j, i]:
                continue
            ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1,
                                       fill=False, edgecolor="black", lw=1.6,
                                       zorder=5))
    if global_best is not None:
        gi, gj = global_best
        if gi >= 0 and gj >= 0 and not miss_T[gj, gi]:
            ax.add_patch(plt.Rectangle((gi - 0.5, gj - 0.5), 1, 1,
                                       fill=False, edgecolor="#d4af37", lw=2.6,
                                       zorder=6))
    return im


def _saturated_cmap(base_name: str, vmax_frac: float, n: int = 256):
    """Compress a base colormap so its full gradient sits in 0…vmax_frac and
    values above saturate at the max colour. Used so cell colours look like
    the original (vmax=0.6 / 0.8) while the colorbar can still span 0-100%."""
    base = plt.get_cmap(base_name)
    n_grad = max(1, int(round(n * vmax_frac)))
    n_sat = n - n_grad
    colors = list(base(np.linspace(0.0, 1.0, n_grad)))
    if n_sat > 0:
        colors += [base(1.0)] * n_sat
    return LinearSegmentedColormap.from_list(f"{base_name}_sat{vmax_frac}",
                                             colors)


# (key, base_cmap, vmax_frac=where the gradient saturates, label)
METRIC_DEFS = [
    ("sep",  "Greens",  0.6, "% Separable"),
    ("ent",  "Oranges", 0.8, "% Entangled"),
    ("dead", "Greys",   1.0, "% Dead"),
]


def main() -> None:
    grids_data = []
    for grid in ENCODER_GRIDS:
        sep, ent, dead, miss = _build_matrices(grid)
        # Per-layer highlighted expansion = argmax(sep - dead) ignoring missing.
        score = np.where(miss, -np.inf, sep - dead)
        best_E_idx = np.full(score.shape[0], -1, dtype=int)
        for i in range(score.shape[0]):
            if np.all(np.isneginf(score[i])):
                continue
            best_E_idx[i] = int(np.argmax(score[i]))
        # Global best across (L, E): coordinates in transposed view (i=L, j=E).
        if np.all(np.isneginf(score)):
            global_best = (-1, -1)
        else:
            flat_idx = int(np.argmax(score))
            gL, gE = np.unravel_index(flat_idx, score.shape)
            global_best = (int(gL), int(gE))
        grids_data.append({"grid": grid, "sep": sep, "ent": ent, "dead": dead,
                           "miss": miss, "best_E_idx": best_E_idx,
                           "global_best": global_best})

    # 1-column paper layout — TRANSPOSED:
    #   rows = metrics (Separable / Entangled / Dead)
    #   cols = encoders (SleepFM / LaBraM / REVE)
    # Column widths proportional to nE per encoder; rows uniform-height
    # (cells will be squat for SleepFM, square-ish for LaBraM, tall for REVE).
    # Transposed view: x-axis = layer, y-axis = expansion.
    # Column widths proportional to nL so cells stay uniform across encoders.
    nLs = [len(g["grid"]["layers"]) for g in grids_data]    # 3, 12, 22
    width_ratios = nLs

    # Wider, less tall — fits a single-column paper width better.
    fig_w = 18.0
    fig_h = 8.5
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = fig.add_gridspec(
        3, 3, width_ratios=width_ratios,
        hspace=0.32, wspace=0.18,
        top=0.86, bottom=0.09, left=0.13, right=0.95,
    )

    # Cell font is uniform across encoders (size 11; cells are equal physical
    # size because width_ratios is proportional to nL).
    FONT_CELL = 11
    def _font_tick(nL: int, nE: int) -> int:
        return FONT_CELL

    last_ims = [None, None, None]   # one per metric row, for colorbar
    grid_axes = [[None, None, None] for _ in range(3)]   # heatmap axes by (ri, ci)

    for ri, (metric_key, base_name, vmax_frac, label) in enumerate(METRIC_DEFS):
        cmap = _saturated_cmap(base_name, vmax_frac)
        for ci, gd in enumerate(grids_data):
            grid = gd["grid"]
            Ls = grid["layers"]; Es = grid["Es"]
            ax = fig.add_subplot(gs[ri, ci])
            grid_axes[ri][ci] = ax
            ax.set_facecolor("white")
            mat = gd[metric_key]
            # vmin/vmax = full 0..1 range; saturation lives in the cmap.
            im = _draw_heatmap(ax, mat, gd["miss"], Ls, Es,
                               cmap=cmap, vmin=0.0, vmax=1.0,
                               font_cell=FONT_CELL,
                               font_tick=_font_tick(len(Ls), len(Es)),
                               best_E_idx=gd["best_E_idx"],
                               global_best=gd["global_best"])
            last_ims[ri] = im

            # Encoder titles on the top row only
            if ri == 0:
                ax.set_title(grid["name"], fontsize=22, pad=10,
                             color="#222222")
            # X-label (Encoder layer) on the bottom row only
            if ri == len(METRIC_DEFS) - 1:
                ax.set_xlabel("Encoder layer", fontsize=18)
            else:
                ax.set_xlabel("")
            # Y-label (SAE expansion) only on leftmost column
            if ci == 0:
                ax.set_ylabel("SAE expansion $E$", fontsize=16)
            else:
                ax.set_ylabel("")

        # Metric label on left margin, rotated 90° (well clear of y-axis label)
        first_ax = grid_axes[ri][0]
        first_ax.text(
            -1.85, 0.5, label,
            transform=first_ax.transAxes, ha="center", va="center",
            fontsize=22, color="#222222",
            rotation=90,
        )
        # Taxonomy glyph between the metric label and the y-axis label.
        # Square axes in figure coords so circles render round regardless of
        # the heatmap aspect ratio.
        fig_w_in, fig_h_in = fig.get_size_inches()
        icon_in = 0.95
        w_fig_icon = icon_in / fig_w_in
        h_fig_icon = icon_in / fig_h_in
        x_axis, y_axis = -1.15, 0.5
        x_disp, y_disp = first_ax.transAxes.transform([x_axis, y_axis])
        x_fig, y_fig = fig.transFigure.inverted().transform([x_disp, y_disp])
        icon_ax = fig.add_axes([
            x_fig - w_fig_icon / 2, y_fig - h_fig_icon / 2,
            w_fig_icon, h_fig_icon,
        ])
        icon_ax.set_xlim(-1, 1)
        icon_ax.set_ylim(-1, 1)
        icon_ax.set_aspect("equal")
        icon_ax.axis("off")
        # Solid glyphs at α=0.9; entangled fills at α=0.7 so the Venn overlap
        # blends visibly darker. Outlines always opaque for crisp edges.
        ALPHA_SOLID = 0.9
        ALPHA_ENT = 0.7
        if metric_key == "sep":
            icon_ax.add_patch(plt.Circle(
                (0, 0), 0.72,
                facecolor=plt.cm.Greens(0.78), edgecolor="none", alpha=ALPHA_SOLID,
            ))
            icon_ax.add_patch(plt.Circle(
                (0, 0), 0.72,
                facecolor="none", edgecolor="black", linewidth=1.6,
            ))
        elif metric_key == "ent":
            # Translucent fills (overlap blends darker) + opaque outlines on top
            # → see-through Venn lens with crisp edges.
            ent_color = plt.cm.Oranges(0.72)
            cL, cR, r = (-0.34, 0), (0.34, 0), 0.60
            icon_ax.add_patch(plt.Circle(
                cR, r, facecolor=ent_color, edgecolor="none", alpha=ALPHA_ENT,
            ))
            icon_ax.add_patch(plt.Circle(
                cL, r, facecolor=ent_color, edgecolor="none", alpha=ALPHA_ENT,
            ))
            icon_ax.add_patch(plt.Circle(
                cR, r, facecolor="none", edgecolor="black", linewidth=1.6,
            ))
            icon_ax.add_patch(plt.Circle(
                cL, r, facecolor="none", edgecolor="black", linewidth=1.6,
            ))
        elif metric_key == "dead":
            icon_ax.add_patch(plt.Circle(
                (0, 0), 0.72,
                facecolor="#dcdcdc", edgecolor="none", alpha=ALPHA_SOLID,
            ))
            icon_ax.add_patch(plt.Circle(
                (0, 0), 0.72,
                facecolor="none", edgecolor="#444444", linewidth=1.6,
                hatch="/////",
            ))

    # Colorbars on the right of each metric row, full grid height
    for ri in range(3):
        row_axes = grid_axes[ri]
        bbs = [a.get_position() for a in row_axes]
        y0 = min(bb.y0 for bb in bbs)
        y1 = max(bb.y1 for bb in bbs)
        x_right = max(bb.x1 for bb in bbs)
        cax = fig.add_axes([x_right + 0.018, y0, 0.010, y1 - y0])
        cbar = fig.colorbar(last_ims[ri], cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("%", fontsize=11, rotation=0, labelpad=6)

    fig.suptitle(
        "Monosemanticity taxonomy across SAE expansion and encoder depth",
        fontsize=26, y=0.98,
    )

    for suffix in (".png", ".pdf"):
        out = HERE / f"figure{suffix}"
        plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
        print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
