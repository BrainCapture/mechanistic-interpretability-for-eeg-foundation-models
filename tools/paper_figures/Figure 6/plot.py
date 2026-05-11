"""Figure 6: Spectrum-level concept steering (abnormal -> normal).

SleepFM layer 2, E=8, k=64. Three-panel layout:

    Panel A (Baseline)         source spectrum + target centroid + bootstrap CI band
    Panel B (Moderate, n=104)  source + steered + target band
    Panel C (Full, n=164)      source + steered + target band

Source data: ``data.npz`` (next to this script). All spectra and metric
values are pre-computed — the script just loads and plots. See README.md
for provenance.

Output::

    Figure 6/figure.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as _mp
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

BANDS = {
    "δ": (0.5, 4, "#c0392b"),
    "θ": (4, 8, "#d35400"),
    "α": (8, 13, "#9a7d0a"),
    "β": (13, 30, "#1e8449"),
    "γ": (30, 45, "#1a5276"),
}
SRC_COL = "#c0392b"
TGT_COL = "#1e8449"
band_tex = {"δ": r"$\delta$", "θ": r"$\theta$", "α": r"$\alpha$",
            "β": r"$\beta$", "γ": r"$\gamma$"}


def main() -> None:
    d = np.load(HERE / "data.npz", allow_pickle=False)
    freqs = d["freqs"]
    src_mean = d["src_mean"]
    tgt_mean, tgt_lo, tgt_hi = d["tgt_mean"], d["tgt_lo"], d["tgt_hi"]
    n_moderate, n_perfect = int(d["n_moderate"]), int(d["n_perfect"])
    moderate_mean = d["moderate_mean"]
    perfect_mean = d["perfect_mean"]
    ci = float(d["ci"])
    src_lbl = str(d["src_lbl"])
    tgt_lbl = str(d["tgt_lbl"])
    title = str(d["title"])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 14,
    })

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), sharey=True,
                             facecolor="white")
    step_colors = {
        n_moderate: "#3b6ea5",
        n_perfect: matplotlib.colormaps["viridis_r"](0.85),
    }
    panel_means = {n_moderate: moderate_mean, n_perfect: perfect_mean}

    y_max = max(
        src_mean.max(),
        tgt_hi.max(),
        moderate_mean.max(),
        perfect_mean.max(),
    ) * 1.12

    panel_specs = [
        ("Baseline", None),
        (rf"Moderate steering  ($n={n_moderate}$)", n_moderate),
        (rf"Full steering  ($n={n_perfect}$)", n_perfect),
    ]

    f_lo, f_hi = 0.5, 45.0
    for ax, (panel_title, n_step) in zip(axes, panel_specs):
        ax.set_facecolor("white")
        for band_lbl, (f0, f1, col) in BANDS.items():
            f0v, f1v = max(f0, f_lo), min(f1, f_hi)
            if f1v <= f0v:
                continue
            ax.axvspan(f0v, f1v, alpha=0.09, color=col, zorder=0)
            ax.text((f0v + f1v) / 2, y_max * 0.965, band_tex[band_lbl],
                    ha="center", va="top", fontsize=13, color=col)

        ax.fill_between(freqs, tgt_lo, tgt_hi, alpha=0.55, color=TGT_COL,
                        zorder=1, linewidth=0)
        ax.plot(freqs, tgt_mean, color=TGT_COL, lw=1.4, ls=":",
                label=f"{tgt_lbl} target", zorder=4)
        ax.plot(freqs, src_mean, color=SRC_COL, lw=2.0,
                label=f"{src_lbl} source", zorder=5)

        if n_step is not None:
            ax.plot(freqs, panel_means[n_step], color=step_colors[n_step],
                    lw=2.0, label="Steered", zorder=6)

        ax.set_xlim(f_lo, f_hi)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(panel_title, pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        for sp_ in ["left", "bottom"]:
            ax.spines[sp_].set_color("#888888")
        ax.yaxis.grid(True, alpha=0.25, linestyle="--", color="#bbbbbb")
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r"Amplitude ($\mu$V)")

    handles = [
        plt.Line2D([], [], color=SRC_COL, lw=2.0,
                   label=f"{src_lbl} source"),
        plt.Line2D([], [], color=TGT_COL, lw=1.4, ls=":",
                   label=f"{tgt_lbl} target"),
        _mp.Patch(facecolor=TGT_COL, alpha=0.55,
                  label=f"{int(ci)}% subject-bootstrap CI"),
        plt.Line2D([], [], color=step_colors[n_moderate], lw=2.0,
                   label=rf"Moderate ($n={n_moderate}$)"),
        plt.Line2D([], [], color=step_colors[n_perfect], lw=2.0,
                   label=rf"Full ($n={n_perfect}$)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=5,
               frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.8, columnspacing=1.8)

    fig.suptitle(title, fontsize=14, y=1.00)
    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(bottom=0.24, top=0.84)

    for suffix in (".png", ".pdf"):
        out = HERE / f"figure{suffix}"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
