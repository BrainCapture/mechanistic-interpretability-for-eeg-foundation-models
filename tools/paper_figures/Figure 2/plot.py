"""Figure 2: SAE-faithfulness layer sweep.

Source data: ``data.json`` (next to this script) — produced by
``tools/probe_layer_sweep_kfold.py`` in the development repo.

Output::

    Figure 2/figure.{png,pdf}
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SUMMARY_PATH = HERE / "data.json"

K_FOLDS = 5  # matches probe_layer_sweep_kfold.py default

mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize":   22,
    "axes.titlesize":   22,
    "xtick.labelsize":  20,
    "ytick.labelsize":  20,
    "legend.fontsize":  20,
    "axes.linewidth":   1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "lines.linewidth":  2.4,
    "lines.markersize": 7.5,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
})

summary = json.loads(SUMMARY_PATH.read_text())

# Wider aspect: 13×3.2 (vs 7.5×3.2 in the original) so the plot stretches
# horizontally without changing point density or font sizes.
fig, ax = plt.subplots(figsize=(13.0, 3.2))
colors = {"sleepfm": "#1f77b4", "reve": "#ff7f0e", "labram": "#2ca02c"}
pretty = {"sleepfm": "SleepFM", "reve": "REVE", "labram": "LaBraM"}
sem_div = math.sqrt(K_FOLDS)

encoder_handles = []
for name, res in summary.items():
    n_total = res["n_layers_total"]
    layers = sorted(int(L) for L in res["by_layer"])
    by = res["by_layer"]

    def get(L, k):
        key = str(L) if str(L) in by else L
        return by[key][k]

    aucs = [get(L, "auc_mean") for L in layers]
    sems = [get(L, "auc_std") / sem_div for L in layers]
    lo = [m - s for m, s in zip(aucs, sems)]
    hi = [m + s for m, s in zip(aucs, sems)]
    xs = [L / max(n_total - 1, 1) for L in layers]
    h, = ax.plot(xs, aucs, "o-", color=colors[name], markeredgewidth=0,
                 label=pretty[name])
    encoder_handles.append(h)
    ax.fill_between(xs, lo, hi, color=colors[name], alpha=0.22, linewidth=0)
    b = res["baseline"]
    ax.axhline(b["auc_mean"], color=colors[name],
               linestyle=(0, (5, 3)), linewidth=1.9, alpha=0.85)

ax.set_xlabel(r"Relative layer depth ($0 = $ first layer, $1 = $ last layer)")
ax.set_ylabel("AUROC")
ax.set_title("SAE-faithfulness layer sweep")
ax.set_xlim(-0.02, 1.02)
ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)

style_handles = [
    Line2D([0], [0], color="0.25", linewidth=2.4, marker="o",
           markersize=7, markeredgewidth=0, label="SAE-substituted"),
    Line2D([0], [0], color="0.25", linewidth=1.9,
           linestyle=(0, (5, 3)), label="Baseline"),
]
ax.legend(handles=encoder_handles + style_handles, frameon=False,
          loc="lower center", ncol=5, handlelength=1.8, columnspacing=2.2,
          bbox_to_anchor=(0.5, -0.65), borderaxespad=0.0)

png = HERE / "figure.png"
pdf = HERE / "figure.pdf"
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
print(f"Wrote {png}")
print(f"      {pdf}")
