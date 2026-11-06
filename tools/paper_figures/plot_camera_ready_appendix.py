"""Camera-ready appendix figures from the seeded reruns (results/camera_ready/summary.json).

  uv run tools/paper_figures/plot_camera_ready_appendix.py

Same serif style as the main NeurIPS figures (Figs 4-5: plot_fig4_camera_ready.py,
plot_fig5_camera_ready.py), not the Nature style used by the SleepFM-2 project.

Writes into paper/camera_ready/figures/:
  app_collateral.png   — normalised AUROC drop, steered (rows) × measured (cols): SAE / LEACE / INLP
  app_native_head.png  — deployed head vs probe vs random clamping, mean ± sd over SAE seeds
                         (SleepFM-TUAB and LaBraM; REVE's operating layer is intermediate, so it
                         has no deployed-head curve and is left out)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})

ROOT = Path(__file__).resolve().parent.parent.parent
CR = Path(os.environ.get("FIG_IN", ROOT / "results" / "camera_ready"))
OUT = Path(os.environ.get("FIG_OUT_DIR", ROOT / "paper" / "camera_ready" / "figures"))
CONCEPTS = ["Pathology", "Age", "Sex", "Med-Psych", "Med-ASM"]
LABELS = {"Pathology": "Abnormality", "Age": "Age", "Sex": "Sex",
          "Med-Psych": "Med. (psych.)", "Med-ASM": "Med. (ASM)"}
# colours shared with Figs 4-5
FAMILY_COLOUR = {"SleepFM": "#3a7ab8", "LaBraM": "#3a8b53", "REVE": "#cc6c30"}
C_HEAD, C_PROBE, C_RAND = "#c0392b", "#1f6fb4", "#888888"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=12)


def _save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def collateral(S, n_seeds):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw=dict(wspace=0.08))
    for ax, key in zip(axes, ["SAE (clamp 25%)", "LEACE", "amnesic (INLP)"]):
        base, mat = S[key]["baseline"], S[key]["matrix"]
        M = np.array([[(base[m]["mean"] - mat[s][m]["mean"]) / max(base[m]["mean"] - 0.5, 1e-6)
                       for m in CONCEPTS] for s in CONCEPTS])
        im = ax.imshow(np.clip(M, 0, 1), cmap="Reds", vmin=0, vmax=1)
        for i in range(len(CONCEPTS)):
            for j in range(len(CONCEPTS)):
                ax.text(j, i, ">1" if M[i, j] > 1 else f"{M[i, j]:.2f}".replace("-0.00", "0.00"),
                        ha="center", va="center",
                        fontsize=11, color="white" if M[i, j] > 0.6 else "black",
                        weight="bold" if i == j else "normal")
        ax.set_xticks(range(len(CONCEPTS)), [LABELS[c] for c in CONCEPTS], rotation=35, ha="right",
                      fontsize=12)
        ax.set_yticks(range(len(CONCEPTS)), [LABELS[c] for c in CONCEPTS] if ax is axes[0] else [],
                      fontsize=12)
        ax.tick_params(length=0)
        ax.set_title({"SAE (clamp 25%)": f"SAE steering (25% clamp, {n_seeds} seeds)",
                      "LEACE": "LEACE", "amnesic (INLP)": "Amnesic probing (INLP)"}[key], fontsize=15, pad=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("Steered / erased concept", fontsize=14)
    axes[1].set_xlabel("Measured concept", fontsize=14)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.015, pad=0.015)
    cb.set_label("Normalised AUROC drop (1 = to chance)", fontsize=12)
    cb.ax.tick_params(labelsize=11)
    cb.outline.set_visible(False)
    _save(fig, "app_collateral")


def native(seed_dirs):
    runs = [json.loads((d / "native_vs_probe.json").read_text()) for d in seed_dirs]
    cells = [(ci, c["name"]) for ci, c in enumerate(runs[0]["cells"]) if c.get("has_native")]
    fig, axes = plt.subplots(1, len(cells), figsize=(10, 3.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (ci, name) in zip(axes, cells):
        fr = np.asarray(runs[0]["cells"][ci]["fracs"])
        for key, col, ls, lw in [("native_rand", C_RAND, (0, (3, 2)), 1.4),
                                 ("probe", C_PROBE, "-", 2.0),
                                 ("native", C_HEAD, "-", 2.0)]:
            Y = np.array([r["cells"][ci][key] for r in runs])
            m, sd = Y.mean(0), Y.std(0, ddof=1)
            ax.plot(fr, m, color=col, ls=ls, lw=lw, zorder=4)
            ax.fill_between(fr, m - sd, m + sd, color=col, alpha=0.15, lw=0, zorder=2)
        ax.axhline(0.5, color="#888", lw=0.5, alpha=0.5, zorder=1)
        ax.set_ylim(0.3, 1.0)
        ax.set_xlim(0, 0.75)
        fam = name.split()[0]
        ax.set_title(name, fontsize=15, color=FAMILY_COLOUR.get(fam, "black"), weight="bold")
        ax.set_xlabel("Fraction of features clamped", fontsize=14)
        _style(ax)
    axes[0].set_ylabel("Abnormality AUROC", fontsize=14)
    handles = [Line2D([], [], color=C_HEAD, lw=2.0), Line2D([], [], color=C_PROBE, lw=2.0),
               Line2D([], [], color=C_RAND, lw=1.4, ls=(0, (3, 2)))]
    fig.legend(handles, ["Deployed head", "Probe (decoded)", "Deployed head, random clamp"],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=13,
               handlelength=2.2, columnspacing=2.0)
    _save(fig, "app_native_head")


def main():
    S = json.loads((CR / "summary.json").read_text())
    seed_dirs = [CR / s for s in S["seeds"]]
    collateral(S, S["SAE (clamp 25%)"]["baseline"]["Pathology"]["n"])
    native(seed_dirs)
    print("wrote app_collateral, app_native_head ->", OUT)


if __name__ == "__main__":
    main()
