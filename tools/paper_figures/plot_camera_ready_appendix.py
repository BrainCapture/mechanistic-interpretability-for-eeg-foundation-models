"""Camera-ready appendix figures from the seeded reruns (results/camera_ready/summary.json).

  uv run tools/paper_figures/plot_camera_ready_appendix.py

Writes into paper/camera_ready/figures/:
  app_collateral.png   — normalised AUROC drop, steered (rows) × measured (cols): SAE / LEACE / INLP
  app_native_head.png  — deployed head vs probe vs random clamping, mean ± sd over SAE seeds
  app_matched.png      — off-target preserved at matched target suppression
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mecheeg.nature_fig_style import (  # noqa: E402
    nature_figure, nature_palette, save_nature_figure, set_nature_style, style_axes,
)

CR = Path(os.environ.get("FIG_IN", ROOT / "results" / "camera_ready"))
OUT = Path(os.environ.get("FIG_OUT_DIR", ROOT / "paper" / "camera_ready" / "figures"))
CONCEPTS = ["Pathology", "Age", "Sex", "Med-Psych", "Med-ASM"]
LABELS = {"Pathology": "Abnormality", "Age": "Age", "Sex": "Sex",
          "Med-Psych": "Med. (psych.)", "Med-ASM": "Med. (ASM)"}


def collateral(S):
    import matplotlib.pyplot as plt
    fig, axes = nature_figure(width="double", height_mm=62, ncols=3)
    for ax, key in zip(axes, ["SAE (clamp 25%)", "LEACE", "amnesic (INLP)"]):
        base, mat = S[key]["baseline"], S[key]["matrix"]
        M = np.array([[(base[m]["mean"] - mat[s][m]["mean"]) / max(base[m]["mean"] - 0.5, 1e-6)
                       for m in CONCEPTS] for s in CONCEPTS])
        im = ax.imshow(np.clip(M, 0, 1), cmap="magma_r", vmin=0, vmax=1)
        for i in range(len(CONCEPTS)):
            for j in range(len(CONCEPTS)):
                ax.text(j, i, ">1" if M[i, j] > 1 else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if M[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(CONCEPTS)), [LABELS[c] for c in CONCEPTS], rotation=40, ha="right")
        ax.set_yticks(range(len(CONCEPTS)), [LABELS[c] for c in CONCEPTS] if ax is axes[0] else [])
        ax.set_title({"SAE (clamp 25%)": "SAE steering (25% clamp, 5 seeds)",
                      "LEACE": "LEACE", "amnesic (INLP)": "Amnesic (INLP)"}[key])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("steered / erased concept")
    axes[1].set_xlabel("measured concept")
    cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02)
    cb.set_label("normalised AUROC drop (1 = to chance)")
    save_nature_figure(fig, str(OUT / "app_collateral.png"))
    save_nature_figure(fig, str(OUT / "app_collateral.pdf"))
    plt.close(fig)


def native(seed_dirs):
    import matplotlib.pyplot as plt
    runs = [json.loads((d / "native_vs_probe.json").read_text()) for d in seed_dirs]
    names = [c["name"] for c in runs[0]["cells"]]
    pal = nature_palette(3)
    fig, axes = nature_figure(width="double", height_mm=52, ncols=len(names))
    for ci, (ax, name) in enumerate(zip(axes, names)):
        fr = np.asarray(runs[0]["cells"][ci]["fracs"])
        for key, lab, col, ls in [("native", "deployed head", pal[0], "-"),
                                  ("probe", "probe (decoded)", pal[1], "-"),
                                  ("native_rand", "head, random clamp", "0.5", "--")]:
            if key not in runs[0]["cells"][ci] or not runs[0]["cells"][ci].get("has_native", key == "probe") and key != "probe":
                continue
            Y = np.array([r["cells"][ci][key] for r in runs])
            m, sd = Y.mean(0), Y.std(0, ddof=1)
            ax.plot(fr, m, ls, color=col, lw=1.0, label=lab)
            ax.fill_between(fr, m - sd, m + sd, color=col, alpha=0.18, lw=0)
        ax.axhline(0.5, color="0.7", lw=0.5, ls=":")
        ax.set_ylim(0.3, 1.0); ax.set_xlim(0, 0.75)
        ax.set_title(name + ("" if runs[0]["cells"][ci].get("has_native") else " (probe only)"))
        ax.set_xlabel("fraction of features clamped")
        style_axes(ax)
    axes[0].set_ylabel("abnormality AUROC")
    axes[0].legend(frameon=False, loc="lower left")
    save_nature_figure(fig, str(OUT / "app_native_head.png"))
    save_nature_figure(fig, str(OUT / "app_native_head.pdf"))
    plt.close(fig)


def matched(S):
    import matplotlib.pyplot as plt
    ms = S["matched_suppression"]["methods"]
    order = ["LEACE", "amnesic", "PCA", "SAE", "dense-CAV", "random-dict", "centroid"]
    lab = {"amnesic": "Amnesic (INLP)", "dense-CAV": "Dense CAV", "random-dict": "Random dict.",
           "centroid": "Centroid", "SAE": "SAE (ours)"}
    fig, ax = nature_figure(width="single", height_mm=55)
    pal = nature_palette(2)
    for i, m in enumerate(order):
        v = ms[m]
        ok = all(v["reached"])
        ax.bar(i, v["off_auc"]["mean"], yerr=v["off_auc"]["sd"], color=pal[0] if ok else "0.8",
               edgecolor="none", error_kw=dict(lw=0.6, capsize=1.5))
        if not ok:
            ax.text(i, 0.33, f"target reached\n{sum(v['reached'])}/{len(v['reached'])}",
                    ha="center", fontsize=5, rotation=90, va="bottom")
    ax.axhline(S["matched_suppression"].get("age_baseline", 0.98), color="0.6", lw=0.5, ls=":")
    ax.axhline(0.5, color="0.7", lw=0.5)
    ax.set_xticks(range(len(order)), [lab.get(m, m) for m in order], rotation=40, ha="right")
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("off-target age AUROC preserved")
    ax.set_title("Abnormality suppressed to AUROC $\\approx$ 0.65")
    save_nature_figure(fig, str(OUT / "app_matched.png"))
    save_nature_figure(fig, str(OUT / "app_matched.pdf"))
    plt.close(fig)


def spectral():
    """Band-amplitude shift under steering, mean ± sd over SAE seeds (confound-controlled)."""
    import matplotlib.pyplot as plt
    f = CR / "population_spectral_seeds.json"
    if not f.exists():
        return
    summ = json.loads(f.read_text())["_summary"]
    bands, blab = ["delta", "theta", "alpha", "beta", "gamma"], ["δ", "θ", "α", "β", "γ"]
    panels = [("SleepFM", "SleepFM"), ("LaBraM", "LaBraM"), ("REVE", "REVE")]
    pal = nature_palette(2)
    fig, axes = nature_figure(width="double", height_mm=50, ncols=3, sharey=True)
    for ax, (model, title) in zip(axes, panels):
        x = np.arange(len(bands)); w = 0.38
        for ci, (concept, lab) in enumerate([("classification", "abnormal → normal"), ("age", "child → elderly")]):
            v = summ.get(f"{model} / {concept}")
            if v is None:
                continue
            ax.bar(x + (ci - 0.5) * w, [v[b]["mean"] for b in bands], w, yerr=[v[b]["sd"] for b in bands],
                   color=pal[ci], edgecolor="none", error_kw=dict(lw=0.6, capsize=1.2),
                   label=f"{lab} (n = {v['n_subjects']})")
        ax.axhline(0, color="0.4", lw=0.5)
        ax.set_xticks(x, blab); ax.set_title(title)
        ax.legend(frameon=False, loc="lower right", fontsize=5.5)
        style_axes(ax)
    axes[0].set_ylabel("amplitude shift (µV)")
    save_nature_figure(fig, str(OUT / "app_spectral_seeds.png"))
    save_nature_figure(fig, str(OUT / "app_spectral_seeds.pdf"))
    plt.close(fig)


def main():
    set_nature_style()
    S = json.loads((CR / "summary.json").read_text())
    seed_dirs = [CR / s for s in S["seeds"]]
    base_age = json.loads((seed_dirs[0] / "selectivity_controlled.json").read_text())["baseline"]["Age"]
    S["matched_suppression"]["age_baseline"] = base_age
    collateral(S)
    native(seed_dirs)
    matched(S)
    spectral()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("app_*.png"))))


if __name__ == "__main__":
    main()
