"""Camera-ready Figure 4 — concept encoding and steering selectivity across encoder layers.

Same layout and visual style as the submitted figure
(plot_cross_model_steering_concepts_paper.py), with three corrections:
  * every bar is the mean over 5 independently seeded SAEs (E=1, k=8), error bars are
    the sd across seeds (the submitted bars were single draws and their error bars the
    sd of 3 random rankings);
  * the selectivity axis is not clipped at 0, so negative Δ̃ ("wrecking ball") shows;
  * layers are labelled by their true 1-based index (data computed at those layers).
A dot above a bar marks a cell selective in ≥3/5 seeds (permutation p < 0.05 against
200 random rankings and Δ̃ ≥ 0.05).

Input:  results/camera_ready/fig4/{experiment}.json  (tools/rebuttal/run_fig4_multiseed.sh)
Output: paper/camera_ready/figures/cross_model_steering_concepts.{png,pdf}
        results/camera_ready/fig4/summary.json  (per-cell mean/sd/n_selective)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})

ROOT = Path(__file__).resolve().parent.parent.parent
IN = Path(os.environ.get("FIG_IN", ROOT / "results" / "camera_ready" / "fig4"))
OUT = Path(os.environ.get("FIG_OUT_DIR", ROOT / "paper" / "camera_ready" / "figures"))
STEM = os.environ.get("FIG_STEM", "cross_model_steering_concepts")

# 0-based code index → shown as 1-based label
FAMILIES = [
    ("SleepFM", [0, 1, 2], "sleepfm_finetuned_layer{L}", plt.cm.Blues),
    ("LaBraM", [0, 3, 7, 11], "labram_layer{L}", plt.cm.Greens),
    ("REVE", [0, 5, 10, 15, 21], "reve_qjbe08_layer{L}", plt.cm.Oranges),
]
CONCEPTS = [("Pathology", "Abnormality"), ("Age", "Age"), ("Gender", "Sex"),
            ("Medication (Psychiatric)", "Medication (Psychiatric)"),
            ("Medication (ASM)", "Medication (ASM)")]
SEL_P, SEL_MIN, SEL_FRAC = 0.05, 0.05, 0.6


def cell_stats(rows):
    ex = np.array([r["excess_delta"] if "excess_delta" in r else r["delta_tilde"] - r["perm_null_mean"]
                   for r in rows])
    a0 = np.array([r["auroc0"] for r in rows])
    nsel = int(sum(1 for r, e in zip(rows, ex) if r["p_value"] < SEL_P and e >= SEL_MIN))
    return {"excess_mean": float(ex.mean()), "excess_sd": float(ex.std(ddof=1)) if len(ex) > 1 else 0.0,
            "auroc0_mean": float(a0.mean()), "auroc0_sd": float(a0.std(ddof=1)) if len(a0) > 1 else 0.0,
            "n_seeds": int(len(ex)), "n_selective": nsel, "values": [round(float(x), 4) for x in ex]}


def main():
    data, summary = {}, {}
    for fam, layers, pat, _ in FAMILIES:
        for L in layers:
            exp = pat.format(L=L)
            f = IN / f"{exp}.json"
            if not f.exists():
                print(f"[missing] {exp}")
                continue
            cells = json.loads(f.read_text())["cells"]
            for cname, _ in CONCEPTS:
                rows = [c for c in cells if c["concept"] == cname]
                if rows:
                    data[(cname, exp)] = cell_stats(rows)
                    summary[f"{cname} | {fam} L{L + 1}"] = data[(cname, exp)]
    (IN / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, (ax_top, ax_sel) = plt.subplots(2, 1, figsize=(17.5, 6.6), sharex=True,
                                         gridspec_kw=dict(height_ratios=[1.0, 1.6], hspace=0.08))
    BAR_W, GROUP_GAP = 1.2, 2.4
    ramps = {fam: cmap(np.linspace(0.40, 0.92, len(layers))) for fam, layers, _, cmap in FAMILIES}
    x, centres, seps, first_x, last_x = 0.0, [], [], None, None
    for ci, (cname, disp) in enumerate(CONCEPTS):
        g0 = x
        for fam, layers, pat, _ in FAMILIES:
            for li, L in enumerate(layers):
                exp = pat.format(L=L)
                col = ramps[fam][li][:3]
                xc = x + BAR_W / 2
                first_x = xc if first_x is None else first_x
                last_x = xc
                v = data.get((cname, exp))
                if v is not None:
                    ec = tuple(0.55 * c for c in col)
                    ax_sel.bar(xc, v["excess_mean"], width=BAR_W, color=col, edgecolor="black",
                               linewidth=0.8, zorder=3, yerr=v["excess_sd"],
                               error_kw=dict(lw=1.0, capsize=0, ecolor=ec))
                    if v["n_selective"] >= SEL_FRAC * v["n_seeds"]:
                        ax_sel.plot(xc, v["excess_mean"] + v["excess_sd"] + 0.012, "o", ms=3.2,
                                    color="black", zorder=4)
                    ax_top.bar(xc, v["auroc0_mean"] - 0.5, bottom=0.5, width=BAR_W, color=col,
                               edgecolor="black", linewidth=0.4, zorder=3, yerr=v["auroc0_sd"],
                               error_kw=dict(lw=1.0, capsize=0, ecolor=ec))
                x += BAR_W
        centres.append(((g0 + x) / 2, disp))
        if ci < len(CONCEPTS) - 1:
            seps.append(x + GROUP_GAP / 2)
            x += GROUP_GAP

    ax_sel.axhline(0, color="black", lw=0.6, zorder=2)
    for ax in (ax_top, ax_sel):
        ax.grid(axis="y", alpha=0.18); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=13)
        for sx in seps:
            ax.axvline(sx, color="black", lw=0.4, alpha=0.12, zorder=1)
    ax_top.set_ylim(0.5, 1.0)
    # lift (pt): the moderate label would otherwise sit on the REVE L22 / ASM bar (0.72)
    for thr, lbl, col, lift in [(0.85, "strong", "#1f7a3a", 0), (0.70, "moderate", "#b8860b", 9)]:
        ax_top.axhline(thr, color=col, lw=0.6, ls=(0, (4, 3)), alpha=0.55, zorder=2)
        ax_top.annotate(rf" {lbl} $\geq$ {thr:.2f}", xy=(0.998, thr), xycoords=ax_top.get_yaxis_transform(),
                        xytext=(0, lift), textcoords="offset points", ha="right", va="bottom",
                        fontsize=18, color=col, style="italic", weight="bold", zorder=5)
    ax_sel.set_xticks([])
    for xc, name in centres:
        ax_sel.text(xc, -0.04, name, transform=ax_sel.get_xaxis_transform(), ha="center", va="top",
                    fontsize=18)
    ax_sel.set_xlim(first_x - BAR_W, last_x + BAR_W)
    ax_sel.set_ylabel(r"Excess selectivity $\tilde{\Delta}$", fontsize=18)
    ax_top.set_ylabel("Encoding strength", fontsize=18)
    ax_top.set_title("Concept encoding and steering selectivity across encoder layers", fontsize=26, pad=14)

    # one legend per family, each in a single row centred under its share of the bars
    fig.subplots_adjust(bottom=0.24, left=0.07, right=0.98, top=0.92)
    pos = ax_sel.get_position()
    n_tot = sum(len(lays) for _, lays, _, _ in FAMILIES)
    seen = 0
    for fam, layers, _, _ in FAMILIES:
        handles = [Patch(facecolor=ramps[fam][li][:3], edgecolor="black", linewidth=0.8)
                   for li in range(len(layers))]
        labels = [f"L{L + 1}" for L in layers]
        xc = (seen + len(layers) / 2) / n_tot
        seen += len(layers)
        leg = fig.legend(handles, labels, title=fam, fontsize=16, title_fontsize=18, loc="upper center",
                         bbox_to_anchor=(pos.x0 + xc * pos.width, pos.y0 - 0.075), frameon=False,
                         ncol=len(layers), columnspacing=1.6, handlelength=1.6, handleheight=1.2,
                         handletextpad=0.5, borderpad=0.3)
        leg.get_title().set_fontweight("bold")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{STEM}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{STEM}.pdf", bbox_inches="tight")
    print(f"wrote {OUT / (STEM + '.png')}  ({len(data)} cells)")


if __name__ == "__main__":
    main()
