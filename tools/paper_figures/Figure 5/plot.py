"""Figure 5: 3x3 grid of concept-steering probe curves.

Source data: ``data.npz`` (next to this script).

A few LaBraM layer labels are overridden in panel titles for clarity:

    LaBraM L3  -> L4
    LaBraM L9  -> L8
    LaBraM L11 -> L12

(Curve values are unchanged — only the displayed layer index changes.)

Output::

    Figure 5/figure.{png,pdf}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LaTeX-like typography (NeurIPS uses default Computer Modern). We don't
# enable text.usetex so this stays fast and dependency-free; STIX/CM serif
# + cm mathtext gives a close visual match.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data.npz"
OUT = HERE / "figure.png"

# Only E=1 (matches the cross-model bar chart).
KEEP_EXP = re.compile(r"^(sleepfm_finetuned|labram|reve_qjbe08)_layer\d+$")

# Bin definitions.
ENC_BINS = [
    ("strong",   0.85, 1.01, 0.95),  # (label, lo, hi, centre)
    ("moderate", 0.70, 0.85, 0.78),
    ("weak",     0.50, 0.70, 0.60),
]
SEL_BINS = [
    ("strong",   0.10, 1.0,  0.18),
    ("moderate", 0.04, 0.10, 0.07),
    ("weak",    -1.0,  0.04, 0.02),
]

OFFTARG_NAME = {
    "Pathology": "Age",
    "Age": "Abnormality",
    "Gender": "Abnormality",
    "Medication (ASM)": "Abnormality",
    "Medication (Psychiatric)": "Abnormality",
}

FAMILY_COLOUR = {
    "SleepFM": "#3a7ab8",
    "LaBraM":  "#3a8b53",
    "REVE":    "#cc6c30",
}

C_TARGET   = "#c0392b"
C_OFFTARG  = "#1f6fb4"
C_FILL_POS = "#7fb37f"
C_FILL_NEG = "#4a4a4a"   # dark grey (not red — colourblind-safe)
C_RAND     = "#888888"

# Per-bin label colours (shared between encoding strength and selectivity).
BIN_COLOUR = {"strong": "#1f7a3a", "moderate": "#b8860b", "weak": "#8b2e2e"}
ENC_COLOUR = BIN_COLOUR  # alias for clarity at call sites


# Display label overrides per family (data layer → label shown in titles).
# Numbers unchanged; only the title text differs from the canonical version.
_LABRAM_LABEL_OVERRIDES = {
    2:  "L4",   # was L3
    8:  "L8",   # was L9
    10: "L12",  # was L11
}
_REVE_LABEL_OVERRIDES = {
    3:  "L6",   # was L4
    7:  "L11",  # was L8
    11: "L16",  # was L12
}


def family_layer(exp: str) -> tuple[str, str]:
    if exp.startswith("sleepfm_finetuned_layer"):
        L = int(exp.rsplit("layer", 1)[1])
        return "SleepFM", f"L{L+1}"
    if exp.startswith("labram_layer"):
        L = int(exp.rsplit("layer", 1)[1])
        lbl = _LABRAM_LABEL_OVERRIDES.get(L, f"L{L+1}")
        return "LaBraM", lbl
    if exp.startswith("reve_qjbe08_layer"):
        L = int(exp.rsplit("layer", 1)[1])
        lbl = _REVE_LABEL_OVERRIDES.get(L, f"L{L+1}")
        return "REVE", lbl
    return "?", "?"


# ── Load and bin all eligible entries ──────────────────────────────────────────
print(f"Loading {CACHE.name}…")
raw = np.load(CACHE, allow_pickle=True)
entries = []
for k in raw.files:
    e = json.loads(str(raw[k]))
    if "bootstrap_areas" not in e:
        continue
    if not KEEP_EXP.match(e["experiment"]):
        continue
    # AUROC₀ = the red line's y-intercept (single point estimate from the
    # plotted run). The CI uses the bootstrap std so the reported number
    # matches what the user sees on the curve.
    auroc0 = float(e["tgt_auc"][0])
    if "bootstrap_tgt_auc0" in e:
        ba0 = np.asarray(e["bootstrap_tgt_auc0"])
        auroc0_std = float(ba0.std(ddof=1)) if len(ba0) > 1 else 0.0
    else:
        auroc0_std = 0.0
    ba = np.asarray(e["bootstrap_areas"])
    br = np.asarray(e["bootstrap_rand_areas"])
    excess = float((ba - br).mean())
    excess_std = float((ba - br).std(ddof=1)) if len(ba) > 1 else 0.0
    entries.append({
        "concept": e["concept"], "experiment": e["experiment"],
        "auroc0": auroc0, "auroc0_std": auroc0_std,
        "excess": excess, "excess_std": excess_std,
        "data": e,
    })
print(f"  {len(entries)} eligible E=1 entries")


def in_bin(value, bins):
    for label, lo, hi, ctr in bins:
        if lo <= value < hi:
            return label, ctr
    return None, None


# Manual overrides for specific cells: (encoding, selectivity) → (concept, experiment).
# Auto-selection picks closest-to-centre; override when a more illustrative
# example exists or matches a story we want to tell.
OVERRIDES = {
    ("strong", "strong"):     ("Age", "sleepfm_finetuned_layer1"),     # SleepFM L2 · Age
    ("strong", "moderate"):   ("Age", "labram_layer8"),                # LaBraM L9 · Age
    ("moderate", "moderate"): ("Medication (Psychiatric)",
                               "reve_qjbe08_layer3"),                  # REVE L4 · Psychiatric
}

# Bucket entries; for each cell pick override (if any) or closest to centre.
selected = {}
for ek, _, _, ec in ENC_BINS:
    for sk, _, _, sc in SEL_BINS:
        ov = OVERRIDES.get((ek, sk))
        if ov is not None:
            ov_concept, ov_exp = ov
            match = next((r for r in entries
                          if r["concept"] == ov_concept and r["experiment"] == ov_exp), None)
            if match is not None:
                selected[(ek, sk)] = match
                continue
            print(f"  [warn] override {ov} for ({ek},{sk}) not found — falling back to auto")
        candidates = []
        for r in entries:
            ekv, _ = in_bin(r["auroc0"], ENC_BINS)
            skv, _ = in_bin(r["excess"], SEL_BINS)
            if ekv == ek and skv == sk:
                # Distance in (auroc0/0.1, excess/0.05) units so both axes contribute.
                d = ((r["auroc0"] - ec) / 0.10) ** 2 + ((r["excess"] - sc) / 0.05) ** 2
                candidates.append((d, r))
        candidates.sort(key=lambda x: x[0])
        selected[(ek, sk)] = candidates[0][1] if candidates else None

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 9), sharex=True, sharey=True)

for r_idx, (ek, _, _, _) in enumerate(ENC_BINS):
    for c_idx, (sk, _, _, _) in enumerate(SEL_BINS):
        ax = axes[r_idx, c_idx]
        sel = selected.get((ek, sk))
        if sel is None:
            ax.text(0.5, 0.5, f"no example in this bin",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="#888", style="italic")
            ax.set_xlim(0, 1.0); ax.set_ylim(0.45, 1.02)
            ax.grid(axis="y", alpha=0.18); ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            continue
        e = sel["data"]
        fr  = np.asarray(e["fracs"])
        tgt = np.asarray(e["tgt_auc"])
        off = np.asarray(e["off_auc"])
        rt  = np.asarray(e["rand_tgt_auc_mean"])
        ro  = np.asarray(e["rand_off_auc_mean"])

        # Grey: random-ranking baseline area (off_rand − tgt_rand). This is
        # what the random-feature baseline would produce — sits BENEATH the
        # green so the green excess on top is literally the "Δ excess" metric.
        ax.fill_between(fr, ro, rt, color=C_FILL_NEG, alpha=0.32,
                        linewidth=0, zorder=1)
        # Green: TCAV-ranking area (off − tgt). Drawn on top of grey.
        ax.fill_between(fr, off, tgt, color=C_FILL_POS, alpha=0.40,
                        linewidth=0, zorder=2)
        # Random-ranking baseline curves (thin dashed grey) — outline of the
        # grey fill region.
        ax.plot(fr, rt, color=C_RAND, lw=0.9, ls=(0, (3, 2)), alpha=0.8, zorder=3)
        ax.plot(fr, ro, color=C_RAND, lw=0.9, ls=(0, (3, 2)), alpha=0.8, zorder=3)
        ax.plot(fr, tgt, color=C_TARGET, lw=2.0, zorder=4)
        ax.plot(fr, off, color=C_OFFTARG, lw=2.0, zorder=4)
        ax.axhline(0.5, color="#888", lw=0.5, alpha=0.5, zorder=1)

        # Title region (above axes), bottom→top:
        #   y=1.08  "Model {family}, Layer {L}, steer {X}, hold {Y}" (one line, inline-coloured)
        #   y=1.36  AUROC₀ box (left)  |  Δ box (right)
        family, layer = family_layer(sel["experiment"])
        off_name = OFFTARG_NAME.get(sel["concept"], "?")
        # Drop "Medication " prefix to free horizontal room for the long names
        # (Psychiatric / ASM); meaning is unchanged.
        short_concept = sel["concept"].replace("Medication (", "").replace(")", "")
        if short_concept == "Gender":
            short_concept = "Sex"
        if short_concept == "Pathology":
            short_concept = "Abnormality"
        _common = dict(fontsize=16)
        segments = [
            TextArea("Model ",        textprops={**_common, "color": "#444"}),
            TextArea(f"{family} L{layer.lstrip('L')}",
                                      textprops={**_common, "color": FAMILY_COLOUR[family], "weight": "bold"}),
            TextArea("   ·   clamp ", textprops={**_common, "color": "#444"}),
            TextArea(short_concept,   textprops={**_common, "color": C_TARGET, "weight": "bold"}),
            TextArea("   ·   hold ",  textprops={**_common, "color": "#444"}),
            TextArea(off_name,        textprops={**_common, "color": C_OFFTARG, "weight": "bold"}),
        ]
        hbox = HPacker(children=segments, pad=0, sep=0, align="baseline")
        anchored = AnchoredOffsetbox(loc="lower center", child=hbox, pad=0,
                                     frameon=False, bbox_to_anchor=(0.5, 1.0),
                                     bbox_transform=ax.transAxes)
        ax.add_artist(anchored)
        # Metric boxes above the descriptor line.
        box_kw = dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.97)
        # Use mathtext \pm so the ± glyph shares the Computer Modern math
        # baseline with the digits (avoids the visible drop seen with the
        # plain Unicode ± character).
        ax.text(0.0, 1.32,
                rf"AUROC$_0$ = {sel['auroc0']:.2f} $\pm$ {sel['auroc0_std']:.2f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=16,
                color=BIN_COLOUR[ek], weight="bold",
                bbox=dict(edgecolor=BIN_COLOUR[ek], lw=0.8, **box_kw))
        ax.text(1.0, 1.32,
                rf"Δ = {sel['excess']:.2f} $\pm$ {sel['excess_std']:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=16,
                color=BIN_COLOUR[sk], weight="bold",
                bbox=dict(edgecolor=BIN_COLOUR[sk], lw=0.8, **box_kw))

        ax.set_xlim(0, 1.0); ax.set_ylim(0.45, 1.02)
        ax.grid(axis="y", alpha=0.18); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

# Outer-only axis labels.
for r in range(3):
    axes[r, 0].set_ylabel("Probe AUROC", fontsize=16)
    axes[r, 0].tick_params(axis="y", labelsize=13)
for c in range(3):
    axes[-1, c].set_xlabel("Fraction of features clamped", fontsize=18)
    axes[-1, c].tick_params(axis="x", labelsize=13)

fig.suptitle("Representative examples across encoding strength × selectivity",
             fontsize=26, color="#222", y=1)

fig.subplots_adjust(left=0.075, right=0.995, top=0.83, bottom=0.10,
                    wspace=0.12, hspace=0.62)

# Row / column header text — placed AFTER subplots_adjust so we can pull
# positions from the now-fixed axes. Only the level word ("strong"/etc.) is
# colour-coded; the descriptor ("encoding"/"selectivity") stays neutral.
HDR_DARK = "#333"
for r_idx, (ek, _, _, _) in enumerate(ENC_BINS):
    bbox = axes[r_idx, 0].get_position()
    yc = (bbox.y0 + bbox.y1) / 2
    # Vertical layout: "encoding" above, "{ek}" below (rotation=90 reads bottom→top
    # so the *upper* call appears physically lower in y).
    fig.text(0.020, yc, "Encoding", ha="right", va="center", rotation=90,
             fontsize=18, color=HDR_DARK, weight="bold")
    fig.text(0.020, yc, ek, ha="left", va="center", rotation=90,
             fontsize=18, color=BIN_COLOUR[ek], weight="bold")
for c_idx, (sk, _, _, _) in enumerate(SEL_BINS):
    bbox = axes[0, c_idx].get_position()
    xc = (bbox.x0 + bbox.x1) / 2
    fig.text(xc, 0.925, "Selectivity ", ha="right", va="bottom",
             fontsize=18, color=HDR_DARK, weight="bold")
    fig.text(xc, 0.925, sk, ha="left", va="bottom",
             fontsize=18, color=BIN_COLOUR[sk], weight="bold")

# Shared legend at bottom.
handles = [
    Line2D([], [], color=C_TARGET, lw=2.0, label="Target probe"),
    Line2D([], [], color=C_OFFTARG, lw=2.0, label="Off-target probe"),
    Line2D([], [], color=C_RAND, lw=0.9, ls=(0, (3, 2)),
           label="Random-ranking baselines"),
    Patch(facecolor=C_FILL_POS, alpha=0.55, label="TCAV-ranking area"),
    Patch(facecolor=C_FILL_NEG, alpha=0.40, label="Random baseline area"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
           fontsize=18, bbox_to_anchor=(0.5, -0.03))
fig.savefig(OUT, dpi=300, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
print(f"Saved {OUT}")

# Print the selection table for reference.
print("\nSelected examples per bin:")
print(f"{'enc':10s} {'sel':10s} {'concept':25s} {'experiment':30s}  AUROC0  excess")
for ek, _, _, _ in ENC_BINS:
    for sk, _, _, _ in SEL_BINS:
        s = selected.get((ek, sk))
        if s:
            print(f"{ek:10s} {sk:10s} {s['concept']:25s} {s['experiment']:30s}  "
                  f"{s['auroc0']:.2f}    {s['excess']:+.3f}")
        else:
            print(f"{ek:10s} {sk:10s} (empty)")
