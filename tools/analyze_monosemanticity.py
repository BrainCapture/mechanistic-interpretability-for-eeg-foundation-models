"""Monosemanticity analysis of SAE features.

For each feature we measure how many *distinct demographic fields* show at
least one significant association (BH q < 0.05).  Age groups are first
collapsed into three coarse bins (child / adult) so that a feature
broadly enriched across childhood registers as one age-group association, not
eight.

Key terms
---------
field breadth : number of *distinct* fields with at least one significant
                collapsed category  (0 = inactive, 1 = field-specific,
                2+ = cross-field polysemantic — most concerning)
monosemantic  : field breadth == 1

Plots
-----
1. Semantic load histogram for a single reference experiment
2. Field-level breakdown — stacked bar per load bucket (0/1/2+), coloured
   by contributing field
3. Cross-field co-occurrence heatmap — which pairs of fields co-activate
4. Cross-layer monosemanticity — fraction of features with field breadth == 1
   vs layer, one line per encoder family
5. Cross-expansion monosemanticity — fraction monosemantic vs expansion ratio,
   coloured by encoder

Usage::

    # All experiments (produces cross-layer / cross-expansion comparisons)
    uv run tools/analyze_monosemanticity.py

    # Single experiment detail
    uv run tools/analyze_monosemanticity.py \\
        --reference sleepfm_finetuned_layer2

    # Filter to one encoder family
    uv run tools/analyze_monosemanticity.py --filter reve_qjbe08
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "experiments"
OUT_DIR     = ROOT / "results" / "monosemanticity"

P_THRESH = 0.05
FIELDS   = ["age_group", "gender", "indication_group", "medication_group"]
FIELD_COLORS = {
    "age_group":        "#e6550d",
    "gender":           "#3182bd",
    "indication_group": "#31a354",
    "medication_group": "#756bb1",
}

# Fields whose co-activation is biologically expected (age/diagnosis/medication cluster).
# A polysemantic feature whose active fields all fall within this set is classified
# as "entangled-monosemantic"; otherwise it is "spurious-polysemantic".
ENTANGLEMENT_CLUSTER = {"age_group", "classification", "indication_group", "medication_group"}

TAXONOMY_COLORS = {
    "inactive":    "#bdbdbd",
    "separable":   "#74c476",   # green
    "entangled":   "#fd8d3c",   # orange
    "spurious":    "#d62728",   # red
}
TAXONOMY_LABELS = {
    "inactive":  "Inactive",
    "separable": "Separable-monosemantic",
    "entangled": "Entangled-monosemantic",
    "spurious":  "Spurious-polysemantic",
}

# Collapse fine-grained age categories into three clinically meaningful bins.
# A feature enriched in any sub-category of a bin counts as one association.
AGE_COLLAPSE: dict[str, str] = {
    "0-3":   "child",
    "4-9":   "child",
    "10-19": "child",
    "20-29": "adult",
    "30-39": "adult",
    "40-49": "adult",
    "50-59": "adult",
    "60+":   "adult",
}
# Collapsed age labels in display order
AGE_BINS = ["child", "adult"]

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":   150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_experiment(exp_dir: Path) -> dict | None:
    """Load metadata + feature_meta_enrichment for one experiment.

    Returns None if the experiment has no app_cache or no enrichment data.
    """
    cache_path = exp_dir / "app_cache.pt"
    meta_path  = exp_dir / "metadata.json"
    if not cache_path.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    enr_list = cache.get("feature_meta_enrichment", [])
    if not enr_list:
        return None

    return {"meta": meta, "enr": enr_list, "name": exp_dir.name}


def _sig_collapsed_categories(feat: dict, field: str) -> set[str]:
    """Return the set of significant collapsed categories for one feature/field.

    For age_group, raw categories are mapped through AGE_COLLAPSE so that
    0-3, 4-9, 10-19 all map to child before counting.
    Other fields are returned as-is.
    """
    sig: set[str] = set()
    for cat, _r, p in feat.get(field, []):
        if p >= P_THRESH:
            continue
        if field == "age_group":
            collapsed = AGE_COLLAPSE.get(cat)
            if collapsed:
                sig.add(collapsed)
        else:
            sig.add(cat)
    return sig


def _compute_metrics(enr_list: list) -> dict:
    """Compute per-feature monosemanticity metrics.

    Primary metric is *field breadth* — the number of distinct demographic
    fields a feature is significantly associated with after age-group
    collapsing.  field breadth == 1 → monosemantic.

    Returns arrays of length n_features:
      field_breadth     : number of distinct fields with >= 1 sig. collapsed cat.
      per_field_n_cats  : dict field → array of sig. collapsed-category counts
      age_bin_counts    : array (n_features, 3) — sig. collapsed age bins per feature
    """
    n = len(enr_list)
    field_breadth    = np.zeros(n, dtype=int)
    per_field_n_cats = {f: np.zeros(n, dtype=int) for f in FIELDS}
    age_bin_counts   = np.zeros((n, len(AGE_BINS)), dtype=int)

    for i, feat in enumerate(enr_list):
        for field in FIELDS:
            sig_cats = _sig_collapsed_categories(feat, field)
            per_field_n_cats[field][i] = len(sig_cats)
            if sig_cats:
                field_breadth[i] += 1
            if field == "age_group":
                for j, bin_name in enumerate(AGE_BINS):
                    age_bin_counts[i, j] = int(bin_name in sig_cats)

    return {
        "field_breadth":    field_breadth,
        "per_field_n_cats": per_field_n_cats,
        "age_bin_counts":   age_bin_counts,
        "n_features":       n,
    }


def _compute_taxonomy(metrics: dict) -> np.ndarray:
    """Classify each feature into one of four taxonomy categories.

    Returns an integer array of length n_features:
      0 = inactive, 1 = separable-monosemantic,
      2 = entangled-monosemantic, 3 = spurious-polysemantic
    """
    fb   = metrics["field_breadth"]
    pfnc = metrics["per_field_n_cats"]
    n    = metrics["n_features"]
    tax  = np.zeros(n, dtype=int)   # default: inactive

    for i in range(n):
        if fb[i] == 0:
            tax[i] = 0  # inactive
        elif fb[i] == 1:
            tax[i] = 1  # separable-monosemantic
        else:
            # polysemantic — check which fields are active
            active_fields = {f for f in FIELDS if pfnc[f][i] > 0}
            if active_fields <= ENTANGLEMENT_CLUSTER:
                tax[i] = 2  # entangled-monosemantic (only biologically linked fields)
            else:
                tax[i] = 3  # spurious-polysemantic (at least one unexpected field)
    return tax


def _taxonomy_fractions(taxonomy: np.ndarray) -> dict[str, float]:
    n = len(taxonomy)
    keys = ["inactive", "separable", "entangled", "spurious"]
    return {k: float((taxonomy == i).sum()) / n for i, k in enumerate(keys)}


def _discover_experiments(filter_str: str | None) -> list[dict]:
    exps = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if filter_str and filter_str not in d.name:
            continue
        exp = _load_experiment(d)
        if exp is not None:
            exp["metrics"] = _compute_metrics(exp["enr"])
            exp["taxonomy"] = _compute_taxonomy(exp["metrics"])
            exps.append(exp)
    return exps


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Semantic load histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_load_histogram(metrics: dict, name: str, ax: plt.Axes) -> None:
    """Bar chart of field breadth (0–4 distinct fields)."""
    fb   = metrics["field_breadth"]
    n    = metrics["n_features"]
    xs   = np.arange(0, len(FIELDS) + 1)          # 0 .. 4
    counts = np.array([(fb == x).sum() for x in xs])

    palette = ["#aec7e8", "#ef553b", "#ff7f0e", "#d62728", "#7f7f7f"]
    bars = ax.bar(xs, counts / n * 100,
                  color=palette[:len(xs)], edgecolor="white", linewidth=0.5)
    _ = bars  # suppress unused warning

    ax.set_xlabel("Field breadth\n(# distinct demographic fields activated)")
    ax.set_ylabel("% of features")
    ax.set_title(f"{name}\nField breadth distribution\n(age grouped: child / adult)")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["0\n(inactive)", "1\n(mono-\nsemantic)", "2", "3", "4"][: len(xs)]
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    mono_pct = float((fb == 1).sum()) / n * 100
    dead_pct = float((fb == 0).sum()) / n * 100
    poly_pct = 100 - mono_pct - dead_pct
    ax.text(
        0.97, 0.95,
        f"Inactive:      {dead_pct:5.1f}%\n"
        f"Monosemantic:  {mono_pct:5.1f}%\n"
        f"Polysemantic:  {poly_pct:5.1f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Field-level breakdown (stacked bars per load bucket)
# ─────────────────────────────────────────────────────────────────────────────

def plot_field_breakdown(metrics: dict, name: str, ax: plt.Axes) -> None:
    """Stacked bar: for each field breadth bucket, which fields contribute?"""
    breadth = metrics["field_breadth"]
    pfnc    = metrics["per_field_n_cats"]
    n       = metrics["n_features"]
    xs      = np.arange(0, len(FIELDS) + 1)
    bottoms = np.zeros(len(xs))

    for field in FIELDS:
        # Count features in each breadth bucket that have >= 1 sig. cat in this field
        heights = np.array([
            (pfnc[field][breadth == b] > 0).sum()
            for b in xs
        ], dtype=float) / n * 100
        ax.bar(xs, heights, bottom=bottoms,
               color=FIELD_COLORS[field], label=field.replace("_", " "),
               edgecolor="white", linewidth=0.5)
        bottoms += heights

    ax.set_xlabel("Field breadth (# distinct fields activated)")
    ax.set_ylabel("Features (% of total)")
    ax.set_title(f"{name}\nWhich fields drive each breadth bucket?\n(age: child/adult)")
    ax.set_xticks(xs)
    ax.legend(fontsize=9, loc="upper right")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Cross-field co-occurrence heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_cooccurrence(metrics: dict, name: str, ax: plt.Axes) -> None:
    """Heatmap: for features significant in field A, what fraction are also
    significant in field B?  Diagonal = fraction of features with >= 1 sig.
    category in that field."""
    pfl    = metrics["per_field_n_cats"]
    n      = metrics["n_features"]
    nf     = len(FIELDS)
    mat    = np.zeros((nf, nf))

    for i, fi in enumerate(FIELDS):
        has_fi = pfl[fi] > 0
        for j, fj in enumerate(FIELDS):
            has_fj = pfl[fj] > 0
            if i == j:
                mat[i, j] = has_fi.sum() / n
            else:
                # fraction of fi-active features that are also fj-active
                denom = has_fi.sum()
                mat[i, j] = (has_fi & has_fj).sum() / denom if denom > 0 else 0.0

    labels = [f.replace("_", "\n") for f in FIELDS]
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(nf))
    ax.set_yticks(range(nf))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"{name}\nCross-field co-occurrence\n(diagonal = prevalence)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="fraction")

    for i in range(nf):
        for j in range(nf):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if mat[i, j] < 0.6 else "white")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Cross-layer monosemanticity
# ─────────────────────────────────────────────────────────────────────────────

def _encoder_family(name: str) -> str:
    """Group experiment names into encoder families for the cross-layer plot."""
    for prefix in [
        "sleepfm_finetuned", "sleepfm_pretrained",
        "sleepfm_v2.0", "sleepfm_v2.1", "sleepfm_v2.3",
        "sleepfm_v2.4", "sleepfm_v2.5", "sleepfm_v2.6", "sleepfm_v2.7",
        "reve_qjbe08", "reve_k8",
    ]:
        if name.startswith(prefix):
            return prefix
    return name.rsplit("_layer", 1)[0]


def plot_taxonomy_cross_layer(exps: list[dict], ax: plt.Axes) -> None:
    """Stacked bar showing taxonomy fractions per experiment, ordered by layer."""
    # Sort experiments: group by family then by layer
    sorted_exps = sorted(exps, key=lambda e: (
        _encoder_family(e["name"]),
        e["meta"].get("target_layer", -1),
    ))

    labels = [
        f"{_encoder_family(e['name']).split('_')[-1]}\nL{e['meta'].get('target_layer','?')}"
        for e in sorted_exps
    ]
    xs = np.arange(len(sorted_exps))
    keys = ["inactive", "separable", "entangled", "spurious"]
    bottoms = np.zeros(len(sorted_exps))

    for key in keys:
        vals = np.array([
            _taxonomy_fractions(e["taxonomy"])[key] * 100
            for e in sorted_exps
        ])
        ax.bar(xs, vals, bottom=bottoms,
               color=TAXONOMY_COLORS[key],
               label=TAXONOMY_LABELS[key],
               edgecolor="white", linewidth=0.4)
        bottoms += vals

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("% of SAE features")
    ax.set_title("Feature taxonomy across experiments\n(new three-way definition)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Add % labels for entangled+spurious (polysemantic total) above each bar
    for i, exp in enumerate(sorted_exps):
        fracs = _taxonomy_fractions(exp["taxonomy"])
        poly = (fracs["entangled"] + fracs["spurious"]) * 100
        if poly > 2:
            ax.text(i, bottoms[i] + 0.5, f"{poly:.0f}%",
                    ha="center", va="bottom", fontsize=6, color="#555")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Cross-expansion monosemanticity
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_expansion(exps: list[dict], ax: plt.Axes) -> None:
    """Fraction monosemantic vs SAE expansion ratio."""
    # Group by encoder
    by_encoder: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for exp in exps:
        expansion = exp["meta"].get("expansion", None)
        encoder   = exp["meta"].get("encoder", "unknown")
        fb        = exp["metrics"]["field_breadth"]
        n         = exp["metrics"]["n_features"]
        if expansion is None:
            continue
        mono = float((fb == 1).sum()) / n * 100
        by_encoder[encoder].append((float(expansion), mono))

    cmap   = plt.cm.Set1
    colors = {e: cmap(i % 9) for i, e in enumerate(sorted(by_encoder))}

    for encoder, pts in sorted(by_encoder.items()):
        xs, ys = zip(*sorted(pts))
        ax.scatter(xs, ys, label=encoder, color=colors[encoder], alpha=0.7, s=40)
        # Trend line per encoder
        if len(set(xs)) > 1:
            z = np.polyfit(xs, ys, 1)
            px = np.linspace(min(xs), max(xs), 50)
            ax.plot(px, np.polyval(z, px), color=colors[encoder],
                    linewidth=1, linestyle="--", alpha=0.6)

    ax.set_xlabel("SAE expansion ratio")
    ax.set_ylabel("Monosemantic features (%)")
    ax.set_title("Monosemanticity vs expansion ratio")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        default=None,
        help="Experiment name to use for per-experiment plots (default: first found).",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only load experiments whose name contains this string.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering experiments …")
    exps = _discover_experiments(args.filter)
    if not exps:
        print("No experiments with feature_meta_enrichment found.")
        return
    print(f"Loaded {len(exps)} experiments.")

    # Pick reference experiment for single-experiment plots
    ref_name = args.reference
    if ref_name:
        ref_exps = [e for e in exps if e["name"] == ref_name]
        if not ref_exps:
            print(f"Reference experiment '{ref_name}' not found; using first available.")
            ref = exps[0]
        else:
            ref = ref_exps[0]
    else:
        # Default: prefer sleepfm_finetuned_layer2
        preferred = [e for e in exps if e["name"] == "sleepfm_finetuned_layer2"]
        ref = preferred[0] if preferred else exps[0]

    print(f"Reference experiment: {ref['name']}")

    # ── Figure 1: per-experiment detail ──────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
    fig1.suptitle(
        f"SAE feature monosemanticity — {ref['name']}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plot_load_histogram(ref["metrics"], ref["name"], axes1[0])
    plot_field_breakdown(ref["metrics"], ref["name"], axes1[1])
    plot_cooccurrence(ref["metrics"], ref["name"], axes1[2])
    fig1.tight_layout()
    p1 = out_dir / f"monosemanticity_detail_{ref['name']}.png"
    fig1.savefig(p1, bbox_inches="tight")
    print(f"Saved: {p1}")

    # ── Figure 2: cross-experiment taxonomy breakdown ─────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(
        "SAE feature taxonomy — cross-experiment comparison",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plot_taxonomy_cross_layer(exps, axes2[0])
    plot_cross_expansion(exps, axes2[1])
    fig2.tight_layout()
    p2 = out_dir / "monosemanticity_cross_experiment.png"
    fig2.savefig(p2, bbox_inches="tight")
    print(f"Saved: {p2}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────")
    print(f"{'Experiment':<40} {'n':>5} {'inactive%':>10} {'mono%':>8} {'poly%':>8}")
    print("─" * 75)
    for exp in sorted(exps, key=lambda e: e["name"]):
        fb   = exp["metrics"]["field_breadth"]
        n    = exp["metrics"]["n_features"]
        dead = (fb == 0).sum() / n * 100
        mono = (fb == 1).sum() / n * 100
        poly = (fb >= 2).sum() / n * 100
        print(f"{exp['name']:<40} {n:>5} {dead:>9.1f}% {mono:>7.1f}% {poly:>7.1f}%")


if __name__ == "__main__":
    main()
