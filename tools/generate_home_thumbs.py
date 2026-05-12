"""Generate magazine-style thumbnails for the Streamlit app home page.

One PNG per tab — depicting representative content the user would actually
see when opening that tab. Outputs go to ``app/static/home_thumbs/`` and are
checked into git so the home page renders instantly without recomputing.

Run:  uv run tools/generate_home_thumbs.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "app" / "static" / "home_thumbs"
OUT.mkdir(parents=True, exist_ok=True)

FIGSIZE = (8.0, 4.5)   # 16:9
DPI     = 130

# Discreet, dark-on-light palette
BAND_ORDER  = ["delta", "theta", "alpha", "low-beta", "high-beta", "gamma"]
BAND_COLORS = ["#3d5a80", "#5c8aa9", "#98c1d9", "#e0a96d", "#d2691e", "#9c5232"]
ACCENT_BLUE = "#2864a8"
ACCENT_RED  = "#c0392b"
ACCENT_GREY = "#888"


def _setup_axes(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#bbb")
    ax.tick_params(colors="#555", length=2)


# ─── 1. Feature Explorer ──────────────────────────────────────────────────────

def thumb_feature_explorer() -> Path:
    expl_path = ROOT / "results" / "xae" / "sleepfm_finetuned" / "explanations" / "feature_explanations.json"
    if not expl_path.exists():
        return _placeholder("feature_explorer", "Feature spectral signature")

    expl = json.load(open(expl_path))
    # Pick the feature with the largest absolute peak boost
    feat = max(expl, key=lambda f: abs(f.get("peak_boost_delta", 0.0)))
    band_effects = feat.get("band_effects", {})
    values = [band_effects.get(b, 0.0) for b in BAND_ORDER]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    bars = ax.bar(BAND_ORDER, values, color=BAND_COLORS, edgecolor="white",
                  linewidth=1.5)
    ax.set_ylabel("amplitude effect (Δ z)", fontsize=10, color="#444")
    ax.set_title(f"Feature {feat['feature']} · {feat.get('description', '')}",
                 fontsize=11, color="#333", loc="left", pad=10)
    _setup_axes(ax)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.35)

    out = OUT / "feature_explorer.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── 2. Layer Explorer ────────────────────────────────────────────────────────

def thumb_layer_explorer() -> Path:
    p = ROOT / "results" / "layer_umap" / "sleepfm_finetuned" / "umap_cache.pt"
    if not p.exists():
        return _placeholder("layer_explorer", "Joint UMAP across encoder layers")

    cache = torch.load(p, map_location="cpu", weights_only=False)
    xy_dict = cache["xy"]
    labels  = np.asarray(cache["label"])

    # Use last-layer UMAP (most separated, most representative of the final encoding).
    last_layer = max(int(k) for k in xy_dict.keys())
    xy = np.asarray(xy_dict[last_layer])

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    # Downsample for speed/clarity
    n_max = 4000
    if xy.shape[0] > n_max:
        rng = np.random.default_rng(0)
        idx = rng.choice(xy.shape[0], size=n_max, replace=False)
        xy = xy[idx]; labels = labels[idx]

    abn_mask = labels == 1
    ax.scatter(xy[~abn_mask, 0], xy[~abn_mask, 1], s=5, alpha=0.35,
               c=ACCENT_BLUE, label="Normal", edgecolors="none")
    ax.scatter(xy[abn_mask, 0],  xy[abn_mask, 1],  s=5, alpha=0.55,
               c=ACCENT_RED,  label="Abnormal", edgecolors="none")
    ax.set_title(f"Token UMAP · layer {last_layer}", fontsize=11, color="#333",
                 loc="left", pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbb")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color("#444")

    out = OUT / "layer_explorer.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── 3. TCAV Explorer ─────────────────────────────────────────────────────────

def thumb_tcav_explorer() -> Path:
    candidates = [
        "sleepfm_finetuned_layer2",
        "sleepfm_finetuned_v4_layer1",
        "labram_layer11",
        "reve_qjbe08_exp16_layer21",
    ]
    cache_path = None
    for c in candidates:
        p = ROOT / "results" / "tcav" / c / "tcav_cache.pt"
        if p.exists():
            cache_path = p; break
    if cache_path is None:
        return _placeholder("tcav_explorer", "TCAV scores per concept")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    names   = list(cache["concept_names"])
    scores  = np.asarray(cache.get("model_tcav_scores", []), dtype=float)
    pvals   = np.asarray(cache.get("model_tcav_p_values", []), dtype=float)

    # Drop NaN entries
    keep = np.isfinite(scores)
    names  = [names[i]  for i in range(len(names))  if keep[i]]
    scores = scores[keep]; pvals = pvals[keep] if pvals.size == keep.size else None

    # Sort by score
    order  = np.argsort(scores)
    names  = [names[i] for i in order]
    scores = scores[order]
    if pvals is not None and len(pvals) == len(scores):
        pvals = pvals[order]

    colors = []
    for i, _s in enumerate(scores):
        sig = pvals is not None and i < len(pvals) and pvals[i] < 0.05
        colors.append(ACCENT_BLUE if sig else "#cfd8dc")

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.barh(names, scores, color=colors, edgecolor="white", linewidth=1.0, height=0.7)
    ax.axvline(0.5, color="#bbb", lw=0.8, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_xlabel("TCAV score (Variant C)", fontsize=10, color="#444")
    ax.set_title(f"Concept attribution · {cache_path.parent.name}",
                 fontsize=11, color="#333", loc="left", pad=10)
    _setup_axes(ax)
    ax.tick_params(axis="y", labelsize=9)

    out = OUT / "tcav_explorer.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── 4. Concept Steering ──────────────────────────────────────────────────────

def thumb_concept_steering() -> Path:
    p = ROOT / "tools" / "paper_figures" / "Figure 5" / "data.npz"
    if not p.exists():
        return _placeholder("concept_steering", "Target / off-target AUROC sweep")

    d = np.load(p, allow_pickle=True)
    target_pair = ("Pathology", "sleepfm_finetuned_layer2")
    chosen = None
    for k in d.keys():
        e = json.loads(str(d[k]))
        if (e["concept"], e["experiment"]) == target_pair:
            chosen = e; break
    if chosen is None:
        # fall back to first entry
        chosen = json.loads(str(d[list(d.keys())[0]]))

    fracs    = np.asarray(chosen["fracs"])
    tgt      = np.asarray(chosen["tgt_auc"])
    off      = np.asarray(chosen["off_auc"])
    rand_tgt = np.asarray(chosen.get("rand_tgt_auc_mean", []))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    if rand_tgt.size == fracs.size:
        ax.plot(fracs, rand_tgt, color="#cfd8dc", lw=1.5, linestyle="--",
                label="Random direction")
    ax.plot(fracs, off, color=ACCENT_BLUE, lw=2.0, label="Off-target")
    ax.plot(fracs, tgt, color=ACCENT_RED,  lw=2.4, label="Target")
    ax.fill_between(fracs, tgt, off, color=ACCENT_RED, alpha=0.06)

    ax.set_xlim(0, 1); ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Clamping fraction f", fontsize=10, color="#444")
    ax.set_ylabel("AUROC", fontsize=10, color="#444")
    ax.set_title(f"Steering sweep · {chosen['experiment']} · {chosen['concept']}",
                 fontsize=11, color="#333", loc="left", pad=10)
    _setup_axes(ax)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.35)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color("#444")

    out = OUT / "concept_steering.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── 5. Taxonomy & Steering ───────────────────────────────────────────────────

def thumb_taxonomy() -> Path:
    p = ROOT / "tools" / "paper_figures" / "Figure 3" / "data.json"
    if not p.exists():
        return _placeholder("taxonomy_steering", "Monosemanticity taxonomy")

    d = json.load(open(p))
    # Build separable fraction heatmap for SleepFM finetuned: layers 0..2 × E in 1..64
    expansions = [1, 2, 4, 8, 16, 32, 64]
    layers     = [0, 1, 2]
    H = np.full((len(layers), len(expansions)), np.nan)
    for li, L in enumerate(layers):
        for ei, E in enumerate(expansions):
            key = f"sleepfm_finetuned_layer{L}" if E == 1 else f"sleepfm_finetuned_exp{E}_layer{L}"
            entry = d.get(key)
            if entry:
                H[li, ei] = entry.get("separable", np.nan)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    im = ax.imshow(H, aspect="auto", cmap="YlGnBu", vmin=0, vmax=0.6)
    for li in range(H.shape[0]):
        for ei in range(H.shape[1]):
            v = H[li, ei]
            if np.isfinite(v):
                txt_color = "white" if v > 0.35 else "#333"
                ax.text(ei, li, f"{v:.2f}", ha="center", va="center",
                        color=txt_color, fontsize=9)
    ax.set_xticks(range(len(expansions)))
    ax.set_xticklabels([f"×{e}" for e in expansions], color="#444", fontsize=9)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{L}" for L in layers], color="#444", fontsize=9)
    ax.set_xlabel("Expansion", fontsize=10, color="#444")
    ax.set_title("Separable feature fraction · SleepFM finetuned",
                 fontsize=11, color="#333", loc="left", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors="#555", length=2)

    out = OUT / "taxonomy_steering.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── 6. Attention Explorer ────────────────────────────────────────────────────

def thumb_attention_explorer() -> Path:
    candidates = [
        "sleepfm_finetuned_layer2",
        "sleepfm_finetuned_layer1",
        "sleepfm_finetuned_layer0",
    ]
    cache = None
    for c in candidates:
        p = ROOT / "results" / "experiments" / c / "attention_cache.pt"
        if p.exists():
            cache = torch.load(p, map_location="cpu", weights_only=False); break
    if cache is None:
        return _placeholder("attention_explorer", "Self-attention matrix")

    attn = np.asarray(cache["windows_attn"])   # (W, heads, S, S)
    if attn.ndim != 4:
        return _placeholder("attention_explorer", "Self-attention matrix")
    # Mean over windows, pick the head with highest off-diagonal mass
    mean_attn = attn.mean(axis=0)              # (heads, S, S)
    off_diag = (mean_attn - mean_attn * np.eye(mean_attn.shape[-1])[None]).sum(axis=(1, 2))
    head = int(np.argmax(off_diag))
    M = mean_attn[head]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    im = ax.imshow(M, cmap="magma_r", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Key position", fontsize=10, color="#444")
    ax.set_ylabel("Query position", fontsize=10, color="#444")
    ax.set_title(f"Self-attention · head {head}", fontsize=11, color="#333",
                 loc="left", pad=10)
    ax.tick_params(colors="#555", length=2)
    for s in ax.spines.values():
        s.set_color("#bbb")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors="#555", length=2)

    out = OUT / "attention_explorer.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Placeholder fallback ─────────────────────────────────────────────────────

def _placeholder(slug: str, label: str) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_facecolor("#f4f4f8")
    ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=12,
            color="#888", style="italic")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#dcdce4")
    out = OUT / f"{slug}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Driver ───────────────────────────────────────────────────────────────────

GENERATORS = [
    ("Feature Explorer",    thumb_feature_explorer),
    ("Layer Explorer",      thumb_layer_explorer),
    ("TCAV Explorer",       thumb_tcav_explorer),
    ("Concept Steering",    thumb_concept_steering),
    ("Taxonomy & Steering", thumb_taxonomy),
    ("Attention Explorer",  thumb_attention_explorer),
]


def main() -> int:
    print(f"Writing home-page thumbnails to {OUT.relative_to(ROOT)}/")
    for name, fn in GENERATORS:
        try:
            p = fn()
            print(f"  ✓ {name:24s} → {p.relative_to(ROOT)}")
        except Exception as ex:
            print(f"  ✗ {name:24s} failed: {ex!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
