"""
Clinical Morphology Analysis
==============================
Links SAE feature activations to clinical EEG patterns by combining:
  1. Per-feature firing rates on Normal vs Abnormal windows
  2. XAE spectral signatures (from explain_features)
  3. Cluster assignments (spectral archetypes)
  4. Known spectral fingerprints of clinical EEG morphologies

The dataset labels are binary (0=Normal, 1=Abnormal) where Abnormal
covers multiple sub-types. The preprocessing report tells us the mix:
  - focal spike-and-wave:       1454 windows (18.2%)
  - generalized spike-and-wave: 445+472 windows (11.5%)
  - focal sharp waves:          292 windows (3.7%)
  - diffuse slowing:            202 windows (2.5%)
  - focal slowing:              97 windows (1.2%)

We can't label individual windows by morphology type, but we CAN:
  (a) identify features that fire preferentially on Abnormal
  (b) read their spectral signatures from the XAE
  (c) match those signatures to known clinical morphology spectra

Outputs (in results/xae/morphology/):
  01_feature_label_enrichment.png — per-feature abnormal enrichment + spectra
  02_cluster_clinical_profile.png — cluster × label × spectral summary
  03_clinical_mapping.png         — spectral signature → morphology match
  morphology_report.json          — machine-readable clinical mapping

Usage:  uv run tools/morphology_analysis.py
"""

import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats

# ── project imports ─────────────────────────────────────────────────────────
from sae4eeg.xae import XAETrainer, CLINICAL_BANDS
from sae4eeg.sae import SparseAutoencoder, ActivationExtractor
from sae4eeg.dataset import get_dataloaders, StandardizeLabel
from sae4eeg.encoders import load_encoder

# ── paths & constants ───────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
MODEL_PATH   = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"

_ENCODER_DATA = {
    "sleepfm": ROOT / "data" / "D4-v3-preprocessed-v2",
    "reve":    ROOT / "data" / "D4-v3-preprocessed-v1",
}
_ENCODER_FS = {"sleepfm": 128, "reve": 200}
DATA_PATH    = _ENCODER_DATA["sleepfm"]  # overridden at runtime
XAE_PATH     = ROOT / "results" / "xae" / "xae_checkpoint.pt"
EXPL_PATH    = ROOT / "results" / "xae" / "explanations" / "feature_explanations.json"
OUT_DIR      = ROOT / "results" / "xae" / "morphology"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EMBED        = 128    # overridden at runtime from encoder.embed_dim
EXPANSION    = 1.0
K            = 8
PATCH_SIZE   = 128
FS           = 128
TARGET_LAYER = 2      # overridden at runtime: last encoder layer
S_TOKENS     = 60
MAX_WINDOWS  = 4000        # use all available val windows

# SAE_PATH is resolved at runtime based on encoder name + layer
SAE_PATH     = ROOT / "results" / "features" / "sleepfm" / "sae_sleepfm_exp1_k8_layer2.pt"

LABEL_NAMES  = {0: "Normal", 1: "Abnormal"}
LABEL_COLORS = {0: "#4CAF50", 1: "#E53935"}

BAND_COLORS = {
    "delta":     "#1f77b4",
    "theta":     "#2ca02c",
    "alpha":     "#ff7f0e",
    "low-beta":  "#d62728",
    "high-beta": "#9467bd",
    "gamma":     "#8c564b",
}

# ── Known clinical EEG morphology spectral fingerprints ─────────────────────
# Each morphology is characterised by which bands it boosts/suppresses.
# These are textbook spectral signatures used for matching.
CLINICAL_MORPHOLOGIES = {
    "Spike-and-slow-wave": {
        "description": "Sharp transient (70-200ms) followed by slow wave (200-500ms). "
                       "Hallmark of epileptiform activity.",
        "spectral_profile": {
            "delta": "strong_boost",       # slow wave component
            "theta": "moderate_boost",     # slow wave tail
            "alpha": "suppress",           # disrupted background
            "low-beta": "boost",           # sharp transient
            "high-beta": "boost",          # sharp transient
            "gamma": "neutral",
        },
        "match_weights": {"delta": 2.0, "theta": 1.0, "alpha": -1.5,
                          "low-beta": 1.0, "high-beta": 1.5, "gamma": 0.0},
    },
    "Generalised slowing": {
        "description": "Diffuse increase in slow activity (delta/theta) with reduced "
                       "alpha. Seen in encephalopathy, drowsiness, medication effects.",
        "spectral_profile": {
            "delta": "strong_boost",
            "theta": "strong_boost",
            "alpha": "strong_suppress",
            "low-beta": "suppress",
            "high-beta": "suppress",
            "gamma": "suppress",
        },
        "match_weights": {"delta": 2.0, "theta": 2.0, "alpha": -2.0,
                          "low-beta": -1.0, "high-beta": -1.0, "gamma": -0.5},
    },
    "Alpha rhythm (normal)": {
        "description": "Dominant posterior 8-13Hz rhythm. Present in relaxed wakefulness. "
                       "Suppresses with eye opening (reactivity).",
        "spectral_profile": {
            "delta": "suppress",
            "theta": "neutral",
            "alpha": "strong_boost",
            "low-beta": "neutral",
            "high-beta": "neutral",
            "gamma": "neutral",
        },
        "match_weights": {"delta": -1.0, "theta": 0.0, "alpha": 3.0,
                          "low-beta": 0.0, "high-beta": 0.0, "gamma": 0.0},
    },
    "Beta activity": {
        "description": "Increased fast activity (13-30Hz). Seen with alertness, anxiety, "
                       "or medication effects (e.g., benzodiazepines).",
        "spectral_profile": {
            "delta": "suppress",
            "theta": "suppress",
            "alpha": "neutral",
            "low-beta": "strong_boost",
            "high-beta": "strong_boost",
            "gamma": "boost",
        },
        "match_weights": {"delta": -1.0, "theta": -1.0, "alpha": 0.0,
                          "low-beta": 2.0, "high-beta": 2.0, "gamma": 1.0},
    },
    "Focal sharp waves": {
        "description": "Brief sharp transients (70-200ms) without following slow wave. "
                       "Localised epileptiform marker.",
        "spectral_profile": {
            "delta": "neutral",
            "theta": "neutral",
            "alpha": "suppress",
            "low-beta": "boost",
            "high-beta": "strong_boost",
            "gamma": "boost",
        },
        "match_weights": {"delta": 0.0, "theta": 0.0, "alpha": -1.0,
                          "low-beta": 1.5, "high-beta": 2.5, "gamma": 1.0},
    },
    "Theta slowing": {
        "description": "Increased theta (4-8Hz) with preserved alpha. Mild slowing, "
                       "seen in drowsiness or mild encephalopathy.",
        "spectral_profile": {
            "delta": "neutral",
            "theta": "strong_boost",
            "alpha": "neutral",
            "low-beta": "suppress",
            "high-beta": "suppress",
            "gamma": "suppress",
        },
        "match_weights": {"delta": 0.0, "theta": 3.0, "alpha": 0.0,
                          "low-beta": -1.0, "high-beta": -1.0, "gamma": -0.5},
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def remove_module_from_state_dict(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()}


def load_model(encoder_name: str = "sleepfm", weights_path=None):
    if weights_path is not None:
        kwargs = {"weights_path": weights_path}
    elif encoder_name == "sleepfm":
        kwargs = {"weights_path": MODEL_PATH}
    else:
        kwargs = {}
    encoder = load_encoder(encoder_name, **kwargs)
    encoder.to(DEVICE).eval()
    return encoder


def load_sae():
    ckpt = torch.load(SAE_PATH, weights_only=False)
    sae = SparseAutoencoder(EMBED, expansion=EXPANSION, mode="topk", k=K)
    sae.load_state_dict(ckpt["sae_state_dict"])
    act_mean = ckpt["act_mean"]
    act_std = ckpt["act_std"]
    return sae, act_mean, act_std


def encode_windows(model, sae, act_mean, act_std, loader, max_windows=MAX_WINDOWS):
    """
    Encode EEG windows through encoder → SAE.
    Returns:
        codes_per_window: (N, S, dict_size)
        labels: (N,)

    ``model`` may be an ``EncoderBackend`` or a plain ``SetTransformer``.
    """
    from sae4eeg.encoders import EncoderBackend

    if isinstance(model, EncoderBackend):
        inner_model = model.model
        hook_layers = model.get_hookable_layers()
        call_fn = lambda x: model.encode(x)   # noqa: E731
    else:
        inner_model = model
        hook_layers = None
        call_fn = lambda x: inner_model(x)    # noqa: E731

    sae_dev = sae.to(DEVICE).eval()
    mean_dev = act_mean.to(DEVICE)
    std_dev = act_std.to(DEVICE)

    all_codes, all_labels = [], []
    n = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        all_labels.append(y)
        with torch.no_grad():
            x_dev = x.to(DEVICE)
            extractor = ActivationExtractor(inner_model, layers=hook_layers)
            _ = call_fn(x_dev)
            acts = extractor.get_activations()
            extractor.remove_hooks()

            layer_acts = acts[TARGET_LAYER].to(DEVICE)
            B, S, E = layer_acts.shape
            normed = (layer_acts - mean_dev) / std_dev
            flat = normed.reshape(B * S, E)
            z = F.relu(sae_dev.encoder(flat - sae_dev.b_pre))
            if sae_dev.mode == "topk":
                z = SparseAutoencoder._topk_mask_fn(z, sae_dev.k)
            codes = z.reshape(B, S, -1)
            all_codes.append(codes.cpu())
        n += x.shape[0]
        if n >= max_windows:
            break

    sae.cpu()
    codes = torch.cat(all_codes, dim=0)[:max_windows]
    labels = torch.cat(all_labels, dim=0)[:max_windows]
    return codes, labels


# ═════════════════════════════════════════════════════════════════════════════
# Analysis: Feature × Label statistics
# ═════════════════════════════════════════════════════════════════════════════

def compute_feature_label_stats(codes_per_window, labels):
    """
    For each SAE feature, compute:
      - firing rate on Normal vs Abnormal
      - mean activation on Normal vs Abnormal
      - enrichment ratio = fire_rate_abnormal / fire_rate_normal
      - point-biserial correlation with label
    """
    dict_size = codes_per_window.shape[-1]
    # Per-window mean activation
    mean_codes = codes_per_window.mean(dim=1)  # (N, dict_size)
    # Per-token firing (for firing rate)
    codes_flat = codes_per_window.reshape(-1, dict_size)

    normal_mask = labels == 0
    abnormal_mask = labels == 1
    normal_mask.sum().item()
    abnormal_mask.sum().item()

    # Per-window mean codes split by label
    normal_codes = mean_codes[normal_mask]    # (n_normal, dict_size)
    abnormal_codes = mean_codes[abnormal_mask]

    # Per-token codes split by label (each window has S tokens)
    normal_tokens = codes_per_window[normal_mask].reshape(-1, dict_size)
    abnormal_tokens = codes_per_window[abnormal_mask].reshape(-1, dict_size)

    results = []
    label_arr = labels.numpy().astype(float)

    for i in range(dict_size):
        # Firing rates (fraction of tokens where feature fires)
        fr_normal = (normal_tokens[:, i] > 0).float().mean().item()
        fr_abnormal = (abnormal_tokens[:, i] > 0).float().mean().item()
        fr_overall = (codes_flat[:, i] > 0).float().mean().item()

        # Mean activation (when firing)
        n_active = normal_tokens[:, i][normal_tokens[:, i] > 0]
        a_active = abnormal_tokens[:, i][abnormal_tokens[:, i] > 0]
        mean_act_normal = n_active.mean().item() if len(n_active) > 0 else 0.0
        mean_act_abnormal = a_active.mean().item() if len(a_active) > 0 else 0.0

        # Enrichment ratio
        enrichment = (fr_abnormal / fr_normal) if fr_normal > 0.001 else float('inf')

        # Point-biserial correlation (window-level)
        r, p = sp_stats.pointbiserialr(label_arr, mean_codes[:, i].numpy())

        # Cohen's d (effect size)
        n_vals = normal_codes[:, i].numpy()
        a_vals = abnormal_codes[:, i].numpy()
        pooled_std = np.sqrt((n_vals.std()**2 + a_vals.std()**2) / 2)
        cohens_d = (a_vals.mean() - n_vals.mean()) / pooled_std if pooled_std > 0 else 0.0

        results.append({
            "feature": i,
            "fire_rate_normal": fr_normal,
            "fire_rate_abnormal": fr_abnormal,
            "fire_rate_overall": fr_overall,
            "mean_act_normal": mean_act_normal,
            "mean_act_abnormal": mean_act_abnormal,
            "enrichment_ratio": enrichment,
            "label_correlation": float(r),
            "label_p_value": float(p),
            "cohens_d": float(cohens_d),
        })

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Morphology matching: spectral signature → clinical pattern
# ═════════════════════════════════════════════════════════════════════════════

def match_morphology(band_effects, morphologies=CLINICAL_MORPHOLOGIES):
    """
    Score each clinical morphology against a feature's band-effect vector.
    Returns sorted list of (morphology_name, score).
    """
    band_names = list(CLINICAL_BANDS.keys())
    scores = {}
    for morph_name, morph in morphologies.items():
        weights = morph["match_weights"]
        score = sum(band_effects.get(b, 0.0) * weights.get(b, 0.0)
                    for b in band_names)
        # Normalise by weight magnitude
        w_norm = np.sqrt(sum(w**2 for w in weights.values()))
        e_norm = np.sqrt(sum(band_effects.get(b, 0.0)**2 for b in band_names))
        if w_norm > 0 and e_norm > 0:
            score = score / (w_norm * e_norm)  # cosine similarity
        scores[morph_name] = score

    return sorted(scores.items(), key=lambda x: -x[1])


# ═════════════════════════════════════════════════════════════════════════════
# Plot 1: Feature × Label Enrichment (top features, sorted by |Cohen's d|)
# ═════════════════════════════════════════════════════════════════════════════

def plot_feature_enrichment(label_stats, explanations, save_path, top_n=20):
    """
    Show the top N features by |Cohen's d| with their spectral profiles.
    Two columns: (left) enrichment bars, (right) spectral description.
    """
    # Sort by absolute Cohen's d
    sorted_stats = sorted(label_stats, key=lambda x: abs(x["cohens_d"]), reverse=True)
    top = sorted_stats[:top_n]

    # Build explanation lookup
    expl_map = {e["feature"]: e for e in explanations}

    fig, (ax_bar, ax_spec) = plt.subplots(1, 2, figsize=(18, max(6, top_n * 0.45)),
                                           gridspec_kw={"width_ratios": [1, 1.2]})

    y_pos = np.arange(len(top))
    [s["feature"] for s in top]
    d_values = [s["cohens_d"] for s in top]
    colors = [LABEL_COLORS[1] if d > 0 else LABEL_COLORS[0] for d in d_values]

    # Left: Cohen's d bars
    ax_bar.barh(y_pos, d_values, color=colors, edgecolor="white",
                       linewidth=0.5, alpha=0.85)
    ax_bar.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax_bar.set_yticks(y_pos)
    ylabels = []
    for s in top:
        fi = s["feature"]
        cluster = expl_map[fi].get("cluster", "?")
        sig = ""
        if s["label_p_value"] < 0.001:
            sig = " ***"
        elif s["label_p_value"] < 0.01:
            sig = " **"
        elif s["label_p_value"] < 0.05:
            sig = " *"
        ylabels.append(f"F{fi} (C{cluster}){sig}")
    ax_bar.set_yticklabels(ylabels, fontsize=9)
    ax_bar.set_xlabel("Cohen's d  (←Normal | Abnormal→)", fontsize=11)
    ax_bar.set_title("Feature Selectivity for Normal vs Abnormal",
                     fontsize=12, fontweight="bold")
    ax_bar.invert_yaxis()
    ax_bar.grid(axis="x", alpha=0.2)

    # Annotate bars with firing rates
    for i, s in enumerate(top):
        fr_n = s["fire_rate_normal"] * 100
        fr_a = s["fire_rate_abnormal"] * 100
        side = "left" if s["cohens_d"] > 0 else "right"
        offset = 0.02 if s["cohens_d"] > 0 else -0.02
        ax_bar.text(s["cohens_d"] + offset, i,
                    f"FR: {fr_n:.1f}%→{fr_a:.1f}%",
                    va="center", ha=side, fontsize=7.5, alpha=0.7)

    # Right: spectral profile per feature
    band_names = list(CLINICAL_BANDS.keys())
    n_bands = len(band_names)
    cell_w = 1.0 / n_bands

    ax_spec.set_xlim(0, 1)
    ax_spec.set_ylim(-0.5, len(top) - 0.5)
    ax_spec.invert_yaxis()
    ax_spec.set_xticks([cell_w * (j + 0.5) for j in range(n_bands)])
    ax_spec.set_xticklabels(band_names, fontsize=9, rotation=30, ha="right")
    ax_spec.set_yticks([])

    # Find max for colour normalisation
    all_effects = []
    for s in top:
        e = expl_map[s["feature"]]["band_effects"]
        all_effects.extend(e.values())
    vmax = max(abs(v) for v in all_effects) if all_effects else 1.0

    for i, s in enumerate(top):
        fi = s["feature"]
        e = expl_map[fi]["band_effects"]
        morph_matches = match_morphology(e)
        best_morph, best_score = morph_matches[0]

        for j, band in enumerate(band_names):
            val = e[band]
            # Color: red=boost, blue=suppress, white=neutral
            if val > 0:
                intensity = min(val / vmax, 1.0)
                color = (1.0, 1.0 - intensity * 0.7, 1.0 - intensity * 0.7)
            else:
                intensity = min(abs(val) / vmax, 1.0)
                color = (1.0 - intensity * 0.7, 1.0 - intensity * 0.7, 1.0)

            rect = plt.Rectangle((j * cell_w, i - 0.4), cell_w, 0.8,
                                  facecolor=color, edgecolor="gray",
                                  linewidth=0.5)
            ax_spec.add_patch(rect)
            ax_spec.text((j + 0.5) * cell_w, i, f"{val:+.2f}",
                         ha="center", va="center", fontsize=7,
                         color="black" if abs(val) / vmax < 0.6 else "white")

        # Add morphology match label
        ax_spec.text(1.02, i, f"≈ {best_morph}\n(cos={best_score:.2f})",
                     ha="left", va="center", fontsize=7.5,
                     style="italic", color="#555")

    ax_spec.set_title("Band Effects (Δ log-amp) + Best Morphology Match",
                      fontsize=12, fontweight="bold")

    fig.suptitle(f"Top {top_n} Features by Label Selectivity",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 2: Cluster × Label × Clinical Profile
# ═════════════════════════════════════════════════════════════════════════════

def plot_cluster_clinical_profile(label_stats, explanations, save_path):
    """
    For each cluster: mean Cohen's d, spectral centroid, morphology match.
    """
    expl_map = {e["feature"]: e for e in explanations}
    stat_map = {s["feature"]: s for s in label_stats}

    # Group by cluster
    clusters = defaultdict(list)
    for e in explanations:
        c = e.get("cluster", -1)
        fi = e["feature"]
        clusters[c].append(fi)

    n_clusters = len(clusters)
    cluster_ids = sorted(clusters.keys())
    band_names = list(CLINICAL_BANDS.keys())

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Top-left: Cohen's d per cluster ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    mean_ds = []
    cluster_labels = []
    for c in cluster_ids:
        members = clusters[c]
        ds = [stat_map[fi]["cohens_d"] for fi in members]
        mean_ds.append(np.mean(ds))
        e = expl_map[members[0]]  # representative
        cluster_labels.append(f"C{c} ({len(members)} feats)")

    colors = [LABEL_COLORS[1] if d > 0 else LABEL_COLORS[0] for d in mean_ds]
    ax1.barh(range(n_clusters), mean_ds, color=colors, edgecolor="white", alpha=0.85)
    ax1.set_yticks(range(n_clusters))
    ax1.set_yticklabels(cluster_labels, fontsize=10)
    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Mean Cohen's d  (←Normal | Abnormal→)", fontsize=11)
    ax1.set_title("Cluster Selectivity", fontsize=12, fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.2)

    for i, d in enumerate(mean_ds):
        ax1.text(d + 0.01 * np.sign(d), i, f"{d:+.3f}", va="center",
                 ha="left" if d >= 0 else "right", fontsize=9)

    # ── Top-right: Cluster centroid band effects heatmap ────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    centroids = np.zeros((n_clusters, len(band_names)))
    for i, c in enumerate(cluster_ids):
        members = clusters[c]
        for j, band in enumerate(band_names):
            vals = [expl_map[fi]["band_effects"][band] for fi in members]
            centroids[i, j] = np.mean(vals)

    vmax = np.abs(centroids).max()
    im = ax2.imshow(centroids, cmap="RdBu_r", aspect="auto",
                    vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax2.set_xticks(range(len(band_names)))
    ax2.set_xticklabels(band_names, fontsize=10, rotation=30, ha="right")
    ax2.set_yticks(range(n_clusters))
    ax2.set_yticklabels([f"C{c}" for c in cluster_ids], fontsize=10)
    ax2.set_title("Cluster Spectral Centroids\n(red=↑ blue=↓)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax2, shrink=0.7, label="Δ log-amplitude")

    for i in range(n_clusters):
        for j in range(len(band_names)):
            ax2.text(j, i, f"{centroids[i, j]:+.2f}", ha="center", va="center",
                     fontsize=8,
                     color="white" if abs(centroids[i, j]) > vmax * 0.5 else "black")

    # ── Bottom-left: Morphology match per cluster ───────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    morph_names = list(CLINICAL_MORPHOLOGIES.keys())
    match_matrix = np.zeros((n_clusters, len(morph_names)))
    for i, c in enumerate(cluster_ids):
        members = clusters[c]
        # Average band effects across cluster members
        avg_effects = {}
        for band in band_names:
            avg_effects[band] = np.mean([expl_map[fi]["band_effects"][band]
                                         for fi in members])
        matches = match_morphology(avg_effects)
        match_dict = dict(matches)
        for j, mn in enumerate(morph_names):
            match_matrix[i, j] = match_dict.get(mn, 0.0)

    im3 = ax3.imshow(match_matrix, cmap="RdYlGn", aspect="auto",
                     vmin=-1, vmax=1, interpolation="nearest")
    ax3.set_xticks(range(len(morph_names)))
    ax3.set_xticklabels(morph_names, fontsize=8, rotation=35, ha="right")
    ax3.set_yticks(range(n_clusters))
    ax3.set_yticklabels([f"C{c}" for c in cluster_ids], fontsize=10)
    ax3.set_title("Cluster × Morphology Match (cosine similarity)",
                  fontsize=12, fontweight="bold")
    fig.colorbar(im3, ax=ax3, shrink=0.7, label="Cosine similarity")

    for i in range(n_clusters):
        for j in range(len(morph_names)):
            ax3.text(j, i, f"{match_matrix[i, j]:.2f}", ha="center", va="center",
                     fontsize=7,
                     color="white" if abs(match_matrix[i, j]) > 0.5 else "black")

    # ── Bottom-right: Summary text ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    lines = ["Cluster → Clinical Interpretation\n"]
    for i, c in enumerate(cluster_ids):
        members = clusters[c]
        d = mean_ds[i]
        # Find best morphology
        avg_effects = {}
        for band in band_names:
            avg_effects[band] = np.mean([expl_map[fi]["band_effects"][band]
                                         for fi in members])
        best_morph, best_score = match_morphology(avg_effects)[0]
        selectivity = "Abnormal" if d > 0.02 else "Normal" if d < -0.02 else "Non-selective"
        lines.append(f"C{c} ({len(members)} feats):")
        lines.append(f"  Selective for: {selectivity} (d={d:+.3f})")
        lines.append(f"  Best morphology: {best_morph} (cos={best_score:.2f})")
        lines.append("")

    ax4.text(0.05, 0.95, "\n".join(lines), transform=ax4.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Cluster Clinical Profiles — Spectral Archetypes × EEG Morphologies",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 3: Clinical Mapping — top abnormal-selective features with spectra
# ═════════════════════════════════════════════════════════════════════════════

def plot_clinical_mapping(label_stats, explanations, amplitudes, spectral,
                          save_path, top_n=6):
    """
    For the top N most abnormal-selective features: show their differential
    spectra with clinical band shading and morphology annotation.
    """
    # Sort by Cohen's d descending (most abnormal-selective)
    sorted_stats = sorted(label_stats, key=lambda x: x["cohens_d"], reverse=True)
    top_abnormal = [s for s in sorted_stats if s["cohens_d"] > 0][:top_n]

    if not top_abnormal:
        print("  ⚠ No abnormal-selective features found!")
        return

    expl_map = {e["feature"]: e for e in explanations}
    freqs = spectral.freqs
    amp_diff = amplitudes.numpy()
    list(CLINICAL_BANDS.keys())

    fig, axes = plt.subplots(len(top_abnormal), 1,
                              figsize=(14, 3 * len(top_abnormal)),
                              sharex=True)
    if len(top_abnormal) == 1:
        axes = [axes]

    for row, s in enumerate(top_abnormal):
        fi = s["feature"]
        ax = axes[row]
        cluster = expl_map[fi].get("cluster", "?")
        d = s["cohens_d"]
        p = s["label_p_value"]

        # Plot differential spectrum
        pos = np.maximum(amp_diff[fi], 0)
        neg = np.minimum(amp_diff[fi], 0)
        color = LABEL_COLORS[1]
        ax.fill_between(freqs, 0, pos, alpha=0.3, color=color)
        ax.fill_between(freqs, 0, neg, alpha=0.2, color="#1565C0")
        ax.plot(freqs, amp_diff[fi], linewidth=2, color=color)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        # Shade clinical bands
        for band_name, (lo, hi) in CLINICAL_BANDS.items():
            ax.axvspan(lo, hi, alpha=0.06, color=BAND_COLORS[band_name])

        # Peak annotations
        peak_idx = amp_diff[fi].argmax()
        if amp_diff[fi, peak_idx] > 0:
            ax.annotate(f"↑ {freqs[peak_idx]:.0f} Hz",
                        (freqs[peak_idx], amp_diff[fi, peak_idx]),
                        textcoords="offset points", xytext=(10, 8),
                        fontsize=10, fontweight="bold", color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

        trough_idx = amp_diff[fi].argmin()
        if amp_diff[fi, trough_idx] < 0:
            ax.annotate(f"↓ {freqs[trough_idx]:.0f} Hz",
                        (freqs[trough_idx], amp_diff[fi, trough_idx]),
                        textcoords="offset points", xytext=(10, -12),
                        fontsize=10, fontweight="bold", color="#1565C0",
                        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5))

        # Morphology match
        e = expl_map[fi]["band_effects"]
        best_morph, best_score = match_morphology(e)[0]

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.set_title(f"Feature {fi} (Cluster {cluster})  ·  d={d:+.3f} {sig}  "
                     f"·  ≈ {best_morph} (cos={best_score:.2f})",
                     fontsize=11, fontweight="bold", loc="left", color=color)
        ax.set_ylabel("Δ Amplitude", fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.set_xlim(freqs[0], freqs[-1])

        if row == len(top_abnormal) - 1:
            ax.set_xlabel("Frequency (Hz)", fontsize=11)

    fig.suptitle("Most Abnormal-Selective Features — Spectral Signatures × Clinical Morphology",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global EMBED, TARGET_LAYER, SAE_PATH, DATA_PATH, XAE_PATH, EXPL_PATH, OUT_DIR, FS, PATCH_SIZE
    t0 = time.time()

    # ── CLI args ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Clinical morphology analysis")
    parser.add_argument("--encoder", default="sleepfm",
                        choices=["sleepfm", "reve"],
                        help="Encoder backend (default: sleepfm)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Transformer layer index (default: last layer)")
    parser.add_argument("--weights-path", default=None,
                        help="Path to finetuned checkpoint (REVE .ckpt or SleepFM .pt)")
    parser.add_argument("--tag", default=None,
                        help="Run tag (e.g. 'qjbe08') — scopes SAE, XAE, explanations, "
                             "and output paths to results/**/{encoder}_{tag}/")
    args = parser.parse_args()

    # ── Resolve encoder-scoped paths ─────────────────────────────────────
    run_name = f"{args.encoder}_{args.tag}" if args.tag else args.encoder
    model = load_model(args.encoder, weights_path=args.weights_path)
    EMBED = model.embed_dim
    FS = _ENCODER_FS[args.encoder]
    PATCH_SIZE = FS  # 1-second patches at native sample rate
    n_layers = len(model.get_hookable_layers())
    TARGET_LAYER = args.layer if args.layer is not None else n_layers - 1
    DATA_PATH = _ENCODER_DATA[args.encoder]
    SAE_PATH = (ROOT / "results" / "features" / run_name
                / f"sae_{args.encoder}_exp{EXPANSION}_k{K}_layer{TARGET_LAYER}.pt")
    if args.encoder == "sleepfm" and not args.tag:
        XAE_PATH  = ROOT / "results" / "xae" / "xae_checkpoint.pt"
        EXPL_PATH = ROOT / "results" / "xae" / "explanations" / "feature_explanations.json"
        OUT_DIR   = ROOT / "results" / "xae" / "morphology"
    else:
        XAE_PATH  = ROOT / "results" / "xae" / run_name / "xae_checkpoint.pt"
        EXPL_PATH = ROOT / "results" / "xae" / run_name / "explanations" / "feature_explanations.json"
        OUT_DIR   = ROOT / "results" / "xae" / run_name / "morphology"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  encoder={args.encoder}  embed_dim={EMBED}  "
          f"layer={TARGET_LAYER}  SAE={SAE_PATH.name}")

    print("=" * 70)
    print("  Clinical Morphology Analysis — SAE Features × EEG Patterns")
    print("=" * 70)

    # ── Load models ─────────────────────────────────────────────────────
    print("\n[1/5] Loading models...")
    sae, act_mean, act_std = load_sae()
    print(f"  {args.encoder} + SAE (k={K}, dict_size={sae.dict_size}) loaded")

    # Load XAE for spectral decoding
    xae_trainer = XAETrainer(embed_dim=EMBED, fs=FS, n_fft=PATCH_SIZE)
    xae_trainer.load(str(XAE_PATH))
    spectral = xae_trainer.spectral
    print(f"  XAE loaded ({spectral.n_bins} freq bins)")

    # Load feature explanations
    with open(EXPL_PATH) as f:
        explanations = json.load(f)
    print(f"  Loaded {len(explanations)} feature explanations")

    # ── Load data ───────────────────────────────────────────────────────
    print("\n[2/5] Loading data...")
    gen = get_dataloaders(train_path=str(DATA_PATH),
                          transformer=StandardizeLabel())
    fold, train_loader, val_loader, test_loader = next(gen)
    print(f"  Using fold {fold}, val set for analysis")

    # ── Encode windows ──────────────────────────────────────────────────
    print("\n[3/5] Encoding windows through SetTransformer → SAE...")
    codes, labels = encode_windows(model, sae, act_mean, act_std, val_loader)
    n_normal = (labels == 0).sum().item()
    n_abnormal = (labels == 1).sum().item()
    print(f"  Encoded {len(labels)} windows: {n_normal} Normal, {n_abnormal} Abnormal")
    print(f"  Codes shape: {codes.shape}  (windows × tokens × features)")

    # ── Decode SAE features through XAE ─────────────────────────────────
    print("\n[3.5/5] Decoding SAE feature directions through XAE...")
    # Re-use the contrastive decoding from explain_features
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "tools"))
    from explain_features import decode_all_features
    amplitudes, cos_phases, sin_phases, baseline_amp, _ = decode_all_features(
        sae, xae_trainer, act_mean, act_std
    )
    print(f"  Decoded {sae.dict_size} feature spectral signatures")

    # ── Compute feature × label statistics ──────────────────────────────
    print("\n[4/5] Computing feature × label statistics...")
    label_stats = compute_feature_label_stats(codes, labels)

    # Print summary
    sig_count = sum(1 for s in label_stats if s["label_p_value"] < 0.05)
    abn_selective = sum(1 for s in label_stats
                        if s["cohens_d"] > 0.05 and s["label_p_value"] < 0.05)
    norm_selective = sum(1 for s in label_stats
                         if s["cohens_d"] < -0.05 and s["label_p_value"] < 0.05)
    print(f"  Significant features (p<0.05): {sig_count}/{len(label_stats)}")
    print(f"  Abnormal-selective (d>0.05):    {abn_selective}")
    print(f"  Normal-selective (d<-0.05):     {norm_selective}")

    # Top 5 most abnormal-selective
    top_abn = sorted(label_stats, key=lambda x: x["cohens_d"], reverse=True)[:5]
    print("\n  Top 5 Abnormal-selective features:")
    for s in top_abn:
        fi = s["feature"]
        e = {ex["feature"]: ex for ex in explanations}[fi]
        cluster = e.get("cluster", "?")
        best_morph, best_score = match_morphology(e["band_effects"])[0]
        print(f"    F{fi:3d} (C{cluster}) d={s['cohens_d']:+.3f}  "
              f"FR: {s['fire_rate_normal']*100:.1f}%→{s['fire_rate_abnormal']*100:.1f}%  "
              f"≈ {best_morph} (cos={best_score:.2f})")

    # Top 5 most normal-selective
    top_norm = sorted(label_stats, key=lambda x: x["cohens_d"])[:5]
    print("\n  Top 5 Normal-selective features:")
    for s in top_norm:
        fi = s["feature"]
        e = {ex["feature"]: ex for ex in explanations}[fi]
        cluster = e.get("cluster", "?")
        best_morph, best_score = match_morphology(e["band_effects"])[0]
        print(f"    F{fi:3d} (C{cluster}) d={s['cohens_d']:+.3f}  "
              f"FR: {s['fire_rate_normal']*100:.1f}%→{s['fire_rate_abnormal']*100:.1f}%  "
              f"≈ {best_morph} (cos={best_score:.2f})")

    # ── Generate plots ──────────────────────────────────────────────────
    print("\n[5/5] Generating visualisations...")

    plot_feature_enrichment(
        label_stats, explanations,
        OUT_DIR / "01_feature_label_enrichment.png",
        top_n=20,
    )

    plot_cluster_clinical_profile(
        label_stats, explanations,
        OUT_DIR / "02_cluster_clinical_profile.png",
    )

    plot_clinical_mapping(
        label_stats, explanations, amplitudes, spectral,
        OUT_DIR / "03_clinical_mapping.png",
        top_n=6,
    )

    # ── Save JSON report ────────────────────────────────────────────────
    report = {
        "summary": {
            "n_windows": len(labels),
            "n_normal": n_normal,
            "n_abnormal": n_abnormal,
            "n_features": sae.dict_size,
            "n_significant_p05": sig_count,
            "n_abnormal_selective": abn_selective,
            "n_normal_selective": norm_selective,
        },
        "feature_label_stats": label_stats,
        "cluster_profiles": [],
    }

    expl_map = {e["feature"]: e for e in explanations}
    stat_map = {s["feature"]: s for s in label_stats}
    clusters = defaultdict(list)
    for e in explanations:
        clusters[e.get("cluster", -1)].append(e["feature"])

    band_names = list(CLINICAL_BANDS.keys())
    for c in sorted(clusters.keys()):
        members = clusters[c]
        ds = [stat_map[fi]["cohens_d"] for fi in members]
        avg_effects = {band: np.mean([expl_map[fi]["band_effects"][band]
                                      for fi in members])
                       for band in band_names}
        morph_matches = match_morphology(avg_effects)
        report["cluster_profiles"].append({
            "cluster": c,
            "n_members": len(members),
            "mean_cohens_d": float(np.mean(ds)),
            "avg_band_effects": {k: round(v, 4) for k, v in avg_effects.items()},
            "morphology_matches": [{"morphology": m, "cosine_similarity": round(s, 4)}
                                   for m, s in morph_matches[:3]],
        })

    report_path = OUT_DIR / "morphology_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Saved {report_path}")

    elapsed = time.time() - t0
    print(f"\n✅ Morphology analysis complete in {elapsed:.1f} seconds.")
    print(f"   Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
