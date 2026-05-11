"""
Explain Features — Spectral signatures of SAE features via XAE
================================================================

Takes a trained SAE and a trained XAE and answers the question:
"What does each SAE feature *mean* in terms of EEG frequency content?"

For each SAE feature direction, the XAE decoder produces a spectral
signature (amplitude + phase across frequency bins), giving us a
human-readable, clinically-grounded interpretation.

Outputs (in results/xae/explanations/):
  01_feature_spectra.png       — amplitude spectrum per SAE feature
  02_band_power_heatmap.png    — clinical band power per feature
  03_feature_spectral_cards.png— compact cards with top-activating EEG +
                                  spectral signature side by side
  04_phase_coherence.png       — phase patterns per feature
  feature_explanations.json    — machine-readable feature descriptions

Usage:  uv run tools/explain_features.py

Prerequisites:
  - Trained SAE checkpoint  (results/features/sae_exp1_k8_layer2.pt)
  - Trained XAE checkpoint  (results/xae/xae_checkpoint.pt)
"""

import sys
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── project imports ─────────────────────────────────────────────────────────
from sae4eeg.xae import (XAETrainer, CLINICAL_BANDS)
from sae4eeg.sae import SparseAutoencoder
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
XAE_PATH     = ROOT / "results" / "xae" / "xae_checkpoint.pt"  # overridden at runtime
OUT_DIR      = ROOT / "results" / "xae" / "explanations"       # overridden at runtime

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EMBED        = 128    # overridden at runtime from encoder.embed_dim
EXPANSION    = 1.0
K            = 8
PATCH_SIZE   = 128
FS           = 128
TARGET_LAYER = 2      # overridden at runtime: last encoder layer
S_TOKENS     = 60

# SAE_PATH and SAE_PATH are resolved at runtime based on encoder name + layer
SAE_PATH     = ROOT / "results" / "features" / "sleepfm" / "sae_sleepfm_exp1_k8_layer2.pt"

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

# Colour scheme for clinical bands
BAND_COLORS = {
    "delta":     "#1f77b4",  # blue
    "theta":     "#2ca02c",  # green
    "alpha":     "#ff7f0e",  # orange
    "low-beta":  "#d62728",  # red
    "high-beta": "#9467bd",  # purple
    "gamma":     "#8c564b",  # brown
}

FEATURE_CMAP = plt.cm.Set1

# Clustering config
N_CLUSTERS = 8          # number of spectral archetypes


# ═════════════════════════════════════════════════════════════════════════════
# Clustering: group 128 features into spectral archetypes
# ═════════════════════════════════════════════════════════════════════════════

def cluster_features(amplitudes, spectral, n_clusters=N_CLUSTERS):
    """
    Cluster SAE features by their clinical-band effect profiles using k-means.

    Returns
    -------
    cluster_labels : ndarray (n_features,)  — cluster id per feature
    representatives : list[int]              — feature index closest to each centroid
    centroids      : ndarray (n_clusters, 6) — centroid band-effect vectors
    band_powers    : ndarray (n_features, 6) — per-feature band-effect matrix
    """
    band_names = list(CLINICAL_BANDS.keys())
    n_features = amplitudes.shape[0]

    # Build band-effect matrix (n_features × n_bands)
    band_powers = np.zeros((n_features, len(band_names)))
    for j, band_name in enumerate(band_names):
        mask = spectral.get_band_mask(band_name)
        if mask.sum() > 0:
            for i in range(n_features):
                band_powers[i, j] = amplitudes[i][torch.tensor(mask)].mean().item()

    # Standardise before clustering (bands have different scales)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(band_powers)

    # k-means
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)

    # Find representative feature per cluster: closest to centroid AND
    # with the strongest overall effect (break ties toward more interpretable)
    representatives = []
    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        dists = np.linalg.norm(X_scaled[members] - km.cluster_centers_[c], axis=1)
        # Weight: prefer features close to centroid AND with strong effects
        effect_strength = np.abs(band_powers[members]).max(axis=1)
        # Score: low dist + high effect  (normalise both to [0,1])
        d_norm = dists / (dists.max() + 1e-8)
        e_norm = effect_strength / (effect_strength.max() + 1e-8)
        score = -d_norm + 0.3 * e_norm   # closeness matters more
        best = members[score.argmax()]
        representatives.append(int(best))

    # Centroids in original (un-scaled) space
    centroids = scaler.inverse_transform(km.cluster_centers_)

    return labels, representatives, centroids, band_powers


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def remove_module_from_state_dict(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()}


def load_model(encoder_name: str = "sleepfm", weights_path=None):
    """Load encoder backend by name."""
    if encoder_name == "sleepfm":
        kwargs = {"weights_path": weights_path or MODEL_PATH}
    else:
        kwargs = {"weights_path": weights_path} if weights_path else {}
    encoder = load_encoder(encoder_name, **kwargs)
    encoder.to(DEVICE).eval()
    return encoder


def load_data():
    gen = get_dataloaders(train_path=str(DATA_PATH),
                          transformer=StandardizeLabel())
    fold, train_loader, val_loader, test_loader = next(gen)
    return train_loader, val_loader, test_loader


def load_sae():
    """Load the trained SAE."""
    ckpt = torch.load(SAE_PATH, weights_only=False)
    sae = SparseAutoencoder(EMBED, expansion=EXPANSION, mode="topk", k=K)
    sae.load_state_dict(ckpt["sae_state_dict"])
    act_mean = ckpt["act_mean"]
    act_std = ckpt["act_std"]
    print(f"[✓] Loaded SAE from {SAE_PATH}")
    return sae, act_mean, act_std


def load_xae():
    """Load the trained XAE."""
    trainer = XAETrainer(embed_dim=EMBED, fs=FS, n_fft=PATCH_SIZE)
    trainer.load(str(XAE_PATH))
    print(f"[✓] Loaded XAE from {XAE_PATH}")
    return trainer


def feature_color(i, n_features):
    return FEATURE_CMAP(i / max(n_features - 1, 1))


# ═════════════════════════════════════════════════════════════════════════════
# Core: Extract spectral signature for each SAE feature direction
# ═════════════════════════════════════════════════════════════════════════════

def get_sae_feature_directions(sae: SparseAutoencoder) -> torch.Tensor:
    """
    Extract the decoder weight columns — each column IS the feature direction
    in embedding space.

    Returns:
        directions: (dict_size, embed_dim) — one row per feature
    """
    # SAE decoder: Linear(dict_size → embed_dim), so weight is (embed_dim, dict_size)
    # Each column of the weight matrix is a feature direction
    return sae.decoder.weight.T.detach().clone()  # (dict_size, embed_dim)


def decode_all_features(
    sae: SparseAutoencoder,
    xae_trainer: XAETrainer,
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
):
    """
    Decode every SAE feature direction through the XAE to get spectral
    signatures (and optionally temporal waveforms) using a *contrastive*
    approach.

    A SAE feature direction is meaningful as a *perturbation* — it tells us
    what changes when that feature fires.  So we decode two points:
      baseline  = mean activation  (feature OFF)
      activated = mean + α·direction  (feature ON at a typical magnitude)
    and report the *difference* in spectral output.

    To estimate a realistic activation magnitude α for each feature, we use
    the encoder bias plus a unit step, or simply a fixed scale based on the
    data's standard deviation (since directions are unit-norm in normalised
    space, α ≈ 1–3 std is a typical activation).

    Returns:
        amplitudes:     (dict_size, n_bins)  — differential amplitude (log-scale)
        cos_phases:     (dict_size, n_bins)
        sin_phases:     (dict_size, n_bins)
        baseline_amp:   (n_bins,) — baseline spectrum amplitude
        temporal_diffs: dict[str, Tensor(dict_size, patch_size)]
                        — per-band differential characteristic waveforms
                        (feature ON − baseline, zero-phase reconstruction)
    """
    directions = get_sae_feature_directions(sae)  # (dict_size, embed_dim) normalised-space
    directions.shape[0]

    # ── Build realistic activation points in raw (un-normalised) space ──
    # Baseline: the mean activation (represents "no particular feature firing")
    baseline_raw = act_mean.unsqueeze(0)  # (1, embed_dim)

    # Each direction is unit-norm in normalised space.  To get raw space:
    #   raw_direction = normalised_direction * act_std  (don't add mean — it's a direction, not a point)
    # Scale factor α: how far along the direction a typical activation sits.
    # Using α=2 (≈ 2 std in normalised space) gives a clearly-active-but-not-extreme point.
    alpha = 2.0
    raw_directions = directions * act_std  # (dict_size, embed_dim)
    activated_raw = baseline_raw + alpha * raw_directions  # (dict_size, embed_dim)

    # ── Decode baseline and activated points (spectral) ────────────────
    amp_base, cos_base, sin_base = xae_trainer.decode_direction(
        baseline_raw, denormalise=True
    )
    # amp_base is (1, n_bins) — squeeze to (n_bins,)
    amp_base = amp_base.squeeze(0)
    cos_base = cos_base.squeeze(0)
    sin_base = sin_base.squeeze(0)

    amp_act, cos_act, sin_act = xae_trainer.decode_direction(
        activated_raw, denormalise=True
    )
    # amp_act is (dict_size, n_bins)

    # ── Differential spectrum: what the feature ADDS ────────────────────
    # For amplitude: take the difference in log-amplitude.  Positive values
    # mean the feature boosts that frequency, negative means it suppresses.
    amp_diff = amp_act - amp_base.unsqueeze(0)   # (dict_size, n_bins)

    # For phase: report the activated phase directly (the difference is less
    # meaningful since phase is circular)
    cos_phases = cos_act
    sin_phases = sin_act

    # ── Temporal: phase-agnostic contrastive waveforms ────────────────
    # Use zero-phase iFFT from predicted spectral amplitudes — this gives
    # clean, deterministic "characteristic waveforms" that show what each
    # feature boosts without depending on unpredictable phase.
    _ps = xae_trainer.spectral.n_fft
    _fs = xae_trainer.fs
    base_temporal = xae_trainer.spectral_to_waveforms(amp_base.unsqueeze(0), patch_size=_ps, fs=_fs)
    act_temporal  = xae_trainer.spectral_to_waveforms(amp_act, patch_size=_ps, fs=_fs)
    temporal_diffs = {}
    for band_name in base_temporal:
        base_wave = base_temporal[band_name].squeeze(0)   # (T,)
        act_wave  = act_temporal[band_name]                # (dict_size, T)
        temporal_diffs[band_name] = act_wave - base_wave.unsqueeze(0)

    return amp_diff, cos_phases, sin_phases, amp_base, temporal_diffs


# ═════════════════════════════════════════════════════════════════════════════
# Plot 1: Cluster Overview — archetype band profiles + membership
# ═════════════════════════════════════════════════════════════════════════════

def _archetype_label(centroid, band_names):
    """Short human-readable label for a cluster centroid."""
    idx_max = np.abs(centroid).argmax()
    direction = "↑" if centroid[idx_max] > 0 else "↓"
    return f"{direction} {band_names[idx_max]}"


def plot_cluster_overview(centroids, labels, representatives, band_powers,
                          baseline_amp, spectral, save_path):
    """
    3-panel overview:
      Left:   Heatmap of cluster centroids (bands × clusters)
      Centre: Cluster membership counts with representative label
      Right:  Baseline spectrum for reference
    """
    n_clusters = centroids.shape[0]
    band_names = list(CLINICAL_BANDS.keys())

    # Sort clusters by dominant band index then sign → visually coherent order
    order = sorted(range(n_clusters),
                   key=lambda c: (np.abs(centroids[c]).argmax(),
                                  -centroids[c, np.abs(centroids[c]).argmax()]))
    centroids = centroids[order]
    old_to_new = {old: new for new, old in enumerate(order)}
    labels_reordered = np.array([old_to_new[lbl] for lbl in labels])
    representatives = [representatives[o] for o in order]

    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 1.5, 2], wspace=0.35)

    # ── Left: centroid heatmap ──────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0])
    vmax = np.abs(centroids).max()
    im = ax_heat.imshow(centroids, cmap="RdBu_r", aspect="auto",
                        interpolation="nearest", vmin=-vmax, vmax=vmax)
    ax_heat.set_xticks(range(len(band_names)))
    ax_heat.set_xticklabels(band_names, fontsize=10, rotation=30, ha="right")
    ylabels = []
    for c in range(n_clusters):
        lbl = _archetype_label(centroids[c], band_names)
        n_members = (labels_reordered == c).sum()
        ylabels.append(f"C{c}  {lbl}\n({n_members} feats, rep F{representatives[c]})")
    ax_heat.set_yticks(range(n_clusters))
    ax_heat.set_yticklabels(ylabels, fontsize=9)
    ax_heat.set_title("Cluster Centroids\n(Δ band power: red=↑ blue=↓)",
                      fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax_heat, shrink=0.6, label="Δ log-amplitude")

    for i in range(n_clusters):
        for j in range(len(band_names)):
            ax_heat.text(j, i, f"{centroids[i, j]:+.2f}", ha="center", va="center",
                         fontsize=8,
                         color="white" if abs(centroids[i, j]) > vmax * 0.5 else "black")

    # ── Centre: membership bar chart ────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])
    counts = [(labels_reordered == c).sum() for c in range(n_clusters)]
    colors = [plt.cm.tab10(c / n_clusters) for c in range(n_clusters)]
    ax_bar.barh(range(n_clusters), counts, color=colors,
                       edgecolor="white", linewidth=0.5)
    ax_bar.set_yticks(range(n_clusters))
    ax_bar.set_yticklabels([f"C{c}" for c in range(n_clusters)], fontsize=10)
    ax_bar.set_xlabel("Number of features", fontsize=11)
    ax_bar.set_title("Cluster Sizes", fontsize=12, fontweight="bold")
    ax_bar.invert_yaxis()
    for c, cnt in enumerate(counts):
        ax_bar.text(cnt + 0.5, c, str(cnt), va="center", fontsize=10, fontweight="bold")

    # ── Right: baseline spectrum ────────────────────────────────────────
    ax_base = fig.add_subplot(gs[2])
    freqs = spectral.freqs
    base_linear = spectral.amplitude_linear(baseline_amp).numpy()
    ax_base.fill_between(freqs, 0, base_linear, alpha=0.3, color="gray")
    ax_base.plot(freqs, base_linear, "k-", linewidth=1.5)
    for band_name, (lo, hi) in CLINICAL_BANDS.items():
        ax_base.axvspan(lo, hi, alpha=0.08, color=BAND_COLORS[band_name])
        mid = (lo + hi) / 2
        ax_base.text(mid, ax_base.get_ylim()[1] * 0.85 if ax_base.get_ylim()[1] > 0 else 0.1,
                     band_name, ha="center", fontsize=7,
                     color=BAND_COLORS[band_name], fontweight="bold", alpha=0.8)
    ax_base.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_base.set_ylabel("Amplitude (linear)", fontsize=11)
    ax_base.set_title("Baseline Spectrum\n(mean activation → XAE)", fontsize=12, fontweight="bold")
    ax_base.set_xlim(freqs[0], freqs[-1])
    ax_base.grid(True, alpha=0.2)

    fig.suptitle(f"SAE Feature Clustering — {n_clusters} Spectral Archetypes from 128 Features",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")

    return labels_reordered, representatives


# ═════════════════════════════════════════════════════════════════════════════
# Plot 2: Representative Spectral Cards (one per cluster)
# ═════════════════════════════════════════════════════════════════════════════

def plot_representative_cards(amplitudes, cos_phases, sin_phases, spectral,
                              representatives, labels, save_path):
    """
    One row per cluster representative: differential amplitude spectrum +
    phase polar plot.  Much more readable than 128 rows.
    """
    n_clusters = len(representatives)
    freqs = spectral.freqs
    amp_diff = amplitudes.numpy()
    list(CLINICAL_BANDS.keys())

    fig, axes = plt.subplots(n_clusters, 2, figsize=(16, 3 * n_clusters),
                              gridspec_kw={"width_ratios": [3, 1]})
    if n_clusters == 1:
        axes = axes[None, :]

    for row, feat_idx in enumerate(representatives):
        n_members = (labels == row).sum()
        color = plt.cm.tab10(row / n_clusters)

        # --- Differential amplitude spectrum ---
        ax = axes[row, 0]
        pos = np.maximum(amp_diff[feat_idx], 0)
        neg = np.minimum(amp_diff[feat_idx], 0)
        ax.fill_between(freqs, 0, pos, alpha=0.3, color=color, label="boost")
        ax.fill_between(freqs, 0, neg, alpha=0.2, color="steelblue", label="suppress")
        ax.plot(freqs, amp_diff[feat_idx], linewidth=2, color=color)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        # Shade clinical bands
        for band_name, (lo, hi) in CLINICAL_BANDS.items():
            ax.axvspan(lo, hi, alpha=0.05, color=BAND_COLORS[band_name])

        # Peak boost
        peak_idx = amp_diff[feat_idx].argmax()
        peak_freq = freqs[peak_idx]
        peak_val = amp_diff[feat_idx, peak_idx]
        if peak_val > 0:
            ax.annotate(f"↑ {peak_freq:.1f} Hz",
                        (peak_freq, peak_val),
                        textcoords="offset points", xytext=(10, 8),
                        fontsize=10, fontweight="bold", color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

        # Peak suppress
        trough_idx = amp_diff[feat_idx].argmin()
        trough_freq = freqs[trough_idx]
        trough_val = amp_diff[feat_idx, trough_idx]
        if trough_val < 0:
            ax.annotate(f"↓ {trough_freq:.1f} Hz",
                        (trough_freq, trough_val),
                        textcoords="offset points", xytext=(10, -12),
                        fontsize=10, fontweight="bold", color="steelblue",
                        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5))

        # Most affected band
        band_effects = {}
        for band_name in CLINICAL_BANDS:
            mask = spectral.get_band_mask(band_name)
            if mask.sum() > 0:
                band_effects[band_name] = amp_diff[feat_idx, mask].mean()
        strongest = max(band_effects, key=lambda k: abs(band_effects[k]))
        direction = "↑" if band_effects[strongest] > 0 else "↓"

        ax.set_xlim(freqs[0], freqs[-1])
        ax.set_ylabel("Δ Amplitude", fontsize=9)
        if row == n_clusters - 1:
            ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_title(f"Cluster {row}  ·  Representative F{feat_idx}  "
                     f"({n_members} features)  —  {direction} {strongest}",
                     fontsize=11, fontweight="bold", color=color, loc="left")
        ax.grid(True, alpha=0.15)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)

        # --- Phase polar plot ---
        ax_phase = axes[row, 1]
        ax_phase.set_aspect("equal")

        top5_bins = np.abs(amp_diff[feat_idx]).argsort()[-5:][::-1]
        abs_max = np.abs(amp_diff[feat_idx]).max()
        for rank, bin_idx in enumerate(top5_bins):
            cos_val = cos_phases[feat_idx, bin_idx].item()
            sin_val = sin_phases[feat_idx, bin_idx].item()
            amp_val = abs(amp_diff[feat_idx, bin_idx])
            freq_val = freqs[bin_idx]

            arrow_len = 0.8 * (amp_val / (abs_max + 1e-8))
            arrow_color = color if amp_diff[feat_idx, bin_idx] > 0 else "steelblue"
            ax_phase.annotate("",
                              xy=(cos_val * arrow_len, sin_val * arrow_len),
                              xytext=(0, 0),
                              arrowprops=dict(
                                  arrowstyle="->", color=arrow_color,
                                  lw=2 - rank * 0.3,
                                  alpha=0.9 - rank * 0.15))
            ax_phase.text(cos_val * arrow_len * 1.15,
                          sin_val * arrow_len * 1.15,
                          f"{freq_val:.0f}",
                          fontsize=7, ha="center", va="center",
                          color=color, alpha=0.8)

        theta = np.linspace(0, 2 * np.pi, 100)
        ax_phase.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.5, alpha=0.3)
        ax_phase.axhline(0, color="k", linewidth=0.3, alpha=0.3)
        ax_phase.axvline(0, color="k", linewidth=0.3, alpha=0.3)
        ax_phase.set_xlim(-1.4, 1.4)
        ax_phase.set_ylim(-1.4, 1.4)
        ax_phase.set_title("Phase (top bins)", fontsize=9)
        ax_phase.tick_params(labelsize=7)

    fig.suptitle("Spectral Cards — One Representative per Cluster",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 3: Cluster Spectra Overlay (one trace per cluster, mean of members)
# ═════════════════════════════════════════════════════════════════════════════

def plot_cluster_spectra(amplitudes, spectral, labels, representatives,
                         save_path):
    """
    Overlay the mean differential spectrum of each cluster's members.
    Much cleaner than 128 individual traces.
    """
    n_clusters = len(representatives)
    freqs = spectral.freqs
    amp_diff = amplitudes.numpy()
    list(CLINICAL_BANDS.keys())

    fig, ax = plt.subplots(figsize=(14, 6))

    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        mean_spec = amp_diff[members].mean(axis=0)
        std_spec = amp_diff[members].std(axis=0) if len(members) > 1 else np.zeros_like(mean_spec)
        color = plt.cm.tab10(c / n_clusters)

        # Strongest band for label
        band_effects = {}
        for band_name in CLINICAL_BANDS:
            mask = spectral.get_band_mask(band_name)
            if mask.sum() > 0:
                band_effects[band_name] = mean_spec[mask].mean()
        strongest = max(band_effects, key=lambda k: abs(band_effects[k]))
        direction = "↑" if band_effects[strongest] > 0 else "↓"

        label = f"C{c}: {direction}{strongest} ({len(members)} feats)"
        ax.plot(freqs, mean_spec, linewidth=2.5, color=color, label=label, alpha=0.9)
        ax.fill_between(freqs, mean_spec - std_spec, mean_spec + std_spec,
                        alpha=0.1, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Shade clinical bands
    for band_name, (lo, hi) in CLINICAL_BANDS.items():
        ax.axvspan(lo, hi, alpha=0.05, color=BAND_COLORS[band_name])
        mid = (lo + hi) / 2
        ypos = ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.1
        ax.text(mid, ypos, band_name, ha="center", fontsize=8,
                color=BAND_COLORS[band_name], fontweight="bold", alpha=0.7)

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Δ Amplitude (log-scale)\n↑ boost  |  ↓ suppress", fontsize=11)
    ax.set_title("Cluster Mean Spectra — Differential (feature ON − baseline)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(freqs[0], freqs[-1])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 4: Phase Coherence (representatives only)
# ═════════════════════════════════════════════════════════════════════════════

def plot_phase_coherence(cos_phases, sin_phases, spectral, representatives,
                         labels, save_path):
    """
    Phase structure for cluster representatives only (instead of all 128).
    Top: phase angle heatmap (rows = cluster reps)
    Bottom: pairwise phase similarity between reps
    """
    n_clusters = len(representatives)
    freqs = spectral.freqs

    cos_rep = cos_phases[representatives]
    sin_rep = sin_phases[representatives]
    phases = torch.atan2(sin_rep, cos_rep).numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: phase angle heatmap
    ax = axes[0]
    im = ax.imshow(phases, aspect="auto", cmap="hsv",
                   extent=[freqs[0], freqs[-1], n_clusters - 0.5, -0.5],
                   vmin=-np.pi, vmax=np.pi, interpolation="nearest")
    ylabels = [f"C{c} (F{representatives[c]})" for c in range(n_clusters)]
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_title("Phase Angle — Cluster Representatives × Frequency",
                 fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Phase (rad)")

    # Bottom: pairwise phase similarity between representatives
    ax2 = axes[1]
    phase_vecs = torch.stack([cos_rep, sin_rep], dim=-1)
    sim_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cos_diff = (phase_vecs[i] * phase_vecs[j]).sum(dim=-1)
            sim_matrix[i, j] = cos_diff.mean().item()

    im2 = ax2.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1,
                     aspect="equal", interpolation="nearest")
    ax2.set_xticks(range(n_clusters))
    ax2.set_xticklabels(ylabels, fontsize=9, rotation=30, ha="right")
    ax2.set_yticks(range(n_clusters))
    ax2.set_yticklabels(ylabels, fontsize=9)
    ax2.set_title("Phase Similarity Between Cluster Representatives",
                  fontsize=13, fontweight="bold")
    fig.colorbar(im2, ax=ax2, shrink=0.7, label="Cosine similarity")

    for i in range(n_clusters):
        for j in range(n_clusters):
            ax2.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                     fontsize=9,
                     color="white" if abs(sim_matrix[i, j]) > 0.5 else "black")

    fig.suptitle("Phase Structure — Cluster Representatives", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 5: Temporal Waveforms — contrastive per-band waveforms for reps
# ═════════════════════════════════════════════════════════════════════════════

def plot_temporal_waveforms(temporal_diffs, representatives, labels,
                            save_path, fs=FS):
    """
    Show phase-agnostic contrastive band waveforms for cluster representatives.
    One row per cluster, one column per clinical band.

    The waveforms show what happens in each frequency band when a feature
    fires: the difference between "feature ON" and baseline, reconstructed
    via zero-phase iFFT from predicted spectral amplitudes.

    Because the SetTransformer discards phase information, we use zero-phase
    (all cosines) which produces symmetric "characteristic waveforms" that
    faithfully show frequency content and relative band power without
    predicting the unpredictable.

    The peak amplitude of each oscillation directly reflects how much
    the feature boosts that band.  The oscillation period reveals the
    dominant frequency within the band.
    """
    band_names = list(temporal_diffs.keys())
    n_clusters = len(representatives)
    n_bands = len(band_names)
    T = temporal_diffs[band_names[0]].shape[-1]
    # Centre the time axis at 0 so the symmetric peak is at t=0
    t_axis = (np.arange(T) - T // 2) / fs  # seconds, centred

    band_colors = {
        "delta": "#2196F3", "theta": "#4CAF50", "alpha": "#FF9800",
        "low-beta": "#F44336", "high-beta": "#9C27B0", "gamma": "#795548",
    }

    fig, axes = plt.subplots(n_clusters, n_bands,
                              figsize=(3 * n_bands, 2.5 * n_clusters),
                              squeeze=False)

    for row, feat_idx in enumerate(representatives):
        n_members = (labels == row).sum()

        for col, band_name in enumerate(band_names):
            ax = axes[row, col]
            wave = temporal_diffs[band_name][feat_idx].numpy()
            color = band_colors.get(band_name, "gray")

            # Fill positive/negative differently
            pos = np.maximum(wave, 0)
            neg = np.minimum(wave, 0)
            ax.fill_between(t_axis, 0, pos, alpha=0.3, color=color)
            ax.fill_between(t_axis, 0, neg, alpha=0.15, color="steelblue")
            ax.plot(t_axis, wave, linewidth=1.5, color=color)
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.5, linestyle="--")
            ax.axvline(0, color="black", linewidth=0.3, alpha=0.3, linestyle=":")

            # Peak amplitude annotation (always at centre for zero-phase)
            peak_idx = np.abs(wave).argmax()
            peak_t = t_axis[peak_idx]
            peak_v = wave[peak_idx]
            if abs(peak_v) > 0.01:
                direction = "↑" if peak_v > 0 else "↓"
                ax.annotate(f"{direction}{peak_v:.2f}",
                            (peak_t, peak_v),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7, color=color, fontweight="bold")

            if row == 0:
                ax.set_title(band_name, fontsize=10, fontweight="bold",
                             color=color)
            if row == n_clusters - 1:
                ax.set_xlabel("Time (s)", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"C{row} F{feat_idx}\n({n_members} feats)",
                              fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.15)

    fig.suptitle(
        "Characteristic Waveforms — Feature ON − Baseline (zero-phase)\n"
        "Peak amplitude = band power boost · Oscillation period = dominant frequency",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# JSON export: machine-readable feature explanations
# ═════════════════════════════════════════════════════════════════════════════

def generate_explanations(amplitudes, cos_phases, sin_phases, spectral,
                          cluster_labels=None, representatives=None):
    """
    Generate a structured description of each SAE feature based on its
    *differential* spectral signature (feature ON − baseline) from the XAE.
    Optionally includes cluster assignment.
    """
    n_features = amplitudes.shape[0]
    amp_diff = amplitudes.numpy()  # differential log-amplitude
    explanations = []

    for i in range(n_features):
        freqs = spectral.freqs

        # Peak boost frequency (most positive differential)
        peak_boost_idx = amp_diff[i].argmax()
        peak_boost_freq = freqs[peak_boost_idx]
        peak_boost_val = amp_diff[i, peak_boost_idx]

        # Peak suppress frequency (most negative differential)
        peak_suppress_idx = amp_diff[i].argmin()
        peak_suppress_freq = freqs[peak_suppress_idx]
        peak_suppress_val = amp_diff[i, peak_suppress_idx]

        # Band effects (mean differential per band)
        band_effects = {}
        for band_name in CLINICAL_BANDS:
            mask = spectral.get_band_mask(band_name)
            if mask.sum() > 0:
                band_effects[band_name] = float(amp_diff[i, mask].mean())

        # Strongest boosted band and strongest suppressed band
        boosted_band = max(band_effects, key=band_effects.get)
        suppressed_band = min(band_effects, key=band_effects.get)

        # Top 3 frequency bins by absolute effect
        top3_idx = np.abs(amp_diff[i]).argsort()[-3:][::-1]
        top3_freqs = [
            {"freq_hz": float(freqs[j]), "delta_amplitude": float(amp_diff[i, j]),
             "direction": "boost" if amp_diff[i, j] > 0 else "suppress"}
            for j in top3_idx
        ]

        # Phase at peak boost
        peak_phase = float(torch.atan2(sin_phases[i, peak_boost_idx],
                                        cos_phases[i, peak_boost_idx]).item())

        # Generate description
        desc_parts = []
        if band_effects[boosted_band] > 0:
            desc_parts.append(f"Boosts {boosted_band}")
        if band_effects[suppressed_band] < 0:
            desc_parts.append(f"suppresses {suppressed_band}")
        if not desc_parts:
            desc_parts.append("Minimal spectral effect")
        peak_str = f"(peak boost at {peak_boost_freq:.1f} Hz)"
        description = ", ".join(desc_parts) + " " + peak_str

        entry = {
            "feature": i,
            "description": description,
            "peak_boost_frequency_hz": float(peak_boost_freq),
            "peak_boost_delta": float(peak_boost_val),
            "peak_suppress_frequency_hz": float(peak_suppress_freq),
            "peak_suppress_delta": float(peak_suppress_val),
            "boosted_band": boosted_band,
            "suppressed_band": suppressed_band,
            "band_effects": {k: round(v, 4) for k, v in band_effects.items()},
            "top3_affected_frequencies": top3_freqs,
            "peak_phase_rad": round(peak_phase, 4),
        }
        if cluster_labels is not None:
            entry["cluster"] = int(cluster_labels[i])
            entry["is_representative"] = (representatives is not None
                                           and i in representatives)
        explanations.append(entry)

    return explanations


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global EMBED, TARGET_LAYER, SAE_PATH, DATA_PATH, XAE_PATH, OUT_DIR, FS, PATCH_SIZE
    t0 = time.time()

    # ── CLI args ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Explain SAE features via XAE")
    parser.add_argument("--encoder", default="sleepfm",
                        choices=["sleepfm", "reve"],
                        help="Encoder backend (default: sleepfm)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Transformer layer index (default: last layer)")
    parser.add_argument("--weights-path", default=None,
                        help="Path to finetuned checkpoint (REVE .ckpt or SleepFM .pt)")
    parser.add_argument("--tag", default=None,
                        help="Run tag (e.g. 'qjbe08') — scopes SAE, XAE, and output "
                             "paths to results/**/{encoder}_{tag}/")
    args = parser.parse_args()

    # ── Resolve encoder-scoped paths ─────────────────────────────────────
    run_name = f"{args.encoder}_{args.tag}" if args.tag else args.encoder
    model = load_model(args.encoder, weights_path=args.weights_path)
    model._encoder_name = args.encoder
    EMBED = model.embed_dim
    FS = _ENCODER_FS[args.encoder]
    PATCH_SIZE = FS  # 1-second patches at native sample rate
    n_layers = len(model.get_hookable_layers())
    TARGET_LAYER = args.layer if args.layer is not None else n_layers - 1
    DATA_PATH = _ENCODER_DATA[args.encoder]
    SAE_PATH = (ROOT / "results" / "features" / run_name
                / f"sae_{args.encoder}_exp{EXPANSION}_k{K}_layer{TARGET_LAYER}.pt")
    # Encoder-scoped XAE and output paths
    if args.encoder == "sleepfm" and not args.tag:
        XAE_PATH = ROOT / "results" / "xae" / "xae_checkpoint.pt"
    else:
        XAE_PATH = ROOT / "results" / "xae" / run_name / "xae_checkpoint.pt"
    OUT_DIR = ROOT / "results" / "xae" / run_name / "explanations"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  encoder={args.encoder}  embed_dim={EMBED}  "
          f"layer={TARGET_LAYER}  SAE={SAE_PATH.name}")

    print("=" * 70)
    print("  Explain Features — SAE × XAE Spectral Interpretation")
    print("=" * 70)

    # ── Load everything ─────────────────────────────────────────────────
    sae, act_mean, act_std = load_sae()
    xae_trainer = load_xae()
    spectral = xae_trainer.spectral

    dict_size = sae.dict_size
    print("\n[Config]")
    print(f"  SAE dict size:   {dict_size}")
    print(f"  Frequency bins:  {spectral.n_bins}")
    print(f"  Freq range:      {spectral.f_min}–{spectral.f_max} Hz")
    print(f"  Clinical bands:  {list(CLINICAL_BANDS.keys())}")
    print(f"  Clusters:        {N_CLUSTERS}")
    has_temporal = xae_trainer.temporal_extractor is not None
    if has_temporal:
        print(f"  Temporal head:   ENABLED ({xae_trainer.temporal_extractor.target_dim} dims)")

    # ── Decode all feature directions ───────────────────────────────────
    print("\n=== Decoding SAE features through XAE (contrastive) ===")
    amplitudes, cos_phases, sin_phases, baseline_amp, temporal_diffs = decode_all_features(
        sae, xae_trainer, act_mean, act_std
    )
    print(f"  Decoded {dict_size} feature directions → differential spectral signatures")
    if temporal_diffs is not None:
        print(f"  Also decoded temporal contrastive waveforms "
              f"({len(temporal_diffs)} bands × {list(temporal_diffs.values())[0].shape[-1]} samples)")

    # ── Cluster features into spectral archetypes ───────────────────────
    print(f"\n=== Clustering {dict_size} features into {N_CLUSTERS} archetypes ===")
    cluster_labels, representatives, centroids, band_powers = cluster_features(
        amplitudes, spectral, n_clusters=N_CLUSTERS
    )
    band_names = list(CLINICAL_BANDS.keys())
    for c in range(N_CLUSTERS):
        n_members = (cluster_labels == c).sum()
        lbl = _archetype_label(centroids[c], band_names)
        print(f"  [pre-sort] Cluster {c}: {n_members:3d} features  "
              f"rep=F{representatives[c]}  {lbl}")

    # ── Generate explanations ───────────────────────────────────────────
    print("\n=== Generating feature explanations ===")
    explanations = generate_explanations(
        amplitudes, cos_phases, sin_phases, spectral,
        cluster_labels=cluster_labels, representatives=representatives,
    )

    # Print cluster-grouped summary (concise)
    for c in range(N_CLUSTERS):
        members = [e for e in explanations if e.get("cluster") == c]
        rep = [e for e in members if e.get("is_representative")]
        rep_desc = rep[0]["description"] if rep else "?"
        print(f"  C{c} ({len(members):3d} feats): {rep_desc}")

    # Save JSON
    json_path = OUT_DIR / "feature_explanations.json"
    with open(json_path, "w") as f:
        json.dump(explanations, f, indent=2)
    print(f"  ✓ Saved {json_path}")

    # ── Generate all plots ──────────────────────────────────────────────
    print("\n=== Generating visualisations ===")

    # Plot 1: Cluster overview (heatmap + membership + baseline)
    labels_reordered, reps_reordered = plot_cluster_overview(
        centroids, cluster_labels, representatives, band_powers,
        baseline_amp, spectral,
        OUT_DIR / "01_cluster_overview.png",
    )

    # Plot 2: Representative spectral cards (one per cluster)
    plot_representative_cards(
        amplitudes, cos_phases, sin_phases, spectral,
        reps_reordered, labels_reordered,
        OUT_DIR / "02_representative_cards.png",
    )

    # Plot 3: Cluster mean spectra overlay
    plot_cluster_spectra(
        amplitudes, spectral, labels_reordered, reps_reordered,
        OUT_DIR / "03_cluster_spectra.png",
    )

    # Plot 4: Phase coherence (representatives only)
    plot_phase_coherence(
        cos_phases, sin_phases, spectral, reps_reordered, labels_reordered,
        OUT_DIR / "04_phase_coherence.png",
    )

    # Plot 5: Characteristic waveforms (phase-agnostic, from spectral amp)
    plot_temporal_waveforms(
        temporal_diffs, reps_reordered, labels_reordered,
        OUT_DIR / "05_temporal_waveforms.png",
    )

    elapsed = time.time() - t0
    print(f"\n✅ Feature explanation complete in {elapsed:.1f} seconds.")
    print(f"   Results in {OUT_DIR}/")
    print(f"\n   Archetype Summary ({N_CLUSTERS} clusters from {dict_size} features):")
    for c in range(N_CLUSTERS):
        n_members = (labels_reordered == c).sum()
        rep_feat = reps_reordered[c]
        rep_expl = explanations[rep_feat]
        print(f"   C{c} ({n_members:3d} feats) → rep F{rep_feat:3d}: "
              f"{rep_expl['description']}")


if __name__ == "__main__":
    main()
