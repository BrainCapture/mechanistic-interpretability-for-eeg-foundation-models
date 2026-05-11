"""Probe the Codebook with Synthetic EEG Morphologies.

Synthesizes textbook EEG waveforms (spike-and-slow-wave, spindle,
K-complex, vertex sharp, pure delta, pure alpha), embeds each through
the SetTransformer, and finds the nearest cluster centroids in the
codebook.

For each probe:
  - Shows the synthetic waveform
  - Shows the top-5 nearest cluster reconstructions
  - Reports cosine similarity and Euclidean distance

This answers the question: can the embedding space distinguish
epileptiform from normal morphology?

Output
------
  results/xae/codebook/
    morphology_probe.png
    morphology_probe_summary.txt

Usage::

    uv run tools/probe_morphology.py
"""
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sae4eeg.xae import XAETrainer, CLINICAL_BANDS
from sae4eeg.dataset import get_dataloaders, StandardizeLabel
from sae4eeg.encoders import EncoderBackend, load_encoder
from sae4eeg.sae import ActivationExtractor

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"
XAE_PATH = ROOT / "results" / "xae" / "xae_checkpoint.pt"
CODEBOOK_PATH = ROOT / "results" / "xae" / "codebook" / "codebook.pt"
DATA_PATH = ROOT / "data" / "D4-v3-preprocessed-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FS = 128
PATCH_SIZE = 128
TARGET_LAYER = 2

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]

BAND_COLORS = {
    "delta":     "#1f77b4",
    "theta":     "#2ca02c",
    "alpha":     "#ff7f0e",
    "low-beta":  "#d62728",
    "high-beta": "#9467bd",
    "gamma":     "#8c564b",
}


def remove_module_from_state_dict(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()}


# =====================================================================
# Synthetic waveform generators
# =====================================================================

def _time(n=PATCH_SIZE, fs=FS):
    return np.arange(n) / fs


def _apply_to_channels(waveform_1d, n_channels=19, spatial_profile=None):
    """Replicate a 1D waveform across channels with optional spatial weighting.

    spatial_profile: array of shape (n_channels,) with weights [0,1].
    If None, all channels get the same amplitude.
    """
    w = waveform_1d.copy()
    if spatial_profile is None:
        spatial_profile = np.ones(n_channels)
    out = np.outer(spatial_profile, w)  # (n_channels, n_samples)
    return out.astype(np.float32)


def spike_and_slow_wave(fs=FS, n=PATCH_SIZE):
    """Textbook spike-and-slow-wave complex.

    - Sharp spike: ~70ms duration, large negative peak
    - Slow wave: ~300ms positive deflection following the spike
    - Background: low-amplitude delta
    """
    t = _time(n, fs)
    sig = np.zeros(n, dtype=np.float64)

    # Background: subtle delta
    sig += 0.1 * np.sin(2 * np.pi * 1.5 * t)

    # Spike at ~0.25s: Gaussian-windowed sharp transient
    spike_center = 0.25
    spike_width = 0.025  # ~50ms FWHM -> very sharp
    spike_amp = -3.0     # large negative
    spike = spike_amp * np.exp(-0.5 * ((t - spike_center) / spike_width) ** 2)
    sig += spike

    # Slow wave following spike: broader positive deflection
    slow_center = 0.45
    slow_width = 0.12
    slow_amp = 1.5
    slow = slow_amp * np.exp(-0.5 * ((t - slow_center) / slow_width) ** 2)
    sig += slow

    # Spatial: generalized (all channels), strongest at Fz/Cz
    spatial = np.ones(19) * 0.6
    spatial[4] = 1.0   # Fz
    spatial[9] = 1.0   # Cz
    spatial[3] = 0.9   # F3
    spatial[5] = 0.9   # F4
    return _apply_to_channels(sig.astype(np.float32), spatial_profile=spatial)


def polyspike_wave(fs=FS, n=PATCH_SIZE):
    """Polyspike-and-wave: multiple spikes followed by slow wave."""
    t = _time(n, fs)
    sig = np.zeros(n, dtype=np.float64)

    # Background delta
    sig += 0.08 * np.sin(2 * np.pi * 1.2 * t)

    # 3 spikes in rapid succession
    for offset in [0.15, 0.22, 0.29]:
        spike = -2.5 * np.exp(-0.5 * ((t - offset) / 0.02) ** 2)
        sig += spike

    # Slow wave
    slow = 1.8 * np.exp(-0.5 * ((t - 0.50) / 0.14) ** 2)
    sig += slow

    spatial = np.ones(19) * 0.7
    spatial[4] = 1.0
    spatial[9] = 1.0  # Fz, Cz
    return _apply_to_channels(sig.astype(np.float32), spatial_profile=spatial)


def sleep_spindle(fs=FS, n=PATCH_SIZE):
    """Sleep spindle: 12-14 Hz oscillation, ~0.5s duration, waxing-waning."""
    t = _time(n, fs)
    freq = 13.0
    envelope = np.exp(-0.5 * ((t - 0.5) / 0.15) ** 2)
    sig = 1.5 * envelope * np.sin(2 * np.pi * freq * t)

    # Background delta
    sig += 0.3 * np.sin(2 * np.pi * 1.0 * t)

    spatial = np.ones(19) * 0.5
    spatial[9] = 1.0   # Cz (central max for spindles)
    spatial[8] = 0.9   # C3
    spatial[10] = 0.9  # C4
    spatial[14] = 0.8  # Pz
    return _apply_to_channels(sig.astype(np.float32), spatial_profile=spatial)


def k_complex(fs=FS, n=PATCH_SIZE):
    """K-complex: sharp negative peak + broad positive deflection."""
    t = _time(n, fs)
    sig = np.zeros(n, dtype=np.float64)

    # Background delta
    sig += 0.15 * np.sin(2 * np.pi * 0.8 * t)

    # Sharp negative component
    neg = -2.0 * np.exp(-0.5 * ((t - 0.35) / 0.04) ** 2)
    sig += neg

    # Broad positive component
    pos = 1.2 * np.exp(-0.5 * ((t - 0.55) / 0.10) ** 2)
    sig += pos

    spatial = np.ones(19) * 0.6
    spatial[4] = 1.0
    spatial[9] = 1.0  # Fz, Cz frontocentral max
    spatial[3] = 0.9
    spatial[5] = 0.9
    return _apply_to_channels(sig.astype(np.float32), spatial_profile=spatial)


def vertex_sharp(fs=FS, n=PATCH_SIZE):
    """Vertex sharp wave: brief negative transient at Cz."""
    t = _time(n, fs)
    sig = np.zeros(n, dtype=np.float64)
    sig += 0.1 * np.sin(2 * np.pi * 1.0 * t)

    # Sharp negative at vertex
    sharp = -2.5 * np.exp(-0.5 * ((t - 0.4) / 0.03) ** 2)
    sig += sharp

    spatial = np.ones(19) * 0.3
    spatial[9] = 1.0   # Cz max
    spatial[4] = 0.7   # Fz
    spatial[14] = 0.7  # Pz
    return _apply_to_channels(sig.astype(np.float32), spatial_profile=spatial)


def pure_delta(fs=FS, n=PATCH_SIZE):
    """Pure 1.5 Hz delta wave -- typical deep sleep."""
    t = _time(n, fs)
    sig = 2.0 * np.sin(2 * np.pi * 1.5 * t)
    spatial = np.ones(19) * 0.8
    spatial[3] = 1.0
    spatial[4] = 1.0
    spatial[5] = 1.0  # frontal max
    return _apply_to_channels(sig.astype(np.float32), spatial_profile=spatial)


def pure_alpha(fs=FS, n=PATCH_SIZE):
    """Pure 10 Hz alpha -- eyes-closed posterior rhythm."""
    t = _time(n, fs)
    sig = 1.0 * np.sin(2 * np.pi * 10.0 * t)
    spatial = np.ones(19) * 0.3
    spatial[17] = 1.0
    spatial[18] = 1.0  # O1, O2 posterior max
    spatial[15] = 0.8
    spatial[13] = 0.8  # P4, P3
    return _apply_to_channels(sig.astype(np.float32), spatial_profile=spatial)


def three_hz_spike_wave(fs=FS, n=PATCH_SIZE):
    """3 Hz spike-and-wave -- classic absence seizure pattern."""
    t = _time(n, fs)
    sig = np.zeros(n, dtype=np.float64)

    # Repeating spike-wave at 3 Hz (~3 complexes per second)
    for cycle in range(3):
        center = 0.1 + cycle * 0.33
        # Spike
        spike = -2.5 * np.exp(-0.5 * ((t - center) / 0.02) ** 2)
        sig += spike
        # Wave
        wave = 1.5 * np.exp(-0.5 * ((t - (center + 0.12)) / 0.08) ** 2)
        sig += wave

    spatial = np.ones(19) * 0.8  # generalized
    spatial[4] = 1.0
    spatial[9] = 1.0
    return _apply_to_channels(sig.astype(np.float32), spatial_profile=spatial)


# All morphologies to probe
MORPHOLOGIES = {
    "Spike & Slow Wave":    spike_and_slow_wave,
    "Polyspike & Wave":     polyspike_wave,
    "3 Hz Spike-Wave":      three_hz_spike_wave,
    "K-Complex":            k_complex,
    "Vertex Sharp":         vertex_sharp,
    "Sleep Spindle":        sleep_spindle,
    "Pure Delta (1.5 Hz)":  pure_delta,
    "Pure Alpha (10 Hz)":   pure_alpha,
}


# =====================================================================
# Embedding and matching
# =====================================================================

def embed_synthetic(model, waveform_19ch, target_layer=None):
    """Embed a single synthetic patch through the encoder.

    We need to pass a full (1, C, T) window.
    Returns the activation at target_layer (defaults to module-level TARGET_LAYER).
    """
    layer_idx = TARGET_LAYER if target_layer is None else target_layer
    if isinstance(model, EncoderBackend):
        inner_model = model.model
        hook_layers = model.get_hookable_layers()
        call_fn = lambda x: model.encode(x)
    else:
        inner_model = model
        hook_layers = None
        call_fn = lambda x: inner_model(x)
    inner_model.eval()
    extractor = ActivationExtractor(inner_model, layers=hook_layers)

    x = torch.tensor(waveform_19ch, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    extractor.clear()
    with torch.no_grad():
        _ = call_fn(x)
    acts = extractor.get_activations()
    emb = acts[layer_idx]  # (1, N_tokens, D)
    extractor.remove_hooks()

    # Average over token dimension to get a single D-dim embedding
    return emb.squeeze(0).mean(0).numpy()  # (D,)


def find_nearest_centroids(emb, codebook, top_k=5):
    """Find the top-k nearest cluster centroids to the probe embedding."""
    centroids = codebook["centroids_emb"].numpy()  # (K, 128)

    # Euclidean distance
    dists = np.linalg.norm(centroids - emb, axis=1)

    # Cosine similarity
    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
    cent_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    cosines = (cent_norms * emb_norm).sum(axis=1)

    # Top-k by Euclidean distance
    top_idx = np.argsort(dists)[:top_k]
    return top_idx, dists[top_idx], cosines[top_idx]


def get_dominant_band(codebook, cluster_id, trainer):
    """Get the dominant clinical band for a cluster."""
    freqs = trainer.spectral.freqs
    clean_mask = freqs <= 45.0
    amp_log1p = codebook["centroid_amp_log1p"][cluster_id].numpy()
    band_names = list(CLINICAL_BANDS.keys())
    bp = []
    for bname, (lo, hi) in CLINICAL_BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi) & clean_mask
        amp_lin = np.expm1(amp_log1p[mask])
        bp.append((amp_lin ** 2).mean())
    return band_names[np.argmax(bp)]


# =====================================================================
# Normalisation: match real data statistics
# =====================================================================

def estimate_data_scale(trainer, model, val_loader, n_batches=5):
    """Estimate mean and std of real EEG patches for normalisation."""
    model.eval().to(DEVICE)
    all_vals = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        x = batch[0].to(DEVICE) if isinstance(batch, (list, tuple)) else batch.to(DEVICE)
        all_vals.append(x.cpu().numpy().flatten())
    arr = np.concatenate(all_vals)
    return arr.mean(), arr.std()


# =====================================================================
# Visualisation
# =====================================================================

def plot_probe_results(results, out_path):
    """Multi-row figure: one row per morphology.

    Each row: [synthetic waveform (Cz)] [top-5 nearest cluster waveforms (Cz)]
    """
    n_morphologies = len(results)
    n_matches = 5
    n_cols = 1 + n_matches

    fig, axes = plt.subplots(n_morphologies, n_cols,
                             figsize=(4 * n_cols, 3.2 * n_morphologies))
    if n_morphologies == 1:
        axes = axes[np.newaxis, :]

    t_axis = np.arange(PATCH_SIZE) / FS
    cz_idx = 9  # Cz channel

    for row, res in enumerate(results):
        name = res["name"]
        synth = res["waveform"]  # (19, 128)

        # Column 0: synthetic waveform
        ax = axes[row, 0]
        sig = synth[cz_idx]
        ymax = np.abs(sig).max() * 1.2
        if ymax < 1e-6:
            ymax = 1.0
        ax.plot(t_axis, sig, color="#C62828", linewidth=1.5)
        ax.axhline(0, color="grey", linewidth=0.3)
        ax.set_ylim(-ymax, ymax)
        ax.set_ylabel(name, fontsize=9, fontweight="bold",
                      rotation=0, labelpad=80, ha="right", va="center")
        ax.set_title("Synthetic (Cz)" if row == 0 else "",
                     fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.1)
        if row == n_morphologies - 1:
            ax.set_xlabel("Time (s)", fontsize=8)

        # Columns 1-5: nearest cluster reconstructions
        for col_i in range(n_matches):
            ax = axes[row, 1 + col_i]
            if col_i < len(res["match_cluster_ids"]):
                res["match_cluster_ids"][col_i]
                c_dist = res["match_dists"][col_i]
                c_cos = res["match_cosines"][col_i]
                recon = res["match_waveforms"][col_i]  # (19, 128)
                c_size = res["match_sizes"][col_i]
                dom_band = res["match_dom_bands"][col_i]
                c_rank = res["match_ranks"][col_i]

                sig_r = recon[cz_idx].numpy() if torch.is_tensor(recon) else recon[cz_idx]
                ymax_r = np.abs(sig_r).max() * 1.2
                if ymax_r < 1e-6:
                    ymax_r = 1.0

                ax.plot(t_axis, sig_r, color="#1565C0", linewidth=1.2)
                ax.axhline(0, color="grey", linewidth=0.3)
                ax.set_ylim(-ymax_r, ymax_r)

                band_color = BAND_COLORS.get(dom_band, "black")
                ax.set_title(
                    f"#{c_rank+1} (n={c_size})\n"
                    f"d={c_dist:.1f}  cos={c_cos:.3f}\n{dom_band}",
                    fontsize=7, color=band_color, fontweight="bold",
                ) if row == 0 else ax.set_title(
                    f"#{c_rank+1} (n={c_size})  d={c_dist:.1f}\n"
                    f"cos={c_cos:.3f}  {dom_band}",
                    fontsize=7, color=band_color,
                )
            else:
                ax.set_visible(False)

            ax.tick_params(labelsize=5)
            ax.grid(True, alpha=0.1)
            if row == n_morphologies - 1:
                ax.set_xlabel("Time (s)", fontsize=7)

    fig.suptitle(
        "Morphology Probe: Synthetic EEG vs Nearest Codebook Clusters (Cz)\n"
        "Left = synthetic textbook waveform, Right = top-5 nearest cluster "
        "reconstructions (XAE amp + exemplar phase)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Probe figure -> {out_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    out_dir = ROOT / "results" / "xae" / "codebook"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Morphology Probe: Synthetic EEG -> Codebook Matching")
    print("=" * 72)

    # -- Load model and codebook -------------------------------------
    model = load_encoder("sleepfm", weights_path=MODEL_PATH)
    model.to(DEVICE).eval()

    trainer = XAETrainer(embed_dim=128, device=DEVICE)
    trainer.load(str(XAE_PATH))
    trainer.xae.to(DEVICE).eval()

    codebook = torch.load(CODEBOOK_PATH, weights_only=False)
    n_clusters = codebook["n_clusters"]
    print(f"[ok] Loaded model, XAE, codebook ({n_clusters} clusters)")

    # Load sort_order for rank mapping
    sort_order = codebook["sort_order"].numpy()
    rank_map = {int(sort_order[r]): r for r in range(len(sort_order))}

    # -- Estimate real data scale for normalisation ------------------
    gen = get_dataloaders(train_path=str(DATA_PATH),
                          transformer=StandardizeLabel())
    _, _, val_loader, _ = next(gen)
    data_mean, data_std = estimate_data_scale(trainer, model, val_loader)
    print(f"  Real data scale: mean={data_mean:.4f}, std={data_std:.4f}")

    # -- Probe each morphology ---------------------------------------
    results = []
    lines = []
    lines.append("=" * 90)
    lines.append("  Morphology Probe Summary")
    lines.append("=" * 90)

    for name, gen_fn in MORPHOLOGIES.items():
        print(f"\n  Probing: {name}")
        waveform = gen_fn()  # (19, 128), raw synthetic

        # Normalise to match real data distribution
        w_mean = waveform.mean()
        w_std = waveform.std()
        if w_std > 1e-8:
            waveform_norm = (waveform - w_mean) / w_std * data_std + data_mean
        else:
            waveform_norm = waveform

        # Embed
        emb = embed_synthetic(model, waveform_norm)

        # Find nearest clusters
        top_idx, dists, cosines = find_nearest_centroids(emb, codebook, top_k=5)

        # Collect match info
        match_waveforms = []
        match_sizes = []
        match_dom_bands = []
        match_ranks = []
        for c_id in top_idx:
            match_waveforms.append(codebook["recon_waveforms"][c_id])
            match_sizes.append(int(codebook["cluster_sizes"][c_id]))
            match_dom_bands.append(get_dominant_band(codebook, c_id, trainer))
            match_ranks.append(rank_map.get(c_id, -1))

        results.append({
            "name": name,
            "waveform": waveform,  # unnormalized for display
            "embedding": emb,
            "match_cluster_ids": top_idx,
            "match_dists": dists,
            "match_cosines": cosines,
            "match_waveforms": match_waveforms,
            "match_sizes": match_sizes,
            "match_dom_bands": match_dom_bands,
            "match_ranks": match_ranks,
        })

        # Print summary
        lines.append(f"\n  {name}")
        lines.append(f"  {'':4s}  {'Rank':>5s}  {'Cl':>4s}  {'Size':>5s}  "
                      f"{'Dist':>6s}  {'Cosine':>7s}  {'Band':>8s}")
        lines.append(f"  {'':4s}  " + "-" * 50)
        for i, c_id in enumerate(top_idx):
            r = rank_map.get(c_id, -1)
            print(f"    #{i+1}: cluster {c_id} (rank #{r+1}, n={match_sizes[i]})  "
                  f"dist={dists[i]:.2f}  cos={cosines[i]:.3f}  "
                  f"band={match_dom_bands[i]}")
            lines.append(
                f"  {i+1:>4d}  {r+1:>5d}  {c_id:>4d}  {match_sizes[i]:>5d}  "
                f"{dists[i]:>6.2f}  {cosines[i]:>7.3f}  {match_dom_bands[i]:>8s}"
            )

    # -- Compute inter-morphology distances --------------------------
    print(f"\n{'='*72}")
    print("  Inter-Morphology Embedding Distances")
    print(f"{'='*72}")
    names = [r["name"] for r in results]
    embs = np.array([r["embedding"] for r in results])
    n = len(names)

    lines.append("\n\n  Inter-Morphology Embedding Distances (Euclidean)")
    header = f"  {'':25s}" + "".join(f"{n[:8]:>10s}" for n in names)
    lines.append(header)

    for i in range(n):
        row = f"  {names[i]:25s}"
        for j in range(n):
            d = np.linalg.norm(embs[i] - embs[j])
            row += f"{d:>10.2f}"
            if i < j:
                print(f"    {names[i]:25s} <-> {names[j]:25s}  dist={d:.2f}")
        lines.append(row)

    # -- Save --------------------------------------------------------
    txt_path = out_dir / "morphology_probe_summary.txt"
    txt_path.write_text("\n".join(lines))
    print(f"\n  Summary -> {txt_path}")

    plot_probe_results(results, out_dir / "morphology_probe.png")
    print("\nDone!")


if __name__ == "__main__":
    main()
