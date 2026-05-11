"""
C2 Deep-Dive: Why "Spike-and-Slow-Wave"?
==========================================
This script produces a multi-panel figure that:

 1. EXPLAINS the morphology match
    - Shows C2's measured band-effect vector alongside each clinical
      morphology template.  The match is a cosine similarity between
      the two 6-dimensional vectors (one dimension per clinical band).
    - A bar chart makes the cosine scores for all 6 morphologies
      visually obvious.

 2. SHOWS the spectral signature
    - Full-resolution XAE-decoded differential spectrum for every C2
      feature and the cluster centroid, with clinical band shading.

 3. SHOWS real EEG examples
    - Finds the validation windows where C2 features fire most
      strongly (highest mean C2 activation across the 60 tokens).
    - Plots the raw multi-channel EEG with per-token C2 activation
      overlaid, highlighting the 1-second patches where the features
      fire.

Outputs → results/xae/morphology/c2_deep_dive.png

Usage:  uv run tools/c2_deep_dive.py
"""

import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch

# ── project imports ──────────────────────────────────────────────────────
from sae4eeg.xae import XAETrainer, CLINICAL_BANDS
from sae4eeg.sae import SparseAutoencoder, ActivationExtractor
from sae4eeg.dataset import get_dataloaders, StandardizeLabel
from sae4eeg.encoders import EncoderBackend, load_encoder

# ── constants ────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"

_ENCODER_DATA = {
    "sleepfm": ROOT / "data" / "D4-v3-preprocessed-v2",
    "reve":    ROOT / "data" / "D4-v3-preprocessed-v1",
}
DATA_PATH  = _ENCODER_DATA["sleepfm"]  # overridden at runtime
XAE_PATH   = ROOT / "results" / "xae" / "xae_checkpoint.pt"
EXPL_PATH  = ROOT / "results" / "xae" / "explanations" / "feature_explanations.json"
OUT_DIR    = ROOT / "results" / "xae" / "morphology"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EMBED      = 128    # overridden at runtime from encoder.embed_dim
EXPANSION  = 1.0
K          = 8
PATCH_SIZE = 128
FS         = 128
TARGET_LAYER = 2    # overridden at runtime: last encoder layer
S_TOKENS   = 60

# SAE_PATH is resolved at runtime based on encoder name + layer
SAE_PATH   = ROOT / "results" / "features" / "sleepfm" / "sae_sleepfm_exp1_k8_layer2.pt"

CLUSTER_ID = 2       # the cluster we're investigating
N_EXAMPLES = 5       # how many EEG windows to show

BAND_NAMES  = list(CLINICAL_BANDS.keys())
BAND_RANGES = CLINICAL_BANDS
BAND_COLORS = {
    "delta":     "#1f77b4",
    "theta":     "#2ca02c",
    "alpha":     "#ff7f0e",
    "low-beta":  "#d62728",
    "high-beta": "#9467bd",
    "gamma":     "#8c564b",
}

# Import the morphology templates from morphology_analysis
from tools.morphology_analysis import CLINICAL_MORPHOLOGIES, match_morphology  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════
# Helpers (same as morphology_analysis.py)
# ═════════════════════════════════════════════════════════════════════════

def remove_module_from_state_dict(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()}


def load_all(encoder_name: str = "sleepfm", target_layer: int = None):
    """Load model, SAE, XAE, data loader, and explanations."""
    global EMBED, TARGET_LAYER, SAE_PATH, DATA_PATH

    DATA_PATH = _ENCODER_DATA[encoder_name]

    # Encoder
    kwargs = {"weights_path": MODEL_PATH} if encoder_name == "sleepfm" else {}
    model = load_encoder(encoder_name, **kwargs)
    model.to(DEVICE).eval()
    EMBED = model.embed_dim
    n_layers = len(model.get_hookable_layers())
    TARGET_LAYER = target_layer if target_layer is not None else n_layers - 1
    SAE_PATH = (ROOT / "results" / "features" / encoder_name
                / f"sae_{encoder_name}_exp1_k8_layer{TARGET_LAYER}.pt")
    print(f"  encoder={encoder_name}  embed_dim={EMBED}  "
          f"layer={TARGET_LAYER}  SAE={SAE_PATH.name}")

    # SAE
    ckpt_sae = torch.load(SAE_PATH, weights_only=False)
    sae = SparseAutoencoder(EMBED, expansion=EXPANSION, mode="topk", k=K)
    sae.load_state_dict(ckpt_sae["sae_state_dict"])
    act_mean = ckpt_sae["act_mean"]
    act_std  = ckpt_sae["act_std"]

    # XAE
    xae_trainer = XAETrainer(embed_dim=EMBED, fs=FS, n_fft=PATCH_SIZE)
    xae_trainer.load(str(XAE_PATH))

    # Data
    gen = get_dataloaders(train_path=str(DATA_PATH),
                          transformer=StandardizeLabel())
    fold, train_loader, val_loader, test_loader = next(gen)

    # Explanations
    with open(EXPL_PATH) as f:
        explanations = json.load(f)

    return model, sae, act_mean, act_std, xae_trainer, val_loader, explanations


def encode_with_raw(model, sae, act_mean, act_std, loader, max_windows=2000):
    """
    Encode EEG windows AND keep the raw EEG + labels.
    Returns:
        codes:  (N, S, dict_size)
        labels: (N,)
        raw:    (N, C, T)
    """
    sae_dev = sae.to(DEVICE).eval()
    mean_dev = act_mean.to(DEVICE)
    std_dev  = act_std.to(DEVICE)

    all_codes, all_labels, all_raw = [], [], []
    n = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        all_labels.append(y)
        all_raw.append(x)
        with torch.no_grad():
            x_dev = x.to(DEVICE)
            if isinstance(model, EncoderBackend):
                inner_model = model.model
                hook_layers = model.get_hookable_layers()
            else:
                inner_model = model
                hook_layers = None
            extractor = ActivationExtractor(inner_model, layers=hook_layers)
            _ = inner_model(x_dev)
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
    codes  = torch.cat(all_codes, dim=0)[:max_windows]
    labels = torch.cat(all_labels, dim=0)[:max_windows]
    raw    = torch.cat(all_raw, dim=0)[:max_windows]
    return codes, labels, raw


# ═════════════════════════════════════════════════════════════════════════
# Panel 1: Morphology Match Explained
# ═════════════════════════════════════════════════════════════════════════

def plot_morphology_match(ax_radar, ax_bar, c2_effects):
    """
    Left:  Grouped bar chart — C2 band effects vs spike-and-slow-wave template weights.
    Right: Cosine similarity to ALL morphology templates.
    """
    # ── Grouped bar: C2 vs SASW template ────────────────────────────────
    sasw_weights = CLINICAL_MORPHOLOGIES["Spike-and-slow-wave"]["match_weights"]

    x = np.arange(len(BAND_NAMES))
    width = 0.35

    c2_vals  = [c2_effects[b] for b in BAND_NAMES]
    sw_vals  = [sasw_weights[b] for b in BAND_NAMES]

    # Normalise both to unit length for visual comparison
    c2_norm = np.array(c2_vals) / np.linalg.norm(c2_vals)
    sw_norm = np.array(sw_vals) / np.linalg.norm(sw_vals)

    ax_radar.bar(x - width/2, c2_norm, width, label="C2 centroid (normalised)",
                         color="#E53935", alpha=0.8, edgecolor="white")
    ax_radar.bar(x + width/2, sw_norm, width,
                         label="Spike-and-slow-wave template",
                         color="#1565C0", alpha=0.6, edgecolor="white",
                         hatch="//")

    ax_radar.set_xticks(x)
    ax_radar.set_xticklabels(BAND_NAMES, fontsize=9, rotation=25, ha="right")
    ax_radar.set_ylabel("Normalised weight / effect", fontsize=10)
    ax_radar.axhline(0, color="black", lw=0.7, ls="--")
    ax_radar.legend(fontsize=8, loc="upper right")
    ax_radar.set_title("Why Spike-and-Slow-Wave?\nC2 band effects vs template",
                       fontsize=11, fontweight="bold")
    ax_radar.grid(axis="y", alpha=0.2)

    # Annotate cosine
    cos_val = np.dot(c2_norm, sw_norm)
    ax_radar.text(0.02, 0.95, f"cosine = {cos_val:.3f}",
                  transform=ax_radar.transAxes, fontsize=12, fontweight="bold",
                  color="#E53935", va="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                            edgecolor="#E53935", alpha=0.9))

    # ── Right: Cosine bar chart vs all morphologies ─────────────────────
    matches = match_morphology(c2_effects)
    morph_names = [m for m, _ in matches]
    cos_scores  = [s for _, s in matches]
    colors = ["#E53935" if m == "Spike-and-slow-wave" else "#90A4AE"
              for m in morph_names]

    y_pos = np.arange(len(morph_names))
    ax_bar.barh(y_pos, cos_scores, color=colors, edgecolor="white", alpha=0.85)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(morph_names, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Cosine similarity", fontsize=10)
    ax_bar.set_title("C2 vs All Clinical Morphologies",
                     fontsize=11, fontweight="bold")
    ax_bar.axvline(0, color="black", lw=0.7, ls="--")
    ax_bar.grid(axis="x", alpha=0.2)

    for i, (mn, sc) in enumerate(matches):
        ax_bar.text(sc + 0.01, i, f"{sc:.3f}", va="center", fontsize=9,
                    fontweight="bold" if mn == "Spike-and-slow-wave" else "normal",
                    color="#E53935" if mn == "Spike-and-slow-wave" else "#555")


# ═════════════════════════════════════════════════════════════════════════
# Panel 2: Full-resolution XAE spectra for C2 features
# ═════════════════════════════════════════════════════════════════════════

def plot_c2_spectra(ax, amplitudes, freqs, c2_feature_ids, explanations):
    """
    Overlay the full-resolution differential spectrum of each C2 feature
    + the centroid (mean) in bold.
    """
    {e["feature"]: e for e in explanations}

    # Clinical band shading
    for band_name, (lo, hi) in BAND_RANGES.items():
        ax.axvspan(lo, hi, alpha=0.08, color=BAND_COLORS[band_name],
                   label=f"_{band_name}")

    # Individual C2 features (thin, translucent)
    cmap = plt.cm.Set2
    for i, fi in enumerate(c2_feature_ids):
        spec = amplitudes[fi]
        color = cmap(i / max(len(c2_feature_ids) - 1, 1))
        ax.plot(freqs, spec, linewidth=1.0, alpha=0.5, color=color,
                label=f"F{fi}")

    # Centroid (mean of all C2 features)
    centroid = np.mean([amplitudes[fi] for fi in c2_feature_ids], axis=0)
    ax.plot(freqs, centroid, linewidth=3.0, color="#E53935", alpha=0.9,
            label="C2 centroid", zorder=10)

    # Zero line
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)

    # Band labels at top
    for band_name, (lo, hi) in BAND_RANGES.items():
        mid = (lo + hi) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.1,
                band_name, ha="center", fontsize=7, color=BAND_COLORS[band_name],
                fontweight="bold", alpha=0.8)

    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Δ log-amplitude (activated − baseline)", fontsize=10)
    ax.set_title("C2 Features — Full-Resolution Spectral Signatures from XAE",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=5, loc="lower right")
    ax.grid(True, alpha=0.15)
    ax.set_xlim(freqs[0], freqs[-1])

    # Annotate key finding
    peak_idx = centroid.argmax()
    ax.annotate(f"Peak: {freqs[peak_idx]:.0f} Hz\n(broadband boost,\nstrongest in delta-alpha)",
                xy=(freqs[peak_idx], centroid[peak_idx]),
                xytext=(freqs[peak_idx] + 15, centroid[peak_idx] * 0.95),
                fontsize=8, color="#E53935", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="#E53935", alpha=0.9))


# ═════════════════════════════════════════════════════════════════════════
# Panel 3: Real EEG examples where C2 fires strongly
# ═════════════════════════════════════════════════════════════════════════

def plot_eeg_examples(fig, gs_slot, codes, labels, raw_eeg, c2_feature_ids,
                      n_examples=N_EXAMPLES):
    """
    For each of the top-N windows by C2 activation:
      - Plot raw EEG (subset of channels)
      - Overlay per-token C2 activation strength
      - Highlight the 1-second patches where C2 fires
    """
    # C2 activation per window: sum activation of all C2 features across tokens
    c2_act_per_token = codes[:, :, c2_feature_ids].sum(dim=-1)  # (N, S)
    c2_act_per_window = c2_act_per_token.mean(dim=1)            # (N,)

    # Only abnormal windows (C2 is abnormal-selective)
    abn_mask = labels == 1
    abn_indices = torch.where(abn_mask)[0]
    abn_scores  = c2_act_per_window[abn_mask]

    # Top N by activation
    top_k = min(n_examples, len(abn_scores))
    top_local = torch.argsort(abn_scores, descending=True)[:top_k]
    top_global = abn_indices[top_local]

    inner_gs = gs_slot.subgridspec(top_k, 1, hspace=0.4)

    # Pick 4 channels to show — let's take a central, frontal, temporal, occipital subset
    ch_labels_19 = [
        "Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2",
        "F7","F8","T3","T4","T5","T6","Fz","Cz","Pz"
    ]
    show_channels = [0, 4, 12, 8]  # Fp1, C3, T3, O1
    ch_names_show = [ch_labels_19[i] for i in show_channels]
    n_ch = len(show_channels)

    for rank, (loc_i, glob_i) in enumerate(zip(top_local, top_global)):
        ax = fig.add_subplot(inner_gs[rank])
        win_idx = glob_i.item()
        eeg = raw_eeg[win_idx]  # (19, 7680)
        token_act = c2_act_per_token[win_idx].numpy()  # (60,)
        window_score = c2_act_per_window[win_idx].item()
        label_name = "Abnormal"

        T = eeg.shape[1]
        t_axis = np.arange(T) / FS  # seconds

        # Plot selected channels stacked
        offsets = np.arange(n_ch) * 1.5  # vertical spacing (data is pre-standardised)
        for ci, ch_idx in enumerate(show_channels):
            trace = eeg[ch_idx].numpy()
            # Gentle scale for visibility
            trace_scaled = trace / (np.std(trace) + 1e-9) * 0.4
            ax.plot(t_axis, trace_scaled + offsets[ci], linewidth=0.4,
                    color="#333", alpha=0.7)

        # Highlight tokens where C2 fires
        fire_threshold = np.percentile(token_act[token_act > 0], 25) if (token_act > 0).sum() > 4 else 0.0
        for tok in range(S_TOKENS):
            if token_act[tok] > fire_threshold:
                t_start = tok  # seconds (each token = 1s)
                intensity = min(token_act[tok] / (token_act.max() + 1e-9), 1.0)
                ax.axvspan(t_start, t_start + 1, alpha=0.15 + 0.35 * intensity,
                           color="#E53935", zorder=0)

        # Token activation trace on secondary y-axis
        ax2 = ax.twinx()
        tok_t = np.arange(S_TOKENS) + 0.5  # centre of each 1s patch
        ax2.fill_between(tok_t, 0, token_act, alpha=0.2, color="#E53935")
        ax2.plot(tok_t, token_act, linewidth=1.2, color="#E53935", alpha=0.6)
        ax2.set_ylabel("C2 act", fontsize=7, color="#E53935")
        ax2.tick_params(axis="y", labelsize=6, colors="#E53935")
        ax2.set_ylim(0, token_act.max() * 1.5 + 0.1)

        # Labels
        ax.set_yticks(offsets)
        ax.set_yticklabels(ch_names_show, fontsize=7)
        ax.set_xlim(0, T / FS)
        if rank == top_k - 1:
            ax.set_xlabel("Time (seconds)", fontsize=9)
        ax.set_title(f"#{rank+1}  Window {win_idx} ({label_name})  ·  "
                     f"C2 mean act = {window_score:.2f}  ·  "
                     f"firing tokens: {(token_act > 0).sum()}/{S_TOKENS}",
                     fontsize=9, fontweight="bold", loc="left", color="#E53935")
        ax.grid(axis="x", alpha=0.1)

        # Add spectral inset: Welch PSD of this window (all channels averaged)
        ax_inset = ax.inset_axes([0.82, 0.05, 0.17, 0.85])
        eeg_np = eeg.numpy()
        f_psd, pxx = welch(eeg_np, fs=FS, nperseg=min(256, T), axis=1)
        pxx_mean = pxx.mean(axis=0)
        ax_inset.semilogy(f_psd, pxx_mean, linewidth=0.8, color="#1565C0")
        ax_inset.set_xlim(0.5, 50)
        ax_inset.set_xlabel("Hz", fontsize=5)
        ax_inset.set_ylabel("PSD", fontsize=5)
        ax_inset.tick_params(labelsize=5)
        # Shade clinical bands
        for bname, (blo, bhi) in BAND_RANGES.items():
            ax_inset.axvspan(blo, bhi, alpha=0.1, color=BAND_COLORS[bname])
        ax_inset.set_title("PSD", fontsize=6, pad=1)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── CLI args ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="C2 deep-dive analysis")
    parser.add_argument("--encoder", default="sleepfm",
                        choices=["sleepfm", "reve"],
                        help="Encoder backend (default: sleepfm)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Transformer layer index (default: last layer)")
    args = parser.parse_args()

    print("=" * 70)
    print("  C2 Deep-Dive: Spike-and-Slow-Wave Morphology Analysis")
    print("=" * 70)

    # ── Load everything ─────────────────────────────────────────────────
    print("\n[1/4] Loading models + data...")
    model, sae, act_mean, act_std, xae_trainer, val_loader, explanations = \
        load_all(encoder_name=args.encoder, target_layer=args.layer)
    spectral = xae_trainer.spectral

    # Identify C2 features
    c2_feats = [e for e in explanations if e.get("cluster") == CLUSTER_ID]
    c2_ids   = [e["feature"] for e in c2_feats]
    print(f"  C2 features: {c2_ids}  ({len(c2_ids)} members)")

    # Compute C2 centroid band effects
    c2_effects = {}
    for band in BAND_NAMES:
        c2_effects[band] = np.mean([e["band_effects"][band] for e in c2_feats])
    print(f"  C2 centroid band effects: {c2_effects}")

    # Morphology match scores
    matches = match_morphology(c2_effects)
    print("  Morphology matches:")
    for m, s in matches:
        print(f"    {m:30s}  cos = {s:+.4f}")

    # ── Encode windows + keep raw EEG ───────────────────────────────────
    print("\n[2/4] Encoding validation windows (keeping raw EEG)...")
    codes, labels, raw_eeg = encode_with_raw(model, sae, act_mean, act_std,
                                              val_loader, max_windows=2000)
    n_normal   = (labels == 0).sum().item()
    n_abnormal = (labels == 1).sum().item()
    print(f"  {len(labels)} windows: {n_normal} Normal, {n_abnormal} Abnormal")

    # ── Decode XAE spectra for C2 ──────────────────────────────────────
    print("\n[3/4] Decoding XAE spectral signatures for C2...")
    from tools.explain_features import decode_all_features
    amplitudes, _, _, _ = decode_all_features(sae, xae_trainer, act_mean, act_std)
    amplitudes = amplitudes.numpy()
    freqs = spectral.freqs

    # ── Build the figure ────────────────────────────────────────────────
    print("\n[4/4] Building figure...")

    fig = plt.figure(figsize=(22, 28))
    gs_top = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 2.5],
                               hspace=0.30, top=0.96, bottom=0.03,
                               left=0.06, right=0.97)

    # ── Row 1: Morphology match explanation (2 subplots) ───────────────
    gs_row1 = gs_top[0].subgridspec(1, 2, wspace=0.35)
    ax_template = fig.add_subplot(gs_row1[0])
    ax_cosbar   = fig.add_subplot(gs_row1[1])
    plot_morphology_match(ax_template, ax_cosbar, c2_effects)

    # ── Row 2: Full-resolution spectra ─────────────────────────────────
    ax_spec = fig.add_subplot(gs_top[1])
    plot_c2_spectra(ax_spec, amplitudes, freqs, c2_ids, explanations)

    # ── Row 3: Real EEG examples ───────────────────────────────────────
    plot_eeg_examples(fig, gs_top[2], codes, labels, raw_eeg, c2_ids,
                      n_examples=N_EXAMPLES)

    fig.suptitle(
        "Cluster 2 Deep-Dive — \"Spike-and-Slow-Wave\" Morphology\n"
        f"9 SAE features  ·  Mean Cohen's d = +0.52 (Abnormal-selective)  ·  "
        f"Cosine similarity to spike-and-slow-wave template = {dict(matches)['Spike-and-slow-wave']:.3f}",
        fontsize=14, fontweight="bold", y=0.99
    )

    save_path = OUT_DIR / "c2_deep_dive.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Saved {save_path}")

    elapsed = time.time() - t0
    print(f"\n✅ Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
