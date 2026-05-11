"""
SAE Feature Exploration
========================
Trains (or loads from cache) a SAE on encoder layer activations,
then generates a suite of visualisations to understand what the learned
features capture across Normal vs Abnormal EEG windows.

Outputs (in results/features/{encoder}/):
  sae_{encoder}_exp1_k8_layer{L}.pt  – SAE checkpoint
  feature_stats.json                 – per-feature firing rate, mean activation, label correlation
  01_feature_dashboards.png          – top-activating EEG windows + activation histogram per feature
  02_co_occurrence.png               – feature co-occurrence matrix
  03_umap_latent.png                 – UMAP of sparse codes, coloured by label
  04_temporal_dynamics.png           – feature activations over time for sample recordings
  05_feature_label_corr.png          – feature activation vs binary label correlation
  06_activation_profiles.png         – mean activation by feature, split by label

Usage:
  uv run tools/explore_features.py                  # SleepFM (default)
  uv run tools/explore_features.py --encoder reve   # REVE encoder
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
from scipy import stats

# ── project imports ─────────────────────────────────────────────────────────
from sae4eeg.sae import SAETrainer, SparseAutoencoder, sae_diagnostics
from sae4eeg.dataset import get_dataloaders, StandardizeLabel
from sae4eeg.encoders import load_encoder

# ── paths & constants ───────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
MODEL_PATH  = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"
OUT_DIR     = ROOT / "results" / "features" / "sleepfm"  # overridden at runtime

# Dataset versions keyed by encoder name:
#   v1 (200 Hz) → REVE native rate
#   v2 (128 Hz) → SleepFM native rate
_ENCODER_DATA = {
    "sleepfm": ROOT / "data" / "D4-v3-preprocessed-v2",
    "reve":    ROOT / "data" / "D4-v3-preprocessed-v1",
}
DATA_PATH = _ENCODER_DATA["sleepfm"]  # overridden at runtime

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EMBED       = 128    # overridden at runtime from encoder.embed_dim
EXPANSION   = 1.0
K           = 8
EPOCHS      = 20
BATCH_SIZE  = 256
RESAMPLE_EVERY = 2
TARGET_LAYER   = 2   # overridden at runtime: last layer of encoder

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]
N_CHANNELS = len(CHANNEL_NAMES)
PATCH_SIZE    = 128        # samples per patch → 1 second at 128 Hz
FS            = 128        # sampling rate — overridden at runtime for REVE (200 Hz)
S_TOKENS      = 60         # temporal tokens per window — overridden at runtime
N_CHAN_TOKENS = 1          # token channels per time step (1 for SleepFM; 19 for REVE)

FEATURE_CMAP = plt.cm.Set1  # distinct colours for 8 features

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def remove_module_from_state_dict(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()}


def load_model(encoder_name: str = "sleepfm", weights_path=None):
    """Load encoder by name. Returns encoder backend."""
    if encoder_name == "sleepfm":
        kwargs = {"weights_path": weights_path or MODEL_PATH}
    else:
        kwargs = {"weights_path": weights_path} if weights_path else {}
    encoder = load_encoder(encoder_name, **kwargs)
    encoder.to(DEVICE).eval()
    n_layers = len(encoder.get_hookable_layers())
    print(f"  encoder={encoder_name}  embed_dim={encoder.embed_dim}  "
          f"transformer_layers={n_layers}")
    return encoder


def load_data():
    gen = get_dataloaders(train_path=str(DATA_PATH),
                          transformer=StandardizeLabel())
    fold, train_loader, val_loader, test_loader = next(gen)
    return train_loader, val_loader, test_loader


def feature_color(i):
    return FEATURE_CMAP(i / max(K - 1, 1))


# ═════════════════════════════════════════════════════════════════════════════
# 1. Train SAE  (exp=1.0, k=8, 20 epochs)
# ═════════════════════════════════════════════════════════════════════════════

def train_sae(model, train_loader, val_loader):
    """Train the SAE and return it alongside normalised activations + metadata.
    Caches the trained SAE to avoid retraining on subsequent runs.
    Cache filename encodes encoder name and layer index.
    """
    encoder_tag = getattr(model, "_encoder_name", "encoder")
    sae_cache = OUT_DIR / f"sae_{encoder_tag}_exp{EXPANSION}_k{K}_layer{TARGET_LAYER}.pt"

    trainer = SAETrainer(
        embed_dim=EMBED, expansion=EXPANSION, mode="topk", k=K,
        target_layers=[TARGET_LAYER], epochs=EPOCHS, batch_size=BATCH_SIZE,
        resample_every=RESAMPLE_EVERY,
    )

    if sae_cache.exists():
        print(f"=== Loading cached SAE from {sae_cache} ===")
        checkpoint = torch.load(sae_cache, weights_only=False)
        sae = SparseAutoencoder(EMBED, expansion=EXPANSION, mode="topk", k=K)
        sae.load_state_dict(checkpoint["sae_state_dict"])
        trainer.act_means[TARGET_LAYER] = checkpoint["act_mean"]
        trainer.act_stds[TARGET_LAYER]  = checkpoint["act_std"]
    else:
        print(f"=== Training SAE (exp={EXPANSION}, k={K}, {EPOCHS} epochs, "
              f"layer {TARGET_LAYER}, embed_dim={EMBED}) ===")
        _max_train_tokens = 500_000 if encoder_tag == "reve" else None
        trained_saes = trainer.train(model, train_loader, max_tokens=_max_train_tokens)
        sae = trained_saes[TARGET_LAYER]
        # Cache for next time — store read-only to prevent accidental overwrite
        torch.save({
            "sae_state_dict": sae.state_dict(),
            "act_mean": trainer.act_means[TARGET_LAYER],
            "act_std":  trainer.act_stds[TARGET_LAYER],
            "encoder":  encoder_tag,
            "embed_dim": EMBED,
            "target_layer": TARGET_LAYER,
        }, sae_cache)
        sae_cache.chmod(0o444)
        print(f"  ✓ Cached SAE (read-only) to {sae_cache}")

    # Diagnostics on validation set
    # Cap tokens for large encoders (REVE produces ~13 GB uncapped)
    _max_val_tokens = 200_000 if encoder_tag == "reve" else None
    print("\n=== Collecting validation activations ===")
    val_acts_raw = trainer.collect_activations(model, val_loader, max_tokens=_max_val_tokens)
    val_acts = (val_acts_raw[TARGET_LAYER] - trainer.act_means[TARGET_LAYER]) / trainer.act_stds[TARGET_LAYER]
    print(f"Validation tokens: {val_acts.shape[0]}")
    sae_diagnostics(sae, val_acts, device=DEVICE)

    return sae, trainer, val_acts


def collect_raw_eeg_and_labels(loader, max_windows=500):
    """
    Collect raw EEG windows and their labels from a DataLoader.
    Returns:
        eeg_windows: (N, C, T) tensor of raw EEG
        labels: (N,) tensor of labels
    """
    eeg_list, label_list = [], []
    n = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        eeg_list.append(x)
        label_list.append(y)
        n += x.shape[0]
        if n >= max_windows:
            break
    eeg = torch.cat(eeg_list, dim=0)[:max_windows]
    labels = torch.cat(label_list, dim=0)[:max_windows]
    return eeg, labels


def encode_windows(model, sae, trainer, eeg_windows):
    """
    Run EEG windows through encoder → SAE encoder.
    Returns:
        codes_per_window: (N, S, dict_size) — sparse codes per temporal token

    ``model`` can be a plain ``SetTransformer`` or an ``EncoderBackend``.
    """
    from sae4eeg.sae import ActivationExtractor, SparseAutoencoder
    from sae4eeg.encoders import EncoderBackend

    if isinstance(model, EncoderBackend):
        inner_model  = model.model
        # Only hook the target layer to save memory
        all_hook_layers = model.get_hookable_layers()
        hook_layers  = [all_hook_layers[TARGET_LAYER]]
        call_fn      = lambda x: model.encode(x)   # noqa: E731
        hook_key     = 0  # ActivationExtractor enumerates from 0
    else:
        inner_model  = model
        hook_layers  = None  # ActivationExtractor will use transformer_encoder.layers
        call_fn      = lambda x: inner_model(x)     # noqa: E731
        hook_key     = TARGET_LAYER

    inner_model.eval().to(DEVICE)
    sae_dev = sae.to(DEVICE)
    sae_dev.eval()

    all_codes = []
    bs = 32
    for i in range(0, len(eeg_windows), bs):
        batch = eeg_windows[i:i+bs].to(DEVICE)
        with torch.no_grad():
            extractor = ActivationExtractor(inner_model, layers=hook_layers)
            _ = call_fn(batch)
            acts = extractor.get_activations()
            extractor.remove_hooks()

            layer_acts = acts[hook_key].to(DEVICE)  # (B, S, E) — hook stores on CPU
            B, S, E = layer_acts.shape

            # Normalise with training stats (ensure same device)
            mean = trainer.act_means[TARGET_LAYER].to(DEVICE)
            std  = trainer.act_stds[TARGET_LAYER].to(DEVICE)
            normed = (layer_acts - mean) / std

            # Encode through SAE — ensure flat is on same device as SAE
            flat = normed.reshape(B * S, E)
            # Use sae_dev which is explicitly on DEVICE
            z = torch.nn.functional.relu(sae_dev.encoder(flat - sae_dev.b_pre))
            if sae_dev.mode == "topk":
                z = SparseAutoencoder._topk_mask_fn(z, sae_dev.k)
            codes = z.reshape(B, S, -1)
            all_codes.append(codes.cpu())

    sae.cpu()
    return torch.cat(all_codes, dim=0)  # (N, S, dict_size)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Visualisation Functions
# ═════════════════════════════════════════════════════════════════════════════

# ── Plot 1: Feature Dashboards ─────────────────────────────────────────────

# Colour palette for 19 EEG channels — grouped by brain region
_REGION_COLORS = {
    # Frontal (Fp1, Fp2, F7, F3, Fz, F4, F8) — blues
    0: "#1f77b4", 1: "#4a90d9",
    2: "#2c5f8a", 3: "#3a7cc2", 4: "#6baed6", 5: "#3a7cc2", 6: "#2c5f8a",
    # Central / Temporal (T7, C3, Cz, C4, T8) — greens/teals
    7: "#2ca02c", 8: "#17becf", 9: "#1b9e77", 10: "#17becf", 11: "#2ca02c",
    # Parietal / Temporal (T5, P3, Pz, P4, T6) — oranges/reds
    12: "#d62728", 13: "#e6550d", 14: "#fd8d3c", 15: "#e6550d", 16: "#d62728",
    # Occipital (O1, O2) — purples
    17: "#9467bd", 18: "#8c6bb1",
}

LABEL_COLORS = {0: "#2196F3", 1: "#FF5722"}   # blue for 0, orange-red for 1
LABEL_NAMES  = {0: "Label 0", 1: "Label 1"}

def plot_single_feature_dashboard(feat_idx, codes_per_window, eeg_windows,
                                  labels, save_path, subtitle="",
                                  n_examples=3):
    """
    One figure per feature: n_examples top-activating 60s EEG windows,
    all 19 channels coloured by brain region, feature activation overlay.
    """
    feat_acts = codes_per_window[:, :, feat_idx]    # (N, S)
    mean_act_per_window = feat_acts.mean(dim=1)     # (N,)
    top_idx = mean_act_per_window.argsort(descending=True)[:n_examples]
    color = feature_color(feat_idx)

    fig = plt.figure(figsize=(32, 5 * n_examples))
    outer_gs = gridspec.GridSpec(n_examples, 2, width_ratios=[6, 1],
                                 hspace=0.35, wspace=0.12)

    for rank, w_idx in enumerate(top_idx):
        ax_eeg = fig.add_subplot(outer_gs[rank, 0])
        eeg_window = eeg_windows[w_idx].numpy()            # (C, T)
        # For REVE tokens are (C_tok × S_time); average over channel dimension
        raw_ts = feat_acts[w_idx].numpy()
        if N_CHAN_TOKENS > 1:
            feat_ts = raw_ts.reshape(N_CHAN_TOKENS, S_TOKENS).mean(axis=0)
        else:
            feat_ts = raw_ts
        t_eeg = np.arange(eeg_window.shape[1]) / FS
        t_feat = np.linspace(0, 60, S_TOKENS)

        spacing = 4.0
        for ch_idx in range(N_CHANNELS):
            trace = eeg_window[ch_idx]
            trace_norm = trace / (np.abs(trace).max() + 1e-8)
            offset = (N_CHANNELS - 1 - ch_idx) * spacing
            ax_eeg.plot(t_eeg, trace_norm + offset,
                        color=_REGION_COLORS[ch_idx], linewidth=0.5,
                        alpha=0.85)

        # Feature activation band at the bottom
        feat_norm = feat_ts / (feat_ts.max() + 1e-8)
        ax_eeg.fill_between(t_feat, -spacing,
                            -spacing + feat_norm * spacing * 0.9,
                            alpha=0.5, color=color,
                            label="feature activation" if rank == 0 else None)

        # Channel labels
        ch_positions = [(N_CHANNELS - 1 - i) * spacing
                        for i in range(N_CHANNELS)]
        ax_eeg.set_yticks(ch_positions)
        ax_eeg.set_yticklabels(CHANNEL_NAMES, fontsize=8)
        ax_eeg.set_ylim(-spacing * 1.5, (N_CHANNELS - 0.5) * spacing)
        ax_eeg.set_xlim(0, 60)
        ax_eeg.set_xlabel("Time (s)", fontsize=11)
        ax_eeg.grid(axis="x", alpha=0.2)

        lab = int(labels[w_idx])
        lab_col = LABEL_COLORS[lab]
        ax_eeg.set_title(
            f"▌ Label {lab}  —  example #{rank+1}  |  "
            f"mean act = {mean_act_per_window[w_idx]:.3f}",
            fontsize=12, color=lab_col, fontweight="bold", loc="left",
        )
        if rank == 0:
            ax_eeg.legend(loc="upper right", fontsize=10)

    # ── RIGHT: activation histogram split by label ─────────────────────
    ax_hist = fig.add_subplot(outer_gs[:, 1])
    unique_labels = sorted(set(labels.tolist()))
    for lab_val in unique_labels:
        lab_mask = (labels == lab_val)
        lab_feat = feat_acts[lab_mask].flatten().numpy()
        nonzero = lab_feat[lab_feat > 0]
        if len(nonzero) > 0:
            ax_hist.hist(nonzero, bins=40, alpha=0.55,
                         color=LABEL_COLORS[lab_val], edgecolor="white",
                         linewidth=0.3, orientation="horizontal",
                         label=f"Label {lab_val}")

    all_acts = feat_acts.flatten().numpy()
    fire_rate = (all_acts > 0).mean() * 100
    nonzero_all = all_acts[all_acts > 0]
    if len(nonzero_all) > 0:
        ax_hist.axhline(nonzero_all.mean(), color="red", ls="--", lw=1.2,
                        label=f"mean={nonzero_all.mean():.2f}")
    ax_hist.set_title(f"Activation dist.\n"
                      f"fire rate: {fire_rate:.1f}%", fontsize=11)
    ax_hist.set_ylabel("Activation magnitude", fontsize=10)
    ax_hist.set_xlabel("Count", fontsize=10)
    ax_hist.legend(fontsize=9)

    fig.suptitle(f"Feature {feat_idx} Dashboard  ({subtitle})",
                 fontsize=16, fontweight="bold", color=color, y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


def plot_feature_dashboards_set(codes_per_window, eeg_windows, labels,
                                out_dir, prefix, selection, subtitle_fn):
    """
    Generate one dashboard PNG per feature for a given selection of features.
    """
    for rank, feat_idx in enumerate(selection):
        subtitle = subtitle_fn(feat_idx, rank)
        fname = f"{prefix}_F{feat_idx}.png"
        plot_single_feature_dashboard(
            feat_idx, codes_per_window, eeg_windows, labels,
            out_dir / fname, subtitle=subtitle,
        )


# ── Plot 2: Co-occurrence Matrix ───────────────────────────────────────────
def plot_co_occurrence(codes_per_window, save_path):
    """
    Show which features tend to co-activate on the same token.
    Two clean heatmaps (no cell annotations — 128×128 is too dense).
    Features are sorted by fire rate so structure is easier to spot.
    """
    codes_flat = codes_per_window.reshape(-1, codes_per_window.shape[-1])
    active = (codes_flat > 0).float()  # (N*S, dict_size)
    n_features = active.shape[1]

    # Sort features by fire rate (descending) for visual structure
    fire_rates = active.mean(dim=0).numpy()
    order = np.argsort(fire_rates)[::-1]

    # ── Co-occurrence: P(feat_j active | feat_i active) ─────────────
    co_occur = torch.zeros(n_features, n_features)
    for i in range(n_features):
        mask_i = active[:, i] > 0
        n_i = mask_i.sum().item()
        if n_i == 0:
            continue
        for j in range(n_features):
            co_occur[i, j] = active[mask_i, j].mean().item()

    # Reorder both matrices by fire rate
    co_occur_sorted = co_occur.numpy()[np.ix_(order, order)]
    corr_full = np.corrcoef(codes_flat.numpy().T)
    corr_sorted = corr_full[np.ix_(order, order)]

    # Tick positions: label every 8th feature with its index
    tick_pos = list(range(0, n_features, 8))
    tick_labels = [f"F{order[i]}" for i in tick_pos]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # LEFT: conditional probability
    ax = axes[0]
    im = ax.imshow(co_occur_sorted, cmap="YlOrRd", vmin=0, vmax=0.6,
                   aspect="equal", interpolation="nearest")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Feature j  (sorted by fire rate →)", fontsize=11)
    ax.set_ylabel("Feature i  (sorted by fire rate →)", fontsize=11)
    ax.set_title("P(j active | i active)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Conditional probability")

    # RIGHT: Pearson correlation
    ax2 = axes[1]
    im2 = ax2.imshow(corr_sorted, cmap="RdBu_r", vmin=-0.3, vmax=0.3,
                     aspect="equal", interpolation="nearest")
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
    ax2.set_yticks(tick_pos)
    ax2.set_yticklabels(tick_labels, fontsize=7)
    ax2.set_xlabel("Feature  (sorted by fire rate →)", fontsize=11)
    ax2.set_ylabel("Feature  (sorted by fire rate →)", fontsize=11)
    ax2.set_title("Pearson correlation (activations)", fontsize=13,
                  fontweight="bold")
    fig.colorbar(im2, ax=ax2, shrink=0.75, label="Correlation")

    fig.suptitle("Feature Co-occurrence & Correlation  "
                 "(sorted by fire rate, exp=1.0, k=8)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Plot 3: UMAP of Latent Space ──────────────────────────────────────────
def plot_latent_space(codes_per_window, labels, save_path, max_points=10000):
    """
    3×3 grid: rows = {PCA, t-SNE, UMAP}, cols = {label, dominant feature, total activation}.
    Uses per-window mean codes (one point per 60s window).
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    try:
        from umap import UMAP
        have_umap = True
    except ImportError:
        print("  ⚠ umap-learn not installed — UMAP row will be skipped")
        have_umap = False

    # ── data prep ───────────────────────────────────────────────────────
    mean_codes = codes_per_window.mean(dim=1).numpy()   # (N, dict_size)
    label_arr = labels.numpy()

    if len(mean_codes) > max_points:
        idx = np.random.choice(len(mean_codes), max_points, replace=False)
        mean_codes = mean_codes[idx]
        label_arr = label_arr[idx]

    # ── compute embeddings ──────────────────────────────────────────────
    print(f"  Computing PCA on {len(mean_codes)} windows...")
    emb_pca = PCA(n_components=2, random_state=42).fit_transform(mean_codes)

    print(f"  Computing t-SNE on {len(mean_codes)} windows...")
    emb_tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                    init="pca", learning_rate="auto").fit_transform(mean_codes)

    methods = [("PCA", emb_pca), ("t-SNE", emb_tsne)]
    if have_umap:
        print(f"  Computing UMAP on {len(mean_codes)} windows...")
        emb_umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                        random_state=42).fit_transform(mean_codes)
        methods.append(("UMAP", emb_umap))

    n_rows = len(methods)

    # ── derived quantities ──────────────────────────────────────────────
    dominant = mean_codes.argmax(axis=1)
    unique_dom = np.unique(dominant)
    # Build a distinct colour palette for the dominant features
    dom_cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_dom))
    dom_color_map = {feat: dom_cmap(i) for i, feat in enumerate(unique_dom)}

    total_act = mean_codes.sum(axis=1)

    # ── figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_rows, 3, figsize=(22, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    markersize = 24
    alpha = 0.6

    for row, (method_name, emb) in enumerate(methods):
        # Col 0: Label colouring
        ax = axes[row, 0]
        for lab_val in sorted(set(label_arr.tolist())):
            mask = label_arr == lab_val
            ax.scatter(emb[mask, 0], emb[mask, 1], s=markersize, alpha=alpha,
                       color=LABEL_COLORS[lab_val],
                       label=f"Label {int(lab_val)}", edgecolors="none")
        ax.set_title(f"{method_name} — label", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, markerscale=3, framealpha=0.8)
        ax.set_xlabel(f"{method_name} 1")
        ax.set_ylabel(f"{method_name} 2")

        # Col 1: Dominant feature colouring
        ax2 = axes[row, 1]
        for feat_idx in unique_dom:
            mask = dominant == feat_idx
            ax2.scatter(emb[mask, 0], emb[mask, 1], s=markersize, alpha=alpha,
                        color=dom_color_map[feat_idx],
                        label=f"F{feat_idx}", edgecolors="none")
        ax2.set_title(f"{method_name} — dominant feature", fontsize=13,
                      fontweight="bold")
        # Compact legend: place outside if many features
        ax2.legend(fontsize=8, markerscale=2.5, ncol=3, framealpha=0.8,
                   loc="upper right", handletextpad=0.3, columnspacing=0.8)
        ax2.set_xlabel(f"{method_name} 1")
        ax2.set_ylabel(f"{method_name} 2")

        # Col 2: Total activation magnitude
        ax3 = axes[row, 2]
        sc = ax3.scatter(emb[:, 0], emb[:, 1], s=markersize, alpha=alpha,
                         c=total_act, cmap="magma", edgecolors="none")
        ax3.set_title(f"{method_name} — total activation", fontsize=13,
                      fontweight="bold")
        fig.colorbar(sc, ax=ax3, shrink=0.7, label="Σ activations")
        ax3.set_xlabel(f"{method_name} 1")
        ax3.set_ylabel(f"{method_name} 2")

    fig.suptitle("Latent Space Structure  (per-window mean codes, exp=1.0, k=8)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Plot 4: Temporal Dynamics ──────────────────────────────────────────────
def plot_temporal_dynamics(codes_per_window, labels, save_path,
                          n_examples=6, n_top_features=10):
    """
    Show how the top-N features (by fire rate) evolve over 60s windows.
    3 examples per label class, stacked vertically.
    """
    dict_size = codes_per_window.shape[-1]

    # Find top features by fire rate
    codes_flat = codes_per_window.reshape(-1, dict_size)
    fire_rates = (codes_flat > 0).float().mean(dim=0).numpy()
    top_feats = np.argsort(fire_rates)[::-1][:n_top_features].tolist()

    # Build a vivid colour palette for just these features
    cmap = plt.colormaps.get_cmap("tab10")
    feat_colors = {f: cmap(i / max(n_top_features - 1, 1))
                   for i, f in enumerate(top_feats)}

    # Pick diverse examples: half from each label class
    unique_labels = sorted(set(labels.tolist()))
    examples = []
    per_class = max(1, n_examples // len(unique_labels))
    for lab in unique_labels:
        idx_lab = (labels == lab).nonzero(as_tuple=True)[0]
        total_act = codes_per_window[idx_lab].sum(dim=(1, 2))
        top = total_act.argsort(descending=True)[:per_class]
        examples.extend(idx_lab[top].tolist())

    n_show = len(examples)
    fig, axes = plt.subplots(n_show, 1, figsize=(18, 3 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    t = np.linspace(0, 60, S_TOKENS)

    for ax_idx, w_idx in enumerate(examples):
        ax = axes[ax_idx]
        for feat_idx in top_feats:
            raw_ts = codes_per_window[w_idx, :, feat_idx].numpy()
            if N_CHAN_TOKENS > 1:
                feat_ts = raw_ts.reshape(N_CHAN_TOKENS, S_TOKENS).mean(axis=0)
            else:
                feat_ts = raw_ts
            if feat_ts.max() > 0:
                ax.fill_between(t, 0, feat_ts, alpha=0.3,
                                color=feat_colors[feat_idx])
                ax.plot(t, feat_ts, color=feat_colors[feat_idx],
                        linewidth=1.2, alpha=0.85,
                        label=f"F{feat_idx} ({fire_rates[feat_idx]*100:.0f}%)"
                        if ax_idx == 0 else None)

        lab = int(labels[w_idx])
        lab_col = LABEL_COLORS[lab]
        ax.set_ylabel("Activation", fontsize=10)
        ax.set_title(f"Window {w_idx}  ▌ Label {lab}",
                     fontsize=11, fontweight="bold", color=lab_col, loc="left")
        ax.set_xlim(0, 60)
        ax.grid(True, alpha=0.2)

    axes[0].legend(fontsize=9, ncol=min(n_top_features, 5),
                   loc="upper right", framealpha=0.85,
                   handletextpad=0.4, columnspacing=1.0)
    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    fig.suptitle(f"Temporal Dynamics — top {n_top_features} features by fire rate"
                 f"  (exp=1.0, k=8)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Plot 5: Feature–Label Correlation ──────────────────────────────────────
def plot_feature_label_corr(codes_per_window, labels, save_path):
    """
    Clean overview: which features fire differently across labels.

    LEFT  – Butterfly bar: fire-rate for Label 0 (left, blue) vs Label 1
            (right, orange-red), sorted by the fire-rate *difference*
            (Label 1 − Label 0).  Immediately shows label-selective features.
    RIGHT – Horizontal bars of point-biserial correlation, sorted by |r|,
            coloured by sign (blue → Label 0, orange → Label 1), with
            significance stars.
    """
    dict_size = codes_per_window.shape[-1]
    # Binary activation per (window, time-step, feature)
    active = (codes_per_window > 0).float()
    label_arr = labels.numpy().astype(float)
    mean_codes = codes_per_window.mean(dim=1).numpy()  # (N, dict_size)

    # ── per-label fire rates ──
    fire0 = active[labels == 0].mean(dim=(0, 1)).numpy() * 100  # %
    fire1 = active[labels == 1].mean(dim=(0, 1)).numpy() * 100
    diff  = fire1 - fire0  # positive ⇒ more active in Label 1

    # ── correlations ──
    correlations = np.empty(dict_size)
    p_values     = np.empty(dict_size)
    for i in range(dict_size):
        r, p = stats.pointbiserialr(label_arr, mean_codes[:, i])
        correlations[i] = r
        p_values[i]     = p

    c0, c1 = LABEL_COLORS[0], LABEL_COLORS[1]

    fig, (ax_fly, ax_corr) = plt.subplots(1, 2, figsize=(16, 10),
                                           gridspec_kw={"width_ratios": [1, 1]})

    # ────────────────────────────────────────────────────────────────────
    # LEFT: butterfly bar – fire rate by label, sorted by diff
    # ────────────────────────────────────────────────────────────────────
    order = np.argsort(diff)  # most Label-0-selective on top
    y = np.arange(dict_size)

    ax_fly.barh(y, -fire0[order], height=0.8, color=c0, alpha=0.85,
                label="Label 0", edgecolor="white", linewidth=0.3)
    ax_fly.barh(y,  fire1[order], height=0.8, color=c1, alpha=0.85,
                label="Label 1", edgecolor="white", linewidth=0.3)
    ax_fly.axvline(0, color="black", linewidth=0.6)

    # Tick labels: show every feature but keep them small
    ax_fly.set_yticks(y)
    ax_fly.set_yticklabels([f"F{order[i]}" for i in range(dict_size)],
                           fontsize=6)
    ax_fly.set_xlabel("Fire rate (%)", fontsize=11)
    ax_fly.set_title("Fire Rate by Label  (sorted by Label 1 − Label 0)",
                     fontsize=12, fontweight="bold")
    ax_fly.legend(fontsize=10, loc="lower right")
    ax_fly.grid(True, alpha=0.2, axis="x")
    # Symmetric x limits
    xmax = max(fire0.max(), fire1.max()) * 1.1
    ax_fly.set_xlim(-xmax, xmax)
    ax_fly.set_ylim(-0.5, dict_size - 0.5)

    # ────────────────────────────────────────────────────────────────────
    # RIGHT: correlation bars sorted by |r|, coloured by sign
    # ────────────────────────────────────────────────────────────────────
    abs_order = np.argsort(np.abs(correlations))  # smallest at top
    y2 = np.arange(dict_size)

    bar_colors = [c1 if correlations[abs_order[i]] > 0 else c0
                  for i in range(dict_size)]
    ax_corr.barh(y2, correlations[abs_order], height=0.8, color=bar_colors,
                 alpha=0.85, edgecolor="white", linewidth=0.3)
    ax_corr.axvline(0, color="black", linewidth=0.6)

    # Significance stars on the right edge
    for j in range(dict_size):
        fi = abs_order[j]
        p  = p_values[fi]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig:
            rx = correlations[fi]
            offset = 0.008 if rx >= 0 else -0.008
            ax_corr.text(rx + offset, j, sig, va="center",
                         ha="left" if rx >= 0 else "right",
                         fontsize=6, fontweight="bold", color="#333")

    ax_corr.set_yticks(y2)
    ax_corr.set_yticklabels([f"F{abs_order[i]}" for i in range(dict_size)],
                            fontsize=6)
    ax_corr.set_xlabel("Point-biserial r  (label)", fontsize=11)
    ax_corr.set_title("Feature–Label Correlation  (sorted by |r|)",
                      fontsize=12, fontweight="bold")
    ax_corr.grid(True, alpha=0.2, axis="x")
    ax_corr.set_ylim(-0.5, dict_size - 0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Plot 6: Activation Profiles by Label ───────────────────────────────────
def plot_activation_profiles(codes_per_window, labels, save_path):
    """
    Clean activation comparison between labels.

    TOP    – Grouped horizontal bars: mean activation for Label 0 vs Label 1
             for the top 20 most discriminative features (by |correlation|).
    BOTTOM – Scatter plot of per-feature fire rate: Label 0 (x) vs Label 1 (y).
             Features far from the diagonal are label-selective.  Top 10
             most selective are labelled.
    """
    dict_size = codes_per_window.shape[-1]
    mean_codes = codes_per_window.mean(dim=1).numpy()  # (N, dict_size)
    label_arr  = labels.numpy().astype(float)
    active     = (codes_per_window > 0).float()

    c0, c1 = LABEL_COLORS[0], LABEL_COLORS[1]

    # ── per-label stats ──
    mask0 = labels == 0
    mask1 = labels == 1
    mean_act0 = mean_codes[mask0.numpy()].mean(axis=0)
    mean_act1 = mean_codes[mask1.numpy()].mean(axis=0)
    sem0 = mean_codes[mask0.numpy()].std(axis=0) / np.sqrt(mask0.sum().item())
    sem1 = mean_codes[mask1.numpy()].std(axis=0) / np.sqrt(mask1.sum().item())
    fire0 = active[mask0].mean(dim=(0, 1)).numpy() * 100
    fire1 = active[mask1].mean(dim=(0, 1)).numpy() * 100

    # ── correlations for ranking ──
    corrs = np.array([stats.pointbiserialr(label_arr, mean_codes[:, i])[0]
                      for i in range(dict_size)])

    # ── Figure ──
    fig, (ax_bar, ax_scat) = plt.subplots(2, 1, figsize=(14, 14),
                                           gridspec_kw={"height_ratios": [3, 2]})

    # ────────────────────────────────────────────────────────────────────
    # TOP: grouped horizontal bars – top 20 by |correlation|
    # ────────────────────────────────────────────────────────────────────
    n_show = 20
    top_idx = np.argsort(np.abs(corrs))[::-1][:n_show]  # descending |r|
    # Reverse so largest |r| is at the top of the chart
    top_idx = top_idx[::-1]

    y = np.arange(n_show)
    bar_h = 0.35

    ax_bar.barh(y + bar_h / 2, mean_act0[top_idx], bar_h, xerr=sem0[top_idx],
                capsize=3, color=c0, alpha=0.85, label="Label 0",
                edgecolor="white", linewidth=0.3)
    ax_bar.barh(y - bar_h / 2, mean_act1[top_idx], bar_h, xerr=sem1[top_idx],
                capsize=3, color=c1, alpha=0.85, label="Label 1",
                edgecolor="white", linewidth=0.3)

    ax_bar.set_yticks(y)
    feat_labels = []
    for fi in top_idx:
        r = corrs[fi]
        feat_labels.append(f"F{fi}  (r={r:+.2f})")
    ax_bar.set_yticklabels(feat_labels, fontsize=9)
    ax_bar.set_xlabel("Mean activation (± SEM)", fontsize=11)
    ax_bar.set_title("Top 20 Label-Discriminative Features — Mean Activation",
                     fontsize=13, fontweight="bold")
    ax_bar.legend(fontsize=11, loc="lower right")
    ax_bar.grid(True, alpha=0.2, axis="x")

    # ────────────────────────────────────────────────────────────────────
    # BOTTOM: fire-rate scatter – Label 0 vs Label 1
    # ────────────────────────────────────────────────────────────────────
    ax_scat.scatter(fire0, fire1, s=50, c="#555", alpha=0.5, edgecolors="white",
                    linewidths=0.5, zorder=2)

    # Diagonal
    lim = max(fire0.max(), fire1.max()) * 1.1
    ax_scat.plot([0, lim], [0, lim], "--", color="#999", linewidth=1, zorder=1)
    ax_scat.set_xlim(0, lim)
    ax_scat.set_ylim(0, lim)
    ax_scat.set_aspect("equal")

    # Label the 10 most selective features (largest |fire1 - fire0|)
    selectivity = np.abs(fire1 - fire0)
    top_sel = np.argsort(selectivity)[::-1][:10]
    for fi in top_sel:
        col = c1 if fire1[fi] > fire0[fi] else c0
        ax_scat.annotate(
            f"F{fi}", (fire0[fi], fire1[fi]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, fontweight="bold", color=col,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.6))
        ax_scat.scatter([fire0[fi]], [fire1[fi]], s=70, c=col,
                        edgecolors="white", linewidths=0.5, zorder=3)

    ax_scat.set_xlabel("Fire rate — Label 0  (%)", fontsize=11)
    ax_scat.set_ylabel("Fire rate — Label 1  (%)", fontsize=11)
    ax_scat.set_title("Fire-Rate Selectivity  (diagonal = no preference)",
                      fontsize=13, fontweight="bold")
    ax_scat.grid(True, alpha=0.2)

    fig.tight_layout(h_pad=3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═════════════════════════════════════════════════════════════════════════════

def print_feature_summary(codes_per_window, labels):
    """Print a compact summary table of feature statistics."""
    dict_size = codes_per_window.shape[-1]
    codes_flat = codes_per_window.reshape(-1, dict_size)
    mean_codes = codes_per_window.mean(dim=1)
    label_arr = labels.numpy().astype(float)

    print(f"\n{'='*75}")
    print(f"{'Feat':>4}  {'Fire%':>7}  {'MeanAct':>8}  {'MaxAct':>8}  "
          f"{'Corr(label)':>12}  {'p-value':>10}")
    print(f"{'-'*75}")

    for i in range(dict_size):
        col = codes_flat[:, i].numpy()
        fire_rate = (col > 0).mean() * 100
        mean_act = col[col > 0].mean() if (col > 0).any() else 0
        max_act = col.max()
        r, p = stats.pointbiserialr(label_arr, mean_codes[:, i].numpy())
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  F{i}  {fire_rate:>6.1f}%  {mean_act:>8.3f}  {max_act:>8.3f}  "
              f"{r:>+11.4f}  {p:>10.2e} {sig}")

    print(f"{'='*75}\n")


def save_feature_stats(codes_per_window, labels, path):
    """Save feature statistics to JSON."""
    dict_size = codes_per_window.shape[-1]
    codes_flat = codes_per_window.reshape(-1, dict_size)
    mean_codes = codes_per_window.mean(dim=1)
    label_arr = labels.numpy().astype(float)

    stats_list = []
    for i in range(dict_size):
        col = codes_flat[:, i].numpy()
        r, p = stats.pointbiserialr(label_arr, mean_codes[:, i].numpy())
        stats_list.append({
            "feature": i,
            "fire_rate_pct": float((col > 0).mean() * 100),
            "mean_activation": float(col[col > 0].mean()) if (col > 0).any() else 0.0,
            "max_activation": float(col.max()),
            "label_correlation": float(r),
            "label_p_value": float(p),
        })

    with open(path, "w") as f:
        json.dump(stats_list, f, indent=2)
    print(f"  ✓ Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global EMBED, TARGET_LAYER, OUT_DIR, DATA_PATH, FS, S_TOKENS, N_CHAN_TOKENS
    t0 = time.time()

    # ── CLI args ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="SAE feature exploration")
    parser.add_argument(
        "--encoder", default="sleepfm",
        choices=["sleepfm", "reve"],
        help="Encoder backend (default: sleepfm)",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Transformer layer index for SAE (default: last layer)",
    )
    parser.add_argument(
        "--weights-path", default=None,
        help="Path to finetuned checkpoint (REVE .ckpt or SleepFM .pt)",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Run tag (e.g. 'qjbe08') — scopes results to "
             "results/features/{encoder}_{tag}/",
    )
    args = parser.parse_args()

    # ── Load encoder and derive runtime constants ────────────────────────
    print(f"Loading encoder: {args.encoder} …")
    model = load_model(args.encoder, weights_path=args.weights_path)
    model._encoder_name = args.encoder   # attach tag for cache filename
    EMBED = model.embed_dim
    n_layers = len(model.get_hookable_layers())
    TARGET_LAYER = args.layer if args.layer is not None else n_layers - 1

    # ── Select dataset matching encoder's native sample rate ─────────────
    DATA_PATH = _ENCODER_DATA[args.encoder]
    print(f"  dataset → {DATA_PATH.name}")

    # ── Scope results directory to encoder (+ optional tag) ─────────────
    run_name = f"{args.encoder}_{args.tag}" if args.tag else args.encoder
    OUT_DIR = ROOT / "results" / "features" / run_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  embed_dim={EMBED}  target_layer={TARGET_LAYER}  "
          f"results → {OUT_DIR}/")

    print("Loading data...")
    train_loader, val_loader, test_loader = load_data()

    # ── 1. Train SAE ────────────────────────────────────────────────────
    sae, trainer, val_acts = train_sae(model, train_loader, val_loader)

    # ── 2. Collect raw EEG + labels for visualisation ───────────────────
    print("\n=== Collecting raw EEG windows for visualisation ===")
    eeg_windows, labels = collect_raw_eeg_and_labels(val_loader, max_windows=2000)
    print(f"  EEG windows: {eeg_windows.shape}  Labels: {labels.shape}")
    print(f"  Label distribution: {dict(zip(*np.unique(labels.numpy(), return_counts=True)))}")

    # ── 3. Encode through SAE ───────────────────────────────────────────
    print("\n=== Encoding through SAE ===")
    codes_per_window = encode_windows(model, sae, trainer, eeg_windows)
    print(f"  Sparse codes shape: {codes_per_window.shape}")

    # ── Set encoder-specific display constants ───────────────────────────
    if args.encoder == "reve":
        FS = 200
        # REVE tokens are per-channel: (N_CHANNELS × S_time) total per window
        total_tokens = codes_per_window.shape[1]
        N_CHAN_TOKENS = N_CHANNELS        # 19 channel tokens per time step
        S_TOKENS = total_tokens // N_CHAN_TOKENS  # temporal tokens after aggregation
    else:
        FS = 128
        S_TOKENS = codes_per_window.shape[1]
        N_CHAN_TOKENS = 1

    # ── 4. Summary statistics ───────────────────────────────────────────
    print_feature_summary(codes_per_window, labels)
    save_feature_stats(codes_per_window, labels,
                       OUT_DIR / "feature_stats.json")

    # ── 5. Generate all plots ───────────────────────────────────────────
    print("\n=== Generating Visualisations ===")

    # --- Dashboards: top 5 by fire rate ---
    dict_size = codes_per_window.shape[-1]
    codes_flat = codes_per_window.reshape(-1, dict_size)
    fire_rates = (codes_flat > 0).float().mean(dim=0)
    top5_fire = fire_rates.argsort(descending=True)[:5].tolist()

    print("\n  Top 5 features by fire rate:")
    dash_dir = OUT_DIR / "dashboards"
    dash_dir.mkdir(parents=True, exist_ok=True)
    plot_feature_dashboards_set(
        codes_per_window, eeg_windows, labels, dash_dir,
        prefix="01a_fire",
        selection=top5_fire,
        subtitle_fn=lambda f, r: f"rank #{r+1} by fire rate "
                                 f"({fire_rates[f]*100:.1f}%)",
    )

    # --- Dashboards: top 5 correlated with Label 0 (most negative r) ---
    mean_codes = codes_per_window.mean(dim=1).numpy()
    label_arr = labels.numpy().astype(float)
    corrs = np.array([
        stats.pointbiserialr(label_arr, mean_codes[:, i])[0]
        for i in range(dict_size)
    ])

    # Most negative r → feature fires more for label 0
    top5_label0 = np.argsort(corrs)[:5].tolist()
    print("\n  Top 5 features for Label 0 (most negative r):")
    plot_feature_dashboards_set(
        codes_per_window, eeg_windows, labels, dash_dir,
        prefix="01b_label0",
        selection=top5_label0,
        subtitle_fn=lambda f, r: f"Label 0 feature #{r+1}  "
                                 f"(r={corrs[f]:+.3f})",
    )

    # Most positive r → feature fires more for label 1
    top5_label1 = np.argsort(corrs)[::-1][:5].tolist()
    print("\n  Top 5 features for Label 1 (most positive r):")
    plot_feature_dashboards_set(
        codes_per_window, eeg_windows, labels, dash_dir,
        prefix="01c_label1",
        selection=top5_label1,
        subtitle_fn=lambda f, r: f"Label 1 feature #{r+1}  "
                                 f"(r={corrs[f]:+.3f})",
    )

    plot_co_occurrence(codes_per_window,
                       OUT_DIR / "02_co_occurrence.png")

    plot_latent_space(codes_per_window, labels,
                      OUT_DIR / "03_latent_space.png")

    plot_temporal_dynamics(codes_per_window, labels,
                           OUT_DIR / "04_temporal_dynamics.png")

    plot_feature_label_corr(codes_per_window, labels,
                            OUT_DIR / "05_feature_label_corr.png")

    plot_activation_profiles(codes_per_window, labels,
                              OUT_DIR / "06_activation_profiles.png")

    elapsed = time.time() - t0
    print(f"\n✅ All done in {elapsed / 60:.1f} minutes.")
    print(f"   Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
