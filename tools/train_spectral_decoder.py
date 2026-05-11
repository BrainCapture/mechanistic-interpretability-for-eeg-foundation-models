"""
Train SpectralDecoder — Spectral Explain-AE for EEG token interpretability
================================================================

Trains the SpectralDecoder that maps encoder token embeddings to
narrow-band Fourier components (amplitude + phase).

Once trained, any direction in embedding space (e.g. SAE feature
directions) can be decoded into a spectral signature.  Per-band
time-domain waveforms are reconstructed at display-time via
**zero-phase iFFT** from the predicted amplitudes — no temporal
head needed (see ``spectral_to_waveforms()``).

Outputs (in results/spectral_decoder/{encoder}/):
  spectral_decoder_checkpoint.pt            — trained SpectralDecoder model + normalisation stats
  training_curves.png          — loss curves (amplitude, phase, round-trip)
  spectral_sanity_check.png    — predicted vs ground-truth spectra
  r2_distribution.png          — per-token R² histogram + per-band R²

Usage:
  uv run tools/train_spectral_decoder.py --encoder sleepfm
  uv run tools/train_spectral_decoder.py --encoder reve
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project imports ─────────────────────────────────────────────────────────
from mecheeg.spectral_decoder import (SpectralTargetExtractor, TemporalTargetExtractor,
                 SpectralDecoderTrainer, CLINICAL_BANDS)
from mecheeg.dataset import get_dataloaders, StandardizeLabel, V4ResampleTransform
from mecheeg.encoders import load_encoder

# ── paths & constants ───────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
MODEL_PATH  = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

_V2_DIR = ROOT / "checkpoints" / "pretrained" / "SleepFM v2 Models"
_V2_CHECKPOINTS = {
    "sleepfm_v2.0": _V2_DIR / "settransformer_exp0_cl_cnn_sgd_fp32_128d_640p_lr0.001_20260307_113442" / "best.pt",
    "sleepfm_v2.1": _V2_DIR / "settransformer_exp1_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_114250" / "best.pt",
    "sleepfm_v2.3": _V2_DIR / "settransformer_exp2_cl_mae_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_113651" / "best.pt",
    "sleepfm_v2.4": _V2_DIR / "settransformer_exp4_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_210846" / "best.pt",
    "sleepfm_v2.5": _V2_DIR / "settransformer_exp5_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260308_111156" / "best.pt",
    "sleepfm_v2.6": _V2_DIR / "settransformer_exp2.6_res_1sec_cl_cnn_adamw_bf16_128d_128p_lr0.0003_20260322_011957" / "best.pt",
    "sleepfm_v2.7": _V2_DIR / "settransformer_exp2.7_res_1sec_cl_cnn_adamw_bf16_128d_128p_lr0.0003_20260321_161621" / "best.pt",
}

# Shared v2 config — same data/dim as sleepfm but patch_size=640 and 6 layers
_V2_CFG = dict(
    data_path     = ROOT / "data" / "D4-v3-preprocessed-v2",
    embed_dim     = 128,
    fs            = 128,
    patch_size    = 640,       # 5 s patches at 128 Hz (as trained)
    target_layer  = 5,         # last of 6 transformer layers
    max_tokens    = None,
    pool_channels = False,
)

# v2.6 / v2.7 config — 1-second patches (patch_size=128), 6 layers
_V2_128P_CFG = dict(
    data_path     = ROOT / "data" / "D4-v3-preprocessed-v2",
    embed_dim     = 128,
    fs            = 128,
    patch_size    = 128,       # 1 s patches at 128 Hz
    target_layer  = 5,         # last of 6 transformer layers
    max_tokens    = None,
    pool_channels = False,
)

# Per-encoder settings
_ENCODER_CFG = {
    "sleepfm": dict(
        data_path     = ROOT / "data" / "D4-v3-preprocessed-v2",
        embed_dim     = 128,
        fs            = 128,
        patch_size    = 128,       # 1 s at 128 Hz
        target_layer  = 2,         # last of 3 transformer layers
        max_tokens    = None,      # no cap — fits in RAM
        pool_channels = False,     # SleepFM already outputs channel-pooled tokens
    ),
    "sleepfm_v2.0": _V2_CFG,
    "sleepfm_v2.1": _V2_CFG,
    "sleepfm_v2.3": _V2_CFG,
    "sleepfm_v2.4": _V2_CFG,
    "sleepfm_v2.5": _V2_CFG,
    "sleepfm_v2.6": _V2_128P_CFG,
    "sleepfm_v2.7": _V2_128P_CFG,
    "sleepfm_granular": dict(
        data_path     = ROOT / "data" / "D4-v4-preprocessed-10s",
        embed_dim     = 128,
        fs            = 128,
        patch_size    = 128,       # 1 s at 128 Hz (after V4ResampleTransform)
        target_layer  = 2,         # last of 3 transformer layers
        max_tokens    = None,
        pool_channels = False,
    ),
    "reve": dict(
        data_path     = ROOT / "data" / "D4-v3-preprocessed-v1",
        embed_dim     = 512,
        fs            = 200,
        patch_size    = 200,       # 1 s at 200 Hz
        target_layer  = 21,        # last of 22 transformer layers
        max_tokens    = 500_000,   # 500K × 512-dim ≈ 1 GB; full set ≈ 15 GB OOM
        pool_channels = True,      # average 19-ch tokens → 1 per time step
    ),
    "labram": dict(
        data_path     = ROOT / "data" / "D4-v3-preprocessed-v1",
        embed_dim     = 200,
        fs            = 200,
        patch_size    = 200,       # 1 s at 200 Hz
        target_layer  = 11,        # last of 12 transformer blocks
        max_tokens    = 500_000,   # per-channel tokens (19 × 60 per window) — cap memory
        pool_channels = True,      # average 19-ch tokens → 1 per time step
    ),
}

# Shared training hyper-parameters
EPOCHS       = 150      # max epochs (early stopping will cut short)
BATCH_SIZE   = 2048
LR           = 5e-4
HIDDEN_DIM   = 1024
N_BLOCKS     = 4
PHASE_WEIGHT = 0.2      # reduced: phase is noisy, prioritise amplitude
RECON_WEIGHT = 0.05     # light round-trip check
PATIENCE     = 20       # early stopping patience

# ── Temporal head (DISABLED — phase not recoverable from embeddings) ───
TEMPORAL       = False    # zero-phase iFFT from spectral amp is used instead
TEMPORAL_WEIGHT     = 1.0   # (unused when TEMPORAL=False)
TEMPORAL_CORR_WEIGHT = 1.0  # (unused when TEMPORAL=False)


def parse_args():
    p = argparse.ArgumentParser(description="Train SpectralDecoder spectral decoder")
    _all_encoders = ["sleepfm", "sleepfm_v2.0", "sleepfm_v2.1", "sleepfm_v2.3", "sleepfm_v2.4", "sleepfm_v2.5", "sleepfm_v2.6", "sleepfm_v2.7", "sleepfm_granular", "reve", "labram"]
    p.add_argument("--encoder", default="sleepfm", choices=_all_encoders,
                   help="Encoder backend (default: sleepfm)")
    p.add_argument("--layer", type=int, default=None,
                   help="Override target_layer (default: per-encoder config)")
    p.add_argument("--weights-path", default=None,
                   help="Path to finetuned checkpoint (REVE .ckpt or SleepFM .pt)")
    p.add_argument("--tag", default=None,
                   help="Short label appended to the output directory "
                        "(e.g. 'qjbe08' → results/spectral_decoder/reve_qjbe08/). "
                        "Use to keep finetuned runs separate from the base model.")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def remove_module_from_state_dict(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()}


_GRANULAR_CHECKPOINTS = {
    "sleepfm_granular": ROOT / "checkpoints" / "granular" / "sleepfm_granular.ckpt",
}

_LABRAM_CHECKPOINTS = {
    "labram": ROOT / "checkpoints" / "finetuned" / "labram_binary" / "finetuned.ckpt",
}


def load_model(encoder_name: str, weights_path=None):
    if weights_path is not None:
        resolved = weights_path
    elif encoder_name == "sleepfm":
        resolved = MODEL_PATH
    elif encoder_name in _V2_CHECKPOINTS:
        resolved = _V2_CHECKPOINTS[encoder_name]
    elif encoder_name in _GRANULAR_CHECKPOINTS:
        resolved = _GRANULAR_CHECKPOINTS[encoder_name]
    elif encoder_name in _LABRAM_CHECKPOINTS:
        ckpt = _LABRAM_CHECKPOINTS[encoder_name]
        resolved = ckpt if ckpt.exists() else (ROOT / "checkpoints" / "pretrained" / "labram" / "labram-base.pth")
    else:
        resolved = None
    kwargs = {"weights_path": resolved} if resolved is not None else {}
    model = load_encoder(encoder_name, **kwargs)
    model.to(DEVICE).eval()
    print(f"[✓] Loaded {encoder_name} encoder  (embed_dim={model.embed_dim})")
    return model


def load_data(data_path: Path):
    transform = V4ResampleTransform() if "D4-v4" in str(data_path) else StandardizeLabel()
    gen = get_dataloaders(train_path=str(data_path),
                          transformer=transform)
    fold, train_loader, val_loader, test_loader = next(gen)
    print(f"[✓] Loaded data from {data_path.name}  (fold {fold})")
    return train_loader, val_loader, test_loader


# ═════════════════════════════════════════════════════════════════════════════
# Sanity-check plot: predicted vs ground-truth spectra
# ═════════════════════════════════════════════════════════════════════════════

def plot_sanity_check(trainer: SpectralDecoderTrainer, model, val_loader, save_path,
                      target_layer: int, pool_channels: bool = False,
                      n_examples=6):
    """
    Curated sanity-check plot: show best, median, and worst tokens
    with per-example R², residuals, and band-power bar comparison.
    """
    # Collect a decent number of pairs for statistics
    embeddings, targets, _, _ = trainer.collect_pairs(
        model, val_loader, target_layer, max_tokens=3000,
        pool_channels=pool_channels,
    )

    # Normalise & predict
    emb_norm = (embeddings - trainer.embed_mean) / trainer.embed_std
    trainer.spectral_decoder.eval()
    with torch.no_grad():
        pred_norm = trainer.spectral_decoder.decode(emb_norm)
    pred = pred_norm * trainer.target_std + trainer.target_mean

    spectral = trainer.spectral
    freqs = spectral.freqs
    n_bins = spectral.n_bins

    # Compute per-example R²
    gt_amp_all = targets[:, :n_bins]
    pr_amp_all = pred[:, :n_bins]
    per_r2 = []
    for i in range(len(gt_amp_all)):
        ss_res = ((gt_amp_all[i] - pr_amp_all[i]) ** 2).sum().item()
        ss_tot = ((gt_amp_all[i] - gt_amp_all[i].mean()) ** 2).sum().item()
        r2 = 1 - ss_res / (ss_tot + 1e-8) if ss_tot > 0.01 else float("nan")
        per_r2.append(r2)
    per_r2 = np.array(per_r2)

    # Filter out NaN (dead patches)
    valid_mask = ~np.isnan(per_r2)
    valid_r2 = per_r2[valid_mask]
    valid_idx = np.where(valid_mask)[0]

    # Pick curated examples: 2 best, 2 median, 2 worst (among valid)
    sorted_idx = valid_idx[np.argsort(valid_r2)]
    n_valid = len(sorted_idx)
    picks = []
    # 2 best
    picks.extend(sorted_idx[-2:][::-1].tolist())
    # 2 median
    mid = n_valid // 2
    picks.extend(sorted_idx[mid - 1 : mid + 1].tolist())
    # 2 worst
    picks.extend(sorted_idx[:2].tolist())
    labels = ["Best", "Best", "Median", "Median", "Worst", "Worst"]

    n_show = len(picks)
    fig, axes = plt.subplots(n_show, 3, figsize=(20, 3.5 * n_show))

    band_names = list(CLINICAL_BANDS.keys())
    band_colors = {
        "delta": "#2196F3", "theta": "#4CAF50", "alpha": "#FF9800",
        "low-beta": "#F44336", "high-beta": "#9C27B0", "gamma": "#795548",
    }

    for row, (i, label) in enumerate(zip(picks, labels)):
        r2 = per_r2[i]

        gt_amp, gt_cos, gt_sin = spectral.unpack_targets(targets[i:i+1])
        gt_amp_lin = spectral.amplitude_linear(gt_amp).squeeze().numpy()
        gt_amp_log = gt_amp.squeeze().numpy()

        pr_amp, pr_cos, pr_sin = spectral.unpack_targets(pred[i:i+1])
        pr_amp_lin = spectral.amplitude_linear(pr_amp).squeeze().numpy()
        pr_amp_log = pr_amp.squeeze().numpy()

        # ── Col 1: Amplitude spectrum overlay ────────────────────────
        ax = axes[row, 0]
        ax.fill_between(freqs, gt_amp_lin, alpha=0.15, color="black", label="GT")
        ax.plot(freqs, gt_amp_lin, "k-", linewidth=1.5, alpha=0.8)
        ax.plot(freqs, pr_amp_lin, "r-", linewidth=1.5, alpha=0.85, label="SpectralDecoder pred")

        # Shade clinical bands
        for band_name, (lo, hi) in CLINICAL_BANDS.items():
            ax.axvspan(lo, hi, alpha=0.06, color=band_colors[band_name])

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (linear)")
        ax.set_title(f"{label} — Token {i}  (R² = {r2:.4f})",
                      fontweight="bold",
                      color="green" if r2 > 0.9 else ("orange" if r2 > 0.7 else "red"))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.15)

        # ── Col 2: Residual plot ─────────────────────────────────────
        ax2 = axes[row, 1]
        residual = pr_amp_log - gt_amp_log
        ax2.bar(freqs, residual, width=0.8, alpha=0.7,
                color=["#4CAF50" if r > 0 else "#F44336" for r in residual])
        ax2.axhline(0, color="k", linewidth=0.5)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Residual (log-amp)")
        ax2.set_title(f"Residuals — MAE={np.abs(residual).mean():.4f}")
        ax2.grid(True, alpha=0.15)
        ax2.set_ylim(-max(0.5, np.abs(residual).max() * 1.2),
                      max(0.5, np.abs(residual).max() * 1.2))

        # ── Col 3: Band power comparison ─────────────────────────────
        ax3 = axes[row, 2]
        gt_band_pow = []
        pr_band_pow = []
        for band_name in band_names:
            mask = spectral.get_band_mask(band_name)
            gt_band_pow.append(gt_amp_lin[mask].mean() if mask.sum() > 0 else 0)
            pr_band_pow.append(pr_amp_lin[mask].mean() if mask.sum() > 0 else 0)

        x = np.arange(len(band_names))
        w = 0.35
        ax3.bar(x - w/2, gt_band_pow, w, label="Ground truth",
                          color=[band_colors[b] for b in band_names], alpha=0.7)
        ax3.bar(x + w/2, pr_band_pow, w, label="SpectralDecoder prediction",
                          color=[band_colors[b] for b in band_names], alpha=0.4,
                          edgecolor="red", linewidth=1.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(band_names, rotation=30, ha="right", fontsize=8)
        ax3.set_ylabel("Mean amplitude")
        ax3.set_title("Band power comparison")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.15, axis="y")

    fig.suptitle(
        f"SpectralDecoder Sanity Check — Predicted vs Ground-Truth Spectra\n"
        f"Median R² = {np.nanmedian(per_r2):.4f}    "
        f"(valid: {valid_mask.sum()}/{len(per_r2)} tokens, "
        f"{(valid_r2 > 0.8).sum()} with R²>0.8)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")

    # ── Also save R² distribution plot ──────────────────────────────
    r2_plot_path = save_path.parent / "r2_distribution.png"
    fig2, (ax_hist, ax_band) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of per-example R²
    valid_r2_clipped = np.clip(valid_r2, -0.5, 1.0)
    ax_hist.hist(valid_r2_clipped, bins=50, color="#2196F3", alpha=0.75, edgecolor="white")
    ax_hist.axvline(np.median(valid_r2), color="red", linewidth=2,
                     linestyle="--", label=f"Median = {np.median(valid_r2):.4f}")
    ax_hist.axvline(np.mean(valid_r2), color="orange", linewidth=2,
                     linestyle=":", label=f"Mean = {np.mean(valid_r2):.4f}")
    ax_hist.set_xlabel("R² per token", fontsize=12)
    ax_hist.set_ylabel("Count", fontsize=12)
    ax_hist.set_title("Per-Token Reconstruction Quality", fontweight="bold")
    ax_hist.legend(fontsize=10)
    ax_hist.grid(True, alpha=0.15)

    # Right: per-band R²
    band_r2 = {}
    for band_name in CLINICAL_BANDS:
        mask = spectral.get_band_mask(band_name)
        if mask.sum() == 0:
            continue
        gt_b = gt_amp_all[:, mask]
        pr_b = pr_amp_all[:, mask]
        ss_res = ((gt_b - pr_b) ** 2).sum().item()
        ss_tot = ((gt_b - gt_b.mean(0)) ** 2).sum().item()
        band_r2[band_name] = 1 - ss_res / (ss_tot + 1e-8)

    band_colors_list = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#795548"]
    bars = ax_band.bar(band_r2.keys(), band_r2.values(),
                        color=band_colors_list[:len(band_r2)], alpha=0.8,
                        edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, band_r2.values()):
        ax_band.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax_band.set_ylabel("R²", fontsize=12)
    ax_band.set_title("Reconstruction R² by Clinical Band", fontweight="bold")
    ax_band.set_ylim(0, 1.05)
    ax_band.grid(True, alpha=0.15, axis="y")
    ax_band.tick_params(axis="x", rotation=30)

    fig2.suptitle("SpectralDecoder Reconstruction Quality Summary", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(r2_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Saved {r2_plot_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Training loss curves
# ═════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history, save_path):
    """Plot loss components over training epochs."""
    has_temporal = "temporal_mse" in history and len(history["temporal_mse"]) > 0
    n_cols = 4 if has_temporal else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    epochs = range(1, len(history["total"]) + 1)

    ax = axes[0]
    ax.plot(epochs, history["total"], "b-", linewidth=2, label="Total")
    if "val_total" in history and history["val_total"]:
        ax.plot(epochs, history["val_total"], "b--", linewidth=1.5, label="Val Total")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(epochs, history["amp"], "g-", linewidth=2, label="Amplitude MSE")
    if "val_amp" in history and history["val_amp"]:
        ax.plot(epochs, history["val_amp"], "g--", linewidth=1.5, label="Val Amp")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Amplitude Loss")
    ax.legend()
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    ax.plot(epochs, history["phase"], "m-", linewidth=2, label="Phase Cosine")
    if "val_phase" in history and history["val_phase"]:
        ax.plot(epochs, history["val_phase"], "m--", linewidth=1.5, label="Val Phase")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Phase Loss (amplitude-weighted)")
    ax.legend()
    ax.grid(True, alpha=0.2)

    if has_temporal:
        ax = axes[3]
        ax.plot(epochs, history["temporal_corr"], "c-", linewidth=2,
                label="Train Pearson r")
        if "val_temporal_corr" in history and history["val_temporal_corr"]:
            ax.plot(epochs, history["val_temporal_corr"], "c--", linewidth=1.5,
                    label="Val Pearson r")
        ax2 = ax.twinx()
        ax2.plot(epochs, history["temporal_mse"], "r-", linewidth=1.5,
                 alpha=0.6, label="Train MSE")
        if "val_temporal_mse" in history and history["val_temporal_mse"]:
            ax2.plot(epochs, history["val_temporal_mse"], "r--", linewidth=1.5,
                     alpha=0.6, label="Val MSE")
        ax2.set_ylabel("MSE", color="red", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pearson correlation", color="cyan")
        ax.set_title("Temporal Waveform Loss")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
        ax.grid(True, alpha=0.2)

    fig.suptitle("SpectralDecoder Training Curves", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Temporal sanity check: predicted vs ground-truth band waveforms
# ═════════════════════════════════════════════════════════════════════════════

def plot_temporal_sanity_check(trainer: SpectralDecoderTrainer, model, val_loader,
                                save_path, target_layer: int,
                                pool_channels: bool = False, n_examples=4):
    """
    Show predicted vs ground-truth band-pass filtered waveforms for a few
    validation tokens — one row per token, one column per clinical band.
    """
    if trainer.temporal_extractor is None:
        print("  ⚠ Temporal head not enabled — skipping temporal sanity check")
        return

    # Collect pairs
    embeddings, _, temporal_targets, _ = trainer.collect_pairs(
        model, val_loader, target_layer, max_tokens=2000,
        pool_channels=pool_channels,
    )

    # Normalise & predict
    emb_norm = (embeddings - trainer.embed_mean) / trainer.embed_std
    trainer.spectral_decoder.eval()
    with torch.no_grad():
        pred_norm = trainer.spectral_decoder.decode_temporal(emb_norm)
    pred = pred_norm * trainer.temporal_std + trainer.temporal_mean

    band_names = list(CLINICAL_BANDS.keys())
    n_bands = len(band_names)
    T = trainer.temporal_extractor.patch_size
    fs = trainer.spectral.fs
    t_axis = np.arange(T) / fs  # seconds

    band_colors = {
        "delta": "#2196F3", "theta": "#4CAF50", "alpha": "#FF9800",
        "low-beta": "#F44336", "high-beta": "#9C27B0", "gamma": "#795548",
    }

    # Compute per-token, per-band Pearson r
    all_corr = np.zeros((len(embeddings), n_bands))
    for b in range(n_bands):
        gt_b = temporal_targets[:, b * T : (b + 1) * T]
        pr_b = pred[:, b * T : (b + 1) * T]
        gt_z = gt_b - gt_b.mean(dim=-1, keepdim=True)
        pr_z = pr_b - pr_b.mean(dim=-1, keepdim=True)
        num = (gt_z * pr_z).sum(dim=-1)
        den = (gt_z.norm(dim=-1) * pr_z.norm(dim=-1)).clamp(min=1e-8)
        all_corr[:, b] = (num / den).numpy()

    # Mean correlation across bands per token
    mean_corr = np.nanmean(all_corr, axis=1)

    # Pick curated: best, median, worst, random
    valid = ~np.isnan(mean_corr)
    valid_idx = np.where(valid)[0]
    sorted_idx = valid_idx[np.argsort(mean_corr[valid_idx])]
    picks = []
    pick_labels = []
    # Best
    picks.append(sorted_idx[-1])
    pick_labels.append("Best")
    # Median
    picks.append(sorted_idx[len(sorted_idx) // 2])
    pick_labels.append("Median")
    # 25th percentile
    picks.append(sorted_idx[len(sorted_idx) // 4])
    pick_labels.append("25th pct")
    # Worst
    picks.append(sorted_idx[0])
    pick_labels.append("Worst")
    n_show = len(picks)

    fig, axes = plt.subplots(n_show, n_bands, figsize=(3.5 * n_bands, 3 * n_show),
                              squeeze=False)

    for row, (idx, label) in enumerate(zip(picks, pick_labels)):
        gt_bands = trainer.temporal_extractor.unpack_targets(temporal_targets[idx:idx+1])
        pr_bands = trainer.temporal_extractor.unpack_targets(pred[idx:idx+1])

        for col, band_name in enumerate(band_names):
            ax = axes[row, col]
            gt_wave = gt_bands[band_name].squeeze().numpy()
            pr_wave = pr_bands[band_name].squeeze().numpy()
            r = all_corr[idx, col]

            ax.plot(t_axis, gt_wave, "k-", linewidth=1.5, alpha=0.8, label="GT")
            ax.plot(t_axis, pr_wave, "-", color=band_colors[band_name],
                    linewidth=1.5, alpha=0.85, label="SpectralDecoder")

            ax.set_title(f"{band_name} (r={r:.3f})",
                         fontsize=9, fontweight="bold",
                         color="green" if r > 0.8 else ("orange" if r > 0.5 else "red"))

            if row == n_show - 1:
                ax.set_xlabel("Time (s)", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{label}\nToken {idx}\n(avg r={mean_corr[idx]:.3f})",
                              fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.15)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

    # Summary stats
    band_mean_r = np.nanmean(all_corr, axis=0)
    band_summary = "  ".join(f"{b}={r:.3f}" for b, r in zip(band_names, band_mean_r))
    fig.suptitle(
        f"Temporal Sanity Check — Predicted vs Ground-Truth Band Waveforms\n"
        f"Mean Pearson r per band:  {band_summary}\n"
        f"Overall mean r = {np.nanmean(all_corr):.4f}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t0 = time.time()

    # ── Per-encoder config ───────────────────────────────────────────────
    cfg = _ENCODER_CFG[args.encoder]
    DATA_PATH     = cfg["data_path"]
    EMBED         = cfg["embed_dim"]
    FS            = cfg["fs"]
    PATCH_SIZE    = cfg["patch_size"]
    TARGET_LAYER  = args.layer if args.layer is not None else cfg["target_layer"]
    MAX_TOKENS    = cfg["max_tokens"]
    POOL_CHANNELS = cfg["pool_channels"]
    # SleepFM keeps the legacy path so downstream tools keep working.
    # For REVE with a tag or non-default layer, write to a namespaced subfolder.
    run_name = f"{args.encoder}_{args.tag}" if args.tag else args.encoder
    if args.encoder == "sleepfm" and not args.tag:
        OUT_DIR = ROOT / "results" / "spectral_decoder"
    elif args.layer is not None and args.layer != cfg["target_layer"]:
        OUT_DIR = ROOT / "results" / "spectral_decoder" / run_name / f"layer{TARGET_LAYER}"
    else:
        OUT_DIR = ROOT / "results" / "spectral_decoder" / run_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  SpectralDecoder Training — Spectral Decoder for EEG Tokens  [{args.encoder}]")
    print("=" * 70)
    print(f"  encoder={args.encoder}  embed_dim={EMBED}  fs={FS}  "
          f"patch_size={PATCH_SIZE}  target_layer={TARGET_LAYER}")

    # ── Load model & data ───────────────────────────────────────────────
    model = load_model(args.encoder, weights_path=args.weights_path)
    train_loader, val_loader, test_loader = load_data(DATA_PATH)

    # ── Spectral info ───────────────────────────────────────────────────
    spectral = SpectralTargetExtractor(fs=FS, n_fft=PATCH_SIZE)
    print("\n[Spectral Config]")
    print(f"  Frequency bins:  {spectral.n_bins}  "
          f"({spectral.f_min}–{spectral.f_max} Hz)")
    print(f"  Target dim:      {spectral.target_dim}  "
          f"(= {spectral.n_bins} amp + {spectral.n_bins} cos + {spectral.n_bins} sin)")
    print(f"  Clinical bands:  {list(CLINICAL_BANDS.keys())}")
    print(f"  Freq resolution: {spectral.freqs[1] - spectral.freqs[0]:.2f} Hz")

    if TEMPORAL:
        temporal_ext = TemporalTargetExtractor(fs=FS, patch_size=PATCH_SIZE)
        print("\n[Temporal Config]")
        print(f"  Bands:       {temporal_ext.n_bands}  {temporal_ext.BAND_NAMES}")
        print(f"  Patch size:  {temporal_ext.patch_size} samples "
              f"({temporal_ext.patch_size / FS:.1f} s)")
        print(f"  Target dim:  {temporal_ext.target_dim}  "
              f"(= {temporal_ext.n_bands} bands × {temporal_ext.patch_size} samples)")

    # ── Build trainer ───────────────────────────────────────────────────
    trainer = SpectralDecoderTrainer(
        embed_dim=EMBED,
        fs=FS,
        n_fft=PATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        n_blocks=N_BLOCKS,
        lr=LR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        phase_weight=PHASE_WEIGHT,
        recon_weight=RECON_WEIGHT,
        temporal=TEMPORAL,
        temporal_weight=TEMPORAL_WEIGHT,
        temporal_corr_weight=TEMPORAL_CORR_WEIGHT,
        device=DEVICE,
    )

    print("\n[SpectralDecoder Architecture]")
    print(f"  Input:   {EMBED}-dim token embedding")
    print(f"  Hidden:  {HIDDEN_DIM}  (×{N_BLOCKS} residual blocks)")
    print(f"  Output:  {spectral.target_dim}-dim spectral target")
    if TEMPORAL:
        print(f"           {temporal_ext.target_dim}-dim temporal target "
              f"({temporal_ext.n_bands} bands × {temporal_ext.patch_size} samples)")
    n_params = sum(p.numel() for p in trainer.spectral_decoder.parameters())
    print(f"  Params:  {n_params:,}")
    print(f"  Loss weights:  phase={PHASE_WEIGHT}, recon={RECON_WEIGHT}"
          + (f", temporal={TEMPORAL_WEIGHT} (corr={TEMPORAL_CORR_WEIGHT})"
             if TEMPORAL else ""))
    print(f"  Early stopping patience: {PATIENCE}")

    # ── Train ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Training SpectralDecoder for up to {EPOCHS} epochs (patience={PATIENCE})")
    if TEMPORAL:
        print(f"  Temporal head: ENABLED (6 × {PATCH_SIZE} = {6*PATCH_SIZE} dim waveform output)")
    print(f"{'='*70}\n")

    if MAX_TOKENS:
        print(f"  Token cap: {MAX_TOKENS:,} (encoder RAM limit)")
    if POOL_CHANNELS:
        print("  Channel pooling: ENABLED (avg 19-ch tokens → 1 per time step)")
    trainer.train(
        set_transformer=model,
        dataloader=train_loader,
        target_layer=TARGET_LAYER,
        val_dataloader=val_loader,
        patience=PATIENCE,
        max_tokens=MAX_TOKENS,
        pool_channels=POOL_CHANNELS,
    )

    # ── Save checkpoint ─────────────────────────────────────────────────
    ckpt_path = OUT_DIR / "spectral_decoder_checkpoint.pt"
    trainer.save(str(ckpt_path))

    # ── Training curves ─────────────────────────────────────────────────
    if hasattr(trainer, "history") and trainer.history:
        plot_training_curves(
            trainer.history,
            OUT_DIR / "training_curves.png",
        )

    # ── Sanity-check visualisation ──────────────────────────────────────
    print("\n=== Generating sanity-check plots ===")
    plot_sanity_check(
        trainer, model, val_loader,
        OUT_DIR / "spectral_sanity_check.png",
        target_layer=TARGET_LAYER,
        pool_channels=POOL_CHANNELS,
        n_examples=8,
    )

    # ── Temporal sanity check ───────────────────────────────────────────
    if TEMPORAL:
        print("\n=== Generating temporal sanity-check plots ===")
        plot_temporal_sanity_check(
            trainer, model, val_loader,
            OUT_DIR / "temporal_sanity_check.png",
            target_layer=TARGET_LAYER,
            pool_channels=POOL_CHANNELS,
            n_examples=4,
        )

    # ── R² per frequency band (spectral) ───────────────────────────────
    print("\n=== Per-band reconstruction quality (spectral) ===")
    val_embeddings, val_targets, val_temporal, _ = trainer.collect_pairs(
        model, val_loader, TARGET_LAYER, max_tokens=5000,
        pool_channels=POOL_CHANNELS,
    )
    emb_norm = (val_embeddings - trainer.embed_mean) / trainer.embed_std
    trainer.spectral_decoder.eval()
    with torch.no_grad():
        pred_norm = trainer.spectral_decoder.decode(emb_norm)
    pred = pred_norm * trainer.target_std + trainer.target_mean

    gt_amp = val_targets[:, :spectral.n_bins]
    pr_amp = pred[:, :spectral.n_bins]

    print(f"\n  {'Band':<12s}  {'R²':>8s}  {'MAE':>8s}")
    print(f"  {'-'*32}")
    for band_name in CLINICAL_BANDS:
        mask = torch.tensor(spectral.get_band_mask(band_name))
        if mask.sum() == 0:
            continue
        gt_band = gt_amp[:, mask]
        pr_band = pr_amp[:, mask]
        ss_res = ((gt_band - pr_band) ** 2).sum().item()
        ss_tot = ((gt_band - gt_band.mean(0)) ** 2).sum().item()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        mae = (gt_band - pr_band).abs().mean().item()
        print(f"  {band_name:<12s}  {r2:>8.4f}  {mae:>8.4f}")

    # Overall
    ss_res = ((gt_amp - pr_amp) ** 2).sum().item()
    ss_tot = ((gt_amp - gt_amp.mean(0)) ** 2).sum().item()
    r2_all = 1 - ss_res / (ss_tot + 1e-8)
    mae_all = (gt_amp - pr_amp).abs().mean().item()
    print(f"  {'OVERALL':<12s}  {r2_all:>8.4f}  {mae_all:>8.4f}")

    # ── Temporal per-band Pearson r ─────────────────────────────────────
    if TEMPORAL and val_temporal is not None:
        print("\n=== Per-band waveform Pearson r (temporal) ===")
        with torch.no_grad():
            temp_pred_norm = trainer.spectral_decoder.decode_temporal(emb_norm)
        temp_pred = temp_pred_norm * trainer.temporal_std + trainer.temporal_mean
        T = trainer.temporal_extractor.patch_size
        band_names = list(CLINICAL_BANDS.keys())
        print(f"\n  {'Band':<12s}  {'Mean r':>8s}  {'Median r':>10s}")
        print(f"  {'-'*34}")
        for b, band_name in enumerate(band_names):
            gt_b = val_temporal[:, b * T : (b + 1) * T]
            pr_b = temp_pred[:, b * T : (b + 1) * T]
            gt_z = gt_b - gt_b.mean(dim=-1, keepdim=True)
            pr_z = pr_b - pr_b.mean(dim=-1, keepdim=True)
            num = (gt_z * pr_z).sum(dim=-1)
            den = (gt_z.norm(dim=-1) * pr_z.norm(dim=-1)).clamp(min=1e-8)
            corr = (num / den).numpy()
            print(f"  {band_name:<12s}  {np.nanmean(corr):>8.4f}  "
                  f"{np.nanmedian(corr):>10.4f}")
        # Overall
        all_corr = []
        for b in range(len(band_names)):
            gt_b = val_temporal[:, b * T : (b + 1) * T]
            pr_b = temp_pred[:, b * T : (b + 1) * T]
            gt_z = gt_b - gt_b.mean(dim=-1, keepdim=True)
            pr_z = pr_b - pr_b.mean(dim=-1, keepdim=True)
            num = (gt_z * pr_z).sum(dim=-1)
            den = (gt_z.norm(dim=-1) * pr_z.norm(dim=-1)).clamp(min=1e-8)
            all_corr.append((num / den).numpy())
        all_corr = np.concatenate(all_corr)
        print(f"  {'OVERALL':<12s}  {np.nanmean(all_corr):>8.4f}  "
              f"{np.nanmedian(all_corr):>10.4f}")

    elapsed = time.time() - t0
    print(f"\n✅ SpectralDecoder training complete in {elapsed / 60:.1f} minutes.")
    print(f"   Checkpoint: {ckpt_path}")
    print(f"   Results:    {OUT_DIR}/")


if __name__ == "__main__":
    main()
