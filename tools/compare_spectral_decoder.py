"""
compare_spectral_decoder.py — Cross-model SpectralDecoder reconstruction quality comparison
===================================================================

Loads trained SpectralDecoder checkpoints for all SleepFM variants and computes
amplitude and phase reconstruction metrics on the validation split.

Outputs (in results/spectral_decoder/):
  spectral_decoder_comparison_amplitude.png  — Per-band amplitude R² for every model
  spectral_decoder_comparison_phase.png      — Per-band phase cosine similarity for every model
  spectral_decoder_comparison_overall.png    — Overall amplitude R² + phase cosim summary
  spectral_decoder_comparison_metrics.json  — Machine-readable metrics for the Streamlit comparison page

Usage:
  uv run tools/compare_spectral_decoder.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mecheeg.spectral_decoder import SpectralDecoder, SpectralTargetExtractor, SpectralDecoderTrainer, CLINICAL_BANDS
from mecheeg.dataset import get_dataloaders, StandardizeLabel, V4ResampleTransform
from mecheeg.encoders import load_encoder, MODEL_CARDS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model configs ──────────────────────────────────────────────────────────────
_V2_DIR = ROOT / "checkpoints" / "pretrained" / "SleepFM v2 Models"

MODELS = {
    "sleepfm": dict(
        display_name = MODEL_CARDS["sleepfm"]["display_name"],
        spectral_decoder_ckpt     = ROOT / "results" / "spectral_decoder" / "sleepfm" / "spectral_decoder_checkpoint.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        target_layer = 2,
        patch_size   = 128,
        pool_channels= False,
        weights_path = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt",
    ),
    "sleepfm_v2.0": dict(
        display_name = MODEL_CARDS["sleepfm_v2.0"]["display_name"],
        spectral_decoder_ckpt     = ROOT / "results" / "spectral_decoder" / "sleepfm_v2.0" / "spectral_decoder_checkpoint.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        target_layer = 5,
        patch_size   = 640,
        pool_channels= False,
        weights_path = _V2_DIR / "settransformer_exp0_cl_cnn_sgd_fp32_128d_640p_lr0.001_20260307_113442" / "best.pt",
    ),
    "sleepfm_v2.1": dict(
        display_name = MODEL_CARDS["sleepfm_v2.1"]["display_name"],
        spectral_decoder_ckpt     = ROOT / "results" / "spectral_decoder" / "sleepfm_v2.1" / "spectral_decoder_checkpoint.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        target_layer = 5,
        patch_size   = 640,
        pool_channels= False,
        weights_path = _V2_DIR / "settransformer_exp1_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_114250" / "best.pt",
    ),
    "sleepfm_v2.3": dict(
        display_name = MODEL_CARDS["sleepfm_v2.3"]["display_name"],
        spectral_decoder_ckpt     = ROOT / "results" / "spectral_decoder" / "sleepfm_v2.3" / "spectral_decoder_checkpoint.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        target_layer = 5,
        patch_size   = 640,
        pool_channels= False,
        weights_path = _V2_DIR / "settransformer_exp2_cl_mae_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_113651" / "best.pt",
    ),
    "sleepfm_v2.4": dict(
        display_name = MODEL_CARDS["sleepfm_v2.4"]["display_name"],
        spectral_decoder_ckpt     = ROOT / "results" / "spectral_decoder" / "sleepfm_v2.4" / "spectral_decoder_checkpoint.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        target_layer = 5,
        patch_size   = 640,
        pool_channels= False,
        weights_path = _V2_DIR / "settransformer_exp4_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_210846" / "best.pt",
    ),
    "sleepfm_v2.5": dict(
        display_name = MODEL_CARDS["sleepfm_v2.5"]["display_name"],
        spectral_decoder_ckpt     = ROOT / "results" / "spectral_decoder" / "sleepfm_v2.5" / "spectral_decoder_checkpoint.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        target_layer = 5,
        patch_size   = 640,
        pool_channels= False,
        weights_path = _V2_DIR / "settransformer_exp5_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260308_111156" / "best.pt",
    ),
    "sleepfm_v2.6": dict(
        display_name = MODEL_CARDS["sleepfm_v2.6"]["display_name"],
        spectral_decoder_ckpt     = ROOT / "results" / "spectral_decoder" / "sleepfm_v2.6" / "spectral_decoder_checkpoint.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        target_layer = 5,
        patch_size   = 128,
        pool_channels= False,
        weights_path = _V2_DIR / "settransformer_exp2.6_res_1sec_cl_cnn_adamw_bf16_128d_128p_lr0.0003_20260322_011957" / "best.pt",
    ),
    "sleepfm_v2.7": dict(
        display_name = MODEL_CARDS["sleepfm_v2.7"]["display_name"],
        spectral_decoder_ckpt     = ROOT / "results" / "spectral_decoder" / "sleepfm_v2.7" / "spectral_decoder_checkpoint.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        target_layer = 5,
        patch_size   = 128,
        pool_channels= False,
        weights_path = _V2_DIR / "settransformer_exp2.7_res_1sec_cl_cnn_adamw_bf16_128d_128p_lr0.0003_20260321_161621" / "best.pt",
    ),
    "sleepfm_granular": dict(
        display_name = MODEL_CARDS["sleepfm_granular"]["display_name"],
        spectral_decoder_ckpt     = ROOT / "results" / "spectral_decoder" / "sleepfm_granular" / "spectral_decoder_checkpoint.pt",
        data_path    = ROOT / "data" / "D4-v4-preprocessed-10s",
        target_layer = 2,
        patch_size   = 128,
        pool_channels= False,
        weights_path = ROOT / "checkpoints" / "granular" / "sleepfm_granular.ckpt",
    ),
    "reve": dict(
        display_name = MODEL_CARDS["reve"]["display_name"],
        spectral_decoder_ckpt     = ROOT / "results" / "spectral_decoder" / "reve_qjbe08" / "spectral_decoder_checkpoint.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v1",
        target_layer = 21,
        patch_size   = 200,
        pool_channels= True,
        weights_path = ROOT / "checkpoints" / "finetuned" / "reve_qjbe08.ckpt",
    ),
    "labram": dict(
        display_name = MODEL_CARDS["labram"]["display_name"],
        spectral_decoder_ckpt     = ROOT / "results" / "spectral_decoder" / "labram" / "spectral_decoder_checkpoint.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v1",
        target_layer = 11,
        patch_size   = 200,
        pool_channels= True,
        weights_path = ROOT / "checkpoints" / "finetuned" / "labram_binary" / "finetuned.ckpt",
    ),
}

BAND_COLORS = {
    "delta":     "#2196F3",
    "theta":     "#4CAF50",
    "alpha":     "#FF9800",
    "low-beta":  "#F44336",
    "high-beta": "#9C27B0",
    "gamma":     "#795548",
}

MAX_TOKENS = 5_000   # enough for stable statistics without OOM


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_spectral_decoder(cfg: dict) -> SpectralDecoderTrainer:
    """Restore a trained SpectralDecoderTrainer from checkpoint (weights + normalisation)."""
    ckpt_path = cfg["spectral_decoder_ckpt"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SpectralDecoder checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sc   = ckpt["spectral_config"]
    mc   = ckpt["model_config"]

    trainer = SpectralDecoderTrainer(
        embed_dim   = mc["embed_dim"],
        fs          = sc["fs"],
        n_fft       = sc["n_fft"],
        f_min       = sc.get("f_min", 0.5),
        f_max       = sc.get("f_max", None),
        hidden_dim  = mc.get("hidden_dim", None),
        n_blocks    = mc.get("n_blocks", 3),
        device      = DEVICE,
    )
    trainer.spectral_decoder.load_state_dict(ckpt["spectral_decoder_state_dict"])
    trainer.spectral_decoder.eval().to(DEVICE)
    trainer.embed_mean  = ckpt["embed_mean"].to(DEVICE)
    trainer.embed_std   = ckpt["embed_std"].to(DEVICE)
    trainer.target_mean = ckpt["target_mean"].to(DEVICE)
    trainer.target_std  = ckpt["target_std"].to(DEVICE)
    return trainer


def _compute_metrics(
    trainer: SpectralDecoderTrainer,
    encoder,
    val_loader,
    target_layer: int,
    pool_channels: bool,
) -> dict:
    """
    Collect val pairs, predict, and return amplitude R² + phase cosine sim,
    both overall and per clinical band.
    """
    embeddings, targets, _, _ = trainer.collect_pairs(
        encoder, val_loader, target_layer,
        max_tokens=MAX_TOKENS, pool_channels=pool_channels,
    )

    spectral = trainer.spectral
    n_bins   = spectral.n_bins

    embeddings = embeddings.to(DEVICE)
    emb_norm = (embeddings - trainer.embed_mean) / trainer.embed_std
    with torch.no_grad():
        pred_norm = trainer.spectral_decoder.decode(emb_norm)
    pred = (pred_norm * trainer.target_std + trainer.target_mean).cpu()
    targets = targets.cpu()

    gt_amp   = targets[:, :n_bins]          # (N, n_bins)  log-amplitude
    pr_amp   = pred[:, :n_bins]
    gt_cos   = targets[:, n_bins:2*n_bins]  # (N, n_bins)
    pr_cos   = pred[:, n_bins:2*n_bins]
    gt_sin   = targets[:, 2*n_bins:3*n_bins]
    pr_sin   = pred[:, 2*n_bins:3*n_bins]

    # Overall amplitude R²
    ss_res  = ((gt_amp - pr_amp) ** 2).sum().item()
    ss_tot  = ((gt_amp - gt_amp.mean(0)) ** 2).sum().item()
    amp_r2  = float(1 - ss_res / (ss_tot + 1e-8))

    # Phase cosine similarity: cos(φ_pred − φ_true) = cos·cos + sin·sin
    # Normalise predicted (cos, sin) back to unit circle first
    pr_norm = (pr_cos ** 2 + pr_sin ** 2).sqrt().clamp(min=1e-8)
    pr_cos_u = pr_cos / pr_norm
    pr_sin_u = pr_sin / pr_norm
    phase_cosim = (pr_cos_u * gt_cos + pr_sin_u * gt_sin).mean().item()

    # Per-band metrics
    band_amp_r2   = {}
    band_phase_cosim = {}
    for band_name in CLINICAL_BANDS:
        mask = spectral.get_band_mask(band_name)
        if mask.sum() == 0:
            continue
        g_b = gt_amp[:, mask]
        p_b = pr_amp[:, mask]
        ss_r = ((g_b - p_b) ** 2).sum().item()
        ss_t = ((g_b - g_b.mean(0)) ** 2).sum().item()
        band_amp_r2[band_name] = float(1 - ss_r / (ss_t + 1e-8))

        g_cos = gt_cos[:, mask]; g_sin = gt_sin[:, mask]
        p_cos = pr_cos_u[:, mask]; p_sin = pr_sin_u[:, mask]
        band_phase_cosim[band_name] = (p_cos * g_cos + p_sin * g_sin).mean().item()

    return dict(
        amp_r2=amp_r2,
        phase_cosim=phase_cosim,
        band_amp_r2=band_amp_r2,
        band_phase_cosim=band_phase_cosim,
    )


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_band_comparison(
    results: dict[str, dict],
    metric_key: str,       # "band_amp_r2" or "band_phase_cosim"
    ylabel: str,
    title: str,
    save_path: Path,
    ylim: tuple = (0, 1),
    chance_line: float | None = None,
):
    band_names = list(CLINICAL_BANDS.keys())
    model_keys = list(results.keys())
    n_models   = len(model_keys)
    n_bands    = len(band_names)

    x = np.arange(n_bands)
    bar_w = 0.8 / n_models

    cmap = plt.cm.get_cmap("tab10")
    model_colors = {k: cmap(i) for i, k in enumerate(model_keys)}

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model_key in enumerate(model_keys):
        metrics = results[model_key]
        vals = [metrics[metric_key].get(b, float("nan")) for b in band_names]
        offset = (i - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w,
                      label=results[model_key]["display_name"],
                      color=model_colors[model_key], alpha=0.85,
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold",
                        color=model_colors[model_key])

    if chance_line is not None:
        ax.axhline(chance_line, color="gray", linewidth=1, linestyle="--",
                   label=f"Chance ({chance_line:.1f})")

    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("-", "\n") for b in band_names], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_axisbelow(True)

    # Background band shading
    for j, band_name in enumerate(band_names):
        ax.axvspan(j - 0.5, j + 0.5, alpha=0.04,
                   color=BAND_COLORS.get(band_name, "gray"))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


def _plot_overall_summary(results: dict[str, dict], save_path: Path):
    model_keys   = list(results.keys())
    display_names = [results[k]["display_name"] for k in model_keys]
    amp_vals     = [results[k]["amp_r2"]      for k in model_keys]
    phase_vals   = [results[k]["phase_cosim"] for k in model_keys]

    x = np.arange(len(model_keys))
    w = 0.35
    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(model_keys))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Amplitude R²
    bars = ax1.bar(x, amp_vals, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, amp_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{v:.4f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("R²", fontsize=12)
    ax1.set_title("Overall Amplitude R²", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.2, axis="y")
    ax1.set_axisbelow(True)

    # Phase cosine similarity
    bars2 = ax2.bar(x, phase_vals, color=colors, alpha=0.85,
                    edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars2, phase_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{v:.4f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax2.axhline(0, color="gray", linewidth=1, linestyle="--", label="Chance (0.0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Mean cosine similarity", fontsize=12)
    ax2.set_title("Overall Phase Reconstruction\n(cosine sim; 0 = chance, 1 = perfect)",
                  fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.2, 1.0)
    ax2.grid(True, alpha=0.2, axis="y")
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9)

    fig.suptitle("SpectralDecoder Reconstruction Comparison — All SleepFM Variants",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    out_dir = ROOT / "results" / "spectral_decoder"
    results: dict[str, dict] = {}

    for model_key, cfg in MODELS.items():
        display = cfg["display_name"]
        print(f"\n{'='*60}")
        print(f"  {display}  ({model_key})")
        print(f"{'='*60}")

        if not cfg["spectral_decoder_ckpt"].exists():
            print(f"  SKIP — checkpoint not found: {cfg['spectral_decoder_ckpt']}")
            continue

        # Load encoder
        print("  Loading encoder…", flush=True)
        wp = cfg["weights_path"]
        encoder = load_encoder(model_key, weights_path=str(wp) if wp is not None else None)
        encoder.to(DEVICE).eval()

        # Load SpectralDecoder
        print("  Loading SpectralDecoder checkpoint…", flush=True)
        trainer = _load_spectral_decoder(cfg)

        # Load validation data
        print("  Loading validation data…", flush=True)
        transform = V4ResampleTransform() if "D4-v4" in str(cfg["data_path"]) else StandardizeLabel()
        gen = get_dataloaders(
            train_path=str(cfg["data_path"]),
            transformer=transform,
        )
        _, _, val_loader, _ = next(gen)

        # Compute metrics
        print(f"  Computing metrics (max {MAX_TOKENS:,} tokens)…", flush=True)
        metrics = _compute_metrics(
            trainer, encoder, val_loader,
            target_layer=cfg["target_layer"],
            pool_channels=cfg["pool_channels"],
        )
        metrics["display_name"] = display

        print(f"  → Amplitude R²:          {metrics['amp_r2']:.4f}")
        print(f"  → Phase cosim (overall): {metrics['phase_cosim']:.4f}")
        for band, v in metrics["band_amp_r2"].items():
            pc = metrics["band_phase_cosim"].get(band, float("nan"))
            print(f"       {band:<12} amp R²={v:.4f}   phase cosim={pc:.4f}")

        results[model_key] = metrics

        # Free GPU memory between models
        del encoder, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        print("\nNo models found — nothing to plot.")
        return

    print(f"\n{'='*60}")
    print("  Generating comparison plots…")

    _plot_overall_summary(
        results,
        save_path=out_dir / "spectral_decoder_comparison_overall.png",
    )
    _plot_band_comparison(
        results,
        metric_key="band_amp_r2",
        ylabel="R²",
        title="SpectralDecoder Amplitude Reconstruction R² by Clinical Band",
        save_path=out_dir / "spectral_decoder_comparison_amplitude.png",
        ylim=(0, 1.0),
    )
    _plot_band_comparison(
        results,
        metric_key="band_phase_cosim",
        ylabel="Cosine similarity (1 = perfect, 0 = chance)",
        title="SpectralDecoder Phase Reconstruction Quality by Clinical Band",
        save_path=out_dir / "spectral_decoder_comparison_phase.png",
        ylim=(-0.3, 1.0),
        chance_line=0.0,
    )

    # Save metrics as JSON for the Streamlit comparison page
    json_path = out_dir / "spectral_decoder_comparison_metrics.json"
    serialisable = {
        k: {
            "display_name": v["display_name"],
            "amp_r2":       v["amp_r2"],
            "phase_cosim":  v["phase_cosim"],
            "band_amp_r2":       v["band_amp_r2"],
            "band_phase_cosim":  v["band_phase_cosim"],
        }
        for k, v in results.items()
    }
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  ✓ Saved {json_path}")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
