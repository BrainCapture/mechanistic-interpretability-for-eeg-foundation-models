"""
SAE Hyperparameter Sweep & Visualization
==========================================
Trains SAEs across an expansion × k grid (expansion ∈ {1.0, 1.5, 2.0, 4.0},
k ∈ {1, 2, 4, 8, 16, 32, 48, 64, 96}), evaluates reconstruction R² and dead neuron
percentage on held-out validation data, and generates publication-quality plots.

Use the sweep results to identify the best configuration, then run
explore_features.py (which trains and saves the SAE checkpoint).

Outputs (in results/sae/{encoder}/):
  sweep_results.json         – R², dead %, sparsity for every (expansion, k) pair
  01_r2_vs_k.png             – R² vs k, one curve per expansion factor
  02_dead_vs_k.png           – dead neuron % vs k
  03_pareto_r2_vs_dead.png   – Pareto front: R² vs dead neuron %
  05_heatmap_r2.png          – heatmap of R² across the grid
  06_heatmap_dead.png        – heatmap of dead neuron % across the grid

Usage:
  uv run tools/sweep_and_viz.py                  # defaults to SleepFM
  uv run tools/sweep_and_viz.py --encoder reve   # REVE encoder
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
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, TensorDataset

# ── project imports ─────────────────────────────────────────────────────────
from sae4eeg.sae import SAETrainer, SparseAutoencoder, sae_diagnostics
from sae4eeg.dataset import get_dataloaders, StandardizeLabel
from sae4eeg.encoders import load_encoder

# ── paths & constants ───────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
MODEL_PATH  = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"

# Dataset versions keyed by encoder name:
#   v1 (200 Hz) → REVE native rate
#   v2 (128 Hz) → SleepFM native rate
_ENCODER_DATA = {
    "sleepfm": ROOT / "data" / "D4-v3-preprocessed-v2",
    "reve":    ROOT / "data" / "D4-v3-preprocessed-v1",
}
DATA_PATH = _ENCODER_DATA["sleepfm"]  # overridden at runtime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Runtime values — overridden in main() once the encoder is loaded
EMBED        = 128   # overridden at runtime from encoder.embed_dim
RESULTS_DIR  = ROOT / "results" / "sae" / "sleepfm"  # overridden at runtime

# ── full-training sweep grid ────────────────────────────────────────────────
EXPANSION_VALUES = [1.0, 1.5, 2.0, 4.0]
K_VALUES         = [1, 2, 4, 8, 16, 32, 48, 64, 96]
EPOCHS           = 20
BATCH_SIZE       = 256
RESAMPLE_EVERY   = 2
TARGET_LAYER     = -1      # overridden at runtime: last layer of encoder

# ── historical results from the 4 manual iterations ────────────────────────
HISTORICAL = [
    {"label": "Iter 1\nexp=4 k=32\nresamp=5",   "expansion": 4.0, "k": 32,
     "l0": 32.0, "dead_frac": 0.49, "r2": 0.975,
     "epochs": 20, "resample_every": 5},
    {"label": "Iter 2\nexp=2 k=32\nresamp=5",   "expansion": 2.0, "k": 32,
     "l0": 32.0, "dead_frac": 0.238, "r2": 0.975,
     "epochs": 20, "resample_every": 5},
    {"label": "Iter 3\nexp=2 k=32\nresamp=2",   "expansion": 2.0, "k": 32,
     "l0": 32.0, "dead_frac": 0.137, "r2": 0.975,
     "epochs": 20, "resample_every": 2},
    {"label": "Iter 4\nexp=1.5 k=48\nresamp=2", "expansion": 1.5, "k": 48,
     "l0": 48.0, "dead_frac": 0.073, "r2": 0.986,
     "epochs": 20, "resample_every": 2},
]

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

COLORS = list(mcolors.TABLEAU_COLORS.values())

def _exp_color(exp):
    idx = EXPANSION_VALUES.index(exp) if exp in EXPANSION_VALUES else 0
    return COLORS[idx % len(COLORS)]


def remove_module_from_state_dict(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()}


def load_model(encoder_name: str = "sleepfm"):
    """Load encoder by name. Returns (encoder, embed_dim, n_layers)."""
    kwargs = {}
    if encoder_name == "sleepfm":
        kwargs["weights_path"] = MODEL_PATH
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
    return train_loader, val_loader


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Full-training sweep
# ═════════════════════════════════════════════════════════════════════════════

def run_full_training_grid(model, train_loader, val_loader,
                           embed_dim: int = EMBED,
                           target_layer: int = TARGET_LAYER,
                           max_tokens: int = None):
    """
    For each (expansion, k), do a full 20-epoch SAE training with
    dead neuron resampling, then evaluate on validation data.
    Returns list of dicts with all metrics.
    """
    all_results = []

    # Collect activations once (independent of expansion/k)
    print("=== Collecting training activations (once for all configs) ===")
    collector = SAETrainer(
        embed_dim=embed_dim, expansion=1, mode="topk", k=8,
        target_layers=[target_layer],
    )
    train_acts_raw = collector.collect_activations(model, train_loader,
                                                   max_tokens=max_tokens)
    train_acts_layer = train_acts_raw[target_layer]

    # Normalise once
    act_mean = train_acts_layer.mean(dim=0)
    act_std  = train_acts_layer.std(dim=0).clamp(min=1e-8)
    train_acts = (train_acts_layer - act_mean) / act_std
    print(f"  Training tokens: {train_acts.shape[0]}, dim: {train_acts.shape[1]}")

    # Collect validation activations
    print("=== Collecting validation activations ===")
    val_max = max_tokens // 5 if max_tokens else None
    val_acts_raw = collector.collect_activations(model, val_loader,
                                                 max_tokens=val_max)
    val_acts = (val_acts_raw[target_layer] - act_mean) / act_std
    print(f"  Validation tokens: {val_acts.shape[0]}")

    n_configs = sum(1 for exp in EXPANSION_VALUES
                    for k in K_VALUES if k < int(embed_dim * exp))
    cfg_idx = 0

    for exp in EXPANSION_VALUES:
        dict_size = int(embed_dim * exp)
        valid_k = [k for k in K_VALUES if k < dict_size]
        if not valid_k:
            print(f"\n⚠ exp={exp} (dict={dict_size}) too small, skipping")
            continue

        for k_val in valid_k:
            cfg_idx += 1
            print(f"\n{'='*60}")
            print(f"  [{cfg_idx}/{n_configs}]  exp={exp}  dict={dict_size}  "
                  f"k={k_val}  (k/dict={k_val/dict_size*100:.1f}%)")
            print(f"{'='*60}")

            # Build & train SAE
            sae = SparseAutoencoder(
                input_dim=embed_dim, expansion=exp, mode="topk", k=k_val,
            ).to(DEVICE)
            sae.init_b_pre(train_acts.mean(dim=0))

            opt = torch.optim.Adam(sae.parameters(), lr=3e-4)
            loader = DataLoader(
                TensorDataset(train_acts), batch_size=BATCH_SIZE, shuffle=True
            )

            total_fires = torch.zeros(sae.dict_size, device=DEVICE)

            for epoch in range(EPOCHS):
                epoch_loss = 0.0
                epoch_fires = torch.zeros(sae.dict_size, device=DEVICE)

                for (x,) in loader:
                    x = x.to(DEVICE)
                    loss, recon, sparse, z = sae.loss(x)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sae._normalise_decoder()
                    epoch_loss += loss.item()
                    epoch_fires += (z.detach() > 0).float().sum(dim=0)

                total_fires += epoch_fires
                n_batches = len(loader)
                n_dead_ep = (epoch_fires == 0).sum().item()
                print(f"  Epoch {epoch+1:2d}/{EPOCHS} | "
                      f"loss={epoch_loss/n_batches:.4f}  "
                      f"dead_epoch={n_dead_ep}/{sae.dict_size}")

                # Dead neuron resampling
                if (RESAMPLE_EVERY > 0
                        and (epoch + 1) % RESAMPLE_EVERY == 0
                        and epoch + 1 < EPOCHS):
                    dead_mask = epoch_fires == 0
                    n_dead = dead_mask.sum().item()
                    if n_dead > 0:
                        _resample(sae, train_acts, dead_mask, opt)
                        print(f"    ↳ Resampled {n_dead} dead neurons")

            # Evaluate on validation set
            sae_cpu = sae.cpu()
            diag = sae_diagnostics(sae_cpu, val_acts, device=DEVICE)

            result = {
                "expansion": exp,
                "dict_size": dict_size,
                "k": k_val,
                "l0": diag["l0"],
                "dead_frac": diag["dead_frac"],
                "r2": diag["r2"],
            }
            all_results.append(result)
            print(f"  ▸ VAL  L0={diag['l0']:.1f}  "
                  f"Dead={diag['dead_frac']*100:.1f}%  R²={diag['r2']:.4f}")

    return all_results


@torch.no_grad()
def _resample(sae, acts, dead_mask, opt):
    """Minimal Anthropic-style dead neuron resampling."""
    n_dead = dead_mask.sum().item()
    sample_size = min(len(acts), 8192)
    idx = torch.randperm(len(acts))[:sample_size]
    x_sample = acts[idx].to(sae.encoder.weight.device)
    x_hat, _ = sae(x_sample)
    losses = (x_sample - x_hat).pow(2).sum(dim=-1)

    probs = losses / losses.sum()
    replace_idx = torch.multinomial(probs, n_dead, replacement=True)
    replacement_dirs = x_sample[replace_idx] - sae.b_pre
    norms = replacement_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    replacement_dirs = replacement_dirs / norms
    replacement_dirs += 1e-3 * torch.randn_like(replacement_dirs)

    dead_indices = dead_mask.nonzero(as_tuple=True)[0]
    alive_enc_norm = sae.encoder.weight[~dead_mask].norm(dim=-1).mean()
    sae.encoder.weight[dead_indices] = replacement_dirs * alive_enc_norm * 0.8
    sae.encoder.bias[dead_indices] = 0.0
    sae.decoder.weight[:, dead_indices] = replacement_dirs.T
    sae._normalise_decoder()

    for param in [sae.encoder.weight, sae.encoder.bias,
                  sae.decoder.weight, sae.decoder.bias]:
        state = opt.state.get(param)
        if state:
            if 'exp_avg' in state:
                state['exp_avg'].zero_()
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'].zero_()


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Visualization functions
# ═════════════════════════════════════════════════════════════════════════════

# ── Plot 1: R² vs k ────────────────────────────────────────────────────────
def plot_r2_vs_k(results, save_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for exp in EXPANSION_VALUES:
        pts = sorted([r for r in results if r["expansion"] == exp],
                     key=lambda r: r["k"])
        if not pts:
            continue
        ks  = [r["k"] for r in pts]
        r2s = [r["r2"] for r in pts]
        ax.plot(ks, r2s, "o-", color=_exp_color(exp), linewidth=2,
                label=f"exp={exp}× (dict={int(EMBED*exp)})", markersize=8)

    # Historical full runs as stars
    # for h in HISTORICAL:
    #     ax.plot(h["k"], h["r2"], "*", color=_exp_color(h["expansion"]),
    #             markersize=18, markeredgecolor="black", markeredgewidth=0.8,
    #             zorder=5)

    ax.axhline(0.95, color="green", ls="--", alpha=0.5, linewidth=1,
               label="R² = 0.95 target")
    ax.set_xlabel("k  (top-k active features per token)", fontsize=13)
    ax.set_ylabel("R²  (reconstruction quality)", fontsize=13)
    ax.set_title("Reconstruction Quality vs Sparsity\n"
                 "(20 epochs, full training, val-set evaluation)",
                 fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Plot 2: Dead% vs k (improved) ──────────────────────────────────────────
def plot_dead_vs_k(results, save_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for exp in EXPANSION_VALUES:
        pts = sorted([r for r in results if r["expansion"] == exp],
                     key=lambda r: r["k"])
        if not pts:
            continue
        ks   = [r["k"] for r in pts]
        dead = [r["dead_frac"] * 100 for r in pts]
        ax.plot(ks, dead, "s-", color=_exp_color(exp), linewidth=2,
                label=f"exp={exp}× (dict={int(EMBED*exp)})", markersize=8)
        # Annotate each point
        for k, d in zip(ks, dead):
            ax.annotate(f"{d:.0f}%", (k, d), fontsize=7,
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", color=_exp_color(exp), fontweight="bold")

    # Historical runs
    for h in HISTORICAL:
        ax.plot(h["k"], h["dead_frac"] * 100, "*",
                color=_exp_color(h["expansion"]),
                markersize=18, markeredgecolor="black", markeredgewidth=0.8,
                zorder=5)

    # Target region shading
    ax.axhspan(0, 10, color="green", alpha=0.08, label="Target zone (≤10%)")
    ax.axhline(10, color="red", ls="--", alpha=0.6, linewidth=1.5,
               label="10% target")

    ax.set_xlabel("k  (top-k active features per token)", fontsize=13)
    ax.set_ylabel("Dead neurons  (%)", fontsize=13)
    ax.set_title("Dead Neuron Fraction vs Sparsity\n"
                 "(20 epochs, with resampling every 2 epochs)",
                 fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Plot 3: Pareto frontier (R² vs Dead%) ──────────────────────────────────
def plot_pareto(results, save_path):
    fig, ax = plt.subplots(figsize=(9, 6))

    for exp in EXPANSION_VALUES:
        pts = [r for r in results if r["expansion"] == exp]
        if not pts:
            continue
        dead = [r["dead_frac"] * 100 for r in pts]
        r2   = [r["r2"] for r in pts]
        ax.scatter(dead, r2, s=80, color=_exp_color(exp),
                   label=f"exp={exp}×", zorder=3, edgecolors="white",
                   linewidth=0.5)
        for r in pts:
            ax.annotate(f"k={r['k']}", (r["dead_frac"]*100, r["r2"]),
                        fontsize=7, textcoords="offset points",
                        xytext=(6, 4), alpha=0.85)

    # for h in HISTORICAL:
    #     ax.plot(h["dead_frac"]*100, h["r2"], "*",
    #             color=_exp_color(h["expansion"]),
    #             markersize=20, markeredgecolor="black", markeredgewidth=0.8,
    #             zorder=5)
    #     short = h["label"].replace("\n", " ")
    #     ax.annotate(short, (h["dead_frac"]*100, h["r2"]),
    #                 fontsize=7, textcoords="offset points",
    #                 xytext=(10, -6), alpha=0.9, fontweight="bold")

    # Target region
    ax.axvspan(0, 10, color="green", alpha=0.06)
    ax.axhspan(0.95, 1.0, color="green", alpha=0.06)
    ax.axvline(10, color="red", ls="--", alpha=0.4, label="Dead ≤ 10%")
    ax.axhline(0.95, color="green", ls="--", alpha=0.4, label="R² ≥ 0.95")

    ax.set_xlabel("Dead neurons  (%)", fontsize=13)
    ax.set_ylabel("R²", fontsize=13)
    ax.set_title("Pareto Trade-off: Quality vs Dead Neurons", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Plot 4: k/dict ratio vs Dead% ──────────────────────────────────────────
def plot_ratio_vs_dead(results, save_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for exp in EXPANSION_VALUES:
        pts = sorted([r for r in results if r["expansion"] == exp],
                     key=lambda r: r["k"])
        if not pts:
            continue
        ratio = [r["k"] / r["dict_size"] * 100 for r in pts]
        dead  = [r["dead_frac"] * 100 for r in pts]
        ax.plot(ratio, dead, "D-", color=_exp_color(exp), linewidth=2,
                label=f"exp={exp}×", markersize=8)

    for h in HISTORICAL:
        dsize = int(EMBED * h["expansion"])
        ax.plot(h["k"] / dsize * 100, h["dead_frac"] * 100, "*",
                color=_exp_color(h["expansion"]),
                markersize=18, markeredgecolor="black", markeredgewidth=0.8,
                zorder=5)

    ax.axhspan(0, 10, color="green", alpha=0.08)
    ax.axhline(10, color="red", ls="--", alpha=0.6, linewidth=1.5,
               label="10% target")
    ax.set_xlabel("k / dict_size  (% of dictionary activated)", fontsize=13)
    ax.set_ylabel("Dead neurons  (%)", fontsize=13)
    ax.set_title("Dead Neurons vs Capacity Utilisation Ratio", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Plot 5: Heatmap of R² (adaptive coloring) ──────────────────────────────
def plot_heatmap_r2(results, save_path):
    exp_vals = sorted(set(r["expansion"] for r in results))
    k_vals   = sorted(set(r["k"] for r in results))

    mat = np.full((len(exp_vals), len(k_vals)), np.nan)
    for r in results:
        i = exp_vals.index(r["expansion"])
        j = k_vals.index(r["k"])
        mat[i, j] = r["r2"]

    # Adaptive colour range based on actual data
    valid = mat[~np.isnan(mat)]
    vmin = max(0.0, np.floor(valid.min() * 20) / 20)   # round down to 0.05
    vmax = min(1.0, np.ceil(valid.max() * 20) / 20)     # round up to 0.05
    if vmax - vmin < 0.05:
        vmin = vmax - 0.10

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                   vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels(k_vals, fontsize=11)
    ax.set_yticks(range(len(exp_vals)))
    ax.set_yticklabels([f"{e}×" for e in exp_vals], fontsize=11)
    ax.set_xlabel("k", fontsize=13)
    ax.set_ylabel("Expansion factor", fontsize=13)
    ax.set_title("R² Heatmap  (expansion × k)  —  full 20-epoch training",
                 fontsize=14)

    midpoint = (vmin + vmax) / 2
    for i in range(len(exp_vals)):
        for j in range(len(k_vals)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if v < midpoint else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("R²", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Plot 6: Heatmap of Dead% (improved) ────────────────────────────────────
def plot_heatmap_dead(results, save_path):
    exp_vals = sorted(set(r["expansion"] for r in results))
    k_vals   = sorted(set(r["k"] for r in results))

    mat = np.full((len(exp_vals), len(k_vals)), np.nan)
    for r in results:
        i = exp_vals.index(r["expansion"])
        j = k_vals.index(r["k"])
        mat[i, j] = r["dead_frac"] * 100

    valid = mat[~np.isnan(mat)]
    vmax_val = min(100, np.ceil(valid.max() / 5) * 5)
    if vmax_val < 15:
        vmax_val = 15

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=vmax_val, origin="lower")
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels(k_vals, fontsize=11)
    ax.set_yticks(range(len(exp_vals)))
    ax.set_yticklabels([f"{e}×" for e in exp_vals], fontsize=11)
    ax.set_xlabel("k", fontsize=13)
    ax.set_ylabel("Expansion factor", fontsize=13)
    ax.set_title("Dead Neuron % Heatmap  (expansion × k)  —  full 20-epoch training",
                 fontsize=14)

    for i in range(len(exp_vals)):
        for j in range(len(k_vals)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if v > vmax_val * 0.6 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Dead %", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Plot 7: Feature activation histogram (best checkpoint) ─────────────────
def plot_feature_histogram(model, val_loader, save_path):
    ckpt_path = ROOT / "sae_mvp_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"  ⚠ Checkpoint not found at {ckpt_path}, skipping")
        return

    trainer = SAETrainer(
        embed_dim=EMBED, expansion=1.5, mode="topk", k=48,
        target_layers=[1, 2] if EMBED == 128 else [TARGET_LAYER],
    )
    trainer.load(str(ckpt_path))

    val_acts = trainer.collect_activations(model, val_loader)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, layer_idx in enumerate(sorted(trainer.saes.keys())):
        sae = trainer.saes[layer_idx]
        acts = val_acts[layer_idx]
        if layer_idx in trainer.act_means:
            acts = (acts - trainer.act_means[layer_idx]) / trainer.act_stds[layer_idx]

        codes = trainer.get_feature_activations(sae, acts)
        fire_count = (codes > 0).float().sum(dim=0).numpy()
        fire_rate  = fire_count / len(codes) * 100
        n_dead = int((fire_count == 0).sum())

        ax = axes[ax_idx]
        alive_rates = fire_rate[fire_rate > 0]
        ax.hist(alive_rates, bins=40, edgecolor="black", alpha=0.75,
                color=COLORS[ax_idx], label=f"Alive ({len(alive_rates)})")
        if n_dead > 0:
            # Show dead features as a red bar at x=0
            bin_w = alive_rates.min() * 0.5 if len(alive_rates) > 0 else 1
            ax.bar(0, n_dead, width=max(bin_w, 0.3),
                   color="red", alpha=0.6, label=f"Dead ({n_dead})")
        ax.set_xlabel("Feature fire rate  (% of tokens)", fontsize=11)
        ax.set_ylabel("Number of features", fontsize=11)
        ax.set_title(f"Layer {layer_idx}  —  {n_dead} dead / "
                     f"{len(fire_count)} total  ({n_dead/len(fire_count)*100:.1f}%)",
                     fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Feature Activation Histogram  (best model: exp=1.5, k=48, 20 epochs)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Plot 8: Iteration progress (bar chart) ─────────────────────────────────
def plot_iteration_progress(save_path):
    labels  = [f"Iter {i+1}" for i in range(len(HISTORICAL))]
    configs = [h["label"] for h in HISTORICAL]
    dead    = [h["dead_frac"] * 100 for h in HISTORICAL]
    r2      = [h["r2"] for h in HISTORICAL]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Dead neurons
    bars1 = ax1.bar(labels, dead,
                    color=[COLORS[i] for i in range(len(labels))],
                    edgecolor="black", linewidth=0.8)
    ax1.axhline(10, color="red", ls="--", alpha=0.6, linewidth=1.5,
                label="10% target")
    ax1.axhspan(0, 10, color="green", alpha=0.06)
    ax1.set_ylabel("Dead neurons  (%)", fontsize=13)
    ax1.set_title("Dead Neuron Reduction", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val, cfg in zip(bars1, dead, configs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
        ax1.text(bar.get_x() + bar.get_width()/2, -7,
                 cfg, ha="center", fontsize=7, va="top",
                 color="gray", style="italic")
    ax1.set_ylim(bottom=-10)

    # R²
    bars2 = ax2.bar(labels, r2,
                    color=[COLORS[i] for i in range(len(labels))],
                    edgecolor="black", linewidth=0.8)
    ax2.axhline(0.95, color="green", ls="--", alpha=0.6, linewidth=1.5,
                label="R² ≥ 0.95 target")
    ax2.set_ylabel("R²", fontsize=13)
    ax2.set_title("Reconstruction Quality", fontsize=14)
    ax2.set_ylim(0.94, 1.0)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val, cfg in zip(bars2, r2, configs):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                 f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
        ax2.text(bar.get_x() + bar.get_width()/2, 0.938,
                 cfg, ha="center", fontsize=7, va="top",
                 color="gray", style="italic")

    fig.suptitle("Iterative Improvement Across 4 Training Runs",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ── Summary helpers ─────────────────────────────────────────────────────────
def save_summary(results, path):
    clean = []
    for r in results:
        clean.append({k: (v if not isinstance(v, torch.Tensor) else v.item())
                      for k, v in r.items()})
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"  ✓ Saved {path}")


def print_summary_table(results):
    print(f"\n{'='*70}")
    print(f"{'exp':>5}  {'dict':>5}  {'k':>4}  {'k/dict':>7}  "
          f"{'L0':>6}  {'Dead%':>7}  {'R²':>8}")
    print(f"{'-'*70}")
    for r in sorted(results, key=lambda x: (x["expansion"], x["k"])):
        ratio = r["k"] / r["dict_size"] * 100
        print(f"{r['expansion']:>5.1f}  {r['dict_size']:>5d}  {r['k']:>4d}  "
              f"{ratio:>6.1f}%  {r['l0']:>6.1f}  "
              f"{r['dead_frac']*100:>6.1f}%  {r['r2']:>8.4f}")
    print(f"{'='*70}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="SAE hyperparameter sweep + visualisation"
    )
    p.add_argument(
        "--encoder", default="sleepfm",
        choices=["sleepfm", "reve"],
        help="Encoder backend to use (default: sleepfm)",
    )
    p.add_argument(
        "--layer", type=int, default=None,
        help="Single transformer layer index (default: last layer). "
             "Ignored when --layers is provided.",
    )
    p.add_argument(
        "--layers", default=None,
        help="Comma-separated layer indices to sweep (e.g. 0,4,8,12,16,21). "
             "Results scoped to results/sae/{encoder}/layer_{L}/.",
    )
    p.add_argument(
        "--run-sweep", action="store_true",
        help="Re-run the full training grid (overwrites sweep_results.json)",
    )
    p.add_argument(
        "--max-tokens", type=int, default=None,
        help="Cap activation tokens collected (default: unlimited for sleepfm, "
             "500000 for reve). Set explicitly to override.",
    )
    return p.parse_args()


def _run_sweep_for_layer(model, train_loader, val_loader, layer_idx,
                          results_dir, max_tokens, run_sweep):
    """Run the full sweep for a single layer and save results + plots."""
    global EMBED, TARGET_LAYER, RESULTS_DIR
    TARGET_LAYER = layer_idx
    RESULTS_DIR = results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    sweep_json = results_dir / "sweep_results.json"
    if run_sweep or not sweep_json.exists():
        print(f"\n=== Running Full-Training Grid (layer {layer_idx}) ===")
        results = run_full_training_grid(
            model, train_loader, val_loader,
            embed_dim=EMBED, target_layer=layer_idx,
            max_tokens=max_tokens,
        )
        print_summary_table(results)
        save_summary(results, sweep_json)
    else:
        print(f"\n=== Loading existing sweep results from {sweep_json} ===")
        results = json.load(open(sweep_json))

    print("\n=== Generating Plots ===")
    plot_r2_vs_k(results,        results_dir / "01_r2_vs_k.png")
    plot_dead_vs_k(results,      results_dir / "02_dead_vs_k.png")
    plot_pareto(results,         results_dir / "03_pareto_r2_vs_dead.png")
    plot_ratio_vs_dead(results,  results_dir / "04_ratio_vs_dead.png")
    plot_heatmap_r2(results,     results_dir / "05_heatmap_r2.png")
    plot_heatmap_dead(results,   results_dir / "06_heatmap_dead.png")

    print(f"  Results in {results_dir}/")
    return results


def main():
    global EMBED, TARGET_LAYER, RESULTS_DIR, DATA_PATH
    args = _parse_args()
    t0 = time.time()

    # ── Load encoder ────────────────────────────────────────────────────
    print(f"Loading encoder: {args.encoder} …")
    model = load_model(args.encoder)
    EMBED = model.embed_dim
    n_layers = len(model.get_hookable_layers())

    # ── Resolve target layers ────────────────────────────────────────────
    if args.layers is not None:
        target_layers = [int(x.strip()) for x in args.layers.split(",")]
    else:
        target_layers = [args.layer if args.layer is not None else n_layers - 1]

    # ── Dataset ──────────────────────────────────────────────────────────
    DATA_PATH = _ENCODER_DATA[args.encoder]
    print(f"  dataset → {DATA_PATH.name}")

    # ── max_tokens default ───────────────────────────────────────────────
    max_tokens = args.max_tokens
    if max_tokens is None and args.encoder == "reve":
        max_tokens = 500_000
        print(f"  max_tokens={max_tokens} (reve default; override with --max-tokens)")

    print(f"  embed_dim={EMBED}  layers={target_layers}")
    print("Loading data …")
    train_loader, val_loader = load_data()

    base_dir = ROOT / "results" / "sae" / args.encoder

    for layer_idx in target_layers:
        print(f"\n{'#'*70}")
        print(f"#  Layer {layer_idx}/{n_layers-1}")
        print(f"{'#'*70}")

        # Scope results to a per-layer subdirectory when sweeping multiple layers
        if len(target_layers) > 1:
            results_dir = base_dir / f"layer_{layer_idx}"
        else:
            results_dir = base_dir

        _run_sweep_for_layer(
            model, train_loader, val_loader,
            layer_idx=layer_idx,
            results_dir=results_dir,
            max_tokens=max_tokens,
            run_sweep=args.run_sweep,
        )

    elapsed = time.time() - t0
    print(f"\n✅ All done in {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    main()
