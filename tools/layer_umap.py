"""
layer_umap.py — UMAP visualisation of encoder activations across all layers
============================================================================

For each encoder variant, collects token activations at every transformer
layer and projects them into 2-D with UMAP.  Tokens are coloured by their
spectral codebook cluster (dominant frequency band), making it easy to see
how much spectral information is encoded at each depth.

Outputs (in results/layer_umap/{encoder}/):
  layers_grid.png      — one panel per layer, all in a single figure
  layer_{L}.png        — individual layer panels (higher resolution)

Usage:
  uv run tools/layer_umap.py --encoder sleepfm_v2.1
  uv run tools/layer_umap.py --encoder sleepfm     --max-tokens 10000
  uv run tools/layer_umap.py --all                 # all SleepFM variants
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sae4eeg.dataset import get_dataloaders, StandardizeLabel
from sae4eeg.encoders import load_encoder, MODEL_CARDS
from sae4eeg.sae import SAETrainer
from sae4eeg.xae import CLINICAL_BANDS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_V2_DIR = ROOT / "checkpoints" / "pretrained" / "SleepFM v2 Models"

ENCODER_CONFIGS = {
    "sleepfm_v2.0": dict(
        weights_path = _V2_DIR / "settransformer_exp0_cl_cnn_sgd_fp32_128d_640p_lr0.001_20260307_113442" / "best.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path= ROOT / "results" / "xae" / "sleepfm_v2.0" / "codebook" / "codebook.pt",
    ),
    "sleepfm": dict(
        weights_path = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path= ROOT / "results" / "xae" / "sleepfm" / "codebook" / "codebook.pt",
    ),
    "sleepfm_v2.1": dict(
        weights_path = _V2_DIR / "settransformer_exp1_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_114250" / "best.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path= ROOT / "results" / "xae" / "sleepfm_v2.1" / "codebook" / "codebook.pt",
    ),
    "sleepfm_v2.3": dict(
        weights_path = _V2_DIR / "settransformer_exp2_cl_mae_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_113651" / "best.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path= ROOT / "results" / "xae" / "sleepfm_v2.3" / "codebook" / "codebook.pt",
    ),
    "sleepfm_v2.4": dict(
        weights_path = _V2_DIR / "settransformer_exp4_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_210846" / "best.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path= ROOT / "results" / "xae" / "sleepfm_v2.4" / "codebook" / "codebook.pt",
    ),
    "sleepfm_v2.5": dict(
        weights_path = _V2_DIR / "settransformer_exp5_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260308_111156" / "best.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path= ROOT / "results" / "xae" / "sleepfm_v2.5" / "codebook" / "codebook.pt",
    ),
    "sleepfm_v2.6": dict(
        weights_path = _V2_DIR / "settransformer_exp2.6_res_1sec_cl_cnn_adamw_bf16_128d_128p_lr0.0003_20260322_011957" / "best.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path= ROOT / "results" / "xae" / "sleepfm_v2.6" / "codebook" / "codebook.pt",
    ),
    "sleepfm_v2.7": dict(
        weights_path = _V2_DIR / "settransformer_exp2.7_res_1sec_cl_cnn_adamw_bf16_128d_128p_lr0.0003_20260321_161621" / "best.pt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path= ROOT / "results" / "xae" / "sleepfm_v2.7" / "codebook" / "codebook.pt",
    ),
    "reve_qjbe08": dict(
        load_as      = "reve",
        weights_path = ROOT / "checkpoints" / "finetuned" / "reve_qjbe08.ckpt",
        data_path    = ROOT / "data" / "D4-v3-preprocessed-v1",
        codebook_path= ROOT / "results" / "xae" / "reve_qjbe08" / "codebook" / "codebook.pt",
    ),
}

# One colour per clinical band — used consistently across all plots
BAND_COLORS = {
    "delta":     "#2196F3",
    "theta":     "#4CAF50",
    "alpha":     "#FF9800",
    "low-beta":  "#F44336",
    "high-beta": "#9C27B0",
    "gamma":     "#795548",
    "other":     "#9E9E9E",
}


def parse_args():
    p = argparse.ArgumentParser(description="Layer UMAP visualisation")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--encoder", choices=list(ENCODER_CONFIGS.keys()),
                       help="Single encoder to process")
    group.add_argument("--all", action="store_true",
                       help="Process all configured encoders")
    p.add_argument("--max-tokens", type=int, default=8_000,
                   help="Max val tokens to collect per layer (default: 8000)")
    p.add_argument("--layers", default=None,
                   help="Comma-separated layer subset (default: all layers)")
    return p.parse_args()


def _collect_all_layers(encoder, val_loader, all_layer_ids, max_tokens):
    """Collect activations at every layer in a single pass over the data."""
    trainer = SAETrainer(
        embed_dim=encoder.embed_dim,
        target_layers=all_layer_ids,
        device=DEVICE,
    )
    return trainer.collect_activations(encoder, val_loader, max_tokens=max_tokens)


def _assign_codebook_labels(acts: torch.Tensor, codebook: dict) -> np.ndarray:
    """Assign each token to its nearest codebook centroid (in embedding space)."""
    centroids = codebook["centroids_emb"]   # (K, E) float tensor
    if torch.is_tensor(centroids):
        centroids = centroids.float().numpy()
    acts_np = acts.float().numpy()

    # Cosine similarity for fast nearest-centroid
    acts_norm = acts_np / (np.linalg.norm(acts_np, axis=1, keepdims=True) + 1e-8)
    cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    sim = acts_norm @ cent_norm.T                    # (N, K)
    labels = sim.argmax(axis=1)                      # (N,)
    return labels


def _band_labels_from_codebook(codebook: dict, cluster_ids: np.ndarray) -> list[str]:
    """Map cluster IDs → dominant band name (string) per token."""
    band_per_cluster = codebook["cluster_band_label"]  # list[str], length K
    return [band_per_cluster[int(c)] for c in cluster_ids]


def _run_umap(acts: np.ndarray) -> np.ndarray:
    import umap as umap_module
    reducer = umap_module.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.3,
        random_state=42, metric="euclidean",
    )
    return reducer.fit_transform(acts).astype(np.float32)


def _plot_layer_grid(
    coords_per_layer: dict[int, np.ndarray],
    band_labels_per_layer: dict[int, list[str]],
    display_name: str,
    out_path: Path,
):
    """Render a grid of UMAP scatterplots — one panel per layer."""
    layer_ids = sorted(coords_per_layer.keys())
    n_layers  = len(layer_ids)

    ncols = min(n_layers, 6)
    nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    band_names = list(CLINICAL_BANDS.keys()) + ["other"]

    for ax_idx, layer in enumerate(layer_ids):
        ax = axes[ax_idx]
        coords = coords_per_layer[layer]
        band_labels = band_labels_per_layer[layer]

        # Plot one scatter per band for legend ordering
        for band in band_names:
            mask = np.array([b == band for b in band_labels])
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=BAND_COLORS.get(band, "#9E9E9E"),
                s=3, alpha=0.5, linewidths=0,
                label=band,
            )

        ax.set_title(f"Layer {layer}", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    # Hide unused axes
    for ax_idx in range(len(layer_ids), len(axes)):
        axes[ax_idx].set_visible(False)

    # Shared legend on the last used axes
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower right", fontsize=8,
                   markerscale=4, framealpha=0.8,
                   bbox_to_anchor=(0.98, 0.02))

    fig.suptitle(
        f"Layer Activation UMAPs — {display_name}\n"
        f"(coloured by spectral codebook cluster — dominant band)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved grid: {out_path}")


def _plot_individual_layers(
    coords_per_layer: dict[int, np.ndarray],
    band_labels_per_layer: dict[int, list[str]],
    display_name: str,
    out_dir: Path,
):
    """High-res individual layer plots."""
    band_names = list(CLINICAL_BANDS.keys()) + ["other"]

    for layer, coords in coords_per_layer.items():
        band_labels = band_labels_per_layer[layer]
        fig, ax = plt.subplots(figsize=(7, 7))
        for band in band_names:
            mask = np.array([b == band for b in band_labels])
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=BAND_COLORS.get(band, "#9E9E9E"),
                s=4, alpha=0.55, linewidths=0,
                label=f"{band} (n={mask.sum()})",
            )
        ax.legend(fontsize=8, markerscale=3, loc="best", framealpha=0.8)
        ax.set_title(f"{display_name} — Layer {layer}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        save_path = out_dir / f"layer_{layer:02d}.png"
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"  ✓ Saved {len(coords_per_layer)} individual layer plots to {out_dir}/")


def process_encoder(encoder_key: str, max_tokens: int, layer_subset=None):
    cfg = ENCODER_CONFIGS[encoder_key]
    load_as = cfg.get("load_as", encoder_key)
    display = MODEL_CARDS.get(encoder_key, MODEL_CARDS.get(load_as, {})).get("display_name", encoder_key)
    codebook_path = cfg["codebook_path"]

    print(f"\n{'='*60}")
    print(f"  {display}  ({encoder_key})")
    print(f"{'='*60}")

    if not codebook_path.exists():
        print(f"  SKIP — codebook not found: {codebook_path}")
        print(f"  Run: uv run tools/build_codebook.py --encoder {encoder_key}")
        return

    out_dir = ROOT / "results" / "layer_umap" / encoder_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    print("  Loading encoder…", flush=True)
    encoder = load_encoder(load_as, weights_path=str(cfg["weights_path"]))
    encoder.to(DEVICE).eval()
    n_layers = len(encoder.get_hookable_layers())
    all_layer_ids = list(range(n_layers)) if layer_subset is None else layer_subset
    print(f"  {n_layers} layers, processing: {all_layer_ids}")

    # Load codebook
    print("  Loading codebook…", flush=True)
    codebook = torch.load(codebook_path, map_location="cpu", weights_only=False)
    print(f"  Codebook: {codebook['n_clusters']} clusters")

    # Load val data
    print("  Loading val data…", flush=True)
    gen = get_dataloaders(
        train_path=str(cfg["data_path"]),
        transformer=StandardizeLabel(),
    )
    _, _, val_loader, _ = next(gen)

    # Collect activations at all target layers
    print(f"  Collecting activations (max {max_tokens:,} tokens per layer)…", flush=True)
    layer_acts = _collect_all_layers(encoder, val_loader, all_layer_ids, max_tokens)

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # UMAP + cluster labelling per layer
    coords_per_layer = {}
    band_labels_per_layer = {}

    for layer in all_layer_ids:
        if layer not in layer_acts:
            print(f"  Layer {layer}: no activations, skipping")
            continue
        acts = layer_acts[layer]   # (N, E)
        print(f"  Layer {layer}: {acts.shape[0]} tokens, running UMAP…", flush=True)

        cluster_ids = _assign_codebook_labels(acts, codebook)
        band_labels = _band_labels_from_codebook(codebook, cluster_ids)

        acts_np = acts.float().numpy()
        coords  = _run_umap(acts_np)

        coords_per_layer[layer]     = coords
        band_labels_per_layer[layer] = band_labels
        print(f"  Layer {layer}: UMAP done ({coords.shape})")

    if not coords_per_layer:
        print("  No layers processed.")
        return

    # Grid plot
    _plot_layer_grid(
        coords_per_layer, band_labels_per_layer, display,
        out_dir / "layers_grid.png",
    )
    # Individual plots
    _plot_individual_layers(
        coords_per_layer, band_labels_per_layer, display, out_dir,
    )


def main():
    args = parse_args()
    layer_subset = None
    if args.layers:
        layer_subset = sorted(int(x) for x in args.layers.split(","))

    if args.all:
        for enc_key in ENCODER_CONFIGS:
            process_encoder(enc_key, args.max_tokens, layer_subset)
    else:
        process_encoder(args.encoder, args.max_tokens, layer_subset)


if __name__ == "__main__":
    main()
