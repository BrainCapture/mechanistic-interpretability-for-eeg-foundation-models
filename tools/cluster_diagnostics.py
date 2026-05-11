"""Cluster Diagnostics for Embedding-Space Codebook.

Produces a comprehensive diagnostic figure with:

  1. K-sweep metrics  -- Inertia (elbow), Silhouette, Calinski-Harabasz,
     Davies-Bouldin for a range of cluster counts.
  2. UMAP 2D scatter -- 128-d embeddings projected to 2D via UMAP,
     colored by cluster assignment (for the chosen k).
  3. Cluster size histogram -- distribution of cluster sizes.
  4. Waveform quality scatter -- reconstruction r vs cluster size.
  5. Dominant-band pie chart -- how many clusters are dominated by each
     clinical EEG band.
  6. Intra-cluster variance histogram -- how tight each cluster is.

Output
------
  results/xae/codebook/
    cluster_diagnostics.png
    k_sweep_metrics.png

Usage::

    uv run tools/cluster_diagnostics.py
    uv run tools/cluster_diagnostics.py --ks 50 100 150 200 300 500
"""
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sae4eeg.xae import XAETrainer, CLINICAL_BANDS
from sae4eeg.dataset import get_dataloaders, StandardizeLabel
from sae4eeg.encoders import EncoderBackend, load_encoder
from sae4eeg.sae import ActivationExtractor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "D4-v3-preprocessed-v2"
MODEL_PATH = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"
XAE_PATH = ROOT / "results" / "xae" / "xae_checkpoint.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FS = 128
PATCH_SIZE = 128
TARGET_LAYER = 2
MAX_TOKENS = 20_000
F_DISPLAY_MAX = 45.0

BAND_COLORS = {
    "delta":     "#1f77b4",
    "theta":     "#2ca02c",
    "alpha":     "#ff7f0e",
    "low-beta":  "#d62728",
    "high-beta": "#9467bd",
    "gamma":     "#8c564b",
    "abnormal":  "#e31a1c",  # bright red for abnormal stratum
}


def remove_module_from_state_dict(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()}


def collect_tokens(trainer, model, val_loader, max_tokens=MAX_TOKENS):
    """Collect embeddings, spectral targets, and raw patches."""
    if isinstance(model, EncoderBackend):
        inner_model = model.model
        hook_layers = model.get_hookable_layers()
    else:
        inner_model = model
        hook_layers = None
    inner_model.eval().to(DEVICE)
    extractor = ActivationExtractor(inner_model, layers=hook_layers)
    embed_list, spec_list, raw_list = [], [], []
    total = 0
    pbar = tqdm(val_loader, desc="Collecting tokens", leave=False)
    for batch in pbar:
        x = batch[0].to(DEVICE) if isinstance(batch, (list, tuple)) else batch.to(DEVICE)
        B, C, T = x.shape
        S = T // PATCH_SIZE
        extractor.clear()
        _ = inner_model(x)
        acts = extractor.get_activations()
        layer_acts = acts[TARGET_LAYER]
        T_used = S * PATCH_SIZE
        patches = x[:, :, :T_used].reshape(B, C, S, PATCH_SIZE)
        patches = patches.permute(0, 2, 1, 3).reshape(B * S, C, PATCH_SIZE)
        spec_targets = trainer.spectral.extract(patches)
        embed_list.append(layer_acts.reshape(B * S, -1).cpu())
        spec_list.append(spec_targets.cpu())
        raw_list.append(patches.cpu())
        total += B * S
        pbar.set_postfix(tokens=f"{total:,}")
        if max_tokens and total >= max_tokens:
            break
    pbar.close()
    extractor.remove_hooks()
    embeddings = torch.cat(embed_list)[:max_tokens]
    spectral = torch.cat(spec_list)[:max_tokens]
    raw_full = torch.cat(raw_list)[:max_tokens]
    return embeddings, spectral, raw_full


# =====================================================================
# K-sweep: cluster metrics for multiple values of k
# =====================================================================

def k_sweep(emb_np, ks, subsample=10000):
    """Run MiniBatchKMeans for each k and compute clustering metrics.

    Uses a subsample for silhouette (which is O(n^2)).
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                 davies_bouldin_score)

    N = len(emb_np)
    sub_idx = np.random.RandomState(42).choice(N, min(subsample, N), replace=False)
    emb_sub = emb_np[sub_idx]

    results = {
        "k": [], "inertia": [], "silhouette": [],
        "calinski_harabasz": [], "davies_bouldin": [],
        "mean_cluster_size": [], "min_cluster_size": [],
        "max_cluster_size": [],
    }

    for k in ks:
        print(f"  k={k:>4d} ... ", end="", flush=True)
        km = MiniBatchKMeans(
            n_clusters=k, random_state=42,
            batch_size=2048, n_init=5, max_iter=300,
        )
        labels_full = km.fit_predict(emb_np)
        labels_sub = km.predict(emb_sub)

        # Cluster sizes
        sizes = np.bincount(labels_full, minlength=k)

        # Silhouette on subsample (full dataset too slow for large k)
        sil = silhouette_score(emb_sub, labels_sub, sample_size=None)
        ch = calinski_harabasz_score(emb_np, labels_full)
        db = davies_bouldin_score(emb_np, labels_full)

        results["k"].append(k)
        results["inertia"].append(km.inertia_)
        results["silhouette"].append(sil)
        results["calinski_harabasz"].append(ch)
        results["davies_bouldin"].append(db)
        results["mean_cluster_size"].append(sizes.mean())
        results["min_cluster_size"].append(sizes.min())
        results["max_cluster_size"].append(sizes.max())

        print(f"inertia={km.inertia_:.0f}  sil={sil:.3f}  "
              f"CH={ch:.0f}  DB={db:.3f}  "
              f"sizes=[{sizes.min()},{sizes.max()}]")

    return results


def plot_k_sweep(results, out_path):
    """Plot 4-panel figure with clustering metrics vs k."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ks = results["k"]

    # Inertia (elbow)
    ax = axes[0, 0]
    ax.plot(ks, results["inertia"], "o-", color="#1565C0", linewidth=2, markersize=8)
    ax.set_xlabel("Number of clusters (k)", fontsize=11)
    ax.set_ylabel("Inertia", fontsize=11)
    ax.set_title("Elbow Plot (Inertia)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    for k, v in zip(ks, results["inertia"]):
        ax.annotate(f"{v:.0f}", (k, v), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, ha="center")

    # Silhouette
    ax = axes[0, 1]
    ax.plot(ks, results["silhouette"], "o-", color="#2E7D32", linewidth=2, markersize=8)
    ax.set_xlabel("Number of clusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Silhouette Score (higher = better)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    for k, v in zip(ks, results["silhouette"]):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, ha="center")

    # Calinski-Harabasz
    ax = axes[1, 0]
    ax.plot(ks, results["calinski_harabasz"], "o-", color="#E65100", linewidth=2, markersize=8)
    ax.set_xlabel("Number of clusters (k)", fontsize=11)
    ax.set_ylabel("Calinski-Harabasz Index", fontsize=11)
    ax.set_title("Calinski-Harabasz (higher = better)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    for k, v in zip(ks, results["calinski_harabasz"]):
        ax.annotate(f"{v:.0f}", (k, v), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, ha="center")

    # Davies-Bouldin
    ax = axes[1, 1]
    ax.plot(ks, results["davies_bouldin"], "o-", color="#C62828", linewidth=2, markersize=8)
    ax.set_xlabel("Number of clusters (k)", fontsize=11)
    ax.set_ylabel("Davies-Bouldin Index", fontsize=11)
    ax.set_title("Davies-Bouldin (lower = better)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    for k, v in zip(ks, results["davies_bouldin"]):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, ha="center")

    fig.suptitle(
        "Clustering Metrics Sweep -- Embedding Space (128-d)\n"
        f"MiniBatchKMeans on {results.get('n_tokens', '?')} tokens",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  K-sweep plot -> {out_path}")


# =====================================================================
# UMAP + per-cluster diagnostics (for chosen k)
# =====================================================================

def compute_umap(emb_np, n_neighbors=30, min_dist=0.3, random_state=42):
    """Compute 2D UMAP embedding."""
    import umap
    print("  Computing UMAP (128-d -> 2-d) ...")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=min_dist, random_state=random_state,
        metric="euclidean",
    )
    coords = reducer.fit_transform(emb_np)
    return coords


def compute_per_cluster_stats(trainer, emb_np, raw_full, labels, n_clusters):
    """Compute waveform quality r and dominant band for each cluster."""

    freqs = trainer.spectral.freqs
    freq_mask = torch.tensor(trainer.spectral._freq_mask)
    clean_mask = freqs <= F_DISPLAY_MAX
    device = next(trainer.xae.parameters()).device

    # Re-fit to get centroids (or compute from labels)
    centroids = np.zeros((n_clusters, emb_np.shape[1]))
    cluster_sizes = np.zeros(n_clusters, dtype=int)
    exemplar_idx = np.zeros(n_clusters, dtype=int)

    for c in range(n_clusters):
        mask_c = labels == c
        members = emb_np[mask_c]
        cluster_sizes[c] = len(members)
        centroids[c] = members.mean(axis=0)
        dists = np.linalg.norm(members - centroids[c], axis=1)
        exemplar_idx[c] = np.where(mask_c)[0][dists.argmin()]

    # XAE prediction for centroids
    centroids_t = torch.tensor(centroids, dtype=torch.float32, device=device)
    emb_norm = (centroids_t - trainer.embed_mean.to(device)) / trainer.embed_std.to(device)
    trainer.xae.eval()
    with torch.no_grad():
        pred_norm = trainer.xae.decode(emb_norm).cpu()
    pred = pred_norm * trainer.target_std.cpu() + trainer.target_mean.cpu()
    pr_amp, _, _ = trainer.spectral.unpack_targets(pred)
    pr_amp_clean = pr_amp.clone()
    pr_amp_clean[:, ~clean_mask] = 0.0

    # Dominant band per cluster
    band_names = list(CLINICAL_BANDS.keys())
    dom_bands = []
    for c in range(n_clusters):
        bp = []
        for bname, (lo, hi) in CLINICAL_BANDS.items():
            band_mask = (freqs >= lo) & (freqs <= hi) & clean_mask
            amp_lin = np.expm1(pr_amp_clean[c, band_mask].numpy())
            bp.append((amp_lin ** 2).mean())
        dom_bands.append(band_names[np.argmax(bp)])

    # Waveform r for each cluster (XAE amp + exemplar phase vs raw)
    n_freq = PATCH_SIZE // 2 + 1
    C_ch = raw_full.shape[1]
    recon_r = np.zeros(n_clusters)
    for c in range(n_clusters):
        idx = exemplar_idx[c]
        raw_patch = raw_full[idx]
        raw_fft = torch.fft.rfft(raw_patch, dim=-1)
        phase = raw_fft[:, freq_mask].angle()
        amp_lin = torch.expm1(pr_amp_clean[c])
        amp_exp = amp_lin.unsqueeze(0).expand(C_ch, -1)
        spec = torch.zeros(C_ch, n_freq, dtype=torch.cfloat)
        spec[:, freq_mask] = amp_exp * torch.exp(1j * phase)
        rec = torch.fft.irfft(spec, n=PATCH_SIZE, dim=-1).float()
        a = raw_patch - raw_patch.mean(dim=-1, keepdim=True)
        b = rec - rec.mean(dim=-1, keepdim=True)
        num = (a * b).sum(dim=-1)
        den = (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-8)
        recon_r[c] = (num / den).numpy().mean()

    return cluster_sizes, dom_bands, recon_r


def plot_diagnostics(emb_np, umap_coords, labels, centroids_umap,
                     cluster_sizes, dom_bands, recon_r, n_clusters,
                     out_path):
    """6-panel diagnostic figure."""
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28)

    band_names = list(CLINICAL_BANDS.keys())

    # ---- Panel 1: UMAP colored by cluster (scatter) ----------------
    ax = fig.add_subplot(gs[0, 0:2])
    # Color by cluster, use a colormap
    cmap = plt.colormaps.get_cmap("tab20").resampled(min(n_clusters, 20))
    colors_idx = labels % 20
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                    c=colors_idx, cmap=cmap, s=3, alpha=0.35,
                    rasterized=True)
    # Plot centroids
    ax.scatter(centroids_umap[:, 0], centroids_umap[:, 1],
               c="black", s=25, marker="x", linewidths=0.8,
               alpha=0.7, zorder=5, label="centroids")
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(f"UMAP of 128-d Embeddings (k={n_clusters}, {len(emb_np):,} tokens)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.08)

    # ---- Panel 2: UMAP colored by dominant band --------------------
    ax = fig.add_subplot(gs[0, 2])
    # Map each token to its cluster's dominant band color
    token_colors = np.array([BAND_COLORS[dom_bands[lbl]] for lbl in labels])
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
               c=token_colors, s=3, alpha=0.35, rasterized=True)
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=BAND_COLORS[b], label=b)
                       for b in band_names if b in dom_bands]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right",
              title="Dominant band", title_fontsize=9)
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title("UMAP colored by Dominant Band", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.08)

    # ---- Panel 3: Cluster size histogram ---------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(cluster_sizes, bins=30, color="#1565C0", alpha=0.75,
            edgecolor="white", linewidth=0.5)
    ax.axvline(np.median(cluster_sizes), color="#C62828", linestyle="--",
               linewidth=1.5, label=f"median={np.median(cluster_sizes):.0f}")
    ax.axvline(cluster_sizes.mean(), color="#E65100", linestyle=":",
               linewidth=1.5, label=f"mean={cluster_sizes.mean():.1f}")
    ax.set_xlabel("Cluster Size", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Cluster Size Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # ---- Panel 4: Waveform r vs cluster size -----------------------
    ax = fig.add_subplot(gs[1, 1])
    sc_colors = [BAND_COLORS[db] for db in dom_bands]
    ax.scatter(cluster_sizes, recon_r, c=sc_colors, s=40, alpha=0.7,
               edgecolors="white", linewidth=0.3)
    ax.axhline(np.nanmean(recon_r), color="grey", linestyle="--",
               linewidth=1, label=f"mean r={np.nanmean(recon_r):.3f}")
    ax.axhline(np.nanmedian(recon_r), color="grey", linestyle=":",
               linewidth=1, label=f"median r={np.nanmedian(recon_r):.3f}")
    ax.set_xlabel("Cluster Size", fontsize=11)
    ax.set_ylabel("Waveform r (XAE amp + exemplar phase)", fontsize=11)
    ax.set_title("Reconstruction Quality vs Cluster Size", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)
    ax.set_ylim(0, 1.05)

    # ---- Panel 5: Dominant band / stratum pie chart ----------------
    ax = fig.add_subplot(gs[1, 2])
    all_strata = ["abnormal"] + band_names
    band_counts = {b: 0 for b in all_strata}
    for db in dom_bands:
        band_counts[db] = band_counts.get(db, 0) + 1
    # Only show strata with > 0
    labels_pie = [b for b in all_strata if band_counts[b] > 0]
    sizes_pie  = [band_counts[b] for b in labels_pie]
    colors_pie = [BAND_COLORS.get(b, "#aaaaaa") for b in labels_pie]
    wedges, texts, autotexts = ax.pie(
        sizes_pie, labels=labels_pie, colors=colors_pie,
        autopct=lambda p: f"{p:.0f}%\n({int(p*sum(sizes_pie)/100)})",
        startangle=90, textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(f"Dominant Band Distribution (k={n_clusters})",
                 fontsize=13, fontweight="bold")

    fig.suptitle(
        f"Embedding-Space Clustering Diagnostics\n"
        f"128-d SetTransformer embeddings, MiniBatchKMeans, "
        f"k={n_clusters}, {len(emb_np):,} tokens",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Diagnostics plot -> {out_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cluster diagnostics for embedding-space codebook")
    parser.add_argument("--ks", nargs="+", type=int,
                        default=[25, 50, 100, 150, 200, 300, 500],
                        help="K values for sweep (default: 25 50 100 150 200 300 500)")
    parser.add_argument("--chosen-k", type=int, default=200,
                        help="K for detailed diagnostics (default: 200)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help="Max tokens to collect (default: 20000)")
    args = parser.parse_args()

    out_dir = ROOT / "results" / "xae" / "codebook"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Cluster Diagnostics for Embedding-Space Codebook")
    print("=" * 72)

    # -- Load everything ---------------------------------------------
    model = load_encoder("sleepfm", weights_path=MODEL_PATH)
    model.to(DEVICE).eval()

    trainer = XAETrainer(embed_dim=128, device=DEVICE)
    trainer.load(str(XAE_PATH))
    trainer.xae.to(DEVICE).eval()

    gen = get_dataloaders(train_path=str(DATA_PATH),
                          transformer=StandardizeLabel())
    _, _, val_loader, _ = next(gen)
    print("[ok] Loaded model, XAE, and data\n")

    # -- Collect tokens ----------------------------------------------
    embeddings, gt_spectral, raw_full = collect_tokens(
        trainer, model, val_loader, max_tokens=args.max_tokens,
    )
    emb_np = embeddings.numpy()
    N, D = emb_np.shape
    print(f"  -> {N} tokens collected ({D}-d)\n")

    # -- K-sweep metrics ---------------------------------------------
    print("=" * 72)
    print("  K-Sweep Metrics")
    print("=" * 72)
    sweep_results = k_sweep(emb_np, args.ks)
    sweep_results["n_tokens"] = N
    plot_k_sweep(sweep_results, out_dir / "k_sweep_metrics.png")

    # Save raw metrics as JSON
    json_path = out_dir / "k_sweep_metrics.json"
    json_safe = {k: [float(x) if isinstance(x, (np.floating, float))
                      else int(x) if isinstance(x, (np.integer, int))
                      else x for x in v]
                 if isinstance(v, list) else v
                 for k, v in sweep_results.items()}
    json_path.write_text(json.dumps(json_safe, indent=2))
    print(f"  Metrics JSON -> {json_path}")

    # -- Chosen-K detailed diagnostics -------------------------------
    chosen_k = args.chosen_k
    print(f"\n{'='*72}")
    print(f"  Detailed Diagnostics for k={chosen_k}")
    print(f"{'='*72}")

    # Load the saved codebook so we diagnose the actual stratified result
    codebook_path = out_dir / "codebook.pt"
    if codebook_path.exists():
        print(f"  Loading saved codebook from {codebook_path}")
        cb = torch.load(codebook_path, weights_only=False, map_location="cpu")
        labels      = cb["labels"]
        chosen_k    = cb["n_clusters"]
        dom_bands   = cb.get("cluster_band_label", None)
        cluster_sizes = cb["cluster_sizes"]
        recon_r     = cb["exemplar_recon_r"]
        print(f"  Codebook feature: {cb.get('cluster_feature', 'unknown')}")
        print(f"  Actual k = {chosen_k}")
        # Recompute dom_bands from band_powers if not stored
        if dom_bands is None:
            band_names = list(CLINICAL_BANDS.keys())
            dom_bands = [
                max(band_names, key=lambda b: cb["band_powers"][b][c])
                for c in range(chosen_k)
            ]
    else:
        print(f"  No saved codebook found, re-clustering in embedding space (k={chosen_k})...")
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(
            n_clusters=chosen_k, random_state=42,
            batch_size=2048, n_init=5, max_iter=300,
        )
        labels = km.fit_predict(emb_np)
        cluster_sizes, dom_bands, recon_r = compute_per_cluster_stats(
            trainer, emb_np, raw_full, labels, chosen_k,
        )

    cluster_sizes = np.asarray(cluster_sizes, dtype=float)
    umap_coords = compute_umap(emb_np)

    # Centroids in UMAP space
    centroids_umap = np.zeros((chosen_k, 2))
    for c in range(chosen_k):
        mask = labels == c
        if mask.sum() > 0:
            centroids_umap[c] = umap_coords[mask].mean(axis=0)

    # Plot
    plot_diagnostics(
        emb_np, umap_coords, labels, centroids_umap,
        cluster_sizes, dom_bands, recon_r, chosen_k,
        out_dir / "cluster_diagnostics.png",
    )

    # -- Summary stats -----------------------------------------------
    print(f"\n  Summary for k={chosen_k}:")
    print(f"    Cluster sizes: min={cluster_sizes.min()}, "
          f"max={cluster_sizes.max()}, "
          f"median={np.median(cluster_sizes):.0f}")
    print(f"    Waveform r:    mean={np.nanmean(recon_r):.3f}, "
          f"median={np.nanmedian(recon_r):.3f}")
    all_strata = ["abnormal"] + list(CLINICAL_BANDS.keys())
    strata_counts = {b: sum(1 for d in dom_bands if d == b) for b in all_strata}
    print("    Stratum counts: "
          + ", ".join(f"{b}={strata_counts[b]}" for b in all_strata
                      if strata_counts[b] > 0))
    print(f"\n  Done!  Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
