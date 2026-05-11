"""Whitened-Spectral Codebook -- Prototypical EEG Tokens.

Clusters validation tokens in **whitened XAE-predicted spectral amplitude
space** (45 frequency bins, 0.5-45 Hz), then reconstructs realistic
per-channel waveforms for each prototype by combining:

  * XAE-predicted amplitude of the cluster centroid (mean of members)
  * GT per-channel phase borrowed from the exemplar token (the real
    token nearest the centroid in EMBEDDING space, for phase borrowing)

Why whitened spectral space?
-----------------------------
Raw EEG power follows a 1/f law: delta bins (1-4 Hz) carry 10-100x more
power than alpha or beta bins.  If we cluster in raw embedding or raw
spectral space, the 1/f slope dominates -- KMeans minimises squared
Euclidean distance, so it subdivides the enormous delta cloud into ~197
nearly-identical clusters while allocating at most 3 clusters to all
other bands combined.

Whitening fixes this by z-scoring each frequency bin across the token
population (subtract per-bin mean, divide by per-bin std).  Every bin
now contributes equally to the clustering objective regardless of its
absolute power level.  A token with an unusual alpha peak is now as
"distinctive" as a token with an unusual delta amplitude.

Concretely:
  1. Compute XAE-predicted log1p amplitudes for all tokens  (N x 45)
  2. Per-bin mean and std across all tokens
  3. z = (amp - mean) / std                              -- whitened
  4. MiniBatchKMeans on z                                -- cluster
  5. Centroid amplitude = un-whiten the cluster mean     -- reconstruct
  6. Exemplar = nearest real token to centroid in EMBEDDING space
     (128-d) -- used only for per-channel phase borrowing

Frequencies above 45 Hz are zeroed (removes 50 Hz mains noise).

Each card shows:
  - Realistic per-channel waveforms (Fz, Cz, O1)
  - Amplitude spectrum (0.5-45 Hz, from XAE prediction)
  - Clinical-band power bar chart

Cards are laid out 10 per page (2 cols x 5 rows).

Output
------
  results/xae/codebook/
    page_01.png ... page_N.png
    codebook.pt
    codebook_summary.txt

Usage::

    uv run tools/build_codebook.py
    uv run tools/build_codebook.py --n-clusters 200
"""
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sae4eeg.xae import XAETrainer, CLINICAL_BANDS
from sae4eeg.dataset import get_dataloaders, StandardizeLabel, V4ResampleTransform
from sae4eeg.encoders import EncoderBackend, load_encoder
from sae4eeg.sae import ActivationExtractor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"

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

_ENCODER_DATA = {
    "sleepfm":          ROOT / "data" / "D4-v3-preprocessed-v2",
    "sleepfm_v2.0":     ROOT / "data" / "D4-v3-preprocessed-v2",
    "sleepfm_v2.1":     ROOT / "data" / "D4-v3-preprocessed-v2",
    "sleepfm_v2.3":     ROOT / "data" / "D4-v3-preprocessed-v2",
    "sleepfm_v2.4":     ROOT / "data" / "D4-v3-preprocessed-v2",
    "sleepfm_v2.5":     ROOT / "data" / "D4-v3-preprocessed-v2",
    "sleepfm_v2.6":     ROOT / "data" / "D4-v3-preprocessed-v2",
    "sleepfm_v2.7":     ROOT / "data" / "D4-v3-preprocessed-v2",
    "sleepfm_granular": ROOT / "data" / "D4-v4-preprocessed-10s",
    "reve":             ROOT / "data" / "D4-v3-preprocessed-v1",
    "labram":           ROOT / "data" / "D4-v3-preprocessed-v1",
}
_ENCODER_FS = {
    "sleepfm": 128, "sleepfm_v2.0": 128, "sleepfm_v2.1": 128, "sleepfm_v2.3": 128,
    "sleepfm_v2.4": 128, "sleepfm_v2.5": 128, "sleepfm_v2.6": 128, "sleepfm_v2.7": 128,
    "sleepfm_granular": 128,
    "reve": 200,
    "labram": 200,
}
_ENCODER_EMBED = {
    "sleepfm": 128, "sleepfm_v2.0": 128, "sleepfm_v2.1": 128, "sleepfm_v2.3": 128,
    "sleepfm_v2.4": 128, "sleepfm_v2.5": 128, "sleepfm_v2.6": 128, "sleepfm_v2.7": 128,
    "sleepfm_granular": 128,
    "reve": 512,
    "labram": 200,
}
_ENCODER_PATCH = {
    "sleepfm": 128, "sleepfm_v2.0": 640, "sleepfm_v2.1": 640, "sleepfm_v2.3": 640,
    "sleepfm_v2.4": 640, "sleepfm_v2.5": 640, "sleepfm_v2.6": 128, "sleepfm_v2.7": 128,
    "sleepfm_granular": 128,
    "reve": 200,
    "labram": 200,
}
_ENCODER_XAE = {
    "sleepfm": ROOT / "results" / "xae" / "xae_checkpoint.pt",
    "reve":    ROOT / "results" / "xae" / "reve" / "xae_checkpoint.pt",
    "labram":  ROOT / "results" / "xae" / "labram" / "xae_checkpoint.pt",
}
# v2 XAE paths derived dynamically in main()

# Defaults — overridden at runtime based on --encoder
DATA_PATH  = _ENCODER_DATA["sleepfm"]
XAE_PATH   = _ENCODER_XAE["sleepfm"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FS = 128
PATCH_SIZE = 128
TARGET_LAYER = 2
MAX_TOKENS = 20_000

# Display cutoff -- exclude 50 Hz mains noise and above
F_DISPLAY_MAX = 45.0

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]

# Channels to show in each card (frontal, central, occipital)
SHOW_CHANNELS = {"Fz": 4, "Cz": 9, "O1": 17}

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
# Data collection
# =====================================================================

def collect_tokens(trainer, model, val_loader, max_tokens=MAX_TOKENS):
    """Collect embeddings, spectral targets, and raw patches.

    For SleepFM (per-window tokens): each embedding corresponds to a
    cross-channel 1-second patch → raw shape (N, C, PATCH_SIZE).

    For REVE (per-channel tokens): each embedding corresponds to a single-
    channel 1-second patch → raw shape (N, 1, PATCH_SIZE).
    """
    is_backend = isinstance(model, EncoderBackend)
    # Determine if REVE-style per-channel tokens
    reve_mode = is_backend and hasattr(model, "channel_names")

    if is_backend:
        model.eval()
    else:
        model.eval()

    embed_list, spec_list, raw_list = [], [], []
    total = 0

    pbar = tqdm(val_loader, desc="Collecting tokens", leave=False)
    for batch in pbar:
        x = batch[0].to(DEVICE) if isinstance(batch, (list, tuple)) else batch.to(DEVICE)
        B, C, T = x.shape

        with torch.no_grad():
            if is_backend:
                layer_acts = model.encode(x)  # (B, S_total, E)
            else:
                inner_model = model
                extractor = ActivationExtractor(inner_model)
                extractor.clear()
                _ = inner_model(x)
                layer_acts = extractor.get_activations()[TARGET_LAYER]
                extractor.remove_hooks()

        N_tok = layer_acts.shape[1]

        if reve_mode:
            # REVE: N_tok = C_enc * S_reve tokens, each per-(channel, time)
            C_enc   = len(model.channel_names)
            S_reve  = N_tok // C_enc
            stride  = T // S_reve
            max_st  = T - PATCH_SIZE

            patches_list = []
            for s_idx in range(S_reve):
                st = min(s_idx * stride, max_st)
                patches_list.append(x[:, :C_enc, st : st + PATCH_SIZE])  # (B, C_enc, PATCH_SIZE)
            # (B, C_enc, S_reve, PATCH_SIZE)
            patches_bcsp = torch.stack(patches_list, dim=2)
            # Per-channel: (B * C_enc * S_reve, 1, PATCH_SIZE)
            patches_flat = patches_bcsp.permute(0, 1, 2, 3).reshape(
                B * C_enc * S_reve, 1, PATCH_SIZE
            )
            spec_targets = trainer.spectral.extract(patches_flat)
            n_new = B * C_enc * S_reve

            embed_list.append(layer_acts.reshape(n_new, -1).cpu())
            spec_list.append(spec_targets.cpu())
            raw_list.append(patches_flat.cpu())
        else:
            # SleepFM: N_tok = S cross-channel tokens
            S = T // PATCH_SIZE
            T_used  = S * PATCH_SIZE
            patches = x[:, :, :T_used].reshape(B, C, S, PATCH_SIZE)
            patches = patches.permute(0, 2, 1, 3).reshape(B * S, C, PATCH_SIZE)
            spec_targets = trainer.spectral.extract(patches)
            n_new = B * S

            embed_list.append(layer_acts.reshape(n_new, -1).cpu())
            spec_list.append(spec_targets.cpu())
            raw_list.append(patches.cpu())

        total += n_new
        pbar.set_postfix(tokens=f"{total:,}")
        if max_tokens and total >= max_tokens:
            break

    pbar.close()

    embeddings = torch.cat(embed_list)[:max_tokens]
    spectral   = torch.cat(spec_list)[:max_tokens]
    raw_full   = torch.cat(raw_list)[:max_tokens]
    return embeddings, spectral, raw_full


# =====================================================================
# Clustering -- in embedding space
# =====================================================================

def build_codebook(trainer, embeddings, gt_spectral, raw_full,
                   n_clusters=200):
    """Cluster tokens in whitened XAE-predicted spectral amplitude space.

    The 1/f problem:
    ----------------
    Raw EEG amplitude follows a 1/f power law.  Delta bins carry 10-100x
    more energy than alpha/beta bins.  Clustering in raw amplitude space
    (or raw embedding space) means KMeans subdivides the huge delta cloud
    while ignoring clinically important variation in higher bands.

    The fix -- whitening + stratified clustering:
    ---------------------------------------------
    1. Predict XAE log1p amplitudes for every token  (N x n_clean_bins)
    2. Compute per-bin mean and std across the population
    3. z-score per bin: z = (amp - mean) / std
       -> Every frequency bin now has variance 1.  The 1/f slope is removed.
    4. Assign each token its dominant frequency band (6 clinical bands).
    5. Sub-cluster within each stratum in whitened 45-d space.
       Budget: equal floor per band + proportional remainder.

    Exemplar selection:
    -------------------
    The exemplar (used for per-channel phase borrowing) is the real token
    whose EMBEDDING is nearest to the cluster centroid in EMBEDDING space.
    This keeps phase realistic (real EEG) while the amplitude comes from
    the cluster mean in whitened spectral space.
    """
    from sklearn.cluster import MiniBatchKMeans

    n_bins = trainer.spectral.n_bins
    freqs = trainer.spectral.freqs
    freq_mask = torch.tensor(trainer.spectral._freq_mask)

    # Mask to exclude >= 45 Hz bins
    clean_mask = freqs <= F_DISPLAY_MAX
    n_clean = int(clean_mask.sum())
    print(f"  Using {n_clean}/{n_bins} frequency bins "
          f"(0.5-{F_DISPLAY_MAX} Hz, excluding 50 Hz mains)")

    # ------------------------------------------------------------------
    # Step 1: Get XAE-predicted log1p amplitudes for ALL tokens
    # ------------------------------------------------------------------
    emb_np = embeddings.numpy()
    N, D = emb_np.shape

    device = next(trainer.xae.parameters()).device
    all_emb = embeddings.to(device)
    all_emb_norm = ((all_emb - trainer.embed_mean.to(device))
                    / trainer.embed_std.to(device))
    trainer.xae.eval()
    with torch.no_grad():
        pred_all_chunks = []
        for i in range(0, N, 2048):
            pred_all_chunks.append(
                trainer.xae.decode(all_emb_norm[i:i+2048]).cpu())
        pred_all_norm = torch.cat(pred_all_chunks)
    pred_all = (pred_all_norm * trainer.target_std.cpu()
                + trainer.target_mean.cpu())
    pr_amp_all, _, _ = trainer.spectral.unpack_targets(pred_all)  # (N, n_bins)

    # Restrict to clean frequency bins only
    amp_clean = pr_amp_all[:, clean_mask].numpy()   # (N, n_clean)

    # ------------------------------------------------------------------
    # Step 2: Whiten per frequency bin (remove 1/f dominance)
    # ------------------------------------------------------------------
    amp_mean = amp_clean.mean(axis=0)               # (n_clean,)
    amp_std  = amp_clean.std(axis=0).clip(min=1e-6) # (n_clean,) -- avoid /0
    amp_whitened = (amp_clean - amp_mean) / amp_std  # (N, n_clean)  z-scored

    print(f"\n  1/f whitening: per-bin z-score across {N} tokens")
    print(f"    Raw amp range:      "
          f"[{amp_clean.min():.3f}, {amp_clean.max():.3f}]")
    print(f"    Whitened amp range: "
          f"[{amp_whitened.min():.3f}, {amp_whitened.max():.3f}]")
    print(f"    Per-bin mean (raw): delta={amp_mean[:6].mean():.3f}  "
          f"alpha={amp_mean[18:24].mean():.3f}  "
          f"beta={amp_mean[28:38].mean():.3f}")

    # --- Variance explained by each clinical band, before vs after whitening ---
    print("\n  Band variance (fraction of total squared distance from global mean):")
    print(f"  {'Band':<10}  {'Before whiten':>14}  {'After whiten':>13}")
    total_var_raw    = np.var(amp_clean,    axis=0).sum()
    total_var_white  = np.var(amp_whitened, axis=0).sum()
    for bname, (f_lo, f_hi) in CLINICAL_BANDS.items():
        bmask = (freqs[clean_mask] >= f_lo) & (freqs[clean_mask] <= f_hi)
        var_raw   = np.var(amp_clean[:, bmask],    axis=0).sum()
        var_white = np.var(amp_whitened[:, bmask], axis=0).sum()
        print(f"  {bname:<10}  {var_raw/total_var_raw:>13.1%}  "
              f"{var_white/total_var_white:>12.1%}")
    print()  # blank line before clustering output

    # ------------------------------------------------------------------
    # Step 3: Stratified clustering — 6 clinical frequency band strata
    # ------------------------------------------------------------------
    # Each token is assigned to its dominant clinical band (whitened peak).
    # KMeans runs independently within each stratum, with a guaranteed
    # floor budget per band and proportional remainder allocation.

    freqs_clean = freqs[clean_mask]

    # Assign dominant band per token (whitened peak)
    band_list    = list(CLINICAL_BANDS.keys())
    band_masks_c = {b: (freqs_clean >= f_lo) & (freqs_clean <= f_hi)
                    for b, (f_lo, f_hi) in CLINICAL_BANDS.items()}
    band_scores = np.stack(
        [amp_whitened[:, band_masks_c[b]].mean(axis=1) for b in band_list],
        axis=1)                                      # (N, n_bands)
    token_dominant_band_idx = band_scores.argmax(axis=1)

    # Population per band
    band_pop = {b: int((token_dominant_band_idx == i).sum())
                for i, b in enumerate(band_list)}
    print("\n  Token population per dominant band:")
    for b in band_list:
        pct = 100 * band_pop[b] / max(N, 1)
        print(f"    {b:<10}: {band_pop[b]:>6} tokens  ({pct:.1f}%)")

    # Budget: equal floor per band + proportional remainder
    n_bands = len(band_list)
    floor_k = max(1, n_clusters // (n_bands * 2))
    budget  = {b: floor_k for b in band_list}
    leftover = n_clusters - floor_k * n_bands
    total_band_pop = sum(band_pop.values())
    for b in band_list:
        extra = int(round(leftover * band_pop[b] / max(total_band_pop, 1)))
        budget[b] += extra
    # Fix rounding to hit exactly n_clusters
    diff = n_clusters - sum(budget.values())
    budget[band_list[0]] += diff

    print(f"\n  Cluster budget (floor_k={floor_k}):")
    for b in band_list:
        print(f"    {b:<10}: {budget[b]:>4} clusters")

    # Sub-cluster each band stratum independently
    labels              = np.full(N, -1, dtype=int)
    centroids_whitened_list = []
    cluster_band_label  = []
    cluster_offset      = 0

    def _sub_cluster(idx_arr, k, stratum_name):
        """KMeans on amp_whitened[idx_arr], updates global labels/centroids."""
        nonlocal cluster_offset
        n_b      = len(idx_arr)
        k_actual = min(k, n_b)
        if k_actual < k:
            print(f"    WARNING: '{stratum_name}' has only {n_b} tokens, "
                  f"clamping {k}→{k_actual}")
        km_b = MiniBatchKMeans(
            n_clusters=k_actual, random_state=42,
            batch_size=min(2048, n_b), n_init=5, max_iter=300,
        )
        sub_lbl = km_b.fit_predict(amp_whitened[idx_arr])
        for local_c in range(k_actual):
            global_c = cluster_offset + local_c
            labels[idx_arr[sub_lbl == local_c]] = global_c
        centroids_whitened_list.append(km_b.cluster_centers_)
        cluster_band_label.extend([stratum_name] * k_actual)
        cluster_offset += k_actual

    # --- Band strata ---
    for b in band_list:
        b_idx = np.where(token_dominant_band_idx == band_list.index(b))[0]
        if len(b_idx) == 0:
            centroids_whitened_list.append(
                np.zeros((budget[b], n_clean), dtype=np.float32))
            cluster_band_label.extend([b] * budget[b])
            cluster_offset += budget[b]
        else:
            _sub_cluster(b_idx, budget[b], b)

    # Any unassigned tokens -> nearest centroid
    unassigned = np.where(labels == -1)[0]
    if len(unassigned):
        print(f"  Re-assigning {len(unassigned)} unassigned tokens...")
        all_centroids = np.vstack(centroids_whitened_list)
        for idx in unassigned:
            labels[idx] = np.linalg.norm(
                amp_whitened[idx] - all_centroids, axis=1).argmin()

    centroids_whitened = np.vstack(centroids_whitened_list)
    n_clusters_actual  = len(centroids_whitened)

    # Un-whiten centroids back to log1p amplitude space
    centroids_amp_clean = centroids_whitened * amp_std + amp_mean  # (K, n_clean)

    # ------------------------------------------------------------------
    # Step 4: For each cluster, find exemplar in EMBEDDING space
    # (nearest real token to the cluster mean embedding)
    # ------------------------------------------------------------------
    K = n_clusters_actual
    cluster_mean_emb = np.zeros((K, D), dtype=np.float32)
    exemplar_idx = np.zeros(K, dtype=int)
    exemplar_dist = np.zeros(K)
    cluster_sizes = np.zeros(K, dtype=int)
    intra_var_spec = np.zeros(K)  # intra-cluster spread in whitened space

    for c in range(K):
        mask_c = labels == c
        member_indices = np.where(mask_c)[0]
        cluster_sizes[c] = len(member_indices)

        if len(member_indices) == 0:
            continue

        # Mean embedding of cluster members
        cluster_mean_emb[c] = emb_np[mask_c].mean(axis=0)

        # Intra-cluster spread in whitened spectral space
        spec_dists = np.linalg.norm(
            amp_whitened[mask_c] - centroids_whitened[c], axis=1)
        intra_var_spec[c] = spec_dists.mean()

        # Exemplar: nearest token to cluster mean in EMBEDDING space
        emb_dists = np.linalg.norm(emb_np[mask_c] - cluster_mean_emb[c], axis=1)
        best = emb_dists.argmin()
        exemplar_idx[c] = member_indices[best]
        exemplar_dist[c] = emb_dists[best]

    # Build full centroid amplitude array (all n_bins), zeros outside clean range
    pr_amp_clean = np.zeros((K, n_bins), dtype=np.float32)
    pr_amp_clean[:, clean_mask] = centroids_amp_clean

    gt_amp_all, _, _ = trainer.spectral.unpack_targets(gt_spectral)

    # -- Reconstruct realistic per-channel waveforms -----------------
    n_freq = PATCH_SIZE // 2 + 1
    C_ch = raw_full.shape[1]   # 19 channels

    # Extract per-channel phase from each exemplar raw patch
    exemplar_patches = raw_full[exemplar_idx]              # (K, 19, 128)
    exemplar_fft = torch.fft.rfft(exemplar_patches, dim=-1)  # (K, 19, 65)
    exemplar_phase = exemplar_fft[:, :, freq_mask].angle()  # (K, 19, n_bins)

    # Build complex spectrum: XAE amp (clean) x exemplar phase per channel
    pr_amp_clean_t = torch.tensor(pr_amp_clean)             # (K, n_bins)
    amp_lin = torch.expm1(pr_amp_clean_t)                   # (K, n_bins)
    amp_exp = amp_lin.unsqueeze(1).expand(-1, C_ch, -1)     # (K, 19, n_bins)

    full_spectrum = torch.zeros(K, C_ch, n_freq, dtype=torch.cfloat)
    full_spectrum[:, :, freq_mask] = amp_exp * torch.exp(1j * exemplar_phase)
    recon_waveforms = torch.fft.irfft(full_spectrum, n=PATCH_SIZE, dim=-1).float()
    # recon_waveforms: (K, 19, 128)

    # -- Band powers from XAE-predicted centroid amplitude -----------
    band_powers = {}
    pr_amp_clean_np = pr_amp_clean  # already numpy
    for band_name, (f_lo, f_hi) in CLINICAL_BANDS.items():
        band_mask = (freqs >= f_lo) & (freqs <= f_hi) & clean_mask
        amp_lin_band = np.expm1(pr_amp_clean_np[:, band_mask])
        band_powers[band_name] = (amp_lin_band ** 2).mean(axis=1)  # (K,)

    # -- Spectral R2 between XAE and GT for each exemplar ------------
    exemplar_r2 = []
    for c in range(K):
        idx = exemplar_idx[c]
        gt_a = gt_amp_all[idx, clean_mask].numpy()
        pr_a = pr_amp_all[idx, clean_mask].numpy()
        ss_res = ((gt_a - pr_a) ** 2).sum()
        ss_tot = ((gt_a - gt_a.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8) if ss_tot > 0.01 else float("nan")
        exemplar_r2.append(r2)

    # -- Waveform Pearson r: reconstructed vs actual exemplar --------
    exemplar_recon_r = []
    for c in range(K):
        idx = exemplar_idx[c]
        raw_sig = raw_full[idx]                     # (19, 128)
        rec_sig = recon_waveforms[c]                # (19, 128)
        # Mean Pearson r across channels
        a = raw_sig - raw_sig.mean(dim=-1, keepdim=True)
        b = rec_sig - rec_sig.mean(dim=-1, keepdim=True)
        num = (a * b).sum(dim=-1)
        den = (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-8)
        r_ch = (num / den).numpy()
        exemplar_recon_r.append(np.nanmean(r_ch))

    # -- Sort clusters by total power (dominant -> weakest) ----------
    total_power = np.expm1(pr_amp_clean_np[:, clean_mask]).sum(axis=1)
    sort_order = np.argsort(-total_power)

    codebook = {
        "n_clusters": K,
        "labels": labels,
        "centroids_emb": cluster_mean_emb,
        "centroids_whitened": centroids_whitened,
        "amp_mean": amp_mean,
        "amp_std": amp_std,
        "cluster_feature": "stratified_whitened_spectral_45bins",
        "cluster_budget": budget,
        "cluster_band_label": cluster_band_label,
        "centroid_amp_log1p": pr_amp_clean,
        "clean_mask": clean_mask,
        "exemplar_idx": exemplar_idx,
        "exemplar_dist": exemplar_dist,
        "exemplar_raw_patches": exemplar_patches,  # (K, C, PATCH_SIZE) raw signal
        "cluster_sizes": cluster_sizes,
        "intra_var": intra_var_spec,
        "band_powers": band_powers,
        "recon_waveforms": recon_waveforms,
        "exemplar_r2": np.array(exemplar_r2),
        "exemplar_recon_r": np.array(exemplar_recon_r),
        "sort_order": sort_order,
        "freqs": freqs,
        "freqs_clean": freqs[clean_mask],
    }

    # Print quality stats
    r_arr = np.array(exemplar_recon_r)
    print("\n  Waveform quality (XAE amp + exemplar phase vs raw exemplar):")
    print(f"    Mean r = {np.nanmean(r_arr):.3f},  "
          f"Median r = {np.nanmedian(r_arr):.3f}")

    # Band diversity: clusters per stratum (guaranteed by stratification)
    all_band_names = list(CLINICAL_BANDS.keys())
    dom_counts = {b: cluster_band_label.count(b) for b in all_band_names}
    print("\n  Cluster band diversity (stratum, guaranteed by stratification):")
    for b, cnt in dom_counts.items():
        bar = "█" * cnt
        print(f"    {b:<10}: {cnt:>4} clusters  {bar}")

    return codebook


# =====================================================================
# Visualisation -- compact card pages
# =====================================================================

def plot_codebook_pages(codebook, out_dir, cards_per_page=10):
    """Generate pages of prototype cards for manual inspection."""
    n_clusters = codebook["n_clusters"]
    sort_order = codebook["sort_order"]
    centroid_amp = codebook["centroid_amp_log1p"]
    recon_waveforms = codebook["recon_waveforms"]
    band_powers = codebook["band_powers"]
    cluster_sizes = codebook["cluster_sizes"]
    freqs_clean = codebook["freqs_clean"]
    clean_mask = codebook["clean_mask"]
    exemplar_recon_r = codebook["exemplar_recon_r"]

    band_names = list(CLINICAL_BANDS.keys())
    t_axis = np.arange(PATCH_SIZE) / FS
    ch_items = list(SHOW_CHANNELS.items())

    n_pages = int(np.ceil(n_clusters / cards_per_page))
    cols_per_page = 2
    rows_per_page = int(np.ceil(cards_per_page / cols_per_page))

    for page in range(n_pages):
        start = page * cards_per_page
        end = min(start + cards_per_page, n_clusters)
        page_clusters = sort_order[start:end]

        fig = plt.figure(figsize=(30, 4.2 * rows_per_page))
        outer = gridspec.GridSpec(
            rows_per_page, cols_per_page, figure=fig,
            hspace=0.50, wspace=0.25,
        )

        for card_i, c in enumerate(page_clusters):
            row = card_i // cols_per_page
            col = card_i % cols_per_page

            inner = gridspec.GridSpecFromSubplotSpec(
                len(ch_items), 2, subplot_spec=outer[row, col],
                hspace=0.15, wspace=0.30,
                width_ratios=[2.5, 1],
            )

            rank = start + card_i
            size = cluster_sizes[c]
            recon_r = exemplar_recon_r[c]

            # -- Left panel: Reconstructed per-channel waveforms ----
            n_ch_recon = recon_waveforms.shape[1]
            for ci, (ch_name, ch_idx) in enumerate(ch_items):
                ax = fig.add_subplot(inner[ci, 0])
                # Channel-pooled XAE outputs a single waveform per token; fall
                # back to that lone channel rather than indexing out of bounds.
                ch_use = ch_idx if ch_idx < n_ch_recon else 0
                sig = recon_waveforms[c, ch_use].numpy()
                ymax = np.abs(sig).max() * 1.15
                if ymax < 1e-8:
                    ymax = 1.0

                ax.plot(t_axis, sig, color="black", linewidth=1.0)
                ax.axhline(0, color="grey", linewidth=0.3)
                ax.set_ylim(-ymax, ymax)
                ax.set_ylabel(ch_name, fontsize=8, fontweight="bold",
                              rotation=0, labelpad=20, va="center")
                ax.tick_params(labelsize=5, left=False, labelleft=False)
                ax.grid(True, alpha=0.08)

                if ci == 0:
                    bp = [band_powers[b][c] for b in band_names]
                    dom_band = band_names[np.argmax(bp)]
                    title_color = ("#1565C0" if size >= 20
                                   else "#EF6C00" if size >= 5
                                   else "#C62828")
                    ax.set_title(
                        f"#{rank+1}  (n={size})  --  {dom_band}  "
                        f"[r={recon_r:.2f}]",
                        fontsize=9, fontweight="bold", color=title_color,
                    )

                if ci == len(ch_items) - 1:
                    ax.tick_params(bottom=True, labelbottom=True, labelsize=6)
                    ax.set_xlabel("Time (s)", fontsize=7)
                else:
                    ax.set_xticklabels([])

            # -- Right panel: Spectrum (top 2 rows) + Bars (bottom) --
            inner_right = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=inner[:2, 1],
                hspace=0.45,
            )

            # Amplitude spectrum
            ax_spec = fig.add_subplot(inner_right[0])
            amp_lin = np.expm1(centroid_amp[c, clean_mask])
            ax_spec.fill_between(freqs_clean, amp_lin, alpha=0.15,
                                 color="black")
            ax_spec.plot(freqs_clean, amp_lin, "k-", linewidth=0.9)
            for bname, (lo, hi) in CLINICAL_BANDS.items():
                ax_spec.axvspan(lo, hi, alpha=0.06, color=BAND_COLORS[bname])
            ax_spec.set_xlabel("Hz", fontsize=6)
            ax_spec.set_ylabel("amp", fontsize=6)
            ax_spec.tick_params(labelsize=5)
            ax_spec.grid(True, alpha=0.08)
            ax_spec.set_xlim(0, F_DISPLAY_MAX + 1)

            # Band power bars
            ax_bar = fig.add_subplot(inner_right[1])
            bp = [band_powers[b][c] for b in band_names]
            bp_total = sum(bp) + 1e-12
            bp_frac = [p / bp_total * 100 for p in bp]
            colors = [BAND_COLORS[b] for b in band_names]
            short_names = ["\u03b4", "\u03b8", "\u03b1",
                           "l\u03b2", "h\u03b2", "\u03b3"]
            ax_bar.barh(range(len(band_names)), bp_frac,
                        color=colors, height=0.65, alpha=0.85)
            ax_bar.set_yticks(range(len(band_names)))
            ax_bar.set_yticklabels(short_names, fontsize=6,
                                   fontweight="bold")
            ax_bar.set_xlabel("% power", fontsize=6)
            ax_bar.tick_params(labelsize=5)
            ax_bar.invert_yaxis()
            ax_bar.set_xlim(0, 100)
            ax_bar.grid(axis="x", alpha=0.1)

        fig.suptitle(
            f"Embedding-Space Codebook -- Page {page+1}/{n_pages}  "
            f"(prototypes {start+1}-{end}, sorted by total power)\n"
            f"XAE-predicted amplitude + exemplar per-channel phase  "
            f"(0.5-{F_DISPLAY_MAX:.0f} Hz, 50 Hz mains excluded)",
            fontsize=12, fontweight="bold", y=1.01,
        )

        save_path = out_dir / f"page_{page+1:02d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Page {page+1}/{n_pages}: prototypes {start+1}-{end}")


# =====================================================================
# Text summary
# =====================================================================

def write_summary(codebook, out_path):
    """Write a text summary of all prototypes."""
    sort_order = codebook["sort_order"]
    cluster_sizes = codebook["cluster_sizes"]
    intra_var = codebook["intra_var"]
    exemplar_r2 = codebook["exemplar_r2"]
    exemplar_recon_r = codebook["exemplar_recon_r"]
    band_powers = codebook["band_powers"]
    band_names = list(CLINICAL_BANDS.keys())

    lines = []
    lines.append("=" * 90)
    lines.append("  Embedding-Space Codebook Summary")
    lines.append("  Clustered in 128-d SetTransformer embedding space")
    lines.append("  Waveforms: XAE amplitude (centroid) + GT phase (exemplar)")
    lines.append(f"  Frequencies: 0.5-{F_DISPLAY_MAX:.0f} Hz "
                 f"(50 Hz mains noise excluded)")
    lines.append("=" * 90)
    lines.append(f"  Total prototypes: {codebook['n_clusters']}")
    lines.append(f"  Total tokens:     {len(codebook['labels'])}")
    lines.append("")
    lines.append(f"  {'#':>4s}  {'Cl':>4s}  {'Size':>5s}  {'EmbVar':>6s}  "
                 f"{'SpecR2':>6s}  {'WavR':>5s}  {'Dom.':>6s}  "
                 f"{'d%':>5s} {'t%':>5s} {'a%':>5s} {'lb%':>5s} "
                 f"{'hb%':>5s} {'g%':>5s}")
    lines.append("  " + "-" * 88)

    for rank, c in enumerate(sort_order):
        size = cluster_sizes[c]
        var = intra_var[c]
        r2 = exemplar_r2[c]
        wr = exemplar_recon_r[c]
        bp = [band_powers[b][c] for b in band_names]
        bp_total = sum(bp) + 1e-12
        bp_pct = [p / bp_total * 100 for p in bp]
        dom_band = band_names[np.argmax(bp)]

        lines.append(
            f"  {rank+1:>4d}  {c:>4d}  {size:>5d}  {var:>6.3f}  "
            f"{r2:>6.3f}  {wr:>5.3f}  {dom_band:>6s}  "
            + "  ".join(f"{p:4.1f}" for p in bp_pct)
        )

    out_path.write_text("\n".join(lines))
    print(f"  Summary -> {out_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    global DATA_PATH, XAE_PATH, FS, PATCH_SIZE, TARGET_LAYER

    parser = argparse.ArgumentParser(
        description="Build embedding-space codebook with realistic waveforms")
    _all_encoders = ["sleepfm", "sleepfm_v2.0", "sleepfm_v2.1", "sleepfm_v2.3", "sleepfm_v2.4", "sleepfm_v2.5", "sleepfm_v2.6", "sleepfm_v2.7", "sleepfm_granular", "reve", "labram"]
    parser.add_argument("--encoder", default="sleepfm", choices=_all_encoders,
                        help="Encoder backend (default: sleepfm)")
    parser.add_argument("--weights-path", default=None,
                        help="Path to finetuned checkpoint (REVE .ckpt or SleepFM .pt)")
    parser.add_argument("--n-clusters", type=int, default=200,
                        help="Number of prototypes (default: 200)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help="Max tokens to collect (default: 20000)")
    parser.add_argument("--cards-per-page", type=int, default=10,
                        help="Cards per page (default: 10)")
    parser.add_argument("--tag", default=None,
                        help="Run tag (e.g. 'qjbe08') — scopes XAE and output "
                             "paths to results/**/{encoder}_{tag}/")
    args = parser.parse_args()

    # ── Encoder-specific settings ────────────────────────────────────────
    run_name   = f"{args.encoder}_{args.tag}" if args.tag else args.encoder
    DATA_PATH  = _ENCODER_DATA[args.encoder]
    FS         = _ENCODER_FS[args.encoder]
    PATCH_SIZE = _ENCODER_PATCH[args.encoder]

    # XAE path: legacy flat path for base sleepfm (no tag), run-scoped otherwise
    if args.encoder in _ENCODER_XAE and not args.tag:
        XAE_PATH = _ENCODER_XAE[args.encoder]
    else:
        XAE_PATH = ROOT / "results" / "xae" / run_name / "xae_checkpoint.pt"

    # Encoder weights
    if args.weights_path is not None:
        resolved_weights = args.weights_path
    elif args.encoder == "sleepfm":
        resolved_weights = MODEL_PATH
    elif args.encoder in _V2_CHECKPOINTS:
        resolved_weights = _V2_CHECKPOINTS[args.encoder]
    elif args.encoder == "sleepfm_granular":
        resolved_weights = ROOT / "checkpoints" / "granular" / "sleepfm_granular.ckpt"
    elif args.encoder == "labram":
        ckpt = ROOT / "checkpoints" / "finetuned" / "labram_binary" / "finetuned.ckpt"
        resolved_weights = ckpt if ckpt.exists() else (ROOT / "checkpoints" / "pretrained" / "labram" / "labram-base.pth")
    else:
        resolved_weights = None

    kwargs = {"weights_path": resolved_weights} if resolved_weights is not None else {}
    model = load_encoder(args.encoder, **kwargs)
    model.to(DEVICE).eval()

    n_layers   = len(model.get_hookable_layers())
    TARGET_LAYER = n_layers - 1
    embed_dim  = _ENCODER_EMBED[args.encoder]

    out_dir = ROOT / "results" / "xae" / run_name / "codebook"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Embedding-Space Codebook -- Prototypical EEG Tokens")
    print("  Clustering in 128-d token space")
    print("  Waveforms: XAE amplitude + exemplar per-channel phase")
    print(f"  (0.5-{F_DISPLAY_MAX:.0f} Hz, 50 Hz mains excluded)")
    print("=" * 72)

    # -- Load XAE and data -------------------------------------------
    trainer = XAETrainer(embed_dim=embed_dim, device=DEVICE)
    trainer.load(str(XAE_PATH))
    trainer.xae.to(DEVICE).eval()

    transform = V4ResampleTransform() if "D4-v4" in str(DATA_PATH) else StandardizeLabel()
    gen = get_dataloaders(train_path=str(DATA_PATH),
                          transformer=transform)
    _, _, val_loader, _ = next(gen)
    print(f"[ok] Loaded {args.encoder} model, XAE, and data\n")

    # -- Collect tokens ----------------------------------------------
    embeddings, gt_spectral, raw_full = collect_tokens(
        trainer, model, val_loader, max_tokens=args.max_tokens,
    )
    print(f"  -> {len(embeddings)} tokens collected "
          f"({raw_full.shape[1]} channels, {raw_full.shape[2]} samples)\n")

    # -- Build codebook ----------------------------------------------
    codebook = build_codebook(
        trainer, embeddings, gt_spectral, raw_full,
        n_clusters=args.n_clusters,
    )

    # Print size distribution
    sizes = codebook["cluster_sizes"][codebook["sort_order"]]
    print("\n  Cluster size distribution:")
    print(f"    Min: {sizes.min()},  Max: {sizes.max()},  "
          f"Median: {np.median(sizes):.0f},  Mean: {sizes.mean():.1f}")
    print(f"    Clusters with >=20 tokens: {(sizes >= 20).sum()}")
    print(f"    Clusters with >=10 tokens: {(sizes >= 10).sum()}")
    print(f"    Clusters with <5 tokens:   {(sizes < 5).sum()}")

    # -- Save codebook data ------------------------------------------
    save_data = {
        "n_clusters":          codebook["n_clusters"],
        "cluster_feature":     codebook["cluster_feature"],
        "cluster_budget":      codebook["cluster_budget"],
        "cluster_band_label":  codebook["cluster_band_label"],
        "centroids_emb":       torch.tensor(codebook["centroids_emb"]),
        "centroids_whitened":  torch.tensor(codebook["centroids_whitened"]),
        "amp_mean":            torch.tensor(codebook["amp_mean"]),
        "amp_std":             torch.tensor(codebook["amp_std"]),
        "centroid_amp_log1p":  torch.tensor(codebook["centroid_amp_log1p"]),
        "clean_mask":          torch.tensor(codebook["clean_mask"]),
        "exemplar_idx":        torch.tensor(codebook["exemplar_idx"]),
        "exemplar_raw_patches": codebook["exemplar_raw_patches"],  # (K, C, P)
        "cluster_sizes":       torch.tensor(codebook["cluster_sizes"]),
        "intra_var":           torch.tensor(codebook["intra_var"]),
        "sort_order":          torch.tensor(codebook["sort_order"]),
        "labels":              torch.tensor(codebook["labels"]),
        "recon_waveforms":     codebook["recon_waveforms"],
        "exemplar_recon_r":    torch.tensor(codebook["exemplar_recon_r"]),
        "exemplar_r2":         torch.tensor(codebook["exemplar_r2"]),
        "band_powers":         {b: torch.tensor(v)
                                for b, v in codebook["band_powers"].items()},
        "freqs":               codebook["freqs"],
        "freqs_clean":         codebook["freqs_clean"],
        "f_display_max":       F_DISPLAY_MAX,
    }
    pt_path = out_dir / "codebook.pt"
    torch.save(save_data, pt_path)
    print(f"\n  Codebook data -> {pt_path}")

    # -- Generate visual pages ---------------------------------------
    print(f"\n[Generating codebook pages ({args.cards_per_page} cards/page)...]")
    plot_codebook_pages(codebook, out_dir, cards_per_page=args.cards_per_page)

    # -- Text summary ------------------------------------------------
    write_summary(codebook, out_dir / "codebook_summary.txt")

    print(f"\nDone!  Codebook with {args.n_clusters} prototypes "
          f"saved to {out_dir}/")


if __name__ == "__main__":
    main()
