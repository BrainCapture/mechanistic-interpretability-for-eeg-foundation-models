"""
feature_maximization.py — SAE Feature Activation Maximisation for SleepFM v1.1
================================================================================
Finds prototypical EEG signals that maximally activate each target SAE feature,
following the circuits / feature-visualisation paradigm (Olah et al., 2017).

Two feature-ranking modes
--------------------------
  --rank-by firing_rate   (default) top-N by fraction of tokens firing
  --rank-by tcav          top-N by delta firing rate on the abnormality concept
                          (requires --tcav-cache)

Warm-start initialisation
--------------------------
Each optimisation is seeded with a real data sample that already activates the
target feature, so gradients are non-zero from step 1.  By default uses samples
from class 1 (abnormal) since that is the class of interest.  Override with
--target-class 0.

Objective (per feature i)
--------------------------
  min_x  -max_s z_i(f_l(x_s))  +  λ_l2 * ||x||²  +  λ_tv * TV(x)

where x_s are the token activations at each 1-second patch s.  Using max over
tokens (rather than mean) lets the optimiser concentrate the spike at a single
temporal position — matching the clinical observation that abnormal events are
transient.

Usage
-----
    # Abnormality-sensitive features, warm-started from abnormal data
    uv run tools/feature_maximization.py \\
        --experiment sleepfm_finetuned_layer2 \\
        --rank-by tcav \\
        --tcav-cache results/tcav/sleepfm_finetuned_layer2/tcav_cache.pt

    # Specify features manually
    uv run tools/feature_maximization.py \\
        --experiment sleepfm_finetuned_layer2 \\
        --feature-indices 86,41,49,58,38,44,34,81,40,72

Outputs
-------
    results/feature_viz/<experiment>/
        prototypes.npz      — signals (N, C, T), metadata, loss histories
        feature_viz.png     — signal + spectrum per feature
        feature_ranking.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sae4eeg.encoders import load_encoder
from sae4eeg.sae import SparseAutoencoder

# ── Dataset / model constants (SleepFM defaults; overridden per-experiment) ──
_DATA_PATH   = "data/D4-v3-preprocessed-v2"
_PATCH_SIZE  = 128           # samples per token (1 s @ 128 Hz)
FS           = 128
N_CHANNELS   = 27

# Encoder → data path / patch size / fs (mirrors build_app_cache.py)
_ENCODER_DATA = {
    "sleepfm":            ("data/D4-v3-preprocessed-v2", 128, 128),
    "sleepfm_pretrained": ("data/D4-v3-preprocessed-v2", 128, 128),
    "sleepfm_finetuned":  ("data/D4-v3-preprocessed-v2", 128, 128),
    "sleepfm_granular":   ("data/D4-v4-preprocessed-10s", 128, 128),
    "reve":               ("data/D4-v3-preprocessed-v1", 200, 200),
    "reve_qjbe08":        ("data/D4-v3-preprocessed-v1", 200, 200),
    "labram":             ("data/D4-v3-preprocessed-v1", 200, 200),
}

_CHANNEL_NAMES = [
    "Fp1", "Fp2",
    "F9", "F7", "F3", "Fz", "F4", "F8", "F10",
    "T9", "T7", "C3", "Cz", "C4", "T8", "T10",
    "TP7", "TP8",
    "P9", "P7", "P3", "Pz", "P4", "P8", "P10",
    "O1", "O2",
]

_EEG_BANDS = {
    "δ": (0.5,  4,  "#4477AA"),
    "θ": (4,    8,  "#66CCEE"),
    "α": (8,   13,  "#228833"),
    "β": (13,  30,  "#CCBB44"),
    "γ": (30,  60,  "#EE6677"),
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sae(checkpoint_path: str | Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sae  = SparseAutoencoder(
        input_dim=ckpt["embed_dim"],
        expansion=ckpt["expansion"],
        mode="topk",
        k=ckpt["k"],
    )
    sae.load_state_dict(ckpt["sae_state_dict"])
    sae.eval().requires_grad_(False)
    return sae, ckpt["act_mean"], ckpt["act_std"]


def load_experiment(experiment: str, experiments_dir: str = "results/experiments", device: str = "cpu"):
    meta = json.loads((Path(experiments_dir) / experiment / "metadata.json").read_text())
    backend = load_encoder(meta["encoder"], weights_path=meta["weights_path"])
    backend.to(device).eval()
    backend.model.requires_grad_(False)
    sae, act_mean, act_std = load_sae(meta["sae_checkpoint"])
    sae = sae.to(device)
    return backend.model, sae, act_mean.to(device), act_std.to(device), meta


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Gradient-tracking layer capture
# ─────────────────────────────────────────────────────────────────────────────

class _LayerCapture:
    """Hook that captures a layer output WITHOUT detaching (keeps autograd graph)."""
    def __init__(self, layer: nn.Module):
        self.activation: Optional[torch.Tensor] = None
        self._hook = layer.register_forward_hook(self._fn)

    def _fn(self, _m, _i, output):
        self.activation = output

    def remove(self):
        self._hook.remove()


def _layer_act(raw_model: nn.Module, x: torch.Tensor, target_layer: int) -> torch.Tensor:
    """Forward pass returning target-layer activation tensor (in autograd graph)."""
    cap = _LayerCapture(raw_model.transformer_encoder.layers[target_layer])
    try:
        raw_model(x)
    finally:
        cap.remove()
    if cap.activation is None:
        raise RuntimeError(f"Layer {target_layer} hook did not fire.")
    return cap.activation   # (B, S, E)


@torch.no_grad()
def _layer0_attention_received(raw_model: nn.Module, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Extract per-token attention received from temporal transformer layer 0.

    Replicates the SleepFM forward pass up to layer 0 and calls self_attn with
    need_weights=True — the only way to obtain attention weights since
    TransformerEncoderLayer discards them in its normal forward path.

    Returns
    -------
    Tensor (B, S) — mean attention received by each token (averaged over heads
    and over queries), or None if the model has no temporal transformer.
    """
    if not hasattr(raw_model, "transformer_encoder"):
        return None
    try:
        from einops import rearrange
        emb = raw_model.patch_embedding(x)          # (B, C_ch, S, E)
        B, C_ch, S, E = emb.shape
        emb = rearrange(emb, "b c s e -> (b s) c e")
        emb = raw_model.spatial_pooling(emb)        # (B*S, E)
        emb = emb.view(B, S, E)
        emb = raw_model.positional_encoding(emb)
        emb = raw_model.layer_norm(emb)
        layer0 = raw_model.transformer_encoder.layers[0]
        x_norm = layer0.norm1(emb)
        _, attn = layer0.self_attn(
            x_norm, x_norm, x_norm,
            need_weights=True,
            average_attn_weights=False,             # (B, n_heads, S, S)
        )
        return attn.mean(dim=1).mean(dim=1).cpu()   # (B, S)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature ranking
# ─────────────────────────────────────────────────────────────────────────────

def _make_dataset(data_path: str):
    """Return H5PYDatasetLabeled with the correct transform for the data version."""
    from sae4eeg.dataset import H5PYDatasetLabeled, V4ResampleTransform, StandardizeLabel
    transform = V4ResampleTransform() if "D4-v4" in str(data_path) else StandardizeLabel()
    return H5PYDatasetLabeled(data_path, transform=transform)


def rank_by_firing_rate(
    raw_model, sae, act_mean, act_std, target_layer,
    data_path=_DATA_PATH, max_batches=20, batch_size=8, device="cpu",
) -> torch.Tensor:
    """Returns (n_features,) tensor of firing rates from a data sample."""
    loader = DataLoader(_make_dataset(data_path), batch_size=batch_size, shuffle=True)
    fire_counts = torch.zeros(sae.dict_size)
    total = 0
    for i, batch in enumerate(loader):
        if i >= max_batches: break
        x = batch[0].to(device).float()
        with torch.no_grad():
            act = _layer_act(raw_model, x, target_layer).to(device)
            z   = sae.encode(((act - act_mean) / act_std).reshape(-1, act.shape[-1]))
        fire_counts += (z > 0).float().sum(dim=0).cpu()
        total += z.shape[0]
        print(f"  ranking batch {i+1}/{max_batches} | tokens: {total}", flush=True)
    return fire_counts / max(total, 1)


def rank_by_label_corr(app_cache_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load label_correlation and label_p_value from a pre-built app_cache.pt.
    Returns (|r| values, p_values) as (n_features,) tensors.
    Features with high |r| and low p fire significantly more on one class.
    """
    cache = torch.load(app_cache_path, map_location="cpu", weights_only=False)
    stats = cache["feature_stats"]          # list of dicts, one per feature
    n = len(stats)
    corr = torch.zeros(n)
    pval = torch.ones(n)
    for s in stats:
        fi = s["feature"]
        corr[fi] = abs(s.get("label_correlation", 0.0) or 0.0)
        pval[fi] = s.get("label_p_value", 1.0) or 1.0
    return corr, pval


def rank_by_tcav_abnorm(tcav_cache_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (delta_rates, p_values) for the 'abnormal' concept, shape (n_features,).
    Features with high delta_rate fire significantly more on abnormal than normal examples.
    """
    cache = torch.load(tcav_cache_path, map_location="cpu", weights_only=False)
    names = cache["concept_names"]
    idx   = next(i for i, n in enumerate(names) if "abnorm" in n.lower())
    print(f"  TCAV: using concept '{names[idx]}' (index {idx})")
    return cache["delta_rates"][idx], cache["p_values"][idx]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Warm-start: find real abnormal sample that activates the feature
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def find_warmstart(
    feature_idx: int,
    raw_model: nn.Module,
    sae: SparseAutoencoder,
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
    target_layer: int,
    data_path: str = _DATA_PATH,
    target_class: int = 1,
    n_batches: int = 30,
    batch_size: int = 8,
    device: str = "cpu",
) -> tuple[Optional[torch.Tensor], float, int]:
    """
    Return the FULL recording (C, T_full) from `target_class` that most
    strongly activates `feature_idx`, along with the activation value and the
    index of the peak token.

    We return the full recording (not a crop) because the transformer uses
    self-attention across all S tokens: cropping to fewer tokens changes the
    representation and kills the feature activation.  The caller optimises the
    full window and crops only for display.
    """
    loader = DataLoader(
        _make_dataset(data_path), batch_size=batch_size, shuffle=True, num_workers=0
    )

    best_x:    Optional[torch.Tensor] = None
    best_act:  float = -1.0
    best_tok:  int   = 0

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        x_full, labels, _ch = batch
        mask = labels == target_class
        if mask.sum() == 0:
            continue
        x_cls = x_full[mask].to(device).float()

        act = _layer_act(raw_model, x_cls, target_layer).to(device)   # (B, S, E)
        z   = sae.encode(((act - act_mean) / act_std).reshape(-1, act.shape[-1]))

        B, S = act.shape[:2]
        z_3d = z.reshape(B, S, -1)                                    # (B, S, n_feat)
        act_per_sample, peak_token = z_3d[:, :, feature_idx].max(dim=1)

        batch_best_val, batch_best_idx = act_per_sample.max(dim=0)
        if batch_best_val.item() > best_act:
            best_act = batch_best_val.item()
            best_tok  = peak_token[batch_best_idx].item()
            best_x   = x_cls[batch_best_idx].cpu()                   # full (C, T_full)

        print(f"  warmstart batch {i+1}/{n_batches} | feature {feature_idx} "
              f"| best so far: {best_act:.3f}", end="\r", flush=True)

    print()
    return best_x, best_act, best_tok


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Empirical spatial profile from real data
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_top_activating_patches(
    feature_idx: int,
    raw_model: nn.Module,
    sae: SparseAutoencoder,
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
    target_layer: int,
    data_path: str = _DATA_PATH,
    n_top: int = 100,
    max_batches: int = 60,
    batch_size: int = 8,
    device: str = "cpu",
    use_attention: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scan the dataset and collect the top-n_top EEG patches (C, patch_size)
    where feature_idx has the highest post-TopK activation.

    When use_attention=True (default), patches are ranked by
    norm(activation) × norm(attention_received) so that only tokens which
    both fire the feature AND are contextually attended to are surfaced.
    Falls back to activation-only ranking if the model has no temporal
    transformer (e.g. REVE).

    Returns
    -------
    patches     : (n_top, C, patch_size) float32 — raw EEG patches
    activations : (n_top,)               float32 — feature activation values
    top_joint   : float                          — joint score of the best patch
    """
    loader = DataLoader(
        _make_dataset(data_path), batch_size=batch_size, shuffle=True, num_workers=0
    )

    top_acts:    list[float]      = []
    top_attn:    list[float]      = []
    top_patches: list[np.ndarray] = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x_full, _labels, _ch = batch
        x = x_full.to(device).float()

        act = _layer_act(raw_model, x, target_layer).to(device)   # (B, S, E)
        z   = sae.encode(((act - act_mean) / act_std).reshape(-1, act.shape[-1]))  # (B*S, F)

        B, S, _ = act.shape
        z_3d      = z.reshape(B, S, -1)
        feat_acts = z_3d[:, :, feature_idx]           # (B, S)

        attn_recv = None
        if use_attention:
            attn_recv = _layer0_attention_received(raw_model, x)  # (B, S) or None

        for b in range(B):
            for s in range(S):
                val = feat_acts[b, s].item()
                if val <= 0:
                    continue
                patch = x_full[b, :, s * _PATCH_SIZE:(s + 1) * _PATCH_SIZE].numpy()
                top_acts.append(val)
                top_patches.append(patch)
                top_attn.append(float(attn_recv[b, s]) if attn_recv is not None else 1.0)

        print(f"  spatial profile: batch {i+1}/{max_batches} | "
              f"collected {len(top_acts)} activating patches", end="\r", flush=True)

    print()

    if not top_acts:
        return np.zeros((0, N_CHANNELS, _PATCH_SIZE)), np.array([]), 0.0

    acts_arr = np.array(top_acts)
    attn_arr = np.array(top_attn)

    # Rank by joint score: norm(activation) × norm(attention_received)
    acts_norm = acts_arr / (acts_arr.max() + 1e-8)
    attn_norm = attn_arr / (attn_arr.max() + 1e-8)
    joint     = acts_norm * attn_norm

    order       = np.argsort(joint)[::-1][:n_top]
    patches     = np.stack([top_patches[j] for j in order])
    activations = acts_arr[order]
    top_joint   = float(joint[order[0]])
    print(f"  spatial profile: kept top {len(activations)} patches  "
          f"(max act={activations[0]:.3f}, joint={top_joint:.4f})")
    return patches, activations, top_joint


def empirical_spatial_rms(patches: np.ndarray) -> np.ndarray:
    """Mean per-channel RMS across a set of EEG patches (N, C, T) → (C,)."""
    rms_per_patch = np.sqrt((patches ** 2).mean(axis=-1))   # (N, C)
    return rms_per_patch.mean(axis=0)                        # (C,)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Feature activation maximisation
# ─────────────────────────────────────────────────────────────────────────────

def maximize_feature(
    feature_idx: int,
    raw_model: nn.Module,
    sae: SparseAutoencoder,
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
    target_layer: int,
    init_signal: Optional[torch.Tensor] = None,   # (C, T_full) warm-start
    n_channels: int = N_CHANNELS,
    signal_len: int = 7680,   # default: full 60-token window (60 s @ 128 Hz)
    n_steps: int = 2000,
    lr: float = 0.01,
    l2_coef: float = 0.05,
    tv_coef: float = 0.005,
    supp_coef: float = 0.3,
    supp_warmup: int = 500,   # steps before suppression starts
    supp_ramp: int = 500,     # steps to ramp from 0 → supp_coef
    freq_lo: float = 0.5,     # bandpass low cut (Hz); 0 = no low cut
    freq_hi: float = 40.0,    # bandpass high cut (Hz); 0 = no high cut
    fs: int = FS,             # sample rate for bandpass projection
    jitter_max: int = 4,      # ±samples of random temporal shift per step (0 = off)
    blur_sigma: float = 1.5,  # Gaussian blur sigma in samples applied before forward pass (0 = off)
    ws_proximity_coef: float = 0.0,  # L2 penalty to warm-start; keeps signal near real EEG
    spectral_coef: float = 0.0,      # reward fraction of energy in target band
    spectral_lo: float = 0.0,        # target band low edge (Hz); 0 = disabled
    spectral_hi: float = 0.0,        # target band high edge (Hz)
    sinusoid_blend: float = 0.0,     # blend warm-start with sinusoid at band centre (0–1)
    seed: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[float], int, list[float], list[float], np.ndarray]:
    """
    Gradient ascent on a raw EEG signal to maximise SAE feature `feature_idx`.

    Objective
    ---------
    Uses the *pre-ReLU linear projection* score as the primary objective:
        raw_score_i = (W_enc @ (act_norm - b_pre) + b_enc)[i]
    This has non-zero gradient everywhere, even when feature i is not in the
    TopK active set. Once large enough, the feature enters the top-k naturally.

    Suppression term — curriculum scheduled:
        Phase 1 (steps 0 → supp_warmup):      supp_coef = 0  (find the feature freely)
        Phase 2 (supp_warmup → +supp_ramp):   ramp linearly 0 → supp_coef
        Phase 3 (supp_warmup+supp_ramp → end): full supp_coef
    Starting suppression only once the feature is already active avoids the
    competition between "find feature" and "suppress others" in early steps.

    Warm-start proximity (ws_proximity_coef > 0):
        Adds λ_ws * ||x - x_init||² to the loss. This anchors the optimised
        signal near the warm-start (a real EEG recording), preventing gradient
        ascent from drifting off the EEG manifold into generic smooth noise.

    Spectral concentration (spectral_coef > 0, spectral_lo/hi set):
        Adds -λ_spec * (energy_in_target_band / total_energy) to the loss.
        Rewards the optimizer for concentrating signal energy in the band the
        XAE predicts for this feature (e.g. theta for F81). Combined with
        ws_proximity this tends to produce oscillatory, EEG-like prototypes.

    Returns
    -------
    (optimised_signal, loss_history, peak_token_index,
     feature_act_history, selectivity_history, final_z_at_peak)
    """
    torch.manual_seed(seed)

    if init_signal is not None:
        T = init_signal.shape[-1]
        x = init_signal.clone().to(device).unsqueeze(0).float()    # (1, C, T)
        x = x + torch.randn_like(x) * 0.01
    else:
        T = signal_len
        x = torch.randn(1, n_channels, T, device=device) * 0.1

    # Optional: blend warm-start towards a sinusoid at the target spectral band centre.
    # This biases the initialization toward oscillatory EEG-like structure.
    if sinusoid_blend > 0 and spectral_lo > 0 and spectral_hi > 0:
        center_freq = (spectral_lo + spectral_hi) / 2.0
        t_ax = torch.arange(T, dtype=torch.float32, device=device) / fs
        sino = torch.sin(2 * torch.pi * center_freq * t_ax)  # (T,)
        amp  = x.abs().mean().item() * 2.0  # match rough amplitude
        sino = sino.view(1, 1, T).expand(1, x.shape[1], T) * amp
        x.data = (1.0 - sinusoid_blend) * x.data + sinusoid_blend * sino

    # Store reference for warm-start proximity (frozen, not a learnable param)
    x_init_ref: Optional[torch.Tensor] = x.detach().clone() if ws_proximity_coef > 0 else None

    x = x.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    raw_model.eval()

    # Pre-build Gaussian blur kernel (fixed across steps)
    _blur_kernel: Optional[torch.Tensor] = None
    if blur_sigma > 0:
        _ks = max(3, int(6 * blur_sigma + 1) | 1)  # odd, covers ±3σ
        _k  = torch.arange(_ks, dtype=torch.float32, device=device) - _ks // 2
        _gk = torch.exp(-0.5 * (_k / blur_sigma) ** 2)
        _gk = _gk / _gk.sum()
        _blur_kernel = _gk.view(1, 1, -1)   # (1, 1, K) for conv1d

    loss_hist:         list[float] = []
    feat_act_hist:     list[float] = []
    selectivity_hist:  list[float] = []
    peak_token: int                = 0
    final_z:    np.ndarray         = np.array([])

    for step in range(n_steps):
        optimizer.zero_grad()

        # Transformation robustness: jitter + blur before each forward pass.
        # This forces the optimizer to find patterns that survive small temporal
        # perturbations and mild smoothing, preventing high-frequency artefacts.
        x_aug = x
        if jitter_max > 0:
            shift = torch.randint(-jitter_max, jitter_max + 1, (1,)).item()
            x_aug = torch.roll(x_aug, shifts=int(shift), dims=-1)
        if _blur_kernel is not None:
            C_in  = x_aug.shape[1]
            T_in  = x_aug.shape[2]
            pad   = _blur_kernel.shape[-1] // 2
            # Apply identical kernel to each channel independently
            x_flat = x_aug.view(-1, 1, T_in)
            x_flat = F.conv1d(x_flat, _blur_kernel, padding=pad)
            x_aug  = x_flat.view(1, C_in, T_in)

        act      = _layer_act(raw_model, x_aug, target_layer)           # (1, S, E)
        act_norm = (act - act_mean) / act_std
        flat     = act_norm.reshape(-1, act_norm.shape[-1])             # (S, E)

        # Pre-ReLU linear projection — differentiable everywhere
        pre_relu    = sae.encoder(flat - sae.b_pre)                     # (S, dict_size)
        raw_scores  = pre_relu[:, feature_idx]                          # (S,)
        feature_raw = raw_scores.max()
        peak_token  = int(raw_scores.argmax().item())

        # Post-TopK activations — for selectivity tracking and suppression
        z           = sae.encode(flat)                                  # (S, dict_size)
        z_at_peak   = z[peak_token]                                     # (dict_size,)
        feat_post   = z_at_peak[feature_idx]
        total_act   = z_at_peak.sum().clamp(min=1e-8)
        selectivity = (feat_post / total_act).item()

        # Curriculum suppression schedule
        if step < supp_warmup:
            cur_supp = 0.0
        elif step < supp_warmup + supp_ramp:
            cur_supp = supp_coef * (step - supp_warmup) / supp_ramp
        else:
            cur_supp = supp_coef
        supp = z_at_peak.sum() - z_at_peak[feature_idx]

        reg_l2 = l2_coef * x.pow(2).mean()
        reg_tv = tv_coef * (x[:, :, 1:] - x[:, :, :-1]).pow(2).mean()

        # Warm-start proximity: penalise deviation from the initial real-EEG signal.
        # Keeps the prototype near the EEG manifold rather than drifting to generic noise.
        if x_init_ref is not None and ws_proximity_coef > 0:
            reg_ws = ws_proximity_coef * (x - x_init_ref).pow(2).mean()
        else:
            reg_ws = torch.zeros(1, device=device)

        # Spectral concentration: reward fraction of energy in the XAE target band.
        # Encourages the prototype to have a dominant oscillation at the expected frequency.
        if spectral_coef > 0 and spectral_lo > 0 and spectral_hi > 0:
            X_f      = torch.fft.rfft(x, dim=-1)                            # (1, C, F)
            fq       = torch.fft.rfftfreq(T, d=1.0 / fs).to(device)        # (F,)
            band_m   = (fq >= spectral_lo) & (fq <= spectral_hi)
            e_band   = X_f[:, :, band_m].abs().pow(2).sum()
            e_total  = X_f.abs().pow(2).sum().clamp(min=1e-8)
            reg_spec = -spectral_coef * e_band / e_total
        else:
            reg_spec = torch.zeros(1, device=device)

        loss = -feature_raw + cur_supp * supp + reg_l2 + reg_tv + reg_ws + reg_spec
        loss.backward()
        optimizer.step()

        loss_hist.append(loss.item())
        feat_act_hist.append(feat_post.item())
        selectivity_hist.append(selectivity)

        # Bandpass projection: keep only physiological frequencies.
        # Applied outside autograd so it doesn't affect the gradient graph.
        if freq_lo > 0 or (freq_hi > 0 and freq_hi < fs / 2):
            with torch.no_grad():
                T_sig  = x.shape[-1]
                X_f    = torch.fft.rfft(x.data, dim=-1)
                freqs  = torch.fft.rfftfreq(T_sig, d=1.0 / fs).to(device)
                mask   = torch.ones(freqs.shape, dtype=torch.bool, device=device)
                if freq_lo > 0:
                    mask &= freqs >= freq_lo
                if freq_hi > 0:
                    mask &= freqs <= freq_hi
                X_f[:, :, ~mask] = 0.0
                x.data = torch.fft.irfft(X_f, n=T_sig, dim=-1)

        if (step + 1) % 500 == 0 or step == 0:
            print(
                f"  step {step+1:>5d}/{n_steps} | "
                f"raw={feature_raw.item():+.3f}  post={feat_post.item():+.3f} | "
                f"sel={selectivity:.3f}  supp_λ={cur_supp:.3f} | "
                f"peak_tok={peak_token}",
                flush=True,
            )

    # Record final post-TopK z at peak token
    with torch.no_grad():
        act_f  = _layer_act(raw_model, x, target_layer)
        z_f    = sae.encode(((act_f - act_mean) / act_std).reshape(-1, act_f.shape[-1]))
        final_z = z_f[peak_token].cpu().numpy()

    return x.detach().cpu().squeeze(0), loss_hist, peak_token, feat_act_hist, selectivity_hist, final_z


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_prototypes(
    feature_indices: list[int],
    signals: list[torch.Tensor],
    loss_hists: list[list[float]],
    peak_tokens: list[int],
    scores: Optional[torch.Tensor],
    feat_act_hists: Optional[list[list[float]]] = None,
    selectivity_hists: Optional[list[list[float]]] = None,
    final_zs: Optional[list[np.ndarray]] = None,
    empirical_rms: Optional[list[Optional[np.ndarray]]] = None,
    score_label: str = "fr",
    title_suffix: str = "",
    fs: int = FS,
    out_path: str | Path = "feature_viz.png",
    n_channels_show: int = N_CHANNELS,
    display_tokens: int = 5,    # tokens to show around the peak (±half)
    min_final_act: float = 0.1,  # skip features that never converged
) -> None:
    """
    Four-panel figure per feature:
      Panel 1 — multichannel EEG cropped to ±display_tokens/2 around peak token
      Panel 2 — amplitude spectrum of the peak token
      Panel 3 — empirical spatial profile: mean per-channel RMS across the top-N
                 real EEG patches where the feature fires most strongly in the
                 dataset. This is the honest spatial map — it reflects what the
                 model has actually learned, not the optimizer's trajectory.
      Panel 4 — optimisation dynamics: feature activation + selectivity over steps,
                 plus final z-vector distribution (top-k active features shown)
    """
    # Filter out features that never converged
    converged = [
        (fi, si, lh, pt,
         feat_act_hists[i] if feat_act_hists else None,
         selectivity_hists[i] if selectivity_hists else None,
         final_zs[i] if final_zs else None,
         empirical_rms[i] if empirical_rms else None)
        for i, (fi, si, lh, pt) in enumerate(zip(feature_indices, signals, loss_hists, peak_tokens))
        if (feat_act_hists[i][-1] if feat_act_hists else lh[-1] * -1) >= min_final_act
    ]
    n_skipped = len(feature_indices) - len(converged)
    if n_skipped:
        print(f"[plot] Skipping {n_skipped} features with final activation < {min_final_act} (no warm-start found)")
    if not converged:
        print("[plot] No converged features to plot.")
        return

    n   = len(converged)
    has_dynamics = feat_act_hists is not None and selectivity_hists is not None
    ncols = 4 if has_dynamics else 3
    width_ratios = [2, 1.2, 0.8, 1.2] if has_dynamics else [2, 1.2, 0.8]

    fig = plt.figure(figsize=(16 + (4 if has_dynamics else 0), 4.0 * n))
    gs  = fig.add_gridspec(n, ncols, width_ratios=width_ratios, hspace=0.55, wspace=0.35)

    for row, (feat_idx, sig, _lh, peak_tok, fa_hist_r, sel_hist_r, fz_r, emp_rms_r) in enumerate(converged):
        sig_np  = sig.numpy()          # (C, T_full)
        T_full  = sig_np.shape[-1]
        S_total = T_full // _PATCH_SIZE

        # Crop a window of `display_tokens` patches centred on peak_tok
        half    = display_tokens // 2
        tok_lo  = max(0, peak_tok - half)
        tok_hi  = min(S_total, tok_lo + display_tokens)
        tok_lo  = max(0, tok_hi - display_tokens)       # re-clamp start
        samp_lo = tok_lo * _PATCH_SIZE
        samp_hi = tok_hi * _PATCH_SIZE
        crop    = sig_np[:, samp_lo:samp_hi]            # (C, crop_T)
        T_crop  = crop.shape[-1]
        t_ms    = (np.arange(T_crop) + samp_lo) / fs * 1000   # absolute time

        ch_show  = min(n_channels_show, sig_np.shape[0])
        ch_labels = _CHANNEL_NAMES[:ch_show]

        # ── Panel 1: multichannel EEG (cropped) ───────────────────────────
        ax_sig  = fig.add_subplot(gs[row, 0])
        spacing = np.abs(crop[:ch_show]).max() * 2 + 1e-6
        for c in range(ch_show):
            ax_sig.plot(t_ms, crop[c] + c * spacing, lw=0.7, color="steelblue")

        # Highlight the peak token
        pk_lo_ms = peak_tok * _PATCH_SIZE / fs * 1000
        pk_hi_ms = (peak_tok + 1) * _PATCH_SIZE / fs * 1000
        ax_sig.axvspan(pk_lo_ms, pk_hi_ms, alpha=0.12, color="red", label="peak token")

        score_str = f"  ({score_label}={scores[feat_idx]:.3f})" if scores is not None else ""
        ax_sig.set_title(f"Feature {feat_idx}{score_str}  |  peak tok={peak_tok}",
                         fontsize=9, loc="left")
        ax_sig.set_xlabel("Time (ms)", fontsize=8)
        ax_sig.set_yticks(np.arange(ch_show) * spacing)
        ax_sig.set_yticklabels(ch_labels, fontsize=6)
        ax_sig.tick_params(axis="x", labelsize=7)

        # ── Panel 2: spectrum of the PEAK TOKEN only ──────────────────────
        ax_spec = fig.add_subplot(gs[row, 1])
        peak_patch = sig_np[:, peak_tok * _PATCH_SIZE:(peak_tok + 1) * _PATCH_SIZE]
        freqs = np.fft.rfftfreq(_PATCH_SIZE, d=1.0 / fs)
        amps  = np.abs(np.fft.rfft(peak_patch, axis=-1)).mean(axis=0)
        mask  = freqs <= 60
        ax_spec.fill_between(freqs[mask], amps[mask], alpha=0.6, color="crimson")
        ax_spec.plot(freqs[mask], amps[mask], lw=0.8, color="crimson")
        y_top = amps[mask].max() * 1.08 + 1e-9
        for band, (lo, hi, col) in _EEG_BANDS.items():
            ax_spec.axvspan(lo, hi, alpha=0.10, color=col)
            ax_spec.text((lo + hi) / 2, y_top, band, ha="center",
                         va="bottom", fontsize=7, color=col)
        ax_spec.set_xlabel("Frequency (Hz)", fontsize=8)
        ax_spec.set_xlim(0, 60)
        ax_spec.tick_params(labelsize=7)
        ax_spec.set_title("Peak token spectrum", fontsize=9, loc="left")

        # ── Panel 3: empirical spatial profile from real data ─────────────
        ax_rms = fig.add_subplot(gs[row, 2])
        if emp_rms_r is not None and len(emp_rms_r) >= ch_show:
            rms_ch = emp_rms_r[:ch_show]
            color_rms = "steelblue"
            title_rms = "Spatial RMS (real data)"
        else:
            # Fallback: prototype peak-token RMS (labelled clearly as such)
            rms_ch = np.sqrt((peak_patch[:ch_show] ** 2).mean(axis=-1))
            color_rms = "lightcoral"
            title_rms = "Spatial RMS (prototype)"
        ax_rms.barh(np.arange(ch_show), rms_ch, color=color_rms, alpha=0.8)
        ax_rms.set_yticks(np.arange(ch_show))
        ax_rms.set_yticklabels(ch_labels, fontsize=6)
        ax_rms.set_xlabel("RMS", fontsize=8)
        ax_rms.tick_params(labelsize=7)
        ax_rms.set_title(title_rms, fontsize=9, loc="left")

        # ── Panel 4: optimisation dynamics ────────────────────────────────
        if has_dynamics:
            ax_dyn = fig.add_subplot(gs[row, 3])
            fa_hist  = fa_hist_r   # type: ignore[index]
            sel_hist = sel_hist_r  # type: ignore[index]
            steps = np.arange(1, len(fa_hist) + 1)

            color_feat = "#2196F3"
            color_sel  = "#FF9800"

            ax_dyn.plot(steps, fa_hist, lw=1.0, color=color_feat, label="z_target")
            ax_dyn.set_xlabel("Step", fontsize=8)
            ax_dyn.set_ylabel("Feature activation", fontsize=7, color=color_feat)
            ax_dyn.tick_params(axis="y", labelcolor=color_feat, labelsize=7)
            ax_dyn.tick_params(axis="x", labelsize=7)

            ax_sel = ax_dyn.twinx()
            ax_sel.plot(steps, sel_hist, lw=1.0, color=color_sel, alpha=0.85,
                        linestyle="--", label="selectivity")
            ax_sel.axhline(1.0 / 8, color=color_sel, lw=0.6, alpha=0.4,
                           linestyle=":")  # TopK k=8 floor
            ax_sel.set_ylim(0, 1.05)
            ax_sel.set_ylabel("Selectivity (z_i / Σz)", fontsize=7, color=color_sel)
            ax_sel.tick_params(axis="y", labelcolor=color_sel, labelsize=7)

            # Final selectivity annotation
            final_sel = sel_hist[-1]
            final_fa  = fa_hist[-1]
            ax_dyn.set_title(
                f"Optimisation  |  final z={final_fa:.2f}  sel={final_sel:.3f}",
                fontsize=8, loc="left",
            )

            # Inset: top-8 active features in final z (shows which co-activate)
            if fz_r is not None:
                fz = fz_r
                top8_idx  = np.argsort(fz)[::-1][:8]
                top8_vals = fz[top8_idx]
                colors_bar = ["#F44336" if idx == feat_idx else "#90CAF9"
                              for idx in top8_idx]
                ax_bar = ax_dyn.inset_axes([0.55, 0.12, 0.44, 0.38])
                ax_bar.barh(np.arange(8), top8_vals[::-1], color=colors_bar[::-1],
                            height=0.7)
                ax_bar.set_yticks(np.arange(8))
                ax_bar.set_yticklabels([str(i) for i in top8_idx[::-1]], fontsize=5)
                ax_bar.tick_params(axis="x", labelsize=5)
                ax_bar.set_title("Top-8 z", fontsize=6)
                ax_bar.set_xlabel("activation", fontsize=5)

    ttl = "SAE Feature Prototypes — SleepFM v1.1, Layer 2"
    if title_suffix:
        ttl += f"\n{title_suffix}"
    fig.suptitle(ttl, fontsize=11, y=1.001)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment",      default="sleepfm_finetuned_layer2")
    p.add_argument("--experiments-dir", default="results/experiments")
    p.add_argument("--n-features",      type=int, default=10)
    p.add_argument("--feature-indices", default=None,
                   help="Comma-separated indices; overrides --n-features and ranking")
    p.add_argument("--rank-by",         choices=["firing_rate", "tcav", "label_corr"], default="firing_rate")
    p.add_argument("--tcav-cache",      default=None,
                   help="Path to tcav_cache.pt (required when --rank-by tcav)")
    p.add_argument("--app-cache",       default=None,
                   help="Path to app_cache.pt (required when --rank-by label_corr)")
    p.add_argument("--n-steps",         type=int, default=2000)
    p.add_argument("--lr",              type=float, default=0.01)
    p.add_argument("--l2",              type=float, default=0.05)
    p.add_argument("--tv",              type=float, default=0.005)
    p.add_argument("--supp",            type=float, default=0.3,
                   help="Peak suppression coef (curriculum-ramped)")
    p.add_argument("--supp-warmup",     type=int,   default=500,
                   help="Steps before suppression starts")
    p.add_argument("--supp-ramp",       type=int,   default=500,
                   help="Steps to ramp suppression from 0 to --supp")
    p.add_argument("--n-restarts",      type=int,   default=3,
                   help="Random restarts per feature; keep the run with highest final selectivity")
    p.add_argument("--signal-len",      type=int, default=7680,
                   help="Fallback signal length for no-warmstart Gaussian init "
                        "(default 7680 = 60 tokens = full dataset window)")
    p.add_argument("--no-warmstart",    action="store_true",
                   help="Skip warm-start; use Gaussian noise init (reproduces old behaviour)")
    p.add_argument("--target-class",    type=int, default=1,
                   help="Class to use for warm-start (1=abnormal, 0=normal)")
    p.add_argument("--warmstart-batches", type=int, default=30)
    p.add_argument("--rank-batches",    type=int, default=60)
    p.add_argument("--freq-lo",         type=float, default=0.5,
                   help="Bandpass low cut in Hz applied after each gradient step (0 = off)")
    p.add_argument("--freq-hi",         type=float, default=40.0,
                   help="Bandpass high cut in Hz applied after each gradient step (0 = off)")
    p.add_argument("--jitter-max",      type=int,   default=4,
                   help="Max temporal jitter in samples applied before each forward pass (0 = off)")
    p.add_argument("--blur-sigma",      type=float, default=1.5,
                   help="Gaussian blur sigma (samples) applied before each forward pass (0 = off)")
    p.add_argument("--ws-proximity",    type=float, default=0.0,
                   help="Warm-start proximity coef: L2 penalty ||x - x_init||². "
                        "Anchors optimisation near the real EEG warm-start. "
                        "Try 0.05–0.3 to keep prototypes on the EEG manifold.")
    p.add_argument("--spectral-coef",   type=float, default=0.0,
                   help="Spectral concentration coef: rewards energy in --spectral-lo/hi band. "
                        "Try 0.5–2.0 to push prototype toward a dominant oscillation.")
    p.add_argument("--spectral-lo",     type=float, default=0.0,
                   help="Target band low edge (Hz) for spectral concentration loss. "
                        "Set to 0 to auto-detect from app-cache XAE profile per feature.")
    p.add_argument("--spectral-hi",     type=float, default=0.0,
                   help="Target band high edge (Hz) for spectral concentration loss.")
    p.add_argument("--sinusoid-blend",  type=float, default=0.0,
                   help="Blend init signal with sinusoid at target band centre (0–1). "
                        "0 = pure warm-start, 1 = pure sinusoid. Try 0.3 with --spectral-lo/hi.")
    p.add_argument("--out-dir",         default=None)
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir or f"results/feature_viz/{args.experiment}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    print(f"[main] Loading experiment '{args.experiment}' …")
    raw_model, sae, act_mean, act_std, meta = load_experiment(
        args.experiment, args.experiments_dir, device=args.device
    )
    target_layer = meta["target_layer"]

    # ── Per-encoder dataset routing ───────────────────────────────────────────
    _enc_key = meta.get("encoder", "sleepfm")
    _data_path, _patch_size, _fs = _ENCODER_DATA.get(
        _enc_key, (_DATA_PATH, _PATCH_SIZE, FS)
    )
    # Update global _PATCH_SIZE so function bodies (not defaults) pick up the right value
    globals()["_PATCH_SIZE"] = _patch_size
    print(f"[main] Encoder '{_enc_key}': data={_data_path}  patch={_patch_size}  fs={_fs}")

    # ── Determine feature indices & ranking scores ────────────────────────────
    scores: Optional[torch.Tensor] = None
    score_label = "fr"
    title_suffix = ""

    if args.feature_indices is not None:
        feature_indices = [int(i) for i in args.feature_indices.split(",")]
        print(f"[main] Using specified features: {feature_indices}")

    elif args.rank_by == "label_corr":
        if args.app_cache is None:
            raise ValueError("--app-cache is required with --rank-by label_corr")
        print("\n[main] Ranking features by label correlation (|r|, p < 0.05) …")
        corr, pv = rank_by_label_corr(args.app_cache)
        sig_mask = pv < 0.05
        corr_masked = corr.clone()
        corr_masked[~sig_mask] = -1
        top = torch.topk(corr_masked, k=min(args.n_features, sae.dict_size))
        feature_indices = top.indices.tolist()
        scores = corr
        score_label = "|r|"
        title_suffix = "Ranked by |label correlation| (significant only, p < 0.05)"
        print(f"\n  Top {args.n_features} label-correlated features:")
        print(f"  {'feat':>5}  {'|r|':>8}  {'p':>8}")
        for fi in feature_indices:
            print(f"  {fi:>5}  {corr[fi].item():>8.4f}  {pv[fi].item():>8.4f}")
        json.dump({"feature_indices": feature_indices,
                   "label_correlations": corr.tolist(),
                   "p_values":           pv.tolist()},
                  (out_dir / "feature_ranking.json").open("w"), indent=2)

    elif args.rank_by == "tcav":
        if args.tcav_cache is None:
            raise ValueError("--tcav-cache is required with --rank-by tcav")
        print("\n[main] Ranking features by TCAV abnormality delta-rate …")
        dr, pv = rank_by_tcav_abnorm(args.tcav_cache)
        # Only significant features (BH p < 0.05)
        sig_mask = pv < 0.05
        dr_masked = dr.clone()
        dr_masked[~sig_mask] = -1
        top = torch.topk(dr_masked, k=min(args.n_features, sae.dict_size))
        feature_indices = top.indices.tolist()
        scores = dr
        score_label = "Δfr"
        title_suffix = "Ranked by TCAV abnormality Δfiring-rate (significant only)"
        print(f"\n  Top {args.n_features} abnormality-sensitive features:")
        print(f"  {'feat':>5}  {'Δfr':>8}  {'p':>8}")
        for fi in feature_indices:
            print(f"  {fi:>5}  {dr[fi].item():>8.4f}  {pv[fi].item():>8.4f}")
        json.dump({"feature_indices": feature_indices,
                   "delta_rates":     dr.tolist(),
                   "p_values":        pv.tolist()},
                  (out_dir / "feature_ranking.json").open("w"), indent=2)

    else:
        print("\n[main] Ranking features by firing rate …")
        fr = rank_by_firing_rate(
            raw_model, sae, act_mean, act_std, target_layer,
            data_path=_data_path,
            max_batches=args.rank_batches, device=args.device,
        )
        top = torch.topk(fr, k=min(args.n_features, sae.dict_size))
        feature_indices = top.indices.tolist()
        scores = fr
        score_label = "fr"
        json.dump({"feature_indices": feature_indices, "firing_rates": fr.tolist()},
                  (out_dir / "feature_ranking.json").open("w"), indent=2)

    # ── XAE spectral band lookup (for per-feature spectral concentration) ────────
    # If --app-cache is provided and --spectral-lo/hi are not explicitly set,
    # auto-detect the dominant EEG band per feature from the XAE profile.
    _BAND_EDGES = {
        "delta":    (0.5,  4.0),
        "theta":    (4.0,  8.0),
        "alpha":    (8.0, 13.0),
        "low-beta": (13.0, 20.0),
        "high-beta":(20.0, 30.0),
        "gamma":    (30.0, 45.0),
    }
    feat_spectral_bands: dict[int, tuple[float, float]] = {}
    if args.app_cache and args.spectral_coef > 0 and args.spectral_lo == 0:
        print("\n[main] Loading XAE band info from app_cache for spectral concentration …")
        _ac = torch.load(args.app_cache, map_location="cpu", weights_only=False)
        _band_deltas = _ac["feature_band_deltas"].numpy()   # (n_feat, n_bands)
        _band_names  = _ac["feature_band_names"]            # list[str]
        for fi in feature_indices:
            dom_band = _band_names[int(_band_deltas[fi].argmax())]
            lo, hi   = _BAND_EDGES.get(dom_band, (0.0, 0.0))
            feat_spectral_bands[fi] = (lo, hi)
            print(f"  F{fi}: dominant band = {dom_band}  ({lo}–{hi} Hz)")
    elif args.spectral_lo > 0 and args.spectral_hi > 0:
        # User explicitly set a single band for all features
        for fi in feature_indices:
            feat_spectral_bands[fi] = (args.spectral_lo, args.spectral_hi)

    # ── Empirical spatial profiles (collected once per feature from real data) ──
    print("\n[main] Collecting empirical spatial profiles from real data …")
    N_EXAMPLES = 5   # number of top real patches to save as examples
    emp_rms_list:         list[Optional[np.ndarray]] = []
    example_patches_list: list[Optional[np.ndarray]] = []
    top_joint_scores:     list[float]                 = []
    for feat_idx in feature_indices:
        patches, acts, top_joint = collect_top_activating_patches(
            feature_idx=feat_idx,
            raw_model=raw_model,
            sae=sae,
            act_mean=act_mean,
            act_std=act_std,
            target_layer=target_layer,
            data_path=_data_path,
            max_batches=args.rank_batches,
            device=args.device,
        )
        top_joint_scores.append(top_joint)
        if len(acts) > 0:
            emp_rms_list.append(empirical_spatial_rms(patches))
            ex = patches[:N_EXAMPLES]  # may be fewer than N_EXAMPLES rows
            if len(ex) < N_EXAMPLES:
                pad = np.zeros((N_EXAMPLES - len(ex), ex.shape[1], ex.shape[2]), dtype=ex.dtype)
                ex = np.concatenate([ex, pad], axis=0)
            example_patches_list.append(ex)   # (N_EXAMPLES, C, P)
        else:
            emp_rms_list.append(None)
            example_patches_list.append(None)

    # ── Optimise each feature ─────────────────────────────────────────────────
    signals:           list[torch.Tensor]  = []
    loss_hists:        list[list[float]]   = []
    feat_act_hists:    list[list[float]]   = []
    selectivity_hists: list[list[float]]   = []
    final_zs:          list[np.ndarray]    = []
    peak_tokens:       list[int]           = []
    warmstarts:        list[float]         = []

    for rank, feat_idx in enumerate(feature_indices):
        print(f"\n{'='*60}")
        print(f"Feature {feat_idx}  (rank {rank+1}/{len(feature_indices)})")
        print("="*60)

        # -- warm-start: full window so transformer context is preserved --
        init_sig: Optional[torch.Tensor] = None
        ws_act    = 0.0
        ws_tok    = 0
        if not args.no_warmstart:
            print(f"  Searching for warm-start (class={args.target_class}) …")
            init_sig, ws_act, ws_tok = find_warmstart(
                feature_idx=feat_idx,
                raw_model=raw_model,
                sae=sae,
                act_mean=act_mean,
                act_std=act_std,
                target_layer=target_layer,
                target_class=args.target_class,
                n_batches=args.warmstart_batches,
                device=args.device,
            )
            if init_sig is not None:
                print(f"  Warm-start: feat_act={ws_act:.3f}  peak_tok={ws_tok}")
            else:
                print(f"  No warm-start found — using Gaussian noise")
        warmstarts.append(ws_act)

        # -- gradient ascent with restarts; keep best by final selectivity --
        best_sel   = -1.0
        best_run: tuple = ()  # type: ignore[assignment]
        for restart in range(args.n_restarts):
            seed = rank * 100 + restart
            print(f"  Restart {restart+1}/{args.n_restarts} (seed={seed})")
            _spec_lo, _spec_hi = feat_spectral_bands.get(feat_idx, (args.spectral_lo, args.spectral_hi))
            run = maximize_feature(
                feature_idx=feat_idx,
                raw_model=raw_model,
                sae=sae,
                act_mean=act_mean,
                act_std=act_std,
                target_layer=target_layer,
                init_signal=init_sig,
                signal_len=args.signal_len if args.signal_len != 7680 else _patch_size * 60,
                n_steps=args.n_steps,
                lr=args.lr,
                l2_coef=args.l2,
                tv_coef=args.tv,
                supp_coef=args.supp,
                supp_warmup=args.supp_warmup,
                supp_ramp=args.supp_ramp,
                freq_lo=args.freq_lo,
                freq_hi=args.freq_hi,
                fs=meta["fs"],
                jitter_max=args.jitter_max,
                blur_sigma=args.blur_sigma,
                ws_proximity_coef=args.ws_proximity,
                spectral_coef=args.spectral_coef,
                spectral_lo=_spec_lo,
                spectral_hi=_spec_hi,
                sinusoid_blend=args.sinusoid_blend,
                seed=seed,
                device=args.device,
            )
            final_sel = run[4][-1]   # selectivity_hist[-1]
            print(f"    → final selectivity={final_sel:.3f}  post={run[3][-1]:.2f}")
            if final_sel > best_sel:
                best_sel = final_sel
                best_run = run

        x_opt, lc, peak_tok, fa_hist, sel_hist, fz = best_run
        print(f"  Best restart: selectivity={best_sel:.3f}")
        signals.append(x_opt)
        loss_hists.append(lc)
        feat_act_hists.append(fa_hist)
        selectivity_hists.append(sel_hist)
        final_zs.append(fz)
        peak_tokens.append(peak_tok)

    # ── Save ──────────────────────────────────────────────────────────────────
    _zeros_ecp = np.zeros((N_EXAMPLES, N_CHANNELS, _patch_size))
    emp_rms_arr         = np.stack([r if r is not None else np.zeros(N_CHANNELS) for r in emp_rms_list])
    example_patches_arr = np.stack([e if e is not None else _zeros_ecp for e in example_patches_list])
    np.savez(
        out_dir / "prototypes.npz",
        feature_indices=np.array(feature_indices),
        signals=np.stack([s.numpy() for s in signals]),
        loss_hists=np.array(loss_hists),
        feat_act_hists=np.array(feat_act_hists),
        selectivity_hists=np.array(selectivity_hists),
        final_zs=np.stack(final_zs),
        peak_tokens=np.array(peak_tokens),
        warmstart_activations=np.array(warmstarts),
        empirical_spatial_rms=emp_rms_arr,
        example_patches=example_patches_arr,  # (N, N_EXAMPLES, C, P) — top real patches
        top_joint_scores=np.array(top_joint_scores, dtype=np.float32),  # (N,) attention×activation
        scores=scores.numpy() if scores is not None else np.array([]),
        score_label=np.array(score_label),
        fs=np.array(_fs),
        target_layer=np.array(target_layer),
        experiment=np.array(args.experiment),
    )
    print(f"\n[main] Saved prototypes → {out_dir}/prototypes.npz")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_prototypes(
        feature_indices=feature_indices,
        signals=signals,
        loss_hists=loss_hists,
        peak_tokens=peak_tokens,
        scores=scores,
        feat_act_hists=feat_act_hists,
        selectivity_hists=selectivity_hists,
        final_zs=final_zs,
        empirical_rms=emp_rms_list,
        score_label=score_label,
        title_suffix=title_suffix,
        out_path=out_dir / "feature_viz.png",
    )
    print(f"[main] Done. Results in: {out_dir}/")


if __name__ == "__main__":
    main()
