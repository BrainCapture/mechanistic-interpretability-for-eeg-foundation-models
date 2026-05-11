"""Multi-layer simultaneous concept steering for LaBraM.

Single-layer steering substitutes top-k SAE features at one layer and decodes
locally. Multi-layer steering substitutes at *every* transformer block and
lets the corruption propagate through the rest of the encoder, evaluating on
the final-layer output via a frozen probe.

For each concept:
  1. Pre-compute per-layer (CAV, donor_mean, top-k feature ranking, act_mean,
     act_std, SAE) using the existing 12 LaBraM steering caches.
  2. Train target + off-target probes ONCE on clean final-layer embeddings.
  3. Sample N target + N off-target whole windows from the dataset.
  4. For each k ∈ K_GRID:
       - Register modifying forward hooks on every transformer block (B0..B11):
         strip CLS, normalise, SAE encode, substitute top-k_L features with
         donor_mean_L, SAE decode, denormalise, re-attach CLS, return modified
         tensor as block output.
       - Forward-pass the sampled windows; capture final-layer output.
       - Score frozen probes per token, then average per window for window-
         level AUROC.
  5. Compare against single-layer L11 baseline (k=0..k_max applied at L11
     only, all other layers unchanged).

Usage::

    uv run tools/multi_layer_steering.py --concept Pathology
    uv run tools/multi_layer_steering.py --concepts all
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sae4eeg.dataset import H5PYDatasetLabeled, StandardizeLabel
from sae4eeg.encoders import load_encoder
from sae4eeg.sae import SparseAutoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Helpers (inlined from plot_cross_model_steering_concepts.py to avoid the
# parent script's module-level side-effects when imported) ───────────────────

N_PROBE_PER_GROUP = 3000
TEST_FRAC = 0.25

AGE_CHILD = {"0-3", "4-9"}
AGE_ADULT = {"20-29", "30-39", "40-49", "50-59", "60+"}
ABN_CATS  = {"Abnormal - Epileptiform", "Abnormal - Other"}
NORMAL    = {"Normal"}


def _random_window_mask(subjects: np.ndarray, parity: int,
                         seed: int = 1729) -> np.ndarray:
    """Negative-control labelling: per-window random binary label.
    Keep this in sync with the helper in plot_cross_model_steering_concepts.py."""
    n = len(subjects)
    boundaries = np.zeros(n, dtype=bool)
    boundaries[0] = True
    if n > 1:
        boundaries[1:] = subjects[1:] != subjects[:-1]
    win_idx = np.cumsum(boundaries) - 1
    n_windows = int(win_idx.max()) + 1
    rng = np.random.default_rng(seed)
    win_labels = rng.integers(0, 2, size=n_windows)
    tok_labels = win_labels[win_idx]
    return tok_labels == parity


CONCEPTS = [
    {"name": "Pathology", "tgt_field": "classification", "tgt_pos": ABN_CATS, "tgt_neg": NORMAL,
     "off_field": "age_group", "off_pos": AGE_CHILD, "off_neg": AGE_ADULT,
     "filter_field": None, "filter_set": None},
    {"name": "Age", "tgt_field": "age_group", "tgt_pos": AGE_CHILD, "tgt_neg": AGE_ADULT,
     "off_field": "classification", "off_pos": ABN_CATS, "off_neg": NORMAL,
     "filter_field": None, "filter_set": None},
    {"name": "Gender", "tgt_field": "gender", "tgt_pos": {"male"}, "tgt_neg": {"female"},
     "off_field": "classification", "off_pos": ABN_CATS, "off_neg": NORMAL,
     "filter_field": None, "filter_set": None},
    {"name": "Medicine (ASM)", "tgt_field": "medication_group",
     "tgt_pos": {"ASM"}, "tgt_neg": {"No Current Medication"},
     "off_field": "classification", "off_pos": ABN_CATS, "off_neg": NORMAL,
     "filter_field": None, "filter_set": None},
    {"name": "Medicine (Psychiatric)", "tgt_field": "medication_group",
     "tgt_pos": {"Psychiatric"}, "tgt_neg": {"No Current Medication"},
     "off_field": "classification", "off_pos": ABN_CATS, "off_neg": NORMAL,
     "filter_field": None, "filter_set": None},
    # Negative control: per-window random binary labels.
    {"name": "Random labels (control)", "tgt_field": "subject_id",
     "tgt_synth_pos": lambda s: _random_window_mask(s, 0),
     "tgt_synth_neg": lambda s: _random_window_mask(s, 1),
     "off_field": "classification", "off_pos": ABN_CATS, "off_neg": NORMAL,
     "filter_field": None, "filter_set": None},
]


class TorchProbe:
    def __init__(self, w: torch.Tensor, b: torch.Tensor):
        self.w = w
        self.b = b


def fit_probe(X_train: torch.Tensor, y_train: np.ndarray, C: float = 1.0,
              max_iter: int = 200) -> TorchProbe:
    if X_train.device != DEVICE:
        X_train = X_train.to(DEVICE)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=DEVICE)
    n, d = X_train.shape
    n_pos = float((yt == 1).sum())
    n_neg = float((yt == 0).sum())
    w_pos = n / (2.0 * n_pos) if n_pos > 0 else 1.0
    w_neg = n / (2.0 * n_neg) if n_neg > 0 else 1.0
    sample_weight = torch.where(yt == 1, w_pos * torch.ones_like(yt),
                                w_neg * torch.ones_like(yt))
    w = torch.zeros(d, device=DEVICE, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, device=DEVICE, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS(
        [w, b], lr=1.0, max_iter=max_iter, history_size=20,
        line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9,
    )
    def closure():
        opt.zero_grad()
        logits = X_train @ w + b
        per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, yt, reduction="none")
        bce = (per_sample * sample_weight).sum()
        loss = C * bce + 0.5 * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return TorchProbe(w.detach(), b.detach())


def build_tcav_alignment(Xt_raw: torch.Tensor, y_t: np.ndarray,
                         W_dec: torch.Tensor) -> np.ndarray:
    probe = fit_probe(Xt_raw, y_t, max_iter=300)
    cav = probe.w
    cav = cav / (cav.norm() + 1e-8)
    proj = (cav @ W_dec).cpu().numpy()
    return np.abs(proj)

CACHE_DIR = ROOT / "results" / "steering_cache"
OUT_DIR = ROOT / "paper" / "concept_steering_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_GRID = [0, 2, 4, 8, 16, 32, 64, 128, 200]
N_WINDOWS_PER_GROUP = 200   # total windows: 4 * 200 = 800 per concept
RNG_SEED = 42

# Per-encoder configuration. All k counts the *same value applied at every
# layer* — different feature subsets per layer (per-layer TCAV ranking) but
# the same count. The "deep_layer" is the final encoder block, used for both
# the single-layer baseline (deep-only) and the frozen final-layer probe.
ENCODER_CFG = {
    "labram": {
        "data_path": ROOT / "data" / "D4-v3-preprocessed-v1",
        "weights":   ROOT / "checkpoints" / "finetuned" / "labram_binary" / "finetuned.ckpt",
        "sae_dir":   ROOT / "results" / "features" / "labram",
        "sae_name":  "sae_labram_exp1.0_k8_layer{L}.pt",
        "cache_dir": CACHE_DIR,
        "cache_name": "labram_layer{L}",
        "window_idx_layer": 0,    # L11 cache lacks window_file_idx
        "n_layers":  12,
        "deep_layer": 11,
        "n_prefix":  1,           # CLS token to skip in hooks
    },
    "sleepfm": {
        "data_path": ROOT / "data" / "D4-v3-preprocessed-v2",
        "weights":   ROOT / "checkpoints" / "finetuned" / "sleepfm1.ckpt",
        "sae_dir":   ROOT / "results" / "features" / "sleepfm_finetuned",
        "sae_name":  "sae_sleepfm_exp1.0_k8_layer{L}.pt",
        "cache_dir": CACHE_DIR,
        "cache_name": "sleepfm_finetuned_layer{L}",
        "window_idx_layer": 0,
        "n_layers":  3,
        "deep_layer": 2,
        "n_prefix":  0,           # no CLS for SleepFM
    },
}

# Concept specs piggyback on the single-layer file.
ALL_CONCEPTS = {c["name"]: c for c in CONCEPTS}


# ── Per-layer state ───────────────────────────────────────────────────────────

def build_layer_state(L: int, concept, cfg) -> dict:
    """Build per-layer state for a given concept: SAE, normalisation,
    donor_mean (z-mean of target_neg pool), TCAV-aligned feature ranking.

    Aggressively frees the cache after extracting what we need — loading all
    layer caches at once would blow tens of GB of RAM for LaBraM."""
    cache = torch.load(cfg["cache_dir"] / cfg["cache_name"].format(L=L) / "steering_cache.pt",
                       map_location="cpu", weights_only=False)
    sae_ckpt = torch.load(cfg["sae_dir"] / cfg["sae_name"].format(L=L),
                          map_location="cpu", weights_only=False)

    sae = SparseAutoencoder(
        input_dim=sae_ckpt["embed_dim"],
        expansion=sae_ckpt["expansion"], k=sae_ckpt["k"], mode="topk",
    ).to(DEVICE).eval()
    sae.load_state_dict(sae_ckpt["sae_state_dict"])
    am = sae_ckpt["act_mean"].to(DEVICE).float()
    as_ = sae_ckpt["act_std"].to(DEVICE).float()

    # Sample target_pos and target_neg tokens for CAV training and donor_mean.
    def asarr(field):
        return np.asarray([str(v) for v in cache[field]])

    if concept["filter_field"] is not None:
        filt = np.isin(asarr(concept["filter_field"]),
                       list(concept["filter_set"]))
    else:
        filt = np.ones(len(cache["embeddings"]), dtype=bool)
    tgt_field = asarr(concept["tgt_field"])
    if "tgt_synth_pos" in concept:
        pos_mask = filt & concept["tgt_synth_pos"](tgt_field)
        neg_mask = filt & concept["tgt_synth_neg"](tgt_field)
    else:
        pos_mask = filt & np.isin(tgt_field, list(concept["tgt_pos"]))
        neg_mask = filt & np.isin(tgt_field, list(concept["tgt_neg"]))
    rng = np.random.default_rng(RNG_SEED)
    n = min(N_PROBE_PER_GROUP, int(pos_mask.sum()), int(neg_mask.sum()))
    if n < 200:
        raise ValueError(f"L{L} {concept['name']}: too few tokens "
                         f"(pos={int(pos_mask.sum())}, neg={int(neg_mask.sum())})")
    idx_pos = rng.choice(np.where(pos_mask)[0], n, replace=False)
    idx_neg = rng.choice(np.where(neg_mask)[0], n, replace=False)
    idx = np.concatenate([idx_pos, idx_neg])
    y = np.array([1] * n + [0] * n)

    # Slice only the needed tokens, then immediately drop the full cache.
    sampled_emb = torch.as_tensor(np.asarray(cache["embeddings"][idx])).float()
    del cache, sae_ckpt
    import gc; gc.collect()

    Xt_raw = (sampled_emb.to(DEVICE) - am) / (as_ + 1e-8)
    with torch.no_grad():
        zt = sae.encode(Xt_raw)
    donor_mean = zt[y == 0].mean(dim=0)  # target_neg z-mean

    W_dec = sae.decoder.weight.detach()
    ranking = build_tcav_alignment(Xt_raw, y, W_dec)
    sorted_features = np.argsort(ranking)[::-1].copy()

    return {
        "sae": sae,
        "act_mean": am,
        "act_std": as_,
        "donor": donor_mean,
        "ranking": sorted_features,
        "n_features": int(sae.decoder.weight.shape[1]),
        "n_prefix": cfg["n_prefix"],
    }


# ── Window sampling ───────────────────────────────────────────────────────────

def sample_windows(concept, n_per_group: int, rng, cfg) -> tuple:
    """Draw N target_pos / target_neg / off_pos / off_neg windows. Returns
    (window_file_idx, window_local_idx, target_label, offtarget_label,
     all-window arrays of shape (n_total,))."""
    L = cfg["window_idx_layer"]
    cache = torch.load(cfg["cache_dir"] / cfg["cache_name"].format(L=L) / "steering_cache.pt",
                       map_location="cpu", weights_only=False)

    file_idx = cache["window_file_idx"]
    local_idx = cache["window_local_idx"]
    tpw = int(cache["tokens_per_window"])
    n_windows = len(file_idx)

    def asarr(field):
        return np.asarray([str(v) for v in cache[field]])

    if concept["filter_field"] is not None:
        filt_tok = np.isin(asarr(concept["filter_field"]),
                           list(concept["filter_set"]))
    else:
        filt_tok = np.ones(len(cache["embeddings"]), dtype=bool)

    tgt_tok = asarr(concept["tgt_field"])
    off_tok = asarr(concept["off_field"])

    # Window-level labels: take token-0 of each window (window-uniform fields).
    win0 = np.arange(n_windows) * tpw
    filt_w = filt_tok[win0]
    tgt_w = tgt_tok[win0]
    off_w = off_tok[win0]

    if "tgt_synth_pos" in concept:
        tgt_pos = filt_w & concept["tgt_synth_pos"](tgt_w)
        tgt_neg = filt_w & concept["tgt_synth_neg"](tgt_w)
    else:
        tgt_pos = filt_w & np.isin(tgt_w, list(concept["tgt_pos"]))
        tgt_neg = filt_w & np.isin(tgt_w, list(concept["tgt_neg"]))
    off_pos = filt_w & np.isin(off_w, list(concept["off_pos"]))
    off_neg = filt_w & np.isin(off_w, list(concept["off_neg"]))

    n = min(n_per_group, *(int(m.sum()) for m in (tgt_pos, tgt_neg, off_pos, off_neg)))
    if n < 30:
        raise ValueError(f"{concept['name']}: only {n} per group")
    sets = []
    for mask in (tgt_pos, tgt_neg, off_pos, off_neg):
        sets.append(rng.choice(np.where(mask)[0], n, replace=False))
    win_indices = np.concatenate(sets)
    file_idx_sel = file_idx[win_indices]
    local_idx_sel = local_idx[win_indices]
    y_target = np.concatenate([np.ones(n), np.zeros(n), np.full(n, -1.0), np.full(n, -1.0)])
    y_offtarget = np.concatenate([np.full(n, -1.0), np.full(n, -1.0), np.ones(n), np.zeros(n)])
    return (file_idx_sel, local_idx_sel, y_target, y_offtarget, n,
            tpw)


# ── Multi-layer steering hooks ────────────────────────────────────────────────

class MultiLayerSteer:
    """Registers modifying forward hooks on every transformer block.
    Each hook applies SAE substitution in z-space using its layer's per-layer
    state, then writes the modified activation back into the encoder."""

    def __init__(self, encoder, per_layer: list[dict]):
        self.encoder = encoder
        self.per_layer = per_layer
        self.k_per_layer = [0] * len(per_layer)
        self._handles = []
        blocks = encoder.get_hookable_layers()
        for L, blk in enumerate(blocks):
            self._handles.append(blk.register_forward_hook(self._make_hook(L)))

    def set_k(self, k_per_layer: list[int]):
        assert len(k_per_layer) == len(self.per_layer)
        self.k_per_layer = list(k_per_layer)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_hook(self, L: int):
        pl = self.per_layer[L]
        def hook(module, inputs, output):
            k = self.k_per_layer[L]
            if k <= 0:
                return output  # no-op
            n_pre = pl["n_prefix"]
            cls = output[:, :n_pre, :]
            tok = output[:, n_pre:, :]
            B, N, D = tok.shape
            x = tok.reshape(B * N, D).float()
            am = pl["act_mean"]
            as_ = pl["act_std"]
            x_n = (x - am) / (as_ + 1e-8)
            with torch.no_grad():
                z = pl["sae"].encode(x_n)
                top = pl["ranking"][:k]
                z[:, top] = pl["donor"][top]
                x_hat_n = pl["sae"].decode(z)
            x_hat = x_hat_n * (as_ + 1e-8) + am
            tok_new = x_hat.reshape(B, N, D).to(output.dtype)
            return torch.cat([cls, tok_new], dim=1)
        return hook


# ── Window forward pass ───────────────────────────────────────────────────────

def encoder_forward_with_steering(encoder, dataset, global_idx_sel,
                                  steerer: MultiLayerSteer, batch_size: int = 8) -> torch.Tensor:
    """Re-run the encoder on the selected windows with current steerer.k_per_layer.
    Returns final-layer output (n_windows, tokens_per_window, embed_dim) on CPU."""
    out_chunks = []
    n_total = len(global_idx_sel)
    for i in range(0, n_total, batch_size):
        batch = global_idx_sel[i:i + batch_size]
        xs = [dataset[int(g)][0] for g in batch]
        x = torch.stack(xs).to(DEVICE)
        with torch.no_grad():
            tokens = encoder.encode(x)        # already CLS-stripped
        out_chunks.append(tokens.float().cpu())
    return torch.cat(out_chunks, dim=0)


# ── Probe utilities ───────────────────────────────────────────────────────────

def fit_window_probe(emb_per_window: torch.Tensor, y: np.ndarray):
    """Train logistic-regression probe at token level, with window-level labels.
    emb_per_window: (n_w, tpw, D) on CPU. y: (n_w,) ∈ {0,1}.
    Returns TorchProbe trained on all tokens (token-level features, repeated
    window labels)."""
    n_w, tpw, D = emb_per_window.shape
    X = emb_per_window.reshape(n_w * tpw, D).to(DEVICE)
    y_tok = np.repeat(y, tpw)
    # Use sklearn-style holdout split, train on ¾, score on ¼.
    idx = np.arange(len(y_tok))
    idx_t, idx_v, yt, yv = train_test_split(
        idx, y_tok, test_size=0.25, random_state=RNG_SEED, stratify=y_tok,
    )
    probe = fit_probe(X[idx_t], yt, max_iter=300)
    return probe, idx_v, yv


def score_window_probe(probe, emb_per_window: torch.Tensor, y_window: np.ndarray) -> float:
    """Score per-token, average per window, compute window-level AUROC.
    Avoids per-window selection-bias: every token sees the same (frozen) probe."""
    n_w, tpw, D = emb_per_window.shape
    X = emb_per_window.reshape(n_w * tpw, D).to(DEVICE)
    with torch.no_grad():
        scores_tok = (X @ probe.w + probe.b).cpu().numpy()
    scores_window = scores_tok.reshape(n_w, tpw).mean(axis=1)
    return float(roc_auc_score(y_window, scores_window))


# ── Main sweep ────────────────────────────────────────────────────────────────

def sweep_concept(encoder, dataset, concept, per_layer, cfg):
    rng = np.random.default_rng(RNG_SEED)
    file_idx_sel, local_idx_sel, y_tgt_full, y_off_full, n, tpw = sample_windows(
        concept, N_WINDOWS_PER_GROUP, rng, cfg)

    # Convert (file_idx, local_idx) → global dataset index. Identity index_map
    # means global_idx = cumulative_lengths[file_idx] + local_idx.
    cum = np.asarray(dataset.cumulative_lengths, dtype=np.int64)
    global_idx_sel = cum[file_idx_sel.astype(np.int64)] + local_idx_sel.astype(np.int64)

    # Window-level labels: target = first 2n windows (1,0); off = next 2n (1,0)
    is_tgt = (y_tgt_full >= 0)
    is_off = (y_off_full >= 0)

    # Steerer: identity at first to capture clean baseline.
    steerer = MultiLayerSteer(encoder, per_layer)
    steerer.set_k([0] * len(per_layer))

    print(f"  [{concept['name']}] forward k=0 (clean baseline) on {len(global_idx_sel)} windows…")
    final_clean = encoder_forward_with_steering(
        encoder, dataset, global_idx_sel, steerer)
    print(f"  clean final shape = {tuple(final_clean.shape)}")

    # Train target probe on target windows; off-target probe on off-target windows.
    tgt_emb = final_clean[is_tgt]
    tgt_y_w = y_tgt_full[is_tgt].astype(np.int64)
    off_emb = final_clean[is_off]
    off_y_w = y_off_full[is_off].astype(np.int64)
    tgt_probe, _, _ = fit_window_probe(tgt_emb, tgt_y_w)
    off_probe, _, _ = fit_window_probe(off_emb, off_y_w)
    auc0_tgt = score_window_probe(tgt_probe, tgt_emb, tgt_y_w)
    auc0_off = score_window_probe(off_probe, off_emb, off_y_w)
    print(f"  clean window AUROCs: target={auc0_tgt:.3f}  off-target={auc0_off:.3f}")

    # Sweep multi-layer & single-deep-layer in same loop for efficiency.
    multi_tgt, multi_off = [auc0_tgt], [auc0_off]
    deep_tgt, deep_off = [auc0_tgt], [auc0_off]
    deep_L = cfg["deep_layer"]

    for k in K_GRID[1:]:
        # Multi-layer: same k applied at every block.
        steerer.set_k([k] * len(per_layer))
        final = encoder_forward_with_steering(
            encoder, dataset, global_idx_sel, steerer)
        m_t = score_window_probe(tgt_probe, final[is_tgt], tgt_y_w)
        m_o = score_window_probe(off_probe, final[is_off], off_y_w)
        multi_tgt.append(m_t); multi_off.append(m_o)

        # Deep-layer-only baseline: k applied at the final block only.
        only_k = [0] * len(per_layer); only_k[deep_L] = k
        steerer.set_k(only_k)
        final = encoder_forward_with_steering(
            encoder, dataset, global_idx_sel, steerer)
        s_t = score_window_probe(tgt_probe, final[is_tgt], tgt_y_w)
        s_o = score_window_probe(off_probe, final[is_off], off_y_w)
        deep_tgt.append(s_t); deep_off.append(s_o)

        print(f"  k={k:>3d}  multi(t={m_t:.3f}, o={m_o:.3f})  "
              f"L{deep_L}-only(t={s_t:.3f}, o={s_o:.3f})", flush=True)

    steerer.remove()
    return {
        "k_grid": K_GRID,
        "multi_tgt":   np.array(multi_tgt),
        "multi_off":   np.array(multi_off),
        "deep_tgt":    np.array(deep_tgt),
        "deep_off":    np.array(deep_off),
        "deep_layer":  deep_L,
        "n_windows_per_group": n,
        "tokens_per_window":   tpw,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_grid(results: dict, out_path: Path, encoder_name: str):
    n_concepts = len(results)
    fig, axes = plt.subplots(1, n_concepts, figsize=(4.0 * n_concepts, 4.2),
                             sharey=True)
    if n_concepts == 1:
        axes = [axes]
    deep_label = f"L{next(iter(results.values()))['deep_layer']}-only"
    for ax, (name, r) in zip(axes, results.items()):
        ks = r["k_grid"]
        ax.plot(ks, r["multi_tgt"],   "-o", color="C2", label="multi · target")
        ax.plot(ks, r["multi_off"],   "-s", color="C2", alpha=0.45,
                label="multi · off-target")
        ax.plot(ks, r["deep_tgt"],   "--o", color="C0",
                label=f"{deep_label} · target")
        ax.plot(ks, r["deep_off"],   "--s", color="C0", alpha=0.45,
                label=f"{deep_label} · off-target")
        ax.set_xlabel("k features substituted (per layer)")
        ax.set_title(f"{name}\n({r['n_windows_per_group']} windows/group)")
        ax.set_ylim(0.4, 1.02)
        ax.axhline(0.5, color="grey", lw=0.5, ls=":")
        ax.set_xscale("symlog", linthresh=2)
    axes[0].set_ylabel("Window-level AUROC")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"{encoder_name} multi-layer vs {deep_label} concept steering "
                 f"· frozen final-layer probe", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[saved] {out_path}")
    plt.close(fig)


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="labram", choices=list(ENCODER_CFG.keys()),
                        help="Which encoder family to run on")
    parser.add_argument("--concepts", default="Pathology",
                        help="Comma-separated concept names, or 'all'")
    parser.add_argument("--out", default=None,
                        help="Output figure name (default depends on --encoder)")
    args = parser.parse_args()

    cfg = ENCODER_CFG[args.encoder]
    if args.out is None:
        args.out = f"multi_layer_steering_{args.encoder}.png"

    if args.concepts.lower() == "all":
        names = list(ALL_CONCEPTS.keys())
    else:
        names = [c.strip() for c in args.concepts.split(",")]
    bad = [n for n in names if n not in ALL_CONCEPTS]
    if bad:
        raise SystemExit(f"unknown concepts: {bad}")

    print(f"[Loading {args.encoder} encoder]")
    encoder = load_encoder(args.encoder, weights_path=cfg["weights"]).to(DEVICE)
    if hasattr(encoder, "model"):
        encoder.model.eval()

    print(f"[Loading dataset]")
    dataset = H5PYDatasetLabeled(str(cfg["data_path"]), transform=StandardizeLabel())

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for name in names:
            concept = ALL_CONCEPTS[name]
            print(f"\n=== Concept: {name} ===", flush=True)
            print(f"[Building per-layer state]", flush=True)
            per_layer = []
            for L in range(cfg["n_layers"]):
                print(f"  L{L}…", flush=True)
                per_layer.append(build_layer_state(L, concept, cfg))
            print(f"  built {len(per_layer)} layers", flush=True)
            results[name] = sweep_concept(encoder, dataset, concept, per_layer, cfg)
            for pl in per_layer:
                pl["sae"].to("cpu")
            del per_layer
            torch.cuda.empty_cache()

    out_path = OUT_DIR / args.out
    plot_grid(results, out_path, args.encoder)

    # Save raw curves alongside.
    npz_path = out_path.with_suffix(".npz")
    save_dict = {}
    for n, r in results.items():
        for k, v in r.items():
            save_dict[f"{n}__{k}"] = np.asarray(v)
    np.savez(npz_path, **save_dict)
    print(f"[saved] {npz_path}")


if __name__ == "__main__":
    main()
