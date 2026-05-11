"""Build attention cache for the Attention Explorer app page.

For each feature, finds the top-K windows by mean SAE activation, then
extracts temporal transformer layer-0 attention weights for those windows.

Only supported for SleepFM-family encoders (SetTransformer architecture).
REVE support is skipped gracefully (attention_cache["windows_attn"] = None).

Output: results/experiments/{experiment}/attention_cache.pt

Cache schema
------------
  encoder           str
  experiment        str
  n_features        int
  n_heads           int       (0 if no attention extracted)
  S                 int       tokens per window (60 for SleepFM 1-s patches)
  fs                int       sample rate (128)
  patch_size        int       samples per token (128)
  K                 int       top-K windows stored per feature
  channel_names     list[str] 19 standard 10-20 names

  top_window_idx        LongTensor  (n_features, K)  index into unique windows
  top_window_mean_acts  Tensor      (n_features, K)  peak token SAE activation
  top_window_labels     Tensor      (n_features, K)  diagnostic label (float)

  windows_eeg        Tensor  (n_unique, C, T)              float16
  windows_feat_acts  Tensor  (n_unique, S, n_features)     float16
  windows_attn       Tensor  (n_unique, n_heads, S, S)     float16  or None

Usage::

    uv run tools/build_attention_cache.py --experiment sleepfm_finetuned_layer2
    uv run tools/build_attention_cache.py --experiment sleepfm_finetuned_layer2 \\
        --top-k 5 --max-windows 5000
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sae4eeg.sae import SparseAutoencoder
from sae4eeg.dataset import get_dataloaders, StandardizeLabel
from sae4eeg.encoders import load_encoder, SleepFMBackend

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7",  "C3",  "Cz", "C4", "T8",
    "T5",  "P3",  "Pz", "P4", "T6", "O1", "O2",
]

_ENCODER_DATA = {
    "sleepfm":      "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.0": "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.1": "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.3": "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.4": "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.5": "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.6": "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.7": "data/D4-v3-preprocessed-v2",
    "reve":         "data/D4-v3-preprocessed-v1",
}


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_layer0_attention(backend: SleepFMBackend, x: torch.Tensor) -> torch.Tensor:
    """Replicate SleepFM forward pass up to temporal transformer layer 0 and
    return per-head attention weights.

    Parameters
    ----------
    backend : SleepFMBackend
    x : Tensor (B, C, T) on the correct device

    Returns
    -------
    attn_weights : Tensor (B, n_heads, S, S) float32 on CPU
    """
    from einops import rearrange

    raw_model = backend.model
    with torch.no_grad():
        emb = raw_model.patch_embedding(x)          # (B, C_ch, S, E)
        B, C_ch, S, E = emb.shape
        emb = rearrange(emb, "b c s e -> (b s) c e")
        emb = raw_model.spatial_pooling(emb)        # (B*S, E)
        emb = emb.view(B, S, E)
        emb = raw_model.positional_encoding(emb)
        emb = raw_model.layer_norm(emb)

        layer0 = raw_model.transformer_encoder.layers[0]
        x_norm = layer0.norm1(emb)                  # pre-norm
        _, attn_weights = layer0.self_attn(
            x_norm, x_norm, x_norm,
            need_weights=True,
            average_attn_weights=False,             # keep per-head: (B, n_heads, S, S)
        )
    return attn_weights.cpu()                       # (B, n_heads, S, S)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build attention cache for the Attention Explorer page")
    parser.add_argument("--experiment", "-e", required=True,
                        help="Experiment name (matches results/experiments/<name>/)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K windows to store per feature (default: 5)")
    parser.add_argument("--max-windows", type=int, default=5000,
                        help="Max validation windows to scan in pass 1 (default: 5000)")
    args = parser.parse_args()

    exp_dir = ROOT / "results" / "experiments" / args.experiment
    meta_path = exp_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata.json at {meta_path}\n"
            f"Create one first (see results/experiments/ for examples)."
        )

    meta = json.loads(meta_path.read_text())
    out_path = exp_dir / "attention_cache.pt"

    encoder_name  = meta.get("encoder", "sleepfm")
    encoder_embed = meta.get("embed_dim", 128)
    encoder_patch = meta.get("patch_size", 128)
    data_path     = meta.get(
        "data_path", _ENCODER_DATA.get(encoder_name, "data/D4-v3-preprocessed-v2"))
    weights_path  = meta.get("weights_path", None)
    if weights_path:
        weights_path = ROOT / weights_path

    print("=" * 72)
    print(f"  Building attention cache: {args.experiment}")
    print(f"  Output: {out_path}")
    print("=" * 72)

    # ── Load encoder ─────────────────────────────────────────────────
    if encoder_name == "sleepfm":
        backend = load_encoder(
            "sleepfm",
            weights_path=weights_path
            or ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt",
        )
    else:
        kwargs = {"weights_path": weights_path} if weights_path else {}
        backend = load_encoder(encoder_name, **kwargs)
    backend = backend.to(DEVICE).eval()

    has_attention = isinstance(backend, SleepFMBackend)
    if not has_attention:
        print(f"  NOTE: Attention extraction not supported for {encoder_name}.")
        print(f"  windows_attn will be None in the cache.")

    # ── Load SAE ─────────────────────────────────────────────────────
    sae_path = ROOT / meta["sae_checkpoint"]
    sae_ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
    act_mean = sae_ckpt["act_mean"].to(DEVICE)
    act_std  = sae_ckpt["act_std"].to(DEVICE)

    sae = SparseAutoencoder(
        encoder_embed,
        expansion=sae_ckpt.get("expansion", 1.0),
        mode="topk",
        k=sae_ckpt.get("k", 8),
    )
    sae.load_state_dict(sae_ckpt["sae_state_dict"])
    sae = sae.to(DEVICE).eval()
    n_features = sae.encoder.weight.shape[0]
    print(f"  SAE loaded: {n_features} features, k={sae_ckpt.get('k', 8)}")

    # ── Dataset ──────────────────────────────────────────────────────
    gen1 = get_dataloaders(
        train_path=str(ROOT / data_path),
        transformer=StandardizeLabel(),
    )
    _, _, val_loader, _ = next(gen1)
    print(f"  Dataset ready ({len(val_loader.dataset)} windows in val split)")

    # ── Pass 1: collect peak token SAE activations per window ────────
    print(f"\n[Pass 1] Scanning up to {args.max_windows} windows ...")
    mean_acts_list: list[torch.Tensor] = []   # each (B, n_features) — peak token activation
    labels_list:    list[torch.Tensor] = []

    n_scanned = 0
    for batch in tqdm(val_loader, desc="Pass 1", unit="batch"):
        x = batch[0].to(DEVICE)
        y = batch[1].float() if len(batch) > 1 else torch.zeros(x.shape[0])
        B, C, T = x.shape
        S = T // encoder_patch

        with torch.no_grad():
            acts = backend.encode(x)                           # (B, S, E)
            acts_flat = acts.reshape(B * S, encoder_embed)
            acts_norm = (acts_flat - act_mean) / act_std
            z_flat    = sae.encode(acts_norm)                  # (B*S, n_features)
            z_3d      = z_flat.reshape(B, S, n_features)
            mean_acts = z_3d.max(dim=1).values.cpu()           # (B, n_features) — peak token activation

        mean_acts_list.append(mean_acts)
        labels_list.append(y.cpu())
        n_scanned += B
        if n_scanned >= args.max_windows:
            break

    window_mean_acts = torch.cat(mean_acts_list)   # (N_win, n_features)
    window_labels    = torch.cat(labels_list)       # (N_win,)
    N_win = window_mean_acts.shape[0]
    K = min(args.top_k, N_win)
    print(f"  Scanned {N_win} windows, K={K}")

    # Top-K window indices per feature (global DataLoader positions)
    # argsort dim=0 → for each feature column, sorted window indices
    sorted_idx      = window_mean_acts.argsort(dim=0, descending=True)  # (N_win, n_features)
    top_global_idx  = sorted_idx[:K].T.contiguous()                     # (n_features, K)

    # Gather mean activations and labels for top windows
    feat_range           = torch.arange(n_features).unsqueeze(1).expand(n_features, K)
    top_window_mean_acts = window_mean_acts[top_global_idx, feat_range]  # (n_features, K)
    top_window_labels    = window_labels[top_global_idx]                 # (n_features, K)

    # Unique window global indices needed for pass 2
    unique_global = sorted(torch.unique(top_global_idx.flatten()).tolist())
    global_to_local = {g: i for i, g in enumerate(unique_global)}
    n_unique = len(unique_global)
    print(f"  Unique windows to collect: {n_unique}")

    # Remap feature→window from global to local indices
    top_window_local_idx = torch.tensor(
        [[global_to_local[g.item()] for g in row] for row in top_global_idx],
        dtype=torch.long,
    )  # (n_features, K)

    # ── Pass 2: collect raw EEG + attention for unique windows ────────
    print(f"\n[Pass 2] Collecting raw EEG + attention for {n_unique} windows ...")
    gen2 = get_dataloaders(
        train_path=str(ROOT / data_path),
        transformer=StandardizeLabel(),
    )
    _, _, val_loader2, _ = next(gen2)

    target_set  = set(unique_global)
    max_target  = max(unique_global)
    collected_eeg:   list[torch.Tensor] = []
    collected_acts:  list[torch.Tensor] = []
    collected_attn:  list[torch.Tensor] = []

    global_win = 0
    for batch in tqdm(val_loader2, desc="Pass 2", unit="batch"):
        x = batch[0]
        B, C, T = x.shape
        S = T // encoder_patch

        for i in range(B):
            if global_win in target_set:
                xi = x[i : i + 1].to(DEVICE)

                with torch.no_grad():
                    acts_i   = backend.encode(xi)                      # (1, S, E)
                    af       = acts_i.reshape(S, encoder_embed)
                    af_norm  = (af - act_mean) / act_std
                    z_i      = sae.encode(af_norm)                     # (S, n_features)

                collected_eeg.append(x[i].cpu().half())                # (C, T)
                collected_acts.append(z_i.cpu().half())                # (S, n_features)

                if has_attention:
                    attn_i = _extract_layer0_attention(backend, xi)    # (1, n_heads, S, S)
                    collected_attn.append(attn_i[0].half())            # (n_heads, S, S)

            global_win += 1

        if global_win > max_target:
            break

    windows_eeg_t       = torch.stack(collected_eeg)   # (n_unique, C, T) float16
    windows_feat_acts_t = torch.stack(collected_acts)  # (n_unique, S, n_features) float16
    windows_attn_t: torch.Tensor | None = None
    n_heads = 0
    if has_attention and collected_attn:
        windows_attn_t = torch.stack(collected_attn)   # (n_unique, n_heads, S, S) float16
        n_heads = windows_attn_t.shape[1]

    S_actual = windows_feat_acts_t.shape[1]
    print(f"  EEG shape:       {tuple(windows_eeg_t.shape)}")
    print(f"  Feat acts shape: {tuple(windows_feat_acts_t.shape)}")
    if windows_attn_t is not None:
        print(f"  Attn shape:      {tuple(windows_attn_t.shape)}")

    # ── Save ──────────────────────────────────────────────────────────
    cache = {
        "encoder":      encoder_name,
        "experiment":   args.experiment,
        "n_features":   n_features,
        "n_heads":      n_heads,
        "S":            S_actual,
        "fs":           backend.sample_rate_in,
        "patch_size":   encoder_patch,
        "K":            K,
        "channel_names": CHANNEL_NAMES,

        "top_window_idx":        top_window_local_idx,    # (n_features, K) → local index
        "top_window_mean_acts":  top_window_mean_acts,    # (n_features, K) float32
        "top_window_labels":     top_window_labels,       # (n_features, K) float32

        "windows_eeg":        windows_eeg_t,              # (n_unique, C, T) float16
        "windows_feat_acts":  windows_feat_acts_t,        # (n_unique, S, n_features) float16
        "windows_attn":       windows_attn_t,             # (n_unique, n_heads, S, S) float16 or None
    }

    torch.save(cache, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
