"""Build a lightweight steering cache from the full dataset.

The concept steering scripts (plot_concept_steering_xae_all.py) normally draw
tokens from the app_cache, which only covers the validation split (~10% of
subjects).  This tool collects encoder embeddings and metadata from the
*entire* dataset (train + val + test) and saves a compact cache suitable for
steering visualisations.

Only collects what the steering scripts need:
  - raw encoder embeddings  (N, embed_dim)
  - per-token metadata      (age_group, gender, classification,
                             medication_group, subject_id)

Does NOT run UMAP, codebook assignment, XAE decoding, or feature stats.
Runtime: ~5–15 min depending on GPU / dataset size.

Output:
    results/steering_cache/{experiment}/steering_cache.pt
      keys: embeddings, age_group, gender, classification,
            medication_group, subject_id, experiment, n_tokens, embed_dim

Usage::

    uv run tools/build_steering_cache.py --experiment sleepfm_finetuned_layer2
    uv run tools/build_steering_cache.py --experiment sleepfm_finetuned_layer2 \\
        --max-tokens 200000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT   = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Import shared helpers from build_app_cache
sys.path.insert(0, str(ROOT / "tools"))
import build_app_cache as _bac
from build_app_cache import (
    _collect_tokens,
    _collect_token_metadata,
    _ENCODER_DATA,
    METADATA_FIELDS,
    _load_model,
)

from sae4eeg.dataset import H5PYDatasetLabeled, StandardizeLabel


def _collect_embeddings_dual(model, full_loader, primary_layer, max_tokens):
    """Single-pass capture of embeddings at ``primary_layer`` + final layer.

    Replaces back-to-back ``_collect_tokens`` calls when target_layer !=
    final_layer (every REVE non-L21 and LaBraM non-L11 cache). The hook
    captures the primary layer; ``model.encode()`` returns the final layer
    in the same forward pass — no redundant second pass over the dataset.

    Spectral targets, raw patches, and labels are *not* needed by the
    steering cache, so we drop them entirely (further win vs _collect_tokens).
    """
    from sae4eeg.encoders import EncoderBackend
    from sae4eeg.sae import ActivationExtractor
    from tqdm import tqdm

    assert isinstance(model, EncoderBackend), \
        "dual-layer collection only supports EncoderBackend (REVE/LaBraM)"

    all_layers      = model.get_hookable_layers()
    n_layers        = len(all_layers)
    final_layer_idx = n_layers - 1
    assert primary_layer != final_layer_idx, \
        "primary_layer == final_layer; use _collect_tokens single-pass instead"

    n_prefix_strip = int(getattr(model, "n_prefix_tokens", 0))
    extractor = ActivationExtractor(model.model,
                                    layers=[all_layers[primary_layer]])

    primary_list, final_list = [], []
    total = 0
    tokens_per_window = 1
    pbar = tqdm(full_loader,
                desc=f"Dual collect (L{primary_layer}+L{final_layer_idx})",
                leave=False)
    for batch in pbar:
        x = batch[0].to(DEVICE)
        B = x.shape[0]
        with torch.no_grad():
            extractor.clear()
            final_act   = model.encode(x)                  # already CLS-stripped
            primary_act = extractor.get_activations()[0]   # raw hook output

        # Reshape REVE 4D → 3D (per-channel-per-second tokens).
        if primary_act.dim() == 4:
            B2, Cc, Ss, Ee = primary_act.shape
            primary_act = primary_act.reshape(B2, Cc * Ss, Ee)
        if final_act.dim() == 4:
            B2, Cc, Ss, Ee = final_act.shape
            final_act = final_act.reshape(B2, Cc * Ss, Ee)
        # Strip LaBraM CLS prefix from hook output only — encode() already strips.
        if n_prefix_strip and primary_act.dim() == 3:
            primary_act = primary_act[:, n_prefix_strip:, :]

        S     = primary_act.shape[1]
        n_new = B * S
        if total == 0:
            tokens_per_window = S

        primary_list.append(primary_act.reshape(n_new, -1).cpu())
        final_list.append(final_act.reshape(n_new, -1).cpu())
        total += n_new
        pbar.set_postfix(tokens=f"{total:,}")
        if total >= max_tokens:
            break

    pbar.close()
    extractor.remove_hooks()

    primary_emb = torch.cat(primary_list)[:max_tokens]
    final_emb   = torch.cat(final_list)[:max_tokens]
    return primary_emb, final_emb, tokens_per_window

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default="sleepfm_finetuned_layer2")
parser.add_argument("--max-tokens", type=int, default=0,
                    help="Cap on tokens to collect (0 = no cap, use all)")
args = parser.parse_args()

EXP_DIR  = ROOT / "results" / "experiments" / args.experiment
OUT_DIR  = ROOT / "results" / "steering_cache" / args.experiment
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "steering_cache.pt"

# ── Load experiment metadata ──────────────────────────────────────────────────
meta = json.loads((EXP_DIR / "metadata.json").read_text())
encoder_name = meta.get("encoder", "sleepfm")
data_path    = meta.get("data_path", _ENCODER_DATA.get(encoder_name,
                         "data/D4-v3-preprocessed-v2"))

# Propagate target layer + patch size into build_app_cache module scope —
# _collect_tokens reads them from there. Without this, REVE/sleepfm layers
# != module default would silently get the wrong activations.
if "target_layer" not in meta:
    raise ValueError(f"metadata.json for {args.experiment} missing 'target_layer'")
_bac.TARGET_LAYER = int(meta["target_layer"])
_bac.PATCH_SIZE   = int(meta.get("patch_size", 128))
print(f"  target_layer={_bac.TARGET_LAYER}  patch_size={_bac.PATCH_SIZE}")

print(f"Experiment : {args.experiment}")
print(f"Encoder    : {encoder_name}")
print(f"Data path  : {data_path}")
print(f"Device     : {DEVICE}")

# ── Load encoder & XAE (XAE needed by _collect_tokens) ───────────────────────
print("\n[Loading encoder]")
weights_path = meta.get("weights_path")
model = _load_model(encoder_name,
                    weights_path=ROOT / weights_path if weights_path else None)

print("\n[Loading XAE]")
from sae4eeg.xae import XAETrainer
xae_trainer = XAETrainer(embed_dim=meta["embed_dim"], device=DEVICE)
xae_trainer.load(str(ROOT / meta["xae_checkpoint"]))
xae_trainer.xae.to(DEVICE).eval()

# ── Full dataset loader ───────────────────────────────────────────────────────
print("\n[Building full-dataset loader]")
full_ds = H5PYDatasetLabeled(str(ROOT / data_path), transform=StandardizeLabel())
full_loader = DataLoader(full_ds, batch_size=32, shuffle=False, num_workers=0)

n_subjects = len(np.unique(full_ds.subjects))
print(f"  {len(full_ds)} windows  ·  {n_subjects} subjects")

# ── Collect embeddings ────────────────────────────────────────────────────────
# When target_layer != final_layer (every REVE non-L21 and LaBraM non-L11 case)
# we need embeddings at both layers — once for the SAE/CAV space, once for
# the M2_post probe. Hook the primary layer + read encode()'s last-layer
# output in a single forward pass (~2× faster than two _collect_tokens calls).
max_tokens = args.max_tokens if args.max_tokens > 0 else len(full_ds) * 1000  # 1000 >> any realistic tok/window
n_layers_total = len(model.get_hookable_layers()) if hasattr(model, "get_hookable_layers") else None
final_layer    = (n_layers_total - 1) if n_layers_total is not None else _bac.TARGET_LAYER

if final_layer != _bac.TARGET_LAYER:
    print(f"\n[Dual-layer collect — primary L{_bac.TARGET_LAYER}, final L{final_layer}, cap={max_tokens}]")
    embeddings, embeddings_final, tokens_per_window = _collect_embeddings_dual(
        model, full_loader, primary_layer=int(_bac.TARGET_LAYER),
        max_tokens=max_tokens)
    print(f"  Collected {len(embeddings)} tokens  ({tokens_per_window} tok/window)")
    print(f"  Primary  : {tuple(embeddings.shape)} at layer {_bac.TARGET_LAYER}")
    print(f"  Final    : {tuple(embeddings_final.shape)} at layer {final_layer}")
else:
    print(f"\n[Single-pass collect — target == final layer, cap={max_tokens}]")
    embeddings, _, _, _, tokens_per_window = _collect_tokens(
        model, xae_trainer, full_loader, max_tokens)
    embeddings_final = None  # SleepFM L2 == final, M2 already correct
    print(f"  Collected {len(embeddings)} tokens  ({tokens_per_window} tok/window)")

# ── Collect window indices so M2_post can re-run encoder with injection ─────
# build_steering_cache walks full_loader in default order; tokens are flat so
# token i ↔ window i // tokens_per_window. We store the dataset-level window
# coordinates so the M2_post tool can reload the EEG.
n_windows_used = (len(embeddings) + tokens_per_window - 1) // tokens_per_window
file_idx_arr  = full_ds.index_map["file_indices"][:n_windows_used]
local_idx_arr = full_ds.index_map["local_indices"][:n_windows_used]
if torch.is_tensor(file_idx_arr):
    file_idx_arr = file_idx_arr.numpy()
if torch.is_tensor(local_idx_arr):
    local_idx_arr = local_idx_arr.numpy()

# ── Collect metadata ──────────────────────────────────────────────────────────
print("\n[Collecting metadata]")
fields = ["age_group", "gender", "classification", "medication_group"]
token_meta = _collect_token_metadata(
    full_ds, fields, len(embeddings), tokens_per_window)

# ── Save ──────────────────────────────────────────────────────────────────────
payload = {
    "experiment":         args.experiment,
    "n_tokens":           len(embeddings),
    "embed_dim":          embeddings.shape[1],
    "tokens_per_window":  tokens_per_window,
    "target_layer":       int(_bac.TARGET_LAYER),
    "final_layer":        int(final_layer),
    "embeddings":         embeddings,            # (N, embed_dim) at layer L
    "embeddings_final":   embeddings_final,      # (N, embed_dim_final) or None
    "window_file_idx":    file_idx_arr,          # (n_windows_used,)
    "window_local_idx":   local_idx_arr,         # (n_windows_used,)
    "data_path":          str(data_path),
    **{k: token_meta[k] for k in fields},
    "subject_id":         token_meta["subject_id"],
}

torch.save(payload, OUT_PATH)
print(f"\nSaved {OUT_PATH}")
print(f"  tokens={len(embeddings)}  subjects={len(np.unique(token_meta['subject_id']))}")
print(f"  size={OUT_PATH.stat().st_size / 1e6:.1f} MB")
