"""
train_sae_layers.py — Train a SAE on every transformer layer of an encoder
===========================================================================
Collects activations for ALL (or a specified subset of) layers in a single
forward pass through the dataset, then trains one SAE per layer sequentially.

Checkpoints are saved to:
  results/features/{encoder}/sae_{encoder}_exp{E}_k{K}_layer{L}.pt

Use --tag to namespace a finetuned run so results don't overwrite the base model:
  results/features/{encoder}_{tag}/sae_{encoder}_exp{E}_k{K}_layer{L}.pt

Usage
-----
  # All layers (default)
  uv run tools/train_sae_layers.py --encoder reve

  # Finetuned REVE, all layers, results in results/features/reve_qjbe08/
  uv run tools/train_sae_layers.py --encoder reve \
      --weights-path checkpoints/reve_qjbe08.ckpt --tag qjbe08

  # Subset of layers
  uv run tools/train_sae_layers.py --encoder reve --layers 0,5,10,15,21

  # SleepFM
  uv run tools/train_sae_layers.py --encoder sleepfm
"""

import argparse
import time
from pathlib import Path

import torch

from mecheeg.sae import SAETrainer, SparseAutoencoder, sae_diagnostics
from mecheeg.dataset import get_dataloaders, StandardizeLabel, V4ResampleTransform
from mecheeg.encoders import load_encoder

# ── paths & constants ────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

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

# SAE hyper-parameters (match the settings used in explore_features.py)
EXPANSION      = 1.0
K              = 8
EPOCHS         = 20
BATCH_SIZE     = 256
RESAMPLE_EVERY = 2


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train SAE on all encoder layers")
    _all_encoders = ["sleepfm", "sleepfm_v2.0", "sleepfm_v2.1", "sleepfm_v2.3", "sleepfm_v2.4", "sleepfm_v2.5", "sleepfm_v2.6", "sleepfm_v2.7", "sleepfm_granular", "reve", "labram"]
    p.add_argument("--encoder", default="reve", choices=_all_encoders,
                   help="Encoder backend (default: reve)")
    p.add_argument("--layers", default=None,
                   help="Comma-separated layer indices to train on "
                        "(default: all layers)")
    p.add_argument("--expansion", type=float, default=EXPANSION,
                   help=f"SAE expansion factor (default: {EXPANSION})")
    p.add_argument("--k", type=int, default=K,
                   help=f"TopK sparsity (default: {K})")
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help=f"Training epochs per layer (default: {EPOCHS})")
    p.add_argument("--max-tokens", type=int, default=500_000,
                   help="Max activation tokens to collect per layer (default: 500000). "
                        "Caps memory use; 500K tokens at dim=512 ≈ 1 GB.")
    p.add_argument("--collect-batch-size", type=int, default=16,
                   help="Batch size for encoder forward passes during collection "
                        "(default: 16). Increase for faster collection on larger GPUs.")
    p.add_argument("--weights-path", default=None,
                   help="Path to finetuned encoder checkpoint (.ckpt or .pt). "
                        "For REVE: PyTorch Lightning .ckpt with 'encoder.*' keys. "
                        "For SleepFM: overrides the default sleepfm_weights.pt.")
    p.add_argument("--tag", default=None,
                   help="Short label appended to the output directory "
                        "(e.g. 'qjbe08' → results/features/reve_qjbe08/). "
                        "Use to keep finetuned runs separate from the base model.")
    p.add_argument("--force", action="store_true",
                   help="Retrain even if checkpoint already exists")
    return p.parse_args()


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
        # Fall back to the upstream pretrained file when the local finetune is missing.
        resolved = ckpt if ckpt.exists() else (ROOT / "checkpoints" / "pretrained" / "labram" / "labram-base.pth")
    else:
        resolved = None
    kwargs = {"weights_path": resolved} if resolved is not None else {}
    encoder = load_encoder(encoder_name, **kwargs)
    encoder.to(DEVICE).eval()
    n_layers = len(encoder.get_hookable_layers())
    print(f"  encoder={encoder_name}  embed_dim={encoder.embed_dim}  "
          f"layers={n_layers}")
    return encoder


def main():
    args = parse_args()
    t0 = time.time()

    # ── Load encoder ─────────────────────────────────────────────────────────
    print(f"\nLoading encoder: {args.encoder} …")
    encoder = load_model(args.encoder, weights_path=args.weights_path)
    n_layers = len(encoder.get_hookable_layers())

    # ── Resolve target layers ─────────────────────────────────────────────────
    if args.layers is not None:
        target_layers = [int(x.strip()) for x in args.layers.split(",")]
    else:
        target_layers = list(range(n_layers))
    print(f"  target layers: {target_layers}")

    # ── Resolve output directory ──────────────────────────────────────────────
    run_name = f"{args.encoder}_{args.tag}" if args.tag else args.encoder
    out_dir = ROOT / "results" / "features" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  output dir: {out_dir.relative_to(ROOT)}")

    def ckpt_path(layer: int) -> Path:
        return out_dir / f"sae_{args.encoder}_exp{args.expansion}_k{args.k}_layer{layer}.pt"

    existing = [L for L in target_layers if ckpt_path(L).exists()]
    if existing and args.force:
        # Require explicit acknowledgement before overwriting trained checkpoints.
        # This prevents accidental loss of SAE weights that downstream caches depend on.
        import sys
        print(f"\n  *** WARNING: --force will OVERWRITE {len(existing)} existing checkpoint(s):")
        for L in existing:
            p = ckpt_path(L)
            print(f"      {p}  ({p.stat().st_size // 1024} KB)")
        print("  This cannot be undone. Type 'yes' to confirm: ", end="", flush=True)
        answer = sys.stdin.readline().strip()
        if answer.lower() != "yes":
            print("  Aborted.")
            return
        # Make writable before overwriting (checkpoints are stored read-only)
        for L in existing:
            ckpt_path(L).chmod(0o644)

    if not args.force:
        pending = [L for L in target_layers if not ckpt_path(L).exists()]
        skipped = existing
        if skipped:
            print(f"  skipping already-trained layers: {skipped}  (use --force to retrain)")
        if not pending:
            print("  all checkpoints exist — nothing to do. Use --force to retrain.")
            return
        target_layers = pending

    # ── Load dataset ──────────────────────────────────────────────────────────
    data_path = _ENCODER_DATA[args.encoder]
    transform = V4ResampleTransform() if "D4-v4" in str(data_path) else StandardizeLabel()
    print(f"\nLoading dataset: {data_path.name} …", flush=True)
    gen = get_dataloaders(train_path=str(data_path),
                          transformer=transform,
                          batch_size=args.collect_batch_size)
    _, train_loader, val_loader, _ = next(gen)
    print(f"  train batches={len(train_loader)}  val batches={len(val_loader)}", flush=True)

    # ── Train one layer at a time (collect → train → save → free) ───────────
    # Collecting all 22 layers simultaneously would require hundreds of GB.
    # Instead we do one forward pass per layer, keeping only one layer's
    # activation buffer in memory at a time.
    for layer_idx in target_layers:
        ckpt = ckpt_path(layer_idx)

        print(f"\n{'='*70}")
        print(f"=== Layer {layer_idx}/{n_layers-1} ===")

        trainer = SAETrainer(
            embed_dim=encoder.embed_dim,
            expansion=args.expansion,
            mode="topk",
            k=args.k,
            target_layers=[layer_idx],
            epochs=args.epochs,
            batch_size=BATCH_SIZE,
            resample_every=RESAMPLE_EVERY,
            device=DEVICE,
        )

        # Collect train activations for this layer only
        t_collect = time.time()
        train_acts_dict = trainer.collect_activations(
            encoder, train_loader, max_tokens=args.max_tokens
        )
        acts = train_acts_dict[layer_idx]
        print(f"  collected {acts.shape[0]} tokens in "
              f"{(time.time()-t_collect)/60:.1f} min")

        # Normalise
        mean = acts.mean(dim=0)
        std  = acts.std(dim=0).clamp(min=1e-8)
        trainer.act_means[layer_idx] = mean
        trainer.act_stds[layer_idx]  = std
        # Pin to GPU so SAE training avoids per-minibatch CPU→GPU transfers
        acts_norm = ((acts - mean) / std).to(DEVICE)
        del acts, train_acts_dict  # free before training

        # Collect val activations for this layer only
        val_acts_dict = trainer.collect_activations(
            encoder, val_loader, max_tokens=args.max_tokens // 5
        )
        val_acts_norm = ((val_acts_dict[layer_idx] - mean) / std).to(DEVICE)
        del val_acts_dict

        # Train SAE
        t_layer = time.time()
        sae = trainer._train_single_sae(acts_norm, layer_idx)
        elapsed_layer = time.time() - t_layer

        print(f"\n  Validation diagnostics (layer {layer_idx}):")
        sae_diagnostics(sae, val_acts_norm, device=DEVICE)
        del acts_norm, val_acts_norm

        # Save checkpoint (same format as explore_features.py)
        torch.save({
            "sae_state_dict":   sae.state_dict(),
            "act_mean":         trainer.act_means[layer_idx],
            "act_std":          trainer.act_stds[layer_idx],
            "embed_dim":        encoder.embed_dim,
            "expansion":        args.expansion,
            "k":                args.k,
            "target_layer":     layer_idx,
            "encoder":          args.encoder,
            "encoder_weights":  str(args.weights_path) if args.weights_path else None,
            "tag":              args.tag,
        }, ckpt)
        # Make read-only immediately to prevent accidental overwrite by future runs.
        ckpt.chmod(0o444)
        print(f"  Saved (read-only) → {ckpt.name}  (train+val: {elapsed_layer/60:.1f} min)")

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"All done. {len(target_layers)} SAEs trained in {total/60:.1f} min.")
    print(f"Checkpoints in: {out_dir}/")


if __name__ == "__main__":
    main()
