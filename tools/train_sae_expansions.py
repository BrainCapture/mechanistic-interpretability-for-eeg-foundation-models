"""train_sae_expansions.py — Train multiple SAEs at the same encoder layer.

Used to materialise an expansion sweep at a *single* layer (e.g. REVE layer 8
at E=2, 4, 8, 16) without re-collecting activations between runs. Activation
collection dominates wall time when E is small, so sharing one collection
pass across the sweep saves ~30 min for a 4-expansion sweep on a T4.

Saves per-expansion checkpoints with the same naming convention as
``tools/train_sae_layers.py``::

    results/features/{encoder}{_tag}/sae_{encoder}_exp{E}.0_k{K}_layer{L}.pt

Usage::

    uv run tools/train_sae_expansions.py --encoder reve --tag qjbe08 \\
        --weights-path checkpoints/finetuned/reve_qjbe08.ckpt \\
        --layer 8 --expansions 2,4,8,16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from sae4eeg.sae import SAETrainer, sae_diagnostics
from sae4eeg.dataset import StandardizeLabel, V4ResampleTransform, get_dataloaders
from sae4eeg.encoders import load_encoder

ROOT   = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mirror the encoder-data + checkpoint maps from train_sae_layers.py.
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
    "sleepfm_granular": ROOT / "data" / "D4-v4-preprocessed-10s",
    "reve":             ROOT / "data" / "D4-v3-preprocessed-v1",
    "labram":           ROOT / "data" / "D4-v3-preprocessed-v1",
}
for k in _V2_CHECKPOINTS:
    _ENCODER_DATA[k] = ROOT / "data" / "D4-v3-preprocessed-v2"
_LABRAM_CHECKPOINTS = {"labram": ROOT / "checkpoints" / "finetuned" / "labram_binary" / "finetuned.ckpt"}
_GRANULAR_CHECKPOINTS = {"sleepfm_granular": ROOT / "checkpoints" / "granular" / "sleepfm_granular.ckpt"}
MODEL_PATH = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"


def _load_encoder(encoder_name: str, weights_path):
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
        resolved = ckpt if ckpt.exists() else (ROOT / "checkpoints" / "pretrained" / "labram" / "labram-base.pth")
    else:
        resolved = None
    kwargs = {"weights_path": resolved} if resolved is not None else {}
    encoder = load_encoder(encoder_name, **kwargs)
    encoder.to(DEVICE).eval()
    return encoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True)
    p.add_argument("--layer",   type=int, required=True)
    p.add_argument("--expansions", required=True,
                   help="Comma-separated expansion factors, e.g. 2,4,8,16")
    p.add_argument("--k", type=str, default="8",
                   help="Active features per token. Either a single int (broadcast "
                        "across all expansions) or a comma-separated list of "
                        "the same length as --expansions (paired per E).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=500_000)
    p.add_argument("--collect-batch-size", type=int, default=16)
    p.add_argument("--weights-path", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    Es = [float(x.strip()) for x in args.expansions.split(",")]
    k_parts = [int(x.strip()) for x in args.k.split(",")]
    if len(k_parts) == 1:
        Ks = [k_parts[0]] * len(Es)
    elif len(k_parts) == len(Es):
        Ks = k_parts
    else:
        raise SystemExit(f"--k must be a single int or {len(Es)} comma-separated ints "
                         f"(got {len(k_parts)})")
    print(f"Encoder={args.encoder}  layer={args.layer}  "
          f"expansions={Es}  k={Ks}")

    run_name = f"{args.encoder}_{args.tag}" if args.tag else args.encoder
    out_dir  = ROOT / "results" / "features" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def ckpt_path(E: float, K: int) -> Path:
        return out_dir / f"sae_{args.encoder}_exp{E}_k{K}_layer{args.layer}.pt"

    todo: list[tuple[float, int]] = []
    for E, K in zip(Es, Ks):
        p_ck = ckpt_path(E, K)
        if p_ck.exists() and not args.force:
            print(f"  [skip] exists: {p_ck.name}  (use --force to retrain)")
        else:
            todo.append((E, K))
    if not todo:
        print("Nothing to train.")
        return

    # ── Load encoder + dataset once ────────────────────────────────────────────
    print("\nLoading encoder …")
    encoder  = _load_encoder(args.encoder, args.weights_path)
    n_layers = len(encoder.get_hookable_layers())
    print(f"  embed_dim={encoder.embed_dim}  n_layers={n_layers}")
    if not (0 <= args.layer < n_layers):
        raise SystemExit(f"layer={args.layer} out of range for {n_layers}-layer encoder")

    data_path = _ENCODER_DATA[args.encoder]
    transform = V4ResampleTransform() if "D4-v4" in str(data_path) else StandardizeLabel()
    print(f"\nLoading dataset: {data_path.name}")
    gen = get_dataloaders(train_path=str(data_path),
                          transformer=transform,
                          batch_size=args.collect_batch_size)
    _, train_loader, val_loader, _ = next(gen)

    # ── Collect activations for the target layer (shared across expansions) ──
    # Use a tiny "collector" SAETrainer just to call collect_activations; we'll
    # build a fresh trainer per expansion below so each gets its own normalisation.
    collector = SAETrainer(
        embed_dim=encoder.embed_dim, expansion=1.0, mode="topk", k=Ks[0],
        target_layers=[args.layer], device=DEVICE,
    )
    t0 = time.time()
    print(f"\nCollecting train activations at layer {args.layer} "
          f"(max_tokens={args.max_tokens:,})…", flush=True)
    train_acts = collector.collect_activations(encoder, train_loader, max_tokens=args.max_tokens)[args.layer]
    print(f"  {train_acts.shape[0]:,} train tokens in {(time.time()-t0)/60:.1f} min")

    print(f"Collecting val activations at layer {args.layer} "
          f"(max_tokens={args.max_tokens // 5:,})…", flush=True)
    t1 = time.time()
    val_acts = collector.collect_activations(encoder, val_loader, max_tokens=args.max_tokens // 5)[args.layer]
    print(f"  {val_acts.shape[0]:,} val tokens in {(time.time()-t1)/60:.1f} min")
    del collector

    mean = train_acts.mean(dim=0)
    std  = train_acts.std(dim=0).clamp(min=1e-8)
    train_norm = ((train_acts - mean) / std).to(DEVICE)
    val_norm   = ((val_acts   - mean) / std).to(DEVICE)
    del train_acts, val_acts
    print(f"\nNormalisation stats: mean[0:3]={mean[:3].tolist()}  std[0:3]={std[:3].tolist()}")

    # ── Train one SAE per (expansion, k) pair ─────────────────────────────────
    for E, K in todo:
        out_p = ckpt_path(E, K)
        print(f"\n{'='*70}")
        print(f"=== Layer {args.layer} · expansion {E} · k={K} "
              f"(n_features={int(encoder.embed_dim * E)}) ===")
        trainer = SAETrainer(
            embed_dim=encoder.embed_dim, expansion=E, mode="topk", k=K,
            target_layers=[args.layer],
            epochs=args.epochs, batch_size=256, resample_every=2, device=DEVICE,
        )
        trainer.act_means[args.layer] = mean
        trainer.act_stds[args.layer]  = std

        t_layer = time.time()
        sae = trainer._train_single_sae(train_norm, args.layer)
        elapsed = time.time() - t_layer

        print(f"\n  Validation diagnostics (layer {args.layer}, E={E}, k={K}):")
        sae_diagnostics(sae, val_norm, device=DEVICE)

        torch.save({
            "sae_state_dict":   sae.state_dict(),
            "act_mean":         mean,
            "act_std":          std,
            "embed_dim":        encoder.embed_dim,
            "expansion":        E,
            "k":                K,
            "target_layer":     args.layer,
            "encoder":          args.encoder,
            "encoder_weights":  str(args.weights_path) if args.weights_path else None,
            "tag":              args.tag,
        }, out_p)
        out_p.chmod(0o444)
        print(f"  Saved (read-only) → {out_p.name}  (training: {elapsed/60:.1f} min)")

    print(f"\nAll done. Total wall time: {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
