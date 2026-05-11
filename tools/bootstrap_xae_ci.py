"""Bootstrap CIs for XAE amplitude R² across encoders.

Uses the same eval pipeline as compare_xae.py, but caches per-token
squared residuals + squared deviations and bootstraps R² over tokens
(N_BOOT iterations) to give 95% CIs on overall + per-band amplitude R².

The point estimates here should match xae_comparison_metrics.json (modulo
sampling noise from collect_pairs's max_tokens cap).

Usage:
  uv run tools/bootstrap_xae_ci.py            # primary 3 encoders
  uv run tools/bootstrap_xae_ci.py --include-all  # every entry in MODELS

Outputs:
  results/xae/xae_bootstrap_ci.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sae4eeg.xae import CLINICAL_BANDS  # noqa: E402
from sae4eeg.dataset import get_dataloaders, StandardizeLabel, V4ResampleTransform  # noqa: E402
from sae4eeg.encoders import load_encoder  # noqa: E402

# Reuse model registry + XAE loader from compare_xae.py.
from compare_xae import MODELS as _BASE_MODELS, _load_xae  # noqa: E402

# Add the paper's primary SleepFM checkpoint (binary-finetuned), which isn't
# in compare_xae's registry. Reuse the "sleepfm" backend factory.
from sae4eeg.encoders import MODEL_CARDS  # noqa: E402

MODELS = dict(_BASE_MODELS)
MODELS["sleepfm_finetuned"] = dict(
    display_name="SleepFM finetuned (paper primary)",
    xae_ckpt=ROOT / "results" / "xae" / "sleepfm_finetuned" / "xae_checkpoint.pt",
    data_path=ROOT / "data" / "D4-v3-preprocessed-v2",
    target_layer=2,
    patch_size=128,
    pool_channels=False,
    weights_path=ROOT / "checkpoints" / "finetuned" / "sleepfm1.ckpt",
    encoder_key="sleepfm_finetuned",  # used to dispatch to load_encoder
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 20_000
N_BOOT = 1000
SEED = 17

PRIMARY = ("sleepfm_finetuned", "labram", "reve")


def _collect_predictions(trainer, encoder, val_loader, target_layer,
                         pool_channels, max_tokens):
    """Run the encoder + XAE forward pass; return ground-truth and predicted
    log-amplitude tensors of shape (N, n_bins) on CPU."""
    embeddings, targets, _, _ = trainer.collect_pairs(
        encoder, val_loader, target_layer,
        max_tokens=max_tokens, pool_channels=pool_channels,
    )
    n_bins = trainer.spectral.n_bins
    embeddings = embeddings.to(DEVICE)
    emb_norm = (embeddings - trainer.embed_mean) / trainer.embed_std
    with torch.no_grad():
        pred_norm = trainer.xae.decode(emb_norm)
    pred = (pred_norm * trainer.target_std + trainer.target_mean).cpu()
    targets = targets.cpu()
    return targets[:, :n_bins].numpy(), pred[:, :n_bins].numpy()


def _bootstrap_r2(gt: np.ndarray, pr: np.ndarray, mask: np.ndarray | None,
                  n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Bootstrap R² over tokens (axis 0), pooling all bins inside `mask`.
    Returns (point estimate, 2.5% quantile, 97.5% quantile)."""
    if mask is not None:
        gt_b = gt[:, mask]
        pr_b = pr[:, mask]
    else:
        gt_b = gt
        pr_b = pr
    # Per-token squared residual + squared deviation, summed over bins.
    res_per_tok = ((gt_b - pr_b) ** 2).sum(axis=1)
    mean_b = gt_b.mean(axis=0, keepdims=True)
    dev_per_tok = ((gt_b - mean_b) ** 2).sum(axis=1)

    n = len(res_per_tok)
    point = 1.0 - res_per_tok.sum() / (dev_per_tok.sum() + 1e-12)

    idx = rng.integers(0, n, size=(n_boot, n))
    res_b = res_per_tok[idx].sum(axis=1)
    dev_b = dev_per_tok[idx].sum(axis=1)
    r2_b = 1.0 - res_b / (dev_b + 1e-12)
    lo, hi = np.quantile(r2_b, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--include-all", action="store_true",
                   help="Run every entry in MODELS (default: 3 primary encoders).")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    args = p.parse_args()

    keys = list(MODELS.keys()) if args.include_all else list(PRIMARY)
    rng = np.random.default_rng(SEED)
    out: dict[str, dict] = {}

    for model_key in keys:
        cfg = MODELS[model_key]
        if not cfg["xae_ckpt"].exists():
            print(f"[skip — no checkpoint] {model_key}")
            continue

        print(f"\n=== {cfg['display_name']} ({model_key}) ===", flush=True)
        wp = cfg["weights_path"]
        # Some entries override the encoder factory key (e.g. sleepfm_finetuned).
        backend_key = cfg.get("encoder_key", model_key)
        encoder = load_encoder(backend_key, weights_path=str(wp) if wp is not None else None)
        encoder.to(DEVICE).eval()
        trainer = _load_xae(cfg)

        transform = V4ResampleTransform() if "D4-v4" in str(cfg["data_path"]) else StandardizeLabel()
        gen = get_dataloaders(
            train_path=str(cfg["data_path"]),
            transformer=transform,
        )
        _, _, val_loader, _ = next(gen)

        print(f"  Collecting val predictions (max {args.max_tokens:,} tokens)…", flush=True)
        gt, pr = _collect_predictions(
            trainer, encoder, val_loader,
            cfg["target_layer"], cfg["pool_channels"], args.max_tokens,
        )
        n_tokens, n_bins = gt.shape
        print(f"  Bootstrapping R² (n_tokens={n_tokens}, n_boot={args.n_boot})…",
              flush=True)

        # Overall R² over all bins.
        pt, lo, hi = _bootstrap_r2(gt, pr, mask=None, n_boot=args.n_boot, rng=rng)
        rec = {
            "display_name": cfg["display_name"],
            "n_tokens": int(n_tokens),
            "n_bins": int(n_bins),
            "amp_r2": {"point": pt, "ci_low": lo, "ci_high": hi},
            "band_amp_r2": {},
        }
        print(f"  → Amplitude R² (all):  {pt:.4f}  [{lo:.4f}, {hi:.4f}]")

        # Per-band R².
        spectral = trainer.spectral
        for band in CLINICAL_BANDS:
            mask = np.asarray(spectral.get_band_mask(band))
            if mask.sum() == 0:
                continue
            pt_b, lo_b, hi_b = _bootstrap_r2(gt, pr, mask=mask,
                                             n_boot=args.n_boot, rng=rng)
            rec["band_amp_r2"][band] = {"point": pt_b, "ci_low": lo_b, "ci_high": hi_b}
            print(f"      {band:<10}  {pt_b:.4f}  [{lo_b:.4f}, {hi_b:.4f}]")

        out[model_key] = rec

        del encoder, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = ROOT / "results" / "xae" / "xae_bootstrap_ci.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
