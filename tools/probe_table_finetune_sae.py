"""Linear probe on top of finetuned EEG encoders, with optional SAE
substitution at the operating layer.

For each encoder (SleepFM, REVE, LaBraM) we:

  1. Load the finetuned encoder + the operating-layer SAE checkpoint
  2. Pre-compute mean-pooled token embeddings on train/val/test, twice:
        baseline           — encoder runs untouched
        sae_layer<L*>      — operating-layer output replaced by SAE
                              reconstruction (encode → decode in normalised
                              activation space, then unnormalised)
  3. Train a multi-seed linear head and report mean ± std test AUROC

This is the unified version of `probe_sae_reconstruction.py` and is the
source of `tab-finetune-sae` in the NeurIPS 2026 paper.

Usage::

    uv run tools/probe_table_finetune_sae.py
    uv run tools/probe_table_finetune_sae.py --encoders sleepfm
    uv run tools/probe_table_finetune_sae.py --seeds 5 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sae4eeg.dataset import StandardizeLabel, get_dataloaders
from sae4eeg.encoders import load_encoder
from sae4eeg.sae import SparseAutoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = ROOT / "results" / "probe_reconstruction"

# bf16 autocast for the encoder forward pass — ~3× speedup on L4/A100.
_AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

def _amp_ctx():
    if DEVICE == "cuda":
        return torch.autocast(device_type="cuda", dtype=_AMP_DTYPE)
    import contextlib
    return contextlib.nullcontext()

# ──────────────────────────────────────────────────────────────────────────────
# Per-encoder configuration
# ──────────────────────────────────────────────────────────────────────────────
ENCODERS: dict[str, dict] = {
    "sleepfm": {
        "load_name":  "sleepfm",
        "weights":    "checkpoints/finetuned/sleepfm_v1_local/finetuned.ckpt",
        "data":       "data/D4-v3-preprocessed-v2",
        "sae_ckpt":   "results/features/sleepfm_finetuned_local/sae_sleepfm_exp1.0_k8_layer2.pt",
        "target_layer": 2,
        "splits":     "checkpoints/finetuned/sleepfm/splits.json",
        "batch_size": 32,
    },
    "reve": {
        "load_name":  "reve",
        "weights":    "checkpoints/finetuned/reve_v1_local/finetuned.ckpt",
        "data":       "data/D4-v3-preprocessed-v1",
        "sae_ckpt":   "results/features/reve_local/sae_reve_exp1.0_k8_layer8.pt",
        "target_layer": 8,
        "splits":     "checkpoints/finetuned/reve/splits.json",
        "batch_size": 8,
    },
    "labram": {
        "load_name":  "labram",
        "weights":    "checkpoints/finetuned/labram_binary/finetuned.ckpt",
        "data":       "data/D4-v3-preprocessed-v1",
        "sae_ckpt":   "results/features/labram/sae_labram_exp1.0_k8_layer11.pt",
        "target_layer": 11,
        "splits":     "checkpoints/finetuned/labram_binary/splits.json",
        "batch_size": 8,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# SAE injection hook
# ──────────────────────────────────────────────────────────────────────────────
class _SAEInjector:
    """Replace a layer's output with its SAE reconstruction."""

    def __init__(self, sae, act_mean, act_std, device):
        self.sae = sae.to(device).eval()
        self.act_mean = act_mean.to(device)
        self.act_std = act_std.to(device)
        self.device = device
        self._handle = None

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            primary, *rest = output
        else:
            primary, rest = output, None
        orig_dtype = primary.dtype
        primary_fp32 = primary.to(self.device).float()
        z_norm = (primary_fp32 - self.act_mean) / self.act_std
        with torch.no_grad():
            z_hat = self.sae.decode(self.sae.encode(z_norm))
        recon = (z_hat * self.act_std + self.act_mean).to(orig_dtype)
        if rest is None:
            return recon
        return (recon, *rest)

    def register(self, layer):
        self._handle = layer.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def load_sae(path: Path) -> tuple[SparseAutoencoder, torch.Tensor, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(
        input_dim=int(ckpt["embed_dim"]),
        expansion=ckpt["expansion"],
        k=int(ckpt["k"]),
        mode="topk",
    )
    sae.load_state_dict(ckpt["sae_state_dict"])
    sae.eval()
    return sae, ckpt["act_mean"], ckpt["act_std"]


# ──────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(backend, loader, injector, max_samples=None):
    """Mean-pool the encoder's token output across the sequence dimension."""
    if injector is not None:
        target_layer = backend.get_hookable_layers()[injector_target]
        injector.register(target_layer)
    try:
        embs, labels = [], []
        n_seen = 0
        for batch in loader:
            x, y, _ = batch
            x = x.to(DEVICE, non_blocking=True)
            with _amp_ctx():
                tokens = backend.encode(x)        # (B, S, E)  — moves to CPU
            embs.append(tokens.mean(dim=1).float())
            labels.append((y > 0).long())
            n_seen += x.shape[0]
            if max_samples and n_seen >= max_samples:
                break
    finally:
        if injector is not None:
            injector.remove()
    return torch.cat(embs), torch.cat(labels)


# ──────────────────────────────────────────────────────────────────────────────
# Linear probe + evaluation
# ──────────────────────────────────────────────────────────────────────────────
def train_probe(train_emb, train_y, val_emb, val_y, *, epochs, lr, batch_size, seed):
    from sklearn.metrics import balanced_accuracy_score
    torch.manual_seed(seed)
    head = nn.Linear(train_emb.shape[1], 2).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(train_emb, train_y)
    ldr = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_state, best_ba = None, -1.0
    for _ in range(epochs):
        head.train()
        for xb, yb in ldr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(head(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            preds = head(val_emb.to(DEVICE)).argmax(-1).cpu().numpy()
        ba = balanced_accuracy_score(val_y.numpy(), preds)
        if ba > best_ba:
            best_ba = ba
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    return head


@torch.no_grad()
def evaluate(head, emb, y, *, n_bootstrap: int = 1000, ci: float = 0.95,
             rng_seed: int = 0):
    """Eval + non-parametric bootstrap 95% CI on AUROC over the test set."""
    from sklearn.metrics import (
        balanced_accuracy_score, roc_auc_score, average_precision_score,
    )
    head.eval()
    logits = head(emb.to(DEVICE))
    probs = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    preds = logits.argmax(-1).cpu().numpy()
    y_np = y.numpy()

    auc_point = float(roc_auc_score(y_np, probs))

    # Bootstrap test-set AUROC. Cheap (~ms for n_bootstrap=1000 on 1k samples).
    rng = np.random.default_rng(rng_seed)
    n = len(y_np)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        # Skip degenerate resamples that contain only one class.
        if len(np.unique(y_np[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_np[idx], probs[idx]))
    boots = np.asarray(boots)
    alpha = (1.0 - ci) / 2.0
    lo, hi = (float(np.quantile(boots, alpha)),
              float(np.quantile(boots, 1.0 - alpha))) if len(boots) else (auc_point, auc_point)

    return {
        "auc":   auc_point,
        "auprc": float(average_precision_score(y_np, probs)),
        "ba":    float(balanced_accuracy_score(y_np, preds)),
        "acc":   float((preds == y_np).mean()),
        "auc_ci_lo": lo,
        "auc_ci_hi": hi,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-encoder pipeline
# ──────────────────────────────────────────────────────────────────────────────
# Module-level so the hook function in extract_embeddings can read it
injector_target: int = 0


def run_encoder(name: str, cfg: dict, args) -> dict:
    global injector_target

    print(f"\n{'='*72}\nEncoder: {name}\n{'='*72}")
    weights = ROOT / cfg["weights"]
    sae_ckpt = ROOT / cfg["sae_ckpt"]
    data_path = ROOT / cfg["data"]
    splits_path = ROOT / cfg["splits"]
    splits_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. encoder
    backend = load_encoder(cfg["load_name"], weights_path=weights)
    backend.to(DEVICE).eval()

    # 2. SAE
    sae, act_mean, act_std = load_sae(sae_ckpt)
    injector = _SAEInjector(sae, act_mean, act_std, DEVICE)
    injector_target = cfg["target_layer"]

    # 3. data loaders (single fold, subject-level split via GroupShuffleSplit)
    transform = StandardizeLabel()
    folds = list(get_dataloaders(
        train_path=str(data_path),
        transformer=transform,
        batch_size=cfg["batch_size"],
        num_workers=2,
        seed=42,
        n_splits=1,
        split_info_path=str(splits_path),
    ))
    _, train_loader, val_loader, test_loader = folds[0]
    print(f"  train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")

    out: dict = {}
    for cond_name, inj in [("baseline", None), (f"sae_layer{cfg['target_layer']}", injector)]:
        print(f"\n  → {cond_name}")
        t0 = time.time()
        train_emb, train_y = extract_embeddings(backend, train_loader, inj, args.max_samples)
        val_emb,   val_y   = extract_embeddings(backend, val_loader,   inj, args.max_samples)
        test_emb,  test_y  = extract_embeddings(backend, test_loader,  inj, args.max_samples)
        print(f"    embeddings extracted ({time.time()-t0:.0f}s)  "
              f"train={tuple(train_emb.shape)} test={tuple(test_emb.shape)}  "
              f"pos_train={int(train_y.sum())}/{len(train_y)} "
              f"pos_test={int(test_y.sum())}/{len(test_y)}")

        seed_metrics = []
        for s in range(args.seeds):
            head = train_probe(
                train_emb, train_y, val_emb, val_y,
                epochs=args.epochs, lr=args.lr,
                batch_size=args.probe_batch_size, seed=s,
            )
            seed_metrics.append(evaluate(head, test_emb, test_y))

        agg = {}
        for key in ("auc", "auprc", "ba", "acc"):
            vals = [m[key] for m in seed_metrics]
            agg[f"{key}_mean"] = round(statistics.mean(vals), 4)
            agg[f"{key}_std"]  = round(statistics.stdev(vals) if len(vals)>1 else 0.0, 4)
        print(f"    AUROC  = {agg['auc_mean']:.4f} ± {agg['auc_std']:.4f}")
        print(f"    AUPRC  = {agg['auprc_mean']:.4f} ± {agg['auprc_std']:.4f}")
        print(f"    BA     = {agg['ba_mean']:.4f} ± {agg['ba_std']:.4f}")
        out[cond_name] = agg

    out["delta_auc"] = round(out["baseline"]["auc_mean"]
                             - out[f"sae_layer{cfg['target_layer']}"]["auc_mean"], 4)
    print(f"\n  Δ AUROC  (baseline − sae) = {out['delta_auc']:+.4f}")

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=list(ENCODERS.keys()),
                   choices=list(ENCODERS.keys()))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Encoders: {args.encoders}")

    summary: dict = {}
    for name in args.encoders:
        cfg = ENCODERS[name]
        summary[name] = run_encoder(name, cfg, args)
        # checkpoint after each encoder so partial results are not lost
        out_path = OUT_DIR / "table_finetune_sae.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote partial summary to {out_path}")

    print("\n" + "="*72)
    print("FINAL SUMMARY (test AUROC, mean ± std over seeds)")
    print("="*72)
    print(f"{'Encoder':<10}  {'baseline AUROC':>18}  {'SAE-replaced AUROC':>22}  {'Δ':>8}")
    for name, res in summary.items():
        sae_key = next(k for k in res if k.startswith("sae_layer"))
        b = res["baseline"]; s = res[sae_key]
        print(f"{name:<10}  "
              f"{b['auc_mean']:.4f} ± {b['auc_std']:.4f}    "
              f"{s['auc_mean']:.4f} ± {s['auc_std']:.4f}    "
              f"{res['delta_auc']:+.4f}")


if __name__ == "__main__":
    main()
