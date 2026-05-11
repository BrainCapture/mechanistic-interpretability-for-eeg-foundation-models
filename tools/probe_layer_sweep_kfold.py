"""K-fold per-layer SAE-faithfulness sweep.

For each (encoder, fold, layer) we

  1. Load the fold's already-finetuned encoder
       results/kfold/{encoder}/fold_{k}/finetuned.ckpt
  2. Train a TopK SAE on the fold's train subjects' layer-`L` activations
       (or load it from disk if already trained → resumable)
  3. Extract mean-pooled embeddings for the fold's test subjects:
        baseline:    no SAE
        sae_layer L: SAE injected at layer L (encode→decode)
  4. Train a 3-seed linear probe on the fold's train embeddings
  5. Save per-cell metrics + per-cell test predictions (for pooled bootstrap)

Aggregation:
  - mean ± std AUROC across the 5 folds (cross-validation variance)
  - bootstrap 95% CI computed on the POOLED test predictions across folds
    (≈ 5× more samples than the single-split sweep, ~2.2× tighter CIs)

Outputs
-------
  results/probe_reconstruction/layer_sweep_kfold.{json,png}
  results/probe_reconstruction_kfold/{encoder}/fold_{k}/
        sae_layer{L}.pt            — per-fold per-layer TopK SAE
        layer{L}_predictions.npz   — y_true + p (intact + sae_replaced)
        layer{L}_metrics.json      — auc, ci, etc.
        baseline_predictions.npz   — no-SAE
        baseline_metrics.json
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mecheeg.dataset import H5PYDatasetLabeled, StandardizeLabel
from mecheeg.encoders import load_encoder
from mecheeg.sae import SparseAutoencoder

# Reuse model builders + block accessors from the k-fold orchestrator
sys.path.insert(0, str(ROOT / "tools"))
from run_kfold_native_head import (   # noqa: E402
    ENC_CFG, build_finetune_model, get_encoder_blocks, _amp_ctx,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR    = ROOT / "results" / "probe_reconstruction"
CACHE_DIR  = ROOT / "results" / "probe_reconstruction_kfold"
KFOLD_DIR  = ROOT / "results" / "kfold"

# Layer subset per encoder
LAYERS = {
    "sleepfm": list(range(3)),
    "labram":  list(range(12)),
    "reve":    list(range(22)),
}


# ──────────────────────────────────────────────────────────────────────────────
# Data plumbing
# ──────────────────────────────────────────────────────────────────────────────
def get_fold_loaders(data_path: Path, fold_meta: dict, batch_size: int,
                     num_workers: int = 4):
    """Build train/test DataLoaders for one fold by subject ID. Train pool is
    the same as the fold's training set (val is unused here — we only need
    train embeddings for the probe + test embeddings for evaluation)."""
    transform = StandardizeLabel()

    def _ds_for(subject_ids: list[int]):
        ds = H5PYDatasetLabeled(str(data_path), transform=transform)
        sub_set = set(int(s) for s in subject_ids)
        mask = np.array([int(s) in sub_set for s in ds._subjects_all.numpy()])
        idx = torch.from_numpy(np.where(mask)[0]).long()
        ds.update_index_map(idx)
        return ds

    train_ds = _ds_for(fold_meta["train_subjects"])
    test_ds  = _ds_for(fold_meta["test_subjects"])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Encoder + per-fold finetuned weights
# ──────────────────────────────────────────────────────────────────────────────
def load_finetuned_encoder(encoder_name: str, fold: int) -> nn.Module:
    """Build the *Binary wrapper for `encoder_name`, then load the per-fold
    finetuned weights from results/kfold/{enc}/fold_{k}/finetuned.ckpt."""
    cfg = ENC_CFG[encoder_name]
    model = build_finetune_model(encoder_name, cfg).to(DEVICE).eval()
    ckpt_path = KFOLD_DIR / encoder_name / f"fold_{fold}" / "finetuned.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Per-layer SAE: train or load
# ──────────────────────────────────────────────────────────────────────────────
def get_or_train_sae(encoder_name: str, fold: int, layer: int,
                     model: nn.Module, train_loader: DataLoader,
                     cfg: dict, max_tokens: int = 100_000) -> tuple[SparseAutoencoder,
                                                                     torch.Tensor,
                                                                     torch.Tensor]:
    """Reuse fold's operating-layer SAE (already in results/kfold/) when it
    matches; otherwise train a fresh one and cache it."""
    fold_dir = CACHE_DIR / encoder_name / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    sae_path = fold_dir / f"sae_layer{layer}.pt"

    if not sae_path.exists() and layer == cfg["target_layer"]:
        # Reuse the operating-layer SAE that the kfold pipeline already trained
        op_sae = KFOLD_DIR / encoder_name / f"fold_{fold}" / "sae.pt"
        if op_sae.exists():
            ckpt = torch.load(op_sae, map_location="cpu", weights_only=False)
            torch.save(ckpt, sae_path)

    if sae_path.exists():
        ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
        sae = SparseAutoencoder(input_dim=cfg["embed_dim"],
                                expansion=ckpt.get("expansion", 1.0),
                                k=ckpt.get("k", 8), mode="topk").to(DEVICE)
        sae.load_state_dict(ckpt["sae_state_dict"])
        return sae, ckpt["act_mean"], ckpt["act_std"]

    # Train fresh SAE: collect activations on the train loader at layer `layer`
    blocks = get_encoder_blocks(encoder_name, model)
    target_block = blocks[layer]
    acts: list[torch.Tensor] = []

    def _hook(_m, _i, output):
        out = output[0] if isinstance(output, tuple) else output
        acts.append(out.detach().reshape(-1, out.shape[-1]).cpu())

    handle = target_block.register_forward_hook(_hook)
    n_seen = 0
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            x, _, _ = batch
            x = x.to(DEVICE, non_blocking=True)
            with _amp_ctx():
                _ = model(x)
            n_seen += acts[-1].shape[0]
            if n_seen >= max_tokens:
                break
    handle.remove()
    A = torch.cat(acts, dim=0)[:max_tokens].float()
    act_mean = A.mean(dim=0)
    act_std  = A.std(dim=0).clamp(min=1e-6)
    A_norm   = (A - act_mean) / act_std

    sae = SparseAutoencoder(input_dim=cfg["embed_dim"], expansion=1.0,
                            k=8, mode="topk").to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    A_norm = A_norm.to(DEVICE)
    bs = 1024
    perm = torch.randperm(A_norm.shape[0])
    for ep in range(20):
        idx_perm = perm[torch.randperm(perm.shape[0])]
        for s in range(0, len(idx_perm), bs):
            xb = A_norm[idx_perm[s:s + bs]]
            opt.zero_grad()
            recon, _ = sae(xb)
            loss = F.mse_loss(recon, xb)
            loss.backward(); opt.step()

    torch.save({
        "sae_state_dict": sae.state_dict(),
        "act_mean": act_mean, "act_std":  act_std,
        "embed_dim": cfg["embed_dim"], "expansion": 1.0, "k": 8,
        "target_layer": layer, "encoder": encoder_name,
    }, sae_path)
    return sae, act_mean, act_std


# ──────────────────────────────────────────────────────────────────────────────
# SAE injection hook
# ──────────────────────────────────────────────────────────────────────────────
class _SAEInjector:
    def __init__(self, sae, act_mean, act_std):
        self.sae = sae.to(DEVICE).eval()
        self.act_mean = act_mean.to(DEVICE)
        self.act_std  = act_std.to(DEVICE)
        self._handle = None

    def _hook_fn(self, _m, _i, output):
        if isinstance(output, tuple):
            primary, *rest = output
        else:
            primary, rest = output, None
        orig_dtype = primary.dtype
        primary_fp32 = primary.to(DEVICE).float()
        z_norm = (primary_fp32 - self.act_mean) / self.act_std
        with torch.no_grad():
            z_hat = self.sae.decode(self.sae.encode(z_norm))
        recon = (z_hat * self.act_std + self.act_mean).to(orig_dtype)
        return recon if rest is None else (recon, *rest)

    def register(self, target_block):
        self._handle = target_block.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ──────────────────────────────────────────────────────────────────────────────
# Embedding extraction (mean-pool) + linear probe
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader,
                       injector: Optional[_SAEInjector] = None,
                       target_block: Optional[nn.Module] = None):
    """Mean-pool the encoder's tokens. Reads the encoder out of the *Binary
    wrapper at model.encoder; we replicate the wrapper's forward minus head."""
    if injector is not None:
        injector.register(target_block)
    embs, ys = [], []
    try:
        for batch in loader:
            x, y, _ = batch
            x = x.to(DEVICE, non_blocking=True)
            with _amp_ctx():
                tokens = _encode_only(model, x)
            embs.append(tokens.mean(dim=1).float().cpu())
            ys.append((y > 0).long())
    finally:
        if injector is not None:
            injector.remove()
    return torch.cat(embs), torch.cat(ys)


def _encode_only(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run the *Binary wrapper's encoder up to its sequence output (B, S, E)."""
    name = type(model).__name__
    if name == "SleepFMBinary":
        pooled, _ = model.encoder(x)
        return pooled.unsqueeze(1)               # (B, 1, E) — already pooled
    if name == "LaBraMBinary":
        if x.shape[1] != model.n_channels:
            x = x[:, :model.n_channels, :]
        tokens = model.encoder.forward_features(x, return_all_tokens=True)
        return tokens[:, 1:, :]
    if name == "REVEBinary":
        if x.shape[1] != model.n_channels:
            x = x[:, :model.n_channels, :]
        B = x.shape[0]
        pos = model.positions_1ch.unsqueeze(0).expand(B, -1, -1)
        out = model.encoder(x.float(), pos)
        if hasattr(out, "last_hidden_state"):
            tokens = out.last_hidden_state
        elif isinstance(out, torch.Tensor):
            tokens = out
        else:
            tokens = out[0]
        if tokens.dim() == 4:
            B2, C, S, E = tokens.shape
            tokens = tokens.reshape(B2, C * S, E)
        return tokens
    raise ValueError(name)


def train_probe(train_emb, train_y, *, epochs=20, lr=1e-3, batch_size=256,
                seed=0):
    torch.manual_seed(seed)
    head = nn.Linear(train_emb.shape[1], 2).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    from torch.utils.data import TensorDataset
    ds = TensorDataset(train_emb, train_y)
    ldr = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        head.train()
        for xb, yb in ldr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(head(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return head


@torch.no_grad()
def probe_predict(head, emb) -> np.ndarray:
    head.eval()
    return torch.softmax(head(emb.to(DEVICE)), -1)[:, 1].cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap CI on pooled predictions
# ──────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(y, p, *, n_bootstrap=1000, ci=0.95, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    n = len(y)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    if not boots:
        return float(roc_auc_score(y, p)), float(roc_auc_score(y, p))
    boots = np.asarray(boots)
    a = (1 - ci) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


# ──────────────────────────────────────────────────────────────────────────────
# Per-cell pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_cell(encoder_name: str, fold: int, layer: Optional[int],
             fold_meta: dict, args) -> dict:
    """One (encoder, fold, layer) cell. layer=None → baseline."""
    cfg = ENC_CFG[encoder_name]
    fold_dir = CACHE_DIR / encoder_name / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    tag = "baseline" if layer is None else f"layer{layer}"
    metrics_path = fold_dir / f"{tag}_metrics.json"
    preds_path   = fold_dir / f"{tag}_predictions.npz"

    if metrics_path.exists() and preds_path.exists() and not args.force:
        return json.loads(metrics_path.read_text())

    print(f"[{encoder_name} fold {fold} {tag}] starting", flush=True)
    model = load_finetuned_encoder(encoder_name, fold)
    train_loader, test_loader = get_fold_loaders(
        ROOT / cfg["data"], fold_meta,
        batch_size=cfg["batch_size"], num_workers=args.num_workers,
    )

    injector = None
    target_block = None
    if layer is not None:
        sae, act_mean, act_std = get_or_train_sae(
            encoder_name, fold, layer, model, train_loader, cfg,
        )
        injector = _SAEInjector(sae, act_mean, act_std)
        target_block = get_encoder_blocks(encoder_name, model)[layer]

    t0 = time.time()
    train_emb, train_y = extract_embeddings(model, train_loader, injector, target_block)
    test_emb,  test_y  = extract_embeddings(model, test_loader,  injector, target_block)
    print(f"  embeddings extracted ({time.time()-t0:.0f}s)  "
          f"train={tuple(train_emb.shape)} test={tuple(test_emb.shape)}", flush=True)

    seed_aucs, seed_preds = [], []
    for s in range(args.seeds):
        head = train_probe(train_emb, train_y,
                           epochs=args.epochs, lr=args.lr,
                           batch_size=args.probe_batch_size, seed=s)
        p = probe_predict(head, test_emb)
        seed_preds.append(p)
        seed_aucs.append(roc_auc_score(test_y.numpy(), p))
    auc_mean = float(np.mean(seed_aucs))
    auc_std  = float(np.std(seed_aucs, ddof=1)) if args.seeds > 1 else 0.0

    # Save *averaged* predictions across seeds (used for pooled-bootstrap CI)
    p_avg = np.mean(np.stack(seed_preds, axis=0), axis=0)
    np.savez_compressed(preds_path, y=test_y.numpy().astype(np.int8), p=p_avg.astype(np.float32))

    ci_lo, ci_hi = bootstrap_ci(test_y.numpy(), p_avg, n_bootstrap=args.n_bootstrap)
    summary = {
        "encoder": encoder_name, "fold": fold, "layer": layer,
        "auc_mean": round(auc_mean, 4), "auc_std": round(auc_std, 4),
        "auc_ci_lo": round(ci_lo, 4), "auc_ci_hi": round(ci_hi, 4),
        "n_test": int(len(test_y)),
    }
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"  AUROC = {auc_mean:.4f} ± {auc_std:.4f}  "
          f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation across folds
# ──────────────────────────────────────────────────────────────────────────────
def aggregate(encoder_name: str, layer: Optional[int], k: int,
              n_bootstrap: int) -> dict:
    """Mean ± std across folds + bootstrap CI on POOLED predictions."""
    tag = "baseline" if layer is None else f"layer{layer}"
    aucs, ys, ps = [], [], []
    for fold in range(1, k + 1):
        fold_dir = CACHE_DIR / encoder_name / f"fold_{fold}"
        m = json.loads((fold_dir / f"{tag}_metrics.json").read_text())
        aucs.append(m["auc_mean"])
        npz = np.load(fold_dir / f"{tag}_predictions.npz")
        ys.append(npz["y"])
        ps.append(npz["p"])
    y_pool = np.concatenate(ys)
    p_pool = np.concatenate(ps)
    ci_lo, ci_hi = bootstrap_ci(y_pool, p_pool, n_bootstrap=n_bootstrap)
    auc_pool = float(roc_auc_score(y_pool, p_pool))
    return {
        "auc_per_fold": [round(a, 4) for a in aucs],
        "auc_mean":   round(float(np.mean(aucs)), 4),
        "auc_std":    round(float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0, 4),
        "auc_pooled": round(auc_pool, 4),
        "auc_ci_lo":  round(ci_lo, 4),
        "auc_ci_hi":  round(ci_hi, 4),
        "n_test_pooled": int(len(y_pool)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────
def plot(summary: dict, args):
    """Paper-style figure (NeurIPS): serif fonts, mathtext, vector PDF + 300 dpi PNG."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize":   13,
        "axes.titlesize":   13,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  12,
        "axes.linewidth":   0.7,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth":  1.4,
        "lines.markersize": 4.5,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.02,
    })

    # Wider aspect (intrinsic 7.5x2.6) so when scaled to NeurIPS \textwidth
    # the figure renders as a flatter, easier-to-scan landscape plot.
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    colors = {"sleepfm": "#1f77b4", "reve": "#ff7f0e", "labram": "#2ca02c"}
    pretty = {"sleepfm": "SleepFM", "reve": "REVE", "labram": "LaBraM"}

    import math
    from matplotlib.lines import Line2D

    n_folds = args.k
    sem_div = math.sqrt(n_folds)

    encoder_handles = []
    for name, res in summary.items():
        n_total = res["n_layers_total"]
        layers = sorted(int(L) for L in res["by_layer"])
        get   = lambda L, k: res["by_layer"][str(L) if str(L) in res["by_layer"] else L][k]
        aucs  = [get(L, "auc_mean") for L in layers]
        sems  = [get(L, "auc_std") / sem_div for L in layers]
        lo    = [m - s for m, s in zip(aucs, sems)]
        hi    = [m + s for m, s in zip(aucs, sems)]
        xs = [L / max(n_total - 1, 1) for L in layers]
        h, = ax.plot(xs, aucs, "o-", color=colors[name], markeredgewidth=0,
                     label=pretty[name])
        encoder_handles.append(h)
        ax.fill_between(xs, lo, hi, color=colors[name], alpha=0.22, linewidth=0)
        b = res["baseline"]
        # Visible dashed baseline (no faint band — keeps the figure clean).
        ax.axhline(b["auc_mean"], color=colors[name],
                   linestyle=(0, (5, 3)), linewidth=1.1, alpha=0.85)

    ax.set_xlabel(r"Relative layer depth ($0 = $ first layer, $1 = $ last layer)")
    ax.set_ylabel("AUROC")
    ax.set_title("SAE-faithfulness layer sweep")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)

    # One legend row: encoders (colour) + line styles (solid SAE vs dashed baseline).
    style_handles = [
        Line2D([0], [0], color="0.25", linewidth=1.4, marker="o",
               markersize=4, markeredgewidth=0, label="SAE-substituted"),
        Line2D([0], [0], color="0.25", linewidth=1.1,
               linestyle=(0, (5, 3)), label="Baseline"),
    ]
    ax.legend(handles=encoder_handles + style_handles, frameon=False,
              loc="lower center", ncol=5, handlelength=1.8, columnspacing=1.6,
              bbox_to_anchor=(0.5, -0.42), borderaxespad=0.0)

    png = OUT_DIR / "layer_sweep_kfold.png"
    pdf = OUT_DIR / "layer_sweep_kfold.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"\nWrote plot → {png.relative_to(ROOT)}\n            → {pdf.relative_to(ROOT)}")


def cfg_op(name): return {"sleepfm": 2, "reve": 8, "labram": 11}[name]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=list(LAYERS.keys()))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    splits_path = KFOLD_DIR / f"kfold_splits_k{args.k}_seed42.json"
    splits = json.loads(splits_path.read_text())
    print(f"Loaded splits from {splits_path.relative_to(ROOT)}")

    summary: dict = {}
    for enc in args.encoders:
        n_total = len(LAYERS[enc])  # — quick proxy; matches encoder block count
        per_layer: dict = {}
        # baseline (no SAE) per fold
        for fold in range(1, args.k + 1):
            run_cell(enc, fold, None, splits[f"fold_{fold}"], args)
        baseline_agg = aggregate(enc, None, args.k, args.n_bootstrap)
        # each layer per fold
        for layer in LAYERS[enc]:
            for fold in range(1, args.k + 1):
                run_cell(enc, fold, layer, splits[f"fold_{fold}"], args)
            per_layer[layer] = aggregate(enc, layer, args.k, args.n_bootstrap)
            per_layer[layer]["delta_auc"] = round(
                baseline_agg["auc_mean"] - per_layer[layer]["auc_mean"], 4)
            print(f"  → {enc} layer {layer:2d}: "
                  f"AUROC = {per_layer[layer]['auc_mean']:.4f} ± {per_layer[layer]['auc_std']:.4f}  "
                  f"95% CI [{per_layer[layer]['auc_ci_lo']:.4f}, {per_layer[layer]['auc_ci_hi']:.4f}]",
                  flush=True)
        summary[enc] = {
            "n_layers_total": n_total,
            "baseline":       baseline_agg,
            "by_layer":       per_layer,
        }
        # checkpoint after each encoder
        (OUT_DIR / "layer_sweep_kfold.json").write_text(json.dumps(summary, indent=2))
        print(f"  → wrote partial summary to {OUT_DIR/'layer_sweep_kfold.json'}")

    print("\n" + "=" * 72)
    print("FINAL SUMMARY (test AUROC, mean ± std across folds + pooled CI)")
    print("=" * 72)
    for enc, res in summary.items():
        b = res["baseline"]
        print(f"\n{enc}: baseline {b['auc_mean']:.4f} ± {b['auc_std']:.4f}  "
              f"95% CI [{b['auc_ci_lo']:.4f}, {b['auc_ci_hi']:.4f}]")
        for L, r in sorted(res["by_layer"].items(), key=lambda kv: int(kv[0])):
            print(f"  layer {int(L):2d}: {r['auc_mean']:.4f} ± {r['auc_std']:.4f}  "
                  f"95% CI [{r['auc_ci_lo']:.4f}, {r['auc_ci_hi']:.4f}]  Δ={r['delta_auc']:+.4f}")

    try:
        plot(summary, args)
    except Exception as e:
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
