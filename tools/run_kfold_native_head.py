"""K-fold native-head finetune + SAE-faithfulness check for Table 2.

For each (encoder, fold) we

  1. Build the encoder + scalar logit head from pretrained weights.
  2. Finetune end-to-end with aggressive early stopping on the fold's
     val split.
  3. Retrain the operating-layer SAE on the new encoder's activations.
  4. Compute test AUROC two ways with the **native head**:
        intact:        encoder → head
        SAE-replaced:  encoder (with SAE injection at ℓ*) → head
  5. Save per-fold metrics; skip the (encoder, fold) cell if metrics.json
     already exists (resumable).

After all folds finish, aggregate (mean ± std) per encoder.

Outputs
-------
  results/kfold/kfold_splits.json
  results/kfold/{encoder}/fold_{k}/finetuned.ckpt
                                 /sae.pt
                                 /metrics.json
  results/kfold/summary.json

Usage
-----
  uv run tools/run_kfold_native_head.py                    # all encoders, k=5
  uv run tools/run_kfold_native_head.py --encoders sleepfm --k 5
  uv run tools/run_kfold_native_head.py --only-encoder reve --only-fold 2  # single cell
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    balanced_accuracy_score, f1_score,
)
from sklearn.model_selection import GroupKFold

from mecheeg.dataset import H5PYDatasetLabeled, StandardizeLabel
from mecheeg.encoders import load_encoder
from mecheeg.sae import SparseAutoencoder, SAETrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KFOLD_DIR = ROOT / "results" / "kfold"

# bf16 autocast for forward+backward and eval (L4 supports bf16 natively).
# All cells inside the same run share the same precision so Δ_AUROC
# (intact vs SAE-replaced) compares like-for-like.
_AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

def _amp_ctx():
    if DEVICE == "cuda":
        return torch.autocast(device_type="cuda", dtype=_AMP_DTYPE)
    import contextlib
    return contextlib.nullcontext()

# ──────────────────────────────────────────────────────────────────────────────
# Per-encoder configuration
# ──────────────────────────────────────────────────────────────────────────────
ENC_CFG: dict[str, dict] = {
    "sleepfm": {
        "load_name":     "sleepfm",
        "pretrained":    "checkpoints/pretrained/sleepfm_weights.pt",
        "data":          "data/D4-v3-preprocessed-v2",
        "embed_dim":     128,
        "target_layer":  2,
        "n_layers":      3,
        "batch_size":    32,
        # finetune protocol (aggressive early-stop)
        "max_epochs":    6,
        "freeze_epochs": 1,
        "head_lr":       1e-3,
        "full_lr":       1e-4,
        "patience":      1,
    },
    "reve": {
        "load_name":     "reve",
        "pretrained":    None,  # REVEBackend pulls from HF if weights_path=None
        "data":          "data/D4-v3-preprocessed-v1",
        "embed_dim":     512,
        "target_layer":  8,
        "n_layers":      22,
        "batch_size":    8,
        "max_epochs":    5,
        "freeze_epochs": 1,
        "head_lr":       1e-3,
        "full_lr":       1e-4,
        "patience":      1,
    },
    "labram": {
        "load_name":     "labram",
        "pretrained":    "checkpoints/pretrained/labram/labram-base.pth",
        "data":          "data/D4-v3-preprocessed-v1",
        "embed_dim":     200,
        "target_layer":  11,
        "n_layers":      12,
        "batch_size":    8,
        "max_epochs":    6,
        "freeze_epochs": 2,
        "head_lr":       1e-3,
        "full_lr":       1e-4,
        "patience":      1,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# K-fold splits
# ──────────────────────────────────────────────────────────────────────────────
def make_kfold_splits(data_path: Path, k: int, seed: int = 42) -> dict:
    """Return {fold_1..fold_k: {train_subjects, val_subjects, test_subjects}}.

    GroupKFold guarantees disjoint test subjects across folds. Within each
    fold's train pool we carve off ~10% for val (stratified by subject only).
    """
    base = H5PYDatasetLabeled(str(data_path), transform=StandardizeLabel())
    subjects = base._subjects_all.numpy()
    unique_subjects = np.unique(subjects)

    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, k)            # k disjoint chunks of subjects

    splits: dict = {}
    for i, test_subjects in enumerate(folds):
        train_pool = np.setdiff1d(unique_subjects, test_subjects, assume_unique=True)
        # carve val from train_pool
        rng2 = np.random.default_rng(seed + 1000 + i)
        train_pool_shuf = train_pool.copy()
        rng2.shuffle(train_pool_shuf)
        n_val = max(1, int(0.10 * len(train_pool_shuf)))
        val_subjects   = train_pool_shuf[:n_val]
        train_subjects = train_pool_shuf[n_val:]
        splits[f"fold_{i+1}"] = {
            "train_subjects": sorted(int(s) for s in train_subjects),
            "val_subjects":   sorted(int(s) for s in val_subjects),
            "test_subjects":  sorted(int(s) for s in test_subjects),
        }
    return splits


def get_fold_loaders(data_path: Path, fold_meta: dict, batch_size: int,
                     num_workers: int = 4):
    """Build train/val/test DataLoaders for one fold by subject ID."""
    transform = StandardizeLabel()

    def _ds_for(subject_ids: list[int]):
        ds = H5PYDatasetLabeled(str(data_path), transform=transform)
        sub_set = set(int(s) for s in subject_ids)
        # sample-level mask
        mask = np.array([int(s) in sub_set for s in ds._subjects_all.numpy()])
        idx = torch.from_numpy(np.where(mask)[0]).long()
        ds.update_index_map(idx)
        return ds

    train_ds = _ds_for(fold_meta["train_subjects"])
    val_ds   = _ds_for(fold_meta["val_subjects"])
    test_ds  = _ds_for(fold_meta["test_subjects"])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model builder per encoder
# ──────────────────────────────────────────────────────────────────────────────
def build_finetune_model(encoder_name: str, cfg: dict) -> nn.Module:
    """Encoder + scalar logit head wrapped in a single nn.Module."""
    if encoder_name == "sleepfm":
        backend = load_encoder("sleepfm",
                               weights_path=ROOT / cfg["pretrained"])
        encoder_module = backend.model

        class SleepFMBinary(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder_module
                self.head = nn.Linear(cfg["embed_dim"], 1)

            def forward(self, x):
                pooled, _ = self.encoder(x)
                return self.head(pooled).squeeze(-1)

        return SleepFMBinary()

    if encoder_name == "labram":
        backend = load_encoder("labram",
                               weights_path=ROOT / cfg["pretrained"])
        encoder_module = backend.model

        class LaBraMBinary(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder_module
                self.head = nn.Linear(cfg["embed_dim"], 1)
                self.n_channels = 19

            def forward(self, x):
                if x.shape[1] != self.n_channels:
                    x = x[:, :self.n_channels, :]
                tokens = self.encoder.forward_features(x, return_all_tokens=True)
                pooled = tokens[:, 1:, :].mean(dim=1)
                return self.head(pooled).squeeze(-1)

        return LaBraMBinary()

    if encoder_name == "reve":
        backend = load_encoder("reve", weights_path=None)
        encoder_module = backend.model
        positions_buf = backend._positions_1ch.clone()

        class REVEBinary(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder_module
                self.register_buffer("positions_1ch", positions_buf)
                self.head = nn.Linear(cfg["embed_dim"], 1)
                self.n_channels = 19

            def forward(self, x):
                if x.shape[1] != self.n_channels:
                    x = x[:, :self.n_channels, :]
                B = x.shape[0]
                pos = self.positions_1ch.unsqueeze(0).expand(B, -1, -1)
                output = self.encoder(x.float(), pos)
                if hasattr(output, "last_hidden_state"):
                    tokens = output.last_hidden_state
                elif isinstance(output, torch.Tensor):
                    tokens = output
                else:
                    tokens = output[0]
                if tokens.dim() == 4:
                    B2, C, S, E = tokens.shape
                    tokens = tokens.reshape(B2, C * S, E)
                pooled = tokens.mean(dim=1)
                return self.head(pooled).squeeze(-1)

        return REVEBinary()

    raise ValueError(encoder_name)


def get_encoder_blocks(encoder_name: str, model: nn.Module) -> list[nn.Module]:
    """Return the hookable transformer blocks (matches EncoderBackend.get_hookable_layers)."""
    if encoder_name == "sleepfm":
        return list(model.encoder.transformer_encoder.layers)
    if encoder_name == "labram":
        return list(model.encoder.blocks)
    if encoder_name == "reve":
        blocks = list(model.encoder.transformer.layers)
        return [list(b.children())[-1] for b in blocks]  # FeedForward of each block
    raise ValueError(encoder_name)


# ──────────────────────────────────────────────────────────────────────────────
# Training, eval, SAE injection
# ──────────────────────────────────────────────────────────────────────────────
def _set_encoder_grad(model, on: bool):
    for p in model.encoder.parameters():
        p.requires_grad_(on)


@torch.no_grad()
def collect_predictions(model, loader, sae_injector=None, target_block=None):
    model.eval()
    if sae_injector is not None:
        sae_injector.register(target_block)
    try:
        logits_all, labels_all = [], []
        for batch in loader:
            x, y, _ = batch
            x = x.to(DEVICE, non_blocking=True)
            with _amp_ctx():
                logits = model(x)
            logits_all.append(logits.float().cpu().numpy())
            labels_all.append((y > 0).long().numpy())
    finally:
        if sae_injector is not None:
            sae_injector.remove()
    return np.concatenate(logits_all), np.concatenate(labels_all)


def all_metrics(p, y, threshold: Optional[float] = None) -> dict:
    """Full metric pack at a fixed threshold (default = Youden-J on these p,y)."""
    if threshold is None:
        fpr, tpr, thr = roc_curve(y, p)
        threshold = float(thr[int(np.argmax(tpr - fpr))])
    yhat = (p >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "auroc":               float(roc_auc_score(y, p)),
        "auprc":               float(average_precision_score(y, p)),
        "youden_threshold":    threshold,
        "sensitivity":         sens,
        "specificity":         spec,
        "balanced_accuracy":   0.5 * (sens + spec),
        "accuracy":            (tp + tn) / max(len(y), 1),
        "f1":                  float(f1_score(y, yhat, zero_division=0)),
        "n":                   int(len(y)),
        "n_pos":               int(y.sum()),
    }


def evaluate_loader(model, loader, sae_injector=None, target_block=None,
                    threshold=None) -> dict:
    logits, y = collect_predictions(model, loader, sae_injector, target_block)
    p = 1 / (1 + np.exp(-logits))
    return all_metrics(p, y, threshold=threshold)


def finetune_one_fold(model, cfg, train_loader, val_loader, pos_weight, log_path):
    """Train with two-phase schedule + early stopping on val_auc."""
    log = {"epochs": []}
    best_val_auc = -1.0
    best_state = None
    best_threshold = 0.5
    patience = cfg["patience"]
    epochs_since_best = 0

    _set_encoder_grad(model, False)
    opt = torch.optim.AdamW(model.head.parameters(),
                            lr=cfg["head_lr"], weight_decay=0.01)

    for epoch in range(cfg["max_epochs"]):
        if epoch == cfg["freeze_epochs"]:
            _set_encoder_grad(model, True)
            opt = torch.optim.AdamW(model.parameters(),
                                    lr=cfg["full_lr"], weight_decay=0.05)
            phase = "full"
        elif epoch < cfg["freeze_epochs"]:
            phase = "head"
        else:
            phase = "full"

        model.train()
        running_loss, n_seen = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            x, y, _ = batch
            x = x.to(DEVICE, non_blocking=True)
            y_bin = (y > 0).float().to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with _amp_ctx():
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y_bin, pos_weight=pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += loss.item() * x.shape[0]
            n_seen += x.shape[0]
        train_loss = running_loss / max(n_seen, 1)

        val_metrics = evaluate_loader(model, val_loader)
        val_auc = val_metrics["auroc"]
        et = time.time() - t0
        print(f"    E{epoch+1:2d}/{cfg['max_epochs']}  [{phase:4s}]  "
              f"train_loss={train_loss:.4f}  val_auc={val_auc:.4f}  ({et:.0f}s)",
              flush=True)
        log["epochs"].append({
            "epoch": epoch + 1, "phase": phase,
            "train_loss": train_loss, "val_auc": val_auc, "time_s": et,
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            best_threshold = val_metrics["youden_threshold"]
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best > patience:
                print(f"    early stop (patience={patience})", flush=True)
                break

    model.load_state_dict(best_state)
    log["best_val_auc"] = best_val_auc
    log["best_threshold"] = best_threshold
    log_path.write_text(json.dumps(log, indent=2))
    return best_threshold


# ──────────────────────────────────────────────────────────────────────────────
# SAE retrain at the operating layer
# ──────────────────────────────────────────────────────────────────────────────
def retrain_sae_at_layer(encoder_name: str, model: nn.Module, train_loader,
                         cfg: dict, sae_path: Path,
                         max_tokens: int = 100_000):
    """Collect activations at ℓ* from the new finetuned encoder and train an
    expansion=1, k=8 TopK SAE."""
    print(f"    retraining SAE at layer {cfg['target_layer']} → {sae_path.name}",
          flush=True)
    blocks = get_encoder_blocks(encoder_name, model)
    target_block = blocks[cfg["target_layer"]]

    # Collect activations via forward hook over the train loader
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
    A = torch.cat(acts, dim=0)[:max_tokens].float()        # (N, E) — promote to fp32 for SAE training

    # Normalise (per-feature)
    act_mean = A.mean(dim=0)
    act_std  = A.std(dim=0).clamp(min=1e-6)
    A_norm = (A - act_mean) / act_std

    sae = SparseAutoencoder(input_dim=cfg["embed_dim"], expansion=1.0, k=8, mode="topk")
    sae.to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    A_norm = A_norm.to(DEVICE)
    bs = 1024
    perm = torch.randperm(A_norm.shape[0])
    n_epochs = 20
    for ep in range(n_epochs):
        ep_loss = 0.0
        idx_perm = perm[torch.randperm(perm.shape[0])]
        for s in range(0, len(idx_perm), bs):
            xb = A_norm[idx_perm[s:s + bs]]
            opt.zero_grad()
            recon, _ = sae(xb)
            loss = F.mse_loss(recon, xb)
            loss.backward(); opt.step()
            ep_loss += loss.item() * xb.shape[0]
        ep_loss /= len(idx_perm)
        if (ep + 1) % 5 == 0:
            print(f"      sae epoch {ep+1:2d}/{n_epochs}  loss={ep_loss:.4f}",
                  flush=True)

    torch.save({
        "sae_state_dict": sae.state_dict(),
        "act_mean": act_mean,
        "act_std":  act_std,
        "embed_dim": cfg["embed_dim"],
        "expansion": 1.0,
        "k": 8,
        "target_layer": cfg["target_layer"],
        "encoder": encoder_name,
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
# Per-cell pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_one_cell(encoder_name: str, fold_idx: int, fold_meta: dict, args):
    cfg = ENC_CFG[encoder_name]
    out_dir = KFOLD_DIR / encoder_name / f"fold_{fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"[skip] {encoder_name} fold {fold_idx} — {metrics_path.relative_to(ROOT)} exists")
        return json.loads(metrics_path.read_text())

    print(f"\n[{encoder_name} fold {fold_idx}] starting", flush=True)
    print(f"  train={len(fold_meta['train_subjects'])} subj  "
          f"val={len(fold_meta['val_subjects'])} subj  "
          f"test={len(fold_meta['test_subjects'])} subj")

    train_loader, val_loader, test_loader = get_fold_loaders(
        ROOT / cfg["data"], fold_meta,
        batch_size=cfg["batch_size"], num_workers=args.num_workers,
    )
    print(f"  train_windows={len(train_loader.dataset)}  "
          f"val_windows={len(val_loader.dataset)}  "
          f"test_windows={len(test_loader.dataset)}")

    # 1. build + finetune
    model = build_finetune_model(encoder_name, cfg).to(DEVICE)
    pos_weight = _compute_pos_weight(train_loader).to(DEVICE)
    print(f"  pos_weight = {pos_weight.item():.3f}")
    threshold = finetune_one_fold(model, cfg, train_loader, val_loader,
                                  pos_weight, out_dir / "training_log.json")

    # 2. save finetune ckpt
    ckpt_path = out_dir / "finetuned.ckpt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "encoder": encoder_name,
            "embed_dim": cfg["embed_dim"],
            "fold": fold_idx,
        },
    }, ckpt_path)

    # 3. retrain SAE at ℓ*
    sae_path = out_dir / "sae.pt"
    sae, act_mean, act_std = retrain_sae_at_layer(
        encoder_name, model, train_loader, cfg, sae_path,
    )

    # 4. eval native head (intact + SAE-injected) on val + test
    print("  evaluating native head (intact)…", flush=True)
    val_intact  = evaluate_loader(model, val_loader)
    test_intact = evaluate_loader(model, test_loader, threshold=threshold)

    print("  evaluating native head (SAE-replaced)…", flush=True)
    blocks = get_encoder_blocks(encoder_name, model)
    target_block = blocks[cfg["target_layer"]]
    injector = _SAEInjector(sae, act_mean, act_std)
    val_sae   = evaluate_loader(model, val_loader,  injector, target_block)
    test_sae  = evaluate_loader(model, test_loader, injector, target_block,
                                 threshold=threshold)

    summary = {
        "encoder": encoder_name,
        "fold":    fold_idx,
        "n_subjects": {
            "train": len(fold_meta["train_subjects"]),
            "val":   len(fold_meta["val_subjects"]),
            "test":  len(fold_meta["test_subjects"]),
        },
        "n_windows": {
            "train": len(train_loader.dataset),
            "val":   len(val_loader.dataset),
            "test":  len(test_loader.dataset),
        },
        "youden_threshold": threshold,
        "val":  {"intact": val_intact, "sae_replaced": val_sae},
        "test": {"intact": test_intact, "sae_replaced": test_sae},
        "delta_test_auroc": round(test_intact["auroc"] - test_sae["auroc"], 4),
    }
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"  test AUROC: intact={test_intact['auroc']:.4f}  "
          f"sae={test_sae['auroc']:.4f}  Δ={summary['delta_test_auroc']:+.4f}",
          flush=True)
    return summary


def _compute_pos_weight(loader) -> torch.Tensor:
    """Read labels directly from the underlying H5PYDatasetLabeled — avoids
    a full epoch of multi-worker EEG decoding just to count classes."""
    labels = loader.dataset.labels.numpy()
    n_pos = int((labels > 0).sum())
    n_neg = int((labels == 0).sum())
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────
def aggregate(per_fold: list[dict]) -> dict:
    """Compute mean ± std across folds for each metric."""
    if not per_fold:
        return {}
    keys = ["auroc", "auprc", "sensitivity", "specificity",
            "balanced_accuracy", "accuracy", "f1"]
    out = {"intact": {}, "sae_replaced": {}, "delta_auroc": {}}
    for cond in ("intact", "sae_replaced"):
        for k in keys:
            vals = [f["test"][cond][k] for f in per_fold]
            out[cond][k] = {
                "mean": round(statistics.mean(vals), 4),
                "std":  round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
            }
    deltas = [f["delta_test_auroc"] for f in per_fold]
    out["delta_auroc"] = {
        "mean": round(statistics.mean(deltas), 4),
        "std":  round(statistics.stdev(deltas) if len(deltas) > 1 else 0.0, 4),
    }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=list(ENC_CFG.keys()))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--only-encoder", default=None,
                   help="Restrict to a single encoder (skips the others)")
    p.add_argument("--only-fold", type=int, default=None,
                   help="Restrict to a single fold index (1-based)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if metrics.json already exists")
    return p.parse_args()


def main():
    args = parse_args()
    KFOLD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    # Each encoder pair has a different dataset (sample rate). All encoders share
    # the SAME subject pool but different file layouts; subject IDs match across
    # both data dirs, so one set of subject-level splits suffices. We anchor the
    # split-generation to the SleepFM data (D4-v3-preprocessed-v2) and reuse the
    # subject IDs for the other encoders' loaders.
    splits_path = KFOLD_DIR / f"kfold_splits_k{args.k}_seed{args.seed}.json"
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
        print(f"Loaded splits from {splits_path.relative_to(ROOT)}")
    else:
        print(f"Generating {args.k}-fold subject splits …")
        splits = make_kfold_splits(ROOT / ENC_CFG["sleepfm"]["data"],
                                   k=args.k, seed=args.seed)
        splits_path.write_text(json.dumps(splits, indent=2))
        print(f"Wrote splits → {splits_path.relative_to(ROOT)}")

    encoders_to_run = [args.only_encoder] if args.only_encoder else args.encoders
    folds_to_run = [args.only_fold] if args.only_fold else list(range(1, args.k + 1))

    per_encoder_results: dict[str, list[dict]] = {e: [] for e in encoders_to_run}
    for enc_name in encoders_to_run:
        for fold_idx in folds_to_run:
            fold_meta = splits[f"fold_{fold_idx}"]
            res = run_one_cell(enc_name, fold_idx, fold_meta, args)
            per_encoder_results[enc_name].append(res)
            # checkpoint summary after every cell
            summary = {e: aggregate(per_encoder_results[e])
                       for e in per_encoder_results}
            (KFOLD_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
            print(f"  → wrote partial summary to "
                  f"{(KFOLD_DIR/'summary.json').relative_to(ROOT)}")

    print("\n" + "=" * 72)
    print("FINAL SUMMARY (test, mean ± std over folds)")
    print("=" * 72)
    for enc, agg in summary.items():
        print(f"\n{enc}:")
        for cond in ("intact", "sae_replaced"):
            m = agg[cond]
            print(f"  {cond:12s}  AUROC={m['auroc']['mean']:.4f}±{m['auroc']['std']:.4f}  "
                  f"Sens={m['sensitivity']['mean']:.3f}  Spec={m['specificity']['mean']:.3f}  "
                  f"BA={m['balanced_accuracy']['mean']:.3f}  F1={m['f1']['mean']:.3f}")
        print(f"  Δ AUROC = {agg['delta_auroc']['mean']:+.4f} ± {agg['delta_auroc']['std']:.4f}")


if __name__ == "__main__":
    main()
