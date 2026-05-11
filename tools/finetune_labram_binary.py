"""Fine-tune LaBraM-Base on the D4-v3-preprocessed-v1 dataset (200 Hz, 60 s windows)
with collapsed binary normal/abnormal labels.

Loads the LaBraM-Base pretrained encoder from
``checkpoints/pretrained/labram/labram-base.pth`` (braindecode export of the
official upstream weights), interpolates the temporal embedding from 16 to 60
positions, slices the first 19 channels of the 27-channel D4 montage to match
the standard 10-20 layout, and trains a single linear head on the
mean-pooled patch tokens.

Outputs
-------
  checkpoints/finetuned/labram_binary/finetuned.ckpt   — best val-AUROC ckpt
  checkpoints/finetuned/labram_binary/training_log.json
  checkpoints/finetuned/labram_binary/metrics.json

Usage
-----
  uv run tools/finetune_labram_binary.py
  uv run tools/finetune_labram_binary.py --epochs 12 --freeze-epochs 3 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mecheeg.dataset import StandardizeLabel, get_dataloaders
from mecheeg.labram import labram_base_patch200_200, remap_braindecode_state_dict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

V3_DATA = ROOT / "data" / "D4-v3-preprocessed-v1"
PRETRAIN_PATH = ROOT / "checkpoints" / "pretrained" / "labram" / "labram-base.pth"
OUT_DIR = ROOT / "checkpoints" / "finetuned" / "labram_binary"
SPLITS = OUT_DIR / "splits.json"

N_CHANNELS = 19         # standard 10-20
N_TIME_PATCHES = 60     # 60 × 200-sample patches = 60 s at 200 Hz
EMBED_DIM = 200


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class LaBraMBinary(nn.Module):
    """LaBraM-Base encoder + scalar logit head for binary classification."""

    def __init__(self, encoder: nn.Module, embed_dim: int = EMBED_DIM,
                 n_channels: int = N_CHANNELS):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, 1)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — slice first n_channels (D4 27-ch → standard 10-20 19-ch)
        if x.shape[1] != self.n_channels:
            x = x[:, :self.n_channels, :]
        # Encoder returns (B, 1+C*A, E) all-tokens; pool across patch tokens (skip CLS)
        tokens = self.encoder.forward_features(x, return_all_tokens=True)
        pooled = tokens[:, 1:, :].mean(dim=1)        # (B, E)
        return self.head(pooled).squeeze(-1)         # (B,)


def build_model() -> LaBraMBinary:
    encoder = labram_base_patch200_200(
        max_time_patches=N_TIME_PATCHES, init_values=0.1,
    )
    raw_sd = torch.load(PRETRAIN_PATH, map_location="cpu", weights_only=False)
    sd = remap_braindecode_state_dict(raw_sd, target_time_patches=N_TIME_PATCHES)
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys: {missing[:3]}…")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys: {unexpected[:3]}…")
    print(f"  [ok] LaBraM-Base loaded ({sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params)")
    return LaBraMBinary(encoder=encoder, embed_dim=EMBED_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(batch_size: int, num_workers: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen = get_dataloaders(
        train_path=str(V3_DATA),
        transformer=StandardizeLabel(),
        batch_size=batch_size,
        num_workers=num_workers,
        seed=42,
        split_info_path=str(SPLITS),
    )
    fold, train_loader, val_loader, test_loader = next(gen)
    return fold, train_loader, val_loader, test_loader


def binary_pos_weight(loader) -> torch.Tensor:
    n_pos = n_neg = 0
    for batch in loader:
        _, y, _ = batch
        n_pos += int((y > 0).sum())
        n_neg += int((y == 0).sum())
    pw = n_neg / max(n_pos, 1)
    print(f"  Train neg/pos = {n_neg}/{n_pos}  → pos_weight = {pw:.3f}")
    return torch.tensor([pw], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────────────────────────────────────

def _set_encoder_grad(model: LaBraMBinary, requires_grad: bool):
    for p in model.encoder.parameters():
        p.requires_grad_(requires_grad)


@torch.no_grad()
def collect_predictions(model: LaBraMBinary, loader):
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        x, y, _ = batch
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append((y > 0).long().numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def evaluate(model, loader, pos_weight) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    logits, y = collect_predictions(model, loader)
    loss = F.binary_cross_entropy_with_logits(
        torch.from_numpy(logits), torch.from_numpy(y).float(),
        pos_weight=pos_weight.cpu(),
    ).item()
    try:
        auc = float(roc_auc_score(y, logits))
    except ValueError:
        auc = float("nan")
    return loss, auc


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "sensitivity": sens, "specificity": spec,
        "balanced_accuracy": 0.5 * (sens + spec),
    }


def final_metrics(model, loader, pos_weight):
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    logits, y = collect_predictions(model, loader)
    p = 1 / (1 + np.exp(-logits))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    auprc = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    j_idx = int(np.argmax(j))
    youden_thr = float(thr[j_idx])
    y_pred = (p >= youden_thr).astype(int)
    return {
        "auroc": auc, "auprc": auprc,
        "youden_threshold": youden_thr,
        "youden_metrics": _binary_metrics(y, y_pred),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
    }


def train(model: LaBraMBinary, args, train_loader, val_loader, pos_weight):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / "finetuned.ckpt"
    log_path = OUT_DIR / "training_log.json"

    log: dict = {"epochs": []}
    best_val_auc = -1.0

    _set_encoder_grad(model, False)
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.head_lr, weight_decay=0.01)

    t_total = time.time()
    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            _set_encoder_grad(model, True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.full_lr, weight_decay=0.05)
            phase = "full"
        elif epoch < args.freeze_epochs:
            phase = "head"
        else:
            phase = "full"

        model.train()
        running_loss = 0.0
        n_seen = 0
        t0 = time.time()
        for batch in train_loader:
            x, y, _ = batch
            x = x.to(DEVICE, non_blocking=True)
            y_bin = (y > 0).float().to(DEVICE, non_blocking=True)
            opt.zero_grad()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y_bin, pos_weight=pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += loss.item() * x.shape[0]
            n_seen += x.shape[0]
        train_loss = running_loss / max(n_seen, 1)

        val_loss, val_auc = evaluate(model, val_loader, pos_weight)
        et = time.time() - t0
        print(f"  E{epoch+1:2d}/{args.epochs}  [{phase:4s}]  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={val_auc:.4f}  ({et:.0f}s)")
        log["epochs"].append({
            "epoch": epoch + 1, "phase": phase,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_auc": val_auc, "time_s": et,
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss, "val_auc": val_auc,
                "config": {
                    "encoder": "labram_base_patch200_200",
                    "embed_dim": EMBED_DIM,
                    "n_channels": N_CHANNELS,
                    "max_time_patches": N_TIME_PATCHES,
                    "init_values": 0.1,
                    "patch_size": 200,
                    "sample_rate_hz": 200,
                    "input_window_seconds": 60,
                },
            }, ckpt_path)
            print(f"     ↑ new best val_auc → {ckpt_path.name}")

    log["best_val_auc"] = best_val_auc
    log["wall_time_s"] = time.time() - t_total
    log_path.write_text(json.dumps(log, indent=2))
    return ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--freeze-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--full-lr", type=float, default=1e-4)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")

    fold, train_loader, val_loader, test_loader = get_loaders(args.batch_size, args.num_workers)
    print(f"Fold {fold}: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    pw = binary_pos_weight(train_loader)
    pw = pw.to(DEVICE)

    model = build_model().to(DEVICE)
    ckpt_path = train(model, args, train_loader, val_loader, pw)

    # Reload best ckpt and evaluate on val + test
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.to(DEVICE)
    val_metrics = final_metrics(model, val_loader, pw)
    test_metrics = final_metrics(model, test_loader, pw)
    metrics = {"val": val_metrics, "test": test_metrics}
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
