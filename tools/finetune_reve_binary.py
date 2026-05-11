"""Fine-tune REVE-Base on D4-v3-preprocessed-v1 (200 Hz, 60-s windows) with the
binary normal/abnormal label, using the project's canonical splits.

Reproduces the role of `checkpoints/finetuned/reve_qjbe08.ckpt` but with
verifiable subject-level splits (`checkpoints/finetuned/reve/splits.json`),
so the resulting checkpoint is leakage-free with respect to the SAE / probe
evaluation pipeline.

Outputs
-------
  checkpoints/finetuned/reve_v1_local/finetuned.ckpt
  checkpoints/finetuned/reve_v1_local/training_log.json
  checkpoints/finetuned/reve_v1_local/metrics.json

Usage
-----
  uv run tools/finetune_reve_binary.py
  uv run tools/finetune_reve_binary.py --epochs 8 --freeze-epochs 2 --batch-size 8
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

from sae4eeg.dataset import StandardizeLabel, get_dataloaders
from sae4eeg.encoders import load_encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

V3_DATA = ROOT / "data" / "D4-v3-preprocessed-v1"
SPLITS  = ROOT / "checkpoints" / "finetuned" / "reve" / "splits.json"
OUT_DIR = ROOT / "checkpoints" / "finetuned" / "reve_v1_local"

EMBED_DIM   = 512
N_CHANNELS  = 19   # standard 10-20


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class REVEBinary(nn.Module):
    """REVE-Base backbone + scalar logit head over mean-pooled tokens."""

    def __init__(self, backend, embed_dim: int = EMBED_DIM):
        super().__init__()
        # Register the underlying nn.Module so .to(device) and .parameters() work.
        # REVEBackend itself is an ABC, not an nn.Module.
        self.encoder = backend.model
        self.register_buffer("positions_1ch", backend._positions_1ch.clone())
        self.head = nn.Linear(embed_dim, 1)
        self.n_channels = N_CHANNELS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) at 200 Hz
        if x.shape[1] != self.n_channels:
            x = x[:, :self.n_channels, :]

        B = x.shape[0]
        positions = self.positions_1ch.unsqueeze(0).expand(B, -1, -1)  # (B, C, 3)

        output = self.encoder(x.float(), positions)
        if hasattr(output, "last_hidden_state"):
            tokens = output.last_hidden_state
        elif isinstance(output, torch.Tensor):
            tokens = output
        else:
            tokens = output[0]
        # REVE outputs (B, C, S, E) → (B, C*S, E)
        if tokens.dim() == 4:
            B2, C, S, E = tokens.shape
            tokens = tokens.reshape(B2, C * S, E)
        pooled = tokens.mean(dim=1)                                 # (B, E)
        return self.head(pooled).squeeze(-1)                        # (B,)


def build_model() -> REVEBinary:
    # weights_path=None → load from HuggingFace brain-bzh/reve-base (pretrained)
    backend = load_encoder("reve", weights_path=None)
    model = REVEBinary(backend, embed_dim=EMBED_DIM)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [ok] REVE-Base + linear head: {n_params/1e6:.2f}M params")
    return model


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
def _set_encoder_grad(model: REVEBinary, requires_grad: bool):
    for p in model.encoder.parameters():
        p.requires_grad_(requires_grad)


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        x, y, _ = batch
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append((y > 0).long().numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def evaluate(model, loader, pos_weight):
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


def _binary_metrics(y_true, y_pred):
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
    j_idx = int(np.argmax(tpr - fpr))
    youden_thr = float(thr[j_idx])
    y_pred = (p >= youden_thr).astype(int)
    return {
        "auroc": auc, "auprc": auprc,
        "youden_threshold": youden_thr,
        "youden_metrics": _binary_metrics(y, y_pred),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
    }


def train(model, args, train_loader, val_loader, pos_weight):
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
        running_loss, n_seen = 0.0, 0
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
            # Saved keys already use the "encoder.<...>" prefix expected by
            # REVEBackend.load_state_dict (we register backend.model as
            # self.encoder).  Head weights are kept under "head.*".
            sd = model.state_dict()
            sd_remap = {k: v for k, v in sd.items() if k != "positions_1ch"}
            torch.save({
                "state_dict": sd_remap,
                "epoch": epoch + 1,
                "val_loss": val_loss, "val_auc": val_auc,
                "config": {
                    "encoder":      "reve",
                    "embed_dim":    EMBED_DIM,
                    "n_channels":   N_CHANNELS,
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
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--freeze-epochs", type=int, default=2)
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
    print(f"Fold {fold}: train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    pw = binary_pos_weight(train_loader).to(DEVICE)

    model = build_model().to(DEVICE)
    ckpt_path = train(model, args, train_loader, val_loader, pw)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fresh = build_model().to(DEVICE)
    fresh.load_state_dict(state["state_dict"], strict=False)  # positions_1ch is a buffer
    val_metrics  = final_metrics(fresh, val_loader, pw)
    test_metrics = final_metrics(fresh, test_loader, pw)
    metrics = {"val": val_metrics, "test": test_metrics}
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
