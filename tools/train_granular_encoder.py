"""Fine-tune SleepFM on V4 granular 10-class labels.

Architecture: SleepFM v1.1 (SetTransformer, 3 layers, embed_dim=128)
              + linear classification head (128 → 10 classes).

Training protocol (transfer learning):
  Phase 1 (--freeze-epochs, default 5): encoder frozen, head only.
  Phase 2 (remaining epochs):           full fine-tuning end-to-end.

Loss: CrossEntropyLoss with inverse-frequency class weights computed from
      the training split, so the 83.7% normal class does not dominate.

Data: data/D4-v4-preprocessed-10s  (256 Hz → V4ResampleTransform → 128 Hz)

Checkpoint saved to: checkpoints/granular/sleepfm_granular.ckpt

Usage
-----
  uv run tools/train_granular_encoder.py
  uv run tools/train_granular_encoder.py --epochs 40 --freeze-epochs 8
  uv run tools/train_granular_encoder.py --init-weights checkpoints/pretrained/sleepfm_weights.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

V4_DATA  = ROOT / "data" / "D4-v4-preprocessed-10s"
CKPT_OUT = ROOT / "checkpoints" / "granular" / "sleepfm_granular.ckpt"
INIT_WTS = ROOT / "checkpoints" / "finetuned" / "sleepfm1.ckpt"

N_CLASSES   = 10
EMBED_DIM   = 128
PATCH_SIZE  = 128
N_LAYERS    = 3

# SleepFM v1.1 constructor kwargs
_SLEEPFM_KWARGS = dict(
    in_channels=1,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=8,
    num_layers=N_LAYERS,
    pooling_head=8,
    dropout=0.3,
)


# ─────────────────────────────────────────────────────────────────────────────
# Lightning module
# ─────────────────────────────────────────────────────────────────────────────

class SleepFMGranular(nn.Module):
    """SleepFM encoder + 10-class head (stand-alone nn.Module for training)."""

    def __init__(self, class_weights: torch.Tensor | None = None, freeze_epochs: int = 5):
        super().__init__()
        from sae4eeg.sleepfm import SetTransformer
        self.encoder = SetTransformer(**_SLEEPFM_KWARGS)
        self.head = nn.Linear(EMBED_DIM, N_CLASSES)
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(N_CLASSES),
        )
        self.freeze_epochs = freeze_epochs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.encoder(x)   # (B, embed_dim)
        return self.head(pooled)       # (B, N_CLASSES)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets.long(), weight=self.class_weights)


# ─────────────────────────────────────────────────────────────────────────────
# Weight loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_encoder_weights(model: SleepFMGranular, weights_path: Path) -> None:
    """Load encoder-only weights from a .pt or .ckpt file into model.encoder."""
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("state_dict", ckpt)
    # Strip DataParallel 'module.' prefix
    raw_sd = {(k[len("module."):] if k.startswith("module.") else k): v
              for k, v in raw_sd.items()}
    # Strip PyTorch Lightning 'encoder.' prefix if present
    if any(k.startswith("encoder.") for k in raw_sd):
        raw_sd = {(k[len("encoder."):] if k.startswith("encoder.") else k): v
                  for k, v in raw_sd.items()}
    # Discard head.* keys (incompatible shape from binary head)
    raw_sd = {k: v for k, v in raw_sd.items() if not k.startswith("head.")}
    missing, unexpected = model.encoder.load_state_dict(raw_sd, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys: {missing[:3]}…")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys ignored: {unexpected[:3]}…")
    print(f"  [ok] Encoder initialised from {weights_path.name}")


def _save_lightning_ckpt(model: SleepFMGranular, path: Path, epoch: int, val_loss: float) -> None:
    """Save in PyTorch Lightning .ckpt format (state_dict keys prefixed encoder.*|head.*)."""
    state_dict = {}
    for k, v in model.encoder.state_dict().items():
        state_dict[f"encoder.{k}"] = v
    for k, v in model.head.state_dict().items():
        state_dict[f"head.{k}"] = v
    torch.save({"state_dict": state_dict, "epoch": epoch, "val_loss": val_loss}, path)
    print(f"  [✓] Saved → {path.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Class-weight computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights: w_i = N / (n_classes * count_i)."""
    n = len(labels)
    weights = torch.zeros(N_CLASSES)
    for i in range(N_CLASSES):
        count = int((labels == i).sum())
        weights[i] = n / (N_CLASSES * max(count, 1))
    # Clamp extreme weights to prevent instability from very rare classes
    weights = weights.clamp(max=50.0)
    print("  Class weights: " + "  ".join(f"{i}:{w:.2f}" for i, w in enumerate(weights)))
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _set_encoder_grad(model: SleepFMGranular, requires_grad: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad_(requires_grad)


@torch.no_grad()
def evaluate(model: SleepFMGranular, loader) -> tuple[float, float]:
    """Returns (loss, accuracy) on the given loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch in loader:
        x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y.long()).item() * x.shape[0]
        correct += (logits.argmax(1) == y.long()).sum().item()
        n += x.shape[0]
    return total_loss / n, correct / n


def train(args: argparse.Namespace) -> None:
    from sae4eeg.dataset import get_dataloaders, V4ResampleTransform

    print(f"Device: {DEVICE}")
    print(f"Data:   {V4_DATA}")
    print(f"Output: {CKPT_OUT}")

    # ── Data ─────────────────────────────────────────────────────────────────
    gen = get_dataloaders(
        train_path=str(V4_DATA),
        transformer=V4ResampleTransform(),
        batch_size=args.batch_size,
        num_workers=4,
        seed=42,
        split_info_path=str(ROOT / "results" / "probe_reconstruction" / "splits.json"),
    )
    fold, train_loader, val_loader, _ = next(gen)
    print(f"  fold={fold}  train batches={len(train_loader)}  val batches={len(val_loader)}")

    # Compute class weights from training labels
    train_labels = train_loader.dataset.labels.numpy()
    class_weights = compute_class_weights(train_labels).to(DEVICE)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SleepFMGranular(class_weights=class_weights, freeze_epochs=args.freeze_epochs)
    model.to(DEVICE)

    init_path = Path(args.init_weights)
    if init_path.exists():
        _load_encoder_weights(model, init_path)
    else:
        print(f"  [warn] init weights not found at {init_path} — using random init")

    # ── Optimisers ────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_path = CKPT_OUT.with_suffix(".best.ckpt")
    CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # Phase switch
        if epoch < args.freeze_epochs:
            _set_encoder_grad(model, False)
            phase = "head-only"
        elif epoch == args.freeze_epochs:
            _set_encoder_grad(model, True)
            phase = "full (unfreeze)"
            # Reset optimizer so unfreezed params get a fresh moment buffer
            opt = torch.optim.AdamW(
                model.parameters(), lr=args.lr / 10, weight_decay=0.05
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs - epoch, eta_min=1e-6
            )
        else:
            phase = "full"

        # Train
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = model.loss(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * x.shape[0]
            n += x.shape[0]
        train_loss = total_loss / n

        scheduler.step()

        # Validate
        val_loss, val_acc = evaluate(model, val_loader)

        lr_now = opt.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:3d}/{args.epochs}  [{phase:12s}]  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}  lr={lr_now:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_lightning_ckpt(model, best_path, epoch + 1, val_loss)
            print(f"    ↑ new best  (val_loss={val_loss:.4f})")

    # Copy best → final output path
    import shutil
    shutil.copy(best_path, CKPT_OUT)
    print(f"\nFinal checkpoint → {CKPT_OUT.relative_to(ROOT)}")
    print(f"Best val_loss: {best_val_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune SleepFM with granular 10-class labels")
    p.add_argument("--epochs",        type=int,   default=35,
                   help="Total training epochs (default: 35)")
    p.add_argument("--freeze-epochs", type=int,   default=5,
                   help="Epochs to freeze encoder and train head only (default: 5)")
    p.add_argument("--batch-size",    type=int,   default=64,
                   help="Mini-batch size (default: 64)")
    p.add_argument("--lr",            type=float, default=1e-3,
                   help="Initial learning rate for head-only phase (default: 1e-3)")
    p.add_argument("--init-weights",  default=str(INIT_WTS),
                   help="Path to encoder init weights (.pt or .ckpt); "
                        f"default: {INIT_WTS.relative_to(ROOT)}")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
