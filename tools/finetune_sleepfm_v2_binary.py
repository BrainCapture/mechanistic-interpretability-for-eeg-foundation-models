"""Fine-tune all SleepFM v2 variants on D4-v4 with a collapsed binary label.

For each pretrained SleepFM v2 checkpoint under
``checkpoints/pretrained/SleepFM v2 Models/``:

  1. Build the matching SetTransformer + TokenizerV2 encoder from the saved
     ``config.json`` and load weights from ``best.pt``.
  2. Attach a single linear head (``embed_dim → 1``) on top of the pooled
     embedding.
  3. Fine-tune end-to-end on the V4 dataset with the 10-class labels collapsed
     to ``{0: normal, 1: abnormal}``.  Class imbalance handled by ``pos_weight``
     in BCEWithLogitsLoss.
  4. Evaluate on the held-out validation split: AUROC, AUPRC, plus
     accuracy / sens / spec / F1 / balanced acc at two operating points
     (Youden-J optimum, and the threshold that fixes sens=0.95).

Outputs
-------
  results/finetune_v2_binary/{variant}/
      finetuned.ckpt           — best validation-loss checkpoint
      training_log.json         — per-epoch train/val loss + val AUROC
      metrics.json              — final val metrics summary

  results/finetune_v2_binary/comparison_barplot.png
  results/finetune_v2_binary/all_metrics.json

Usage
-----
  uv run tools/finetune_sleepfm_v2_binary.py
  uv run tools/finetune_sleepfm_v2_binary.py --variants sleepfm_v2.1 sleepfm_v2.6
  uv run tools/finetune_sleepfm_v2_binary.py --epochs 8 --freeze-epochs 2 --batch-size 32
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

V4_DATA = ROOT / "data" / "D4-v4-preprocessed-10s"
V2_DIR  = ROOT / "checkpoints" / "pretrained" / "SleepFM v2 Models"
OUT_DIR = ROOT / "results" / "finetune_v2_binary"
SPLITS  = ROOT / "results" / "probe_reconstruction" / "splits.json"

# Map of model name -> directory under V2_DIR
V2_VARIANTS: dict[str, str] = {
    "sleepfm_v2.0": "settransformer_exp0_cl_cnn_sgd_fp32_128d_640p_lr0.001_20260307_113442",
    "sleepfm_v2.1": "settransformer_exp1_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_114250",
    "sleepfm_v2.3": "settransformer_exp2_cl_mae_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_113651",
    "sleepfm_v2.4": "settransformer_exp4_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_210846",
    "sleepfm_v2.5": "settransformer_exp5_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260308_111156",
    "sleepfm_v2.6": "settransformer_exp2.6_res_1sec_cl_cnn_adamw_bf16_128d_128p_lr0.0003_20260322_011957",
    "sleepfm_v2.7": "settransformer_exp2.7_res_1sec_cl_cnn_adamw_bf16_128d_128p_lr0.0003_20260321_161621",
}

# v1.1 — original SleepFM (3 transformer layers, BN-Tokenizer, patch=128)
V1_CHECKPOINT = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"
ALL_VARIANTS = ["sleepfm_v1.1"] + list(V2_VARIANTS)


# ─────────────────────────────────────────────────────────────────────────────
# Channel-type embedding (used by v2.4 / v2.5 / v2.6, per upstream
# sleepfm_dataset.py ChannelType IntEnum).  Index 29 is PAD.
# ─────────────────────────────────────────────────────────────────────────────

N_CHANNEL_TYPES = 30  # matches channel_emb shape (30, 128)

# ChannelType integer codes (from upstream)
_FRONTAL, _CENTRAL, _TEMPORAL, _PARIETAL, _OCCIPITAL = 0, 1, 2, 3, 4

# Our V4 dataset uses the standard 27-channel layout.  Each entry maps a
# channel name to its ChannelType index.  P7/P8 are EEG_TEMPORAL because in
# old 10-20 naming they are T5/T6 (and the upstream lookup table only contains
# old-style names for posterior temporal positions).  P9/P10/T9/T10/TP7/TP8
# are not in the official lookup; they are mapped by topographic proximity.
STANDARD_27_CHANNELS = [
    "Fp1", "Fp2",
    "F9", "F7", "F3", "Fz", "F4", "F8", "F10",
    "T9", "T7", "C3", "Cz", "C4", "T8", "T10",
    "TP7",                        "TP8",
    "P9", "P7", "P3", "Pz", "P4", "P8", "P10",
                          "O1", "O2",
]

CHANNEL_NAME_TO_TYPE: dict[str, int] = {
    "Fp1": _FRONTAL, "Fp2": _FRONTAL,
    "F9":  _FRONTAL, "F7":  _FRONTAL, "F3": _FRONTAL,
    "Fz":  _FRONTAL,
    "F4":  _FRONTAL, "F8":  _FRONTAL, "F10": _FRONTAL,
    "T9":  _TEMPORAL, "T7": _TEMPORAL,
    "C3":  _CENTRAL,  "Cz": _CENTRAL,  "C4": _CENTRAL,
    "T8":  _TEMPORAL, "T10": _TEMPORAL,
    "TP7": _TEMPORAL, "TP8": _TEMPORAL,
    "P9":  _TEMPORAL, "P10": _TEMPORAL,        # ear-level lateral parietal → temporal
    "P7":  _TEMPORAL, "P8":  _TEMPORAL,        # = T5 / T6 in old 10-20 → TEMPORAL
    "P3":  _PARIETAL, "Pz": _PARIETAL, "P4": _PARIETAL,
    "O1":  _OCCIPITAL, "O2": _OCCIPITAL,
}

CHANNEL_IDS_27: list[int] = [CHANNEL_NAME_TO_TYPE[c] for c in STANDARD_27_CHANNELS]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class SleepFMBinary(nn.Module):
    """SetTransformer encoder + scalar logit head for binary classification."""

    def __init__(self, encoder: nn.Module, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head    = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.encoder(x)             # (B, E)
        return self.head(pooled).squeeze(-1)    # (B,)


def _patched_set_transformer_forward(self, x, mask=None):
    """SetTransformer.forward with optional channel-type embedding injected
    after tokenization.  Activated when ``self.channel_emb`` and
    ``self.channel_ids`` buffers are present.
    """
    from einops import rearrange

    x = self.patch_embedding(x)        # (B, C, S, E)
    B, C, S, E = x.shape

    ch_emb_module = getattr(self, "channel_emb", None)
    ch_ids        = getattr(self, "channel_ids", None)
    if ch_emb_module is not None and ch_ids is not None:
        ch_emb = ch_emb_module(ch_ids[:C])         # (C, E)
        x = x + ch_emb[None, :, None, :]           # broadcast over batch + tokens

    x = rearrange(x, "b c s e -> (b s) c e")
    if mask is not None:
        mask = mask.unsqueeze(1).expand(-1, S, -1)
        mask = rearrange(mask, "b t c -> (b t) c")
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)
    x = self.spatial_pooling(x, mask)
    x = x.view((B, S, E))
    x = self.positional_encoding(x)
    x = self.layer_norm(x)
    x = self.transformer_encoder(x)
    embedding = x.clone()
    x = self.temporal_pooling(x)
    return x, embedding


def _build_v1_encoder() -> tuple[nn.Module, dict, Path]:
    """Build the original SleepFM v1.1 encoder (3 transformer layers, BN-Tokenizer)."""
    from sae4eeg.sleepfm import SetTransformer  # uses original BN-Tokenizer by default

    cfg = {
        "patch_size":     128,
        "embed_dim":      128,
        "num_heads":      8,
        "num_layers":     3,
        "pooling_head":   8,
        "dropout":        0.3,
        "max_seq_length": 128,
        "exp_name":       "v1.1",
        "sampling_freq":  128,
    }

    encoder = SetTransformer(
        in_channels=1,
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        pooling_head=cfg["pooling_head"],
        dropout=cfg["dropout"],
        max_seq_length=cfg["max_seq_length"],
    )

    raw = torch.load(V1_CHECKPOINT, map_location="cpu", weights_only=False)
    sd  = raw.get("state_dict", raw)
    sd  = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    if any(k.startswith("encoder.") for k in sd):
        sd = {(k[len("encoder."):] if k.startswith("encoder.") else k): v for k, v in sd.items()}

    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [info] {len(missing)} missing keys (e.g. {missing[:2]})")
    if unexpected:
        print(f"  [info] {len(unexpected)} unexpected keys ignored "
              f"(e.g. {unexpected[:2]})")

    return encoder, cfg, V1_CHECKPOINT


def build_v2_encoder(variant: str) -> tuple[nn.Module, dict, Path]:
    """Construct the encoder for the given variant.

    Dispatches:
      * ``sleepfm_v1.1`` → original SleepFM (3 layers, BN-Tokenizer)
      * ``sleepfm_v2.*``  → SetTransformer with TokenizerV2

    For variants whose pretrained checkpoint contains ``channel_emb.weight``
    (v2.4, v2.5, v2.6), the encoder is augmented with an ``nn.Embedding(30,
    embed_dim)`` keyed by ChannelType integer.  The 27 channel IDs for our V4
    dataset are stored as a non-persistent buffer.  The forward method is
    monkey-patched to add ``channel_emb[ch_ids]`` to each patch token before
    spatial pooling.
    """
    if variant == "sleepfm_v1.1":
        return _build_v1_encoder()

    from sae4eeg.sleepfm import SetTransformer, TokenizerV2

    ckpt_dir = V2_DIR / V2_VARIANTS[variant]
    cfg = json.loads((ckpt_dir / "config.json").read_text())

    patch_size     = int(cfg["patch_size"])
    embed_dim      = int(cfg["embed_dim"])
    num_heads      = int(cfg["num_heads"])
    num_layers     = int(cfg["num_layers"])
    pooling_head   = int(cfg["pooling_head"])
    dropout        = float(cfg["dropout"])
    max_seq_length = int(cfg.get("max_seq_length", 128))

    encoder = SetTransformer(
        in_channels=1,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        pooling_head=pooling_head,
        dropout=dropout,
        max_seq_length=max_seq_length,
    )
    # v2 models use the LayerNorm+ELU tokenizer
    encoder.patch_embedding = TokenizerV2(input_size=patch_size, output_size=embed_dim)

    # Load pretrained weights
    weights_path = ckpt_dir / "best.pt"
    raw = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd  = raw.get("state_dict", raw)
    sd  = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}

    # If the pretrained checkpoint shipped with a channel_emb table, attach the
    # matching nn.Embedding to the encoder so its weights actually load and are
    # applied at inference time.
    if "channel_emb.weight" in sd:
        emb_table = sd["channel_emb.weight"]
        n_types, e_dim = emb_table.shape
        encoder.channel_emb = nn.Embedding(n_types, e_dim)
        encoder.register_buffer(
            "channel_ids",
            torch.tensor(CHANNEL_IDS_27, dtype=torch.long),
            persistent=False,
        )
        # Monkey-patch forward to apply the embedding after tokenization
        encoder.forward = _patched_set_transformer_forward.__get__(encoder, type(encoder))
        print(f"  [info] channel_emb attached: shape {tuple(emb_table.shape)}, "
              f"applied per-channel using 27-channel ChannelType lookup")

    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [info] {len(missing)} missing keys (e.g. {missing[:2]})")
    if unexpected:
        print(f"  [info] {len(unexpected)} unexpected keys ignored "
              f"(e.g. {unexpected[:2]})")

    return encoder, cfg, weights_path


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(batch_size: int, num_workers: int = 4):
    from sae4eeg.dataset import get_dataloaders, V4ResampleTransform
    gen = get_dataloaders(
        train_path=str(V4_DATA),
        transformer=V4ResampleTransform(),
        batch_size=batch_size,
        num_workers=num_workers,
        seed=42,
        split_info_path=str(SPLITS),
    )
    fold, train_loader, val_loader, _ = next(gen)
    return fold, train_loader, val_loader


def binary_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """Return (n_neg / n_pos) so positive class loss is up-weighted."""
    bin_y = (labels > 0).long().numpy()
    n_pos = int((bin_y == 1).sum())
    n_neg = int((bin_y == 0).sum())
    pw = n_neg / max(n_pos, 1)
    print(f"  Train neg/pos = {n_neg}/{n_pos}  → pos_weight = {pw:.3f}")
    return torch.tensor([pw], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _set_encoder_grad(model: SleepFMBinary, requires_grad: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad_(requires_grad)


@torch.no_grad()
def collect_predictions(model: SleepFMBinary, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        x, y, _ = batch
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append((y > 0).long().numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def evaluate_loss_auc(model, loader, pos_weight):
    """Quick val pass: BCE loss + AUROC."""
    from sklearn.metrics import roc_auc_score
    logits, y = collect_predictions(model, loader)
    loss = F.binary_cross_entropy_with_logits(
        torch.from_numpy(logits),
        torch.from_numpy(y).float(),
        pos_weight=pos_weight.cpu(),
    ).item()
    try:
        auc = float(roc_auc_score(y, logits))
    except ValueError:
        auc = float("nan")
    return loss, auc


def train_one_variant(
    variant: str,
    args: argparse.Namespace,
    train_loader,
    val_loader,
    pos_weight: torch.Tensor,
):
    print(f"\n{'='*70}\n[{variant}]  Building encoder…\n{'='*70}")
    encoder, cfg, _ = build_v2_encoder(variant)

    model = SleepFMBinary(encoder=encoder, embed_dim=int(cfg["embed_dim"]))
    model.to(DEVICE)
    pos_weight = pos_weight.to(DEVICE)

    out_dir = OUT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "finetuned.ckpt"
    log_path  = out_dir / "training_log.json"

    log = {"variant": variant, "config": {k: cfg[k] for k in
            ["patch_size", "embed_dim", "num_heads", "num_layers", "pooling_head"]},
           "epochs": []}

    # Phase 1: head only
    _set_encoder_grad(model, False)
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.head_lr, weight_decay=0.01)

    best_val_loss = float("inf")
    best_val_auc  = -1.0

    t_total = time.time()
    for epoch in range(args.epochs):
        # Phase switch
        if epoch == args.freeze_epochs:
            _set_encoder_grad(model, True)
            opt = torch.optim.AdamW(
                model.parameters(), lr=args.full_lr, weight_decay=0.05,
            )
            phase = "full"
        elif epoch < args.freeze_epochs:
            phase = "head"
        else:
            phase = "full"

        # Train
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
            loss = F.binary_cross_entropy_with_logits(
                logits, y_bin, pos_weight=pos_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_loss += loss.item() * x.shape[0]
            n_seen += x.shape[0]
        train_loss = running_loss / max(n_seen, 1)

        # Validate
        val_loss, val_auc = evaluate_loss_auc(model, val_loader, pos_weight)
        epoch_time = time.time() - t0

        print(f"  E{epoch+1:2d}/{args.epochs}  [{phase:4s}]  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={val_auc:.4f}  ({epoch_time:.0f}s)")

        log["epochs"].append({
            "epoch": epoch + 1, "phase": phase,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_auc": val_auc, "time_s": epoch_time,
        })

        # Save best by val AUROC (more meaningful than loss for imbalanced binary)
        if val_auc > best_val_auc:
            best_val_auc  = val_auc
            best_val_loss = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_auc":  val_auc,
                "variant":  variant,
                "config":   cfg,
            }, ckpt_path)
            print(f"     ↑ new best val_auc → {ckpt_path.name}")

    log["best_val_auc"]  = best_val_auc
    log["best_val_loss"] = best_val_loss
    log["wall_time_s"]   = time.time() - t_total
    log_path.write_text(json.dumps(log, indent=2))
    return ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1   = 2 * prec * sens / max(prec + sens, 1e-12)
    bacc = 0.5 * (sens + spec)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc, "sensitivity": sens, "specificity": spec,
        "precision": prec, "f1": f1, "balanced_accuracy": bacc,
    }


def evaluate_all_metrics(model, loader) -> dict:
    """Return AUROC/AUPRC + metrics at two thresholds."""
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    logits, y = collect_predictions(model, loader)
    proba     = 1.0 / (1.0 + np.exp(-logits))

    auroc = float(roc_auc_score(y, proba))
    auprc = float(average_precision_score(y, proba))

    fpr, tpr, thr = roc_curve(y, proba)
    # Youden-J optimum
    youden = tpr - fpr
    j_idx  = int(np.argmax(youden))
    thr_opt = float(thr[j_idx])

    # 95% sensitivity threshold (largest threshold s.t. tpr >= 0.95)
    mask = tpr >= 0.95
    if mask.any():
        # roc_curve thresholds are in decreasing order; we want the highest
        # threshold among those where tpr >= 0.95 (i.e. tightest spec)
        idx95 = int(np.argmax(mask))  # first index with tpr >= 0.95
        thr_95 = float(thr[idx95])
    else:
        thr_95 = float(thr.min())

    pred_opt = (proba >= thr_opt).astype(int)
    pred_95  = (proba >= thr_95).astype(int)

    return {
        "n_samples":   int(len(y)),
        "n_positive":  int((y == 1).sum()),
        "n_negative":  int((y == 0).sum()),
        "auroc":       auroc,
        "auprc":       auprc,
        "thr_optimal": thr_opt,
        "thr_sens95":  thr_95,
        "metrics_at_optimal": _binary_metrics(y, pred_opt),
        "metrics_at_sens95":  _binary_metrics(y, pred_95),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def make_barplot(all_metrics: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = sorted(all_metrics.keys())
    n = len(variants)

    aurocs    = [all_metrics[v]["auroc"]                          for v in variants]
    auprcs    = [all_metrics[v]["auprc"]                          for v in variants]
    sens_opt  = [all_metrics[v]["metrics_at_optimal"]["sensitivity"] for v in variants]
    spec_opt  = [all_metrics[v]["metrics_at_optimal"]["specificity"] for v in variants]
    bacc_opt  = [all_metrics[v]["metrics_at_optimal"]["balanced_accuracy"] for v in variants]
    f1_opt    = [all_metrics[v]["metrics_at_optimal"]["f1"]          for v in variants]
    spec_95   = [all_metrics[v]["metrics_at_sens95"]["specificity"]  for v in variants]
    bacc_95   = [all_metrics[v]["metrics_at_sens95"]["balanced_accuracy"] for v in variants]

    labels = [v.replace("sleepfm_", "") for v in variants]
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    score_ylim = (0.4, 1.0)

    # Panel A — AUROC + AUPRC
    ax = axes[0, 0]
    w = 0.4
    b1 = ax.bar(x - w/2, aurocs, w, label="AUROC",  color="#3b6ea5")
    b2 = ax.bar(x + w/2, auprcs, w, label="AUPRC",  color="#a53b3b")
    ax.set_ylim(*score_ylim)
    ax.axhline(0.5,  color="grey", lw=0.5, ls="--", alpha=0.6)
    ax.set_ylabel("Score")
    ax.set_title("ROC / PR area under curve (val split)")
    ax.legend(loc="lower right", frameon=False)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.005,
                    f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # Panel B — Metrics at Youden-J optimal threshold
    ax = axes[0, 1]
    w = 0.2
    ax.bar(x - 1.5*w, sens_opt, w, label="Sens",  color="#3b6ea5")
    ax.bar(x - 0.5*w, spec_opt, w, label="Spec",  color="#3ba56e")
    ax.bar(x + 0.5*w, bacc_opt, w, label="BalAcc", color="#a5933b")
    ax.bar(x + 1.5*w, f1_opt,   w, label="F1",    color="#8e3ba5")
    ax.set_ylim(*score_ylim)
    ax.set_ylabel("Score")
    ax.set_title("At Youden-J optimal threshold")
    ax.legend(ncol=2, loc="lower right", frameon=False, fontsize=8)

    # Panel C — Metrics at 95% sensitivity threshold
    ax = axes[1, 0]
    w = 0.4
    ax.bar(x - w/2, spec_95, w, label="Spec @ Sens=0.95",   color="#3ba56e")
    ax.bar(x + w/2, bacc_95, w, label="BalAcc @ Sens=0.95", color="#a5933b")
    ax.set_ylim(*score_ylim)
    ax.set_ylabel("Score")
    ax.set_title("At fixed 95 % sensitivity")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    for i, (s, b) in enumerate(zip(spec_95, bacc_95)):
        ax.text(i - w/2, s + 0.005, f"{s:.2f}", ha="center", fontsize=7)
        ax.text(i + w/2, b + 0.005, f"{b:.2f}", ha="center", fontsize=7)

    # Panel D — operating-point thresholds
    ax = axes[1, 1]
    thrs_opt = [all_metrics[v]["thr_optimal"] for v in variants]
    thrs_95  = [all_metrics[v]["thr_sens95"]  for v in variants]
    w = 0.4
    ax.bar(x - w/2, thrs_opt, w, label="Youden-J",   color="#3b6ea5")
    ax.bar(x + w/2, thrs_95,  w, label="Sens=0.95",  color="#a53b3b")
    ax.set_ylabel("Probability threshold")
    ax.set_title("Operating-point thresholds")
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "SleepFM fine-tuning on D4-v4 (binary normal vs abnormal)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[✓] Barplot → {out_path.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="*", default=list(ALL_VARIANTS),
                   help="Subset of variants to fine-tune (default: v1.1 + all 7 v2)")
    p.add_argument("--epochs",        type=int, default=8)
    p.add_argument("--freeze-epochs", type=int, default=2,
                   help="Number of head-only epochs before unfreezing encoder")
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--num-workers",   type=int, default=4)
    p.add_argument("--head-lr",       type=float, default=3e-3)
    p.add_argument("--full-lr",       type=float, default=3e-5)
    p.add_argument("--skip-train",    action="store_true",
                   help="Skip training; only evaluate existing checkpoints + barplot")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"V4 data: {V4_DATA}")
    print(f"Output: {OUT_DIR}")

    print("\nLoading dataloaders…")
    fold, train_loader, val_loader = get_loaders(args.batch_size, args.num_workers)
    print(f"  fold={fold}  train batches={len(train_loader)}  val batches={len(val_loader)}")

    pos_weight = binary_pos_weight(train_loader.dataset.labels)

    all_metrics: dict[str, dict] = {}
    # If skipping training and re-evaluating, start from the previously saved
    # all_metrics.json so that variants we don't re-evaluate are preserved.
    if args.skip_train and (OUT_DIR / "all_metrics.json").exists():
        try:
            all_metrics = json.loads((OUT_DIR / "all_metrics.json").read_text())
        except Exception:
            all_metrics = {}

    for variant in args.variants:
        if variant not in ALL_VARIANTS:
            print(f"[skip] unknown variant: {variant}")
            continue

        ckpt_path = OUT_DIR / variant / "finetuned.ckpt"

        if not args.skip_train:
            ckpt_path = train_one_variant(
                variant, args, train_loader, val_loader, pos_weight,
            )

        # Evaluate from best checkpoint
        if not ckpt_path.exists():
            print(f"  [skip eval] {variant}: no checkpoint at {ckpt_path}")
            continue

        print(f"\n[{variant}]  Loading best checkpoint for evaluation…")
        encoder, cfg, _ = build_v2_encoder(variant)
        model = SleepFMBinary(encoder, embed_dim=int(cfg["embed_dim"])).to(DEVICE)
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        m = evaluate_all_metrics(model, val_loader)
        all_metrics[variant] = m

        (OUT_DIR / variant / "metrics.json").write_text(json.dumps(m, indent=2))
        print(
            f"  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
            f"BalAcc(opt)={m['metrics_at_optimal']['balanced_accuracy']:.3f}  "
            f"Spec@95Sens={m['metrics_at_sens95']['specificity']:.3f}"
        )

    if all_metrics:
        (OUT_DIR / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2))
        make_barplot(all_metrics, OUT_DIR / "comparison_barplot.png")

        print("\nFinal table:")
        print(f"{'variant':<18s}  {'AUROC':>6s}  {'AUPRC':>6s}  "
              f"{'BalAcc-opt':>10s}  {'Spec@95Sens':>11s}  {'BalAcc@95Sens':>14s}")
        for v, m in sorted(all_metrics.items()):
            print(
                f"{v:<18s}  {m['auroc']:.4f}  {m['auprc']:.4f}  "
                f"{m['metrics_at_optimal']['balanced_accuracy']:>10.4f}  "
                f"{m['metrics_at_sens95']['specificity']:>11.4f}  "
                f"{m['metrics_at_sens95']['balanced_accuracy']:>14.4f}"
            )


if __name__ == "__main__":
    main()
