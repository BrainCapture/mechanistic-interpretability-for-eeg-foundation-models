"""Evaluate the sleepfm_granular 10-class classifier on the validation split.

Outputs:
  results/eval_granular/classification_report.json
  results/eval_granular/confusion_matrix.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

matplotlib.rcParams.update({"font.family": "serif", "font.size": 10})

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import classification_report, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
V4_DATA  = ROOT / "data" / "D4-v4-preprocessed-10s"
CKPT     = ROOT / "checkpoints" / "granular" / "sleepfm_granular.best.ckpt"
OUT      = ROOT / "results" / "eval_granular"
OUT.mkdir(parents=True, exist_ok=True)

N_CLASSES  = 10
EMBED_DIM  = 128
PATCH_SIZE = 128

CLASS_NAMES = [
    "Normal",
    "Diffuse slowing",
    "Focal slowing",
    "Focal sharp waves",
    "Focal spike-wave",
    "Gen. spike-wave",
    "Gen. polyspike-wave",
    "Gen. sharp waves",
    "Burst suppression",
    "Epileptic seizure",
]

_SLEEPFM_KWARGS = dict(
    in_channels=1,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=8,
    num_layers=3,
    pooling_head=8,
    dropout=0.3,
)


def _build_model() -> nn.Module:
    from sae4eeg.sleepfm import SetTransformer

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SetTransformer(**_SLEEPFM_KWARGS)
            self.head = nn.Linear(EMBED_DIM, N_CLASSES)

        def forward(self, x):
            pooled, _ = self.encoder(x)
            return self.head(pooled)

    return Model()


def main():
    from sae4eeg.dataset import get_dataloaders, V4ResampleTransform

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT}")

    # Load model
    model = _build_model().to(DEVICE)
    raw = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys")
    model.eval()
    print("  Model loaded.")

    # Load validation data
    gen = get_dataloaders(
        train_path=str(V4_DATA),
        transformer=V4ResampleTransform(),
        batch_size=64,
        num_workers=4,
        seed=42,
        split_info_path=str(ROOT / "results" / "probe_reconstruction" / "splits.json"),
    )
    _, _, val_loader, _ = next(gen)
    print(f"  Val batches: {len(val_loader)}")

    # Run inference
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0].to(DEVICE), batch[1]
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = logits.argmax(1).cpu()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(y.long())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    # Classification report
    report = classification_report(
        y_true, y_pred,
        labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    with open(OUT / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(classification_report(
        y_true, y_pred,
        labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    short = ["Norm", "DiffSlow", "FocSlow", "FocSharp",
             "FocSW", "GenSW", "GenPSW", "GenSharp", "BrstSup", "Seizure"]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("sleepfm_granular — validation confusion matrix (10-class)")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved confusion matrix → {OUT / 'confusion_matrix.png'}")

    # Binary Confusion Matrix
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im2 = ax2.imshow(cm_bin, cmap="Blues")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Normal", "Abnormal"], fontsize=10)
    ax2.set_yticklabels(["Normal", "Abnormal"], fontsize=10)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Binary (Collapsed Abnormalities)")

    thresh2 = cm_bin.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm_bin[i, j]), ha="center", va="center",
                     fontsize=10, color="white" if cm_bin[i, j] > thresh2 else "black")

    fig2.tight_layout()
    fig2.savefig(OUT / "confusion_matrix_binary.png", dpi=150, bbox_inches="tight")
    print(f"Saved binary matrix    → {OUT / 'confusion_matrix_binary.png'}")

    from sklearn.metrics import cohen_kappa_score, roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Binary metrics: label 0 is Normal (neg), >0 is Abnormal (pos)
    # y_true_bin and y_pred_bin already computed: 0 for normal, 1 for abnormal
    y_prob_bin = 1.0 - y_prob[:, 0] # probability of abnormal
    
    # From cm_bin (0: normal, 1: abnormal)
    TN, FP, FN, TP = cm_bin.ravel()
    sens = TP / (TP + FN) if (TP+FN) > 0 else 0
    spec = TN / (TN + FP) if (TN+FP) > 0 else 0
    precision = TP / (TP + FP) if (TP+FP) > 0 else 0
    f1_bin = 2 * (precision * sens) / (precision + sens) if (precision+sens) > 0 else 0
    bal_acc = (sens + spec) / 2
    auroc = roc_auc_score(y_true_bin, y_prob_bin)
    aucpr = average_precision_score(y_true_bin, y_prob_bin)

    # 95% Sensitivity Threshold
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_prob_bin)
    idx_95 = np.where(tpr >= 0.95)[0][0]
    thresh_95 = thresholds[idx_95]
    
    y_pred_bin_95 = (y_prob_bin >= thresh_95).astype(int)
    cm_bin_95 = confusion_matrix(y_true_bin, y_pred_bin_95)
    TN95, FP95, FN95, TP95 = cm_bin_95.ravel()
    sens95 = TP95 / (TP95 + FN95) if (TP95+FN95) > 0 else 0
    spec95 = TN95 / (TN95 + FP95) if (TN95+FP95) > 0 else 0
    precision95 = TP95 / (TP95 + FP95) if (TP95+FP95) > 0 else 0
    f1_bin_95 = 2 * (precision95 * sens95) / (precision95 + sens95) if (precision95+sens95) > 0 else 0
    bal_acc95 = (sens95 + spec95) / 2
    
    print("\n--- Additional Metrics ---")
    print(f"Multiclass Cohen's Kappa: {kappa:.4f}")
    print("\nBinary (Default argmax threshold):")
    print(f"  Sensitivity (Recall): {sens:.4f}")
    print(f"  Specificity:          {spec:.4f}")
    print(f"  Precision:            {precision:.4f}")
    print(f"  F1 Score:             {f1_bin:.4f}")
    print(f"  Balanced Accuracy:    {bal_acc:.4f}")
    print(f"  AUROC:                {auroc:.4f}")
    print(f"  AUCPR:                {aucpr:.4f}")

    print(f"\nBinary (95% Sensitivity Threshold = {thresh_95:.4f}):")
    print(f"  Sensitivity (Recall): {sens95:.4f}")
    print(f"  Specificity:          {spec95:.4f}")
    print(f"  Precision:            {precision95:.4f}")
    print(f"  F1 Score:             {f1_bin_95:.4f}")
    print(f"  Balanced Accuracy:    {bal_acc95:.4f}")

    print(f"\nSaved report          → {OUT / 'classification_report.json'}")


if __name__ == "__main__":
    main()
