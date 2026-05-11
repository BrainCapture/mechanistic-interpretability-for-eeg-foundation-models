"""Evaluate XAE v2 morphology head against granular EEG labels.

For each granular label class (V4: 0=normal, 1=diffuse_slowing, 2=focal_slowing,
3=focal_sharp_waves, 4=focal_spike_wave, 5=gen_spike_wave, 6=gen_polyspike,
7=gen_sharp, 8=burst_suppression, 9=epileptic_seizure), compute predicted
morphology features per token and report:

  • per-feature R² of predicted vs ground-truth morphology (validation R²)
  • per-feature class-conditional means + boxplots
  • per-feature AUROC for "spike-wave (label 4) vs normal (label 0)"
  • side-by-side: ground-truth vs XAE-predicted morphology distributions

Saves figures under ``results/xae/{encoder}_v2/eval/``.

Usage::

    uv run tools/eval_xae_morphology.py --encoder sleepfm_granular
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from sae4eeg.dataset import (H5PYDatasetLabeled, StandardizeLabel,
                              V4ResampleTransform)
from sae4eeg.encoders import load_encoder
from sae4eeg.sae import ActivationExtractor
from sae4eeg.xae import XAETrainer, MorphologyTargetExtractor

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_NAMES = {
    0: "normal",
    1: "diffuse_slowing",
    2: "focal_slowing",
    3: "focal_sharp_waves",
    4: "focal_spike_wave",
    5: "gen_spike_wave",
    6: "gen_polyspike",
    7: "gen_sharp_waves",
    8: "burst_suppression",
    9: "epileptic_seizure",
}

_GRANULAR_CHECKPOINTS = {
    "sleepfm_granular": ROOT / "checkpoints" / "granular" / "sleepfm_granular.ckpt",
    "sleepfm_finetuned": ROOT / "checkpoints" / "finetuned" / "sleepfm1.ckpt",
}

_ENCODER_CFG = {
    "sleepfm_granular": dict(
        data_path = ROOT / "data" / "D4-v4-preprocessed-10s",
        fs=128, patch_size=128, target_layer=2,
        transform=V4ResampleTransform,
        weights_path=_GRANULAR_CHECKPOINTS["sleepfm_granular"],
        encoder_name="sleepfm_granular",
    ),
    "sleepfm_finetuned": dict(
        data_path = ROOT / "data" / "D4-v4-preprocessed-10s",   # use V4 for granular labels
        fs=128, patch_size=128, target_layer=2,
        transform=V4ResampleTransform,
        weights_path=_GRANULAR_CHECKPOINTS["sleepfm_finetuned"],
        encoder_name="sleepfm",
    ),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="sleepfm_granular",
                   choices=list(_ENCODER_CFG.keys()))
    p.add_argument("--xae-tag", default="v2",
                   help="Looks for results/xae/{encoder}_{tag}/xae_checkpoint.pt")
    p.add_argument("--max-windows", type=int, default=4000,
                   help="Max windows to evaluate (each window has 10 tokens)")
    p.add_argument("--layer", type=int, default=None)
    return p.parse_args()


def load_xae(xae_path: Path) -> XAETrainer:
    trainer = XAETrainer(embed_dim=128, fs=128, n_fft=128,
                         morphology=False)   # rebuilt by load
    trainer.load(str(xae_path))
    if trainer.morphology_extractor is None:
        raise RuntimeError(f"XAE checkpoint {xae_path} has no morphology head.")
    trainer.xae = trainer.xae.to(DEVICE).eval()
    return trainer


@torch.no_grad()
def collect_labelled_morphology(
    encoder, trainer: XAETrainer, data_path: Path,
    transform_cls, max_windows: int, target_layer: int,
):
    """Run the dataset through the encoder and collect, per token:
       - encoder embedding → predicted morphology
       - ground-truth morphology (computed from raw patches)
       - granular per-window label (replicated to each token)
    """
    # Use the dataset directly so we can control batching and grab labels
    ds = H5PYDatasetLabeled(str(data_path), transform=transform_cls())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    fs = trainer.morphology_extractor.fs
    patch_size = trainer.morphology_extractor.patch_size
    morph_ext = trainer.morphology_extractor
    n_features = morph_ext.target_dim

    encoder.to(DEVICE).eval()
    inner_model = encoder.model if hasattr(encoder, "model") else encoder
    extractor = ActivationExtractor(
        inner_model,
        layers=encoder.get_hookable_layers() if hasattr(encoder, "get_hookable_layers") else None,
    )

    all_labels = []
    all_gt = []
    all_pred = []
    all_emb_norm = []
    n_seen = 0

    pbar = tqdm(loader, total=min(len(loader), max_windows // 32 + 1),
                desc="collect")
    for batch in pbar:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            y = batch[1]
        else:
            x, y = batch, None
        x = x.to(DEVICE)
        B, C, T = x.shape
        S = T // patch_size

        extractor.clear()
        if hasattr(encoder, "encode"):
            _ = encoder.encode(x)
        else:
            _ = inner_model(x)
        acts = extractor.get_activations()
        layer_acts = acts[target_layer]               # (B, S, E)

        T_used = S * patch_size
        patches = x[:, :, :T_used].reshape(B, C, S, patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(B * S, C, patch_size)

        # Ground-truth morphology
        gt = morph_ext.extract(patches).cpu()         # (B*S, n_features)

        # Predicted morphology from embeddings
        emb_flat = layer_acts.reshape(B * S, -1).cpu()
        emb_norm = (emb_flat - trainer.embed_mean) / trainer.embed_std
        with torch.no_grad():
            pred_norm = trainer.xae.decode_morphology(emb_norm.to(DEVICE)).cpu()
        pred = pred_norm * trainer.morphology_std + trainer.morphology_mean

        # Replicate window-level label to every token in that window
        if y is not None:
            y_rep = y.unsqueeze(1).expand(B, S).reshape(-1).cpu()
        else:
            y_rep = torch.full((B * S,), -1, dtype=torch.long)

        all_labels.append(y_rep.numpy())
        all_gt.append(gt.numpy())
        all_pred.append(pred.numpy())
        all_emb_norm.append(emb_norm.numpy())

        n_seen += B
        if n_seen >= max_windows:
            break

    extractor.remove_hooks()

    labels = np.concatenate(all_labels)
    gt = np.concatenate(all_gt, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    emb_norm = np.concatenate(all_emb_norm, axis=0)
    print(f"  collected {len(labels):,} tokens "
          f"({n_seen} windows × {gt.shape[1]} features)")
    return labels, gt, pred, emb_norm


def per_feature_r2(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-column R² of pred vs gt. Returns array of shape (n_features,)."""
    ss_res = ((gt - pred) ** 2).sum(axis=0)
    ss_tot = ((gt - gt.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.clip(ss_tot, 1e-8, None)


def per_feature_auroc(values: np.ndarray, labels: np.ndarray,
                      pos_label: int, neg_label: int) -> np.ndarray:
    """AUROC per feature for pos_label vs neg_label."""
    pos_mask = labels == pos_label
    neg_mask = labels == neg_label
    aurocs = []
    for j in range(values.shape[1]):
        v = values[:, j]
        try:
            mask = pos_mask | neg_mask
            y = labels[mask] == pos_label
            score = v[mask]
            au = roc_auc_score(y.astype(int), score)
        except Exception:
            au = float("nan")
        aurocs.append(au)
    return np.asarray(aurocs)


def plot_r2_bar(names, r2_vec, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(0.36 * len(names) + 2, 4.2),
                           facecolor="white")
    xs = np.arange(len(names))
    colors = ["#2196F3" if i < 8 else "#9C27B0" for i in xs]
    ax.bar(xs, np.clip(r2_vec, -0.1, 1), color=colors, alpha=0.85)
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("R²")
    ax.set_ylim(min(-0.1, r2_vec.min() - 0.05), 1.05)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_auroc_compare(names, au_gt, au_pred, out_path: Path,
                       pos_name: str, neg_name: str):
    fig, ax = plt.subplots(figsize=(0.4 * len(names) + 2, 4.6),
                           facecolor="white")
    xs = np.arange(len(names))
    w = 0.42
    ax.bar(xs - w / 2, au_gt, width=w, color="#444",   alpha=0.85,
           label=f"GT morphology  ({pos_name} vs {neg_name})")
    ax.bar(xs + w / 2, au_pred, width=w, color="#c0392b", alpha=0.85,
           label=f"XAE-predicted morphology")
    ax.axhline(0.5, color="#888", lw=0.6, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f"Per-feature AUROC: {pos_name} vs {neg_name}\n"
                 f"GT = computed from raw EEG  ·  XAE = decoded from embedding",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_class_boxplots(values, labels, names, out_path: Path,
                        title: str, max_features: int = 8):
    """Boxplot of values per class for the first max_features features."""
    classes = sorted(int(c) for c in np.unique(labels) if c >= 0)
    feat_show = names[:max_features]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), facecolor="white")
    for k, name in enumerate(feat_show):
        ax = axes[k // 4][k % 4]
        data = []
        labels_used = []
        for c in classes:
            v = values[labels == c, k]
            if len(v) >= 10:
                data.append(v)
                labels_used.append(LABEL_NAMES.get(c, str(c)))
        if not data:
            ax.set_axis_off(); continue
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, c in zip(bp["boxes"], classes[:len(data)]):
            patch.set_facecolor("#3b6e8f" if c == 0 else "#c0392b")
            patch.set_alpha(0.65)
        ax.set_xticklabels(labels_used, rotation=45, ha="right", fontsize=7)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    args = parse_args()
    cfg = _ENCODER_CFG[args.encoder]
    target_layer = args.layer if args.layer is not None else cfg["target_layer"]

    out_dir = ROOT / "results" / "xae" / f"{args.encoder}_{args.xae_tag}" / "eval"
    xae_path = ROOT / "results" / "xae" / f"{args.encoder}_{args.xae_tag}" / "xae_checkpoint.pt"
    if not xae_path.exists():
        sys.stderr.write(f"XAE checkpoint not found at {xae_path}\n")
        raise SystemExit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[XAE] loading {xae_path}")
    trainer = load_xae(xae_path)

    encoder_name = cfg["encoder_name"]
    print(f"[encoder] loading {encoder_name}  weights={cfg['weights_path'].name}")
    encoder = load_encoder(encoder_name, weights_path=cfg["weights_path"])

    labels, gt, pred, emb_norm = collect_labelled_morphology(
        encoder, trainer, cfg["data_path"], cfg["transform"],
        max_windows=args.max_windows, target_layer=target_layer,
    )
    names = trainer.morphology_extractor.feature_names

    # ── 1. R² of XAE morphology head (predicted vs ground truth) ─────────
    r2 = per_feature_r2(gt, pred)
    print("\n=== R² (XAE morphology head vs GT) ===")
    for n, v in zip(names, r2):
        print(f"  {n:<22s}  R² = {v:>7.4f}")
    plot_r2_bar(names, r2, out_dir / "morphology_r2.png",
                "XAE morphology head — per-feature R² on V4 dataset")

    # ── 2. AUROC: focal_spike_wave (4) vs normal (0) ─────────────────────
    pos, neg = 4, 0
    pos_name, neg_name = LABEL_NAMES[pos], LABEL_NAMES[neg]
    au_gt = per_feature_auroc(gt, labels, pos, neg)
    au_pred = per_feature_auroc(pred, labels, pos, neg)
    print(f"\n=== Per-feature AUROC ({pos_name} vs {neg_name}) ===")
    print(f"  {'feature':<22s}  {'GT':>8s}  {'XAE':>8s}")
    for n, ag, ap in zip(names, au_gt, au_pred):
        print(f"  {n:<22s}  {ag:>8.3f}  {ap:>8.3f}")
    plot_auroc_compare(names, au_gt, au_pred,
                       out_dir / "morphology_auroc_spike_vs_normal.png",
                       pos_name, neg_name)

    # ── 3. AUROC: any abnormal (1-9) vs normal (0) — sanity ─────────────
    abn_mask = (labels >= 1) & (labels <= 9)
    nrm_mask = labels == 0
    if abn_mask.sum() and nrm_mask.sum():
        binary = np.where(abn_mask, 1, np.where(nrm_mask, 0, -1))
        keep = binary >= 0
        au_gt_b = []
        au_pred_b = []
        for j in range(gt.shape[1]):
            try:
                au_gt_b.append(roc_auc_score(binary[keep], gt[keep, j]))
                au_pred_b.append(roc_auc_score(binary[keep], pred[keep, j]))
            except Exception:
                au_gt_b.append(float("nan"))
                au_pred_b.append(float("nan"))
        plot_auroc_compare(
            names, np.array(au_gt_b), np.array(au_pred_b),
            out_dir / "morphology_auroc_abnormal_vs_normal.png",
            "abnormal", "normal",
        )

    # ── 4. Class-conditional boxplots (GT and predicted side by side) ────
    plot_class_boxplots(gt, labels, names,
                        out_dir / "morphology_boxplot_gt.png",
                        title="Ground-truth morphology by class")
    plot_class_boxplots(pred, labels, names,
                        out_dir / "morphology_boxplot_pred.png",
                        title="XAE-predicted morphology by class")

    # ── 5. Save numeric summary ─────────────────────────────────────────
    summary = {
        "encoder": args.encoder,
        "target_layer": target_layer,
        "n_tokens": int(len(labels)),
        "n_per_class": {LABEL_NAMES.get(int(c), str(c)): int((labels == c).sum())
                         for c in np.unique(labels) if c >= 0},
        "feature_names": names,
        "r2": dict(zip(names, [float(x) for x in r2])),
        "auroc_focal_spike_vs_normal_gt": dict(zip(names, [float(x) for x in au_gt])),
        "auroc_focal_spike_vs_normal_xae": dict(zip(names, [float(x) for x in au_pred])),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved {summary_path}")
    print(f"\n[done] eval written to {out_dir}/")


if __name__ == "__main__":
    import sys
    main()
