"""Probe V4 granular label classes: encoder embeddings vs SAE z-features.

Tests whether clinical information distinguishing EEG abnormality subtypes is
preserved through the SAE bottleneck.

Two representations compared on a fresh subject-level split of V4 data:
  encoder  — mean-pooled raw layer-2 activations (128-dim)
  sae      — mean-pooled SAE sparse z-features (128-dim, top-k=8 per token)

For each label class i in 1–9 (pos=class i, neg=normal):
  Binary logistic regression probe → AUROC ± std across seeds

Multi-class logistic regression on abnormal-only subset:
  Confusion matrix (row-normalised by true class) for both representations

Outputs
-------
  results/probe_v4_labels/binary_auroc.json
  results/probe_v4_labels/confusion_encoder_layer{L}.png
  results/probe_v4_labels/confusion_sae_layer{L}.png

Usage::

    uv run tools/probe_v4_labels.py
    uv run tools/probe_v4_labels.py --layer 2 --seeds 5 --test-size 0.2
    uv run tools/probe_v4_labels.py --max-windows 5000   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

V4_DATA_PATH = ROOT / "data" / "D4-v4-preprocessed-10s"
SAE_DIR      = ROOT / "results" / "features" / "sleepfm_finetuned"
WEIGHTS_PATH = ROOT / "checkpoints" / "finetuned" / "sleepfm1.ckpt"
OUT_DIR      = ROOT / "results" / "probe_v4_labels"

V4_LABEL_NAMES = {
    0: "normal",
    1: "diffuse_slowing",
    2: "focal_slowing",
    3: "focal_sharp_waves",
    4: "focal_spike_wave",
    5: "gen_spike_wave",
    6: "gen_polyspike_wave",
    7: "gen_sharp_waves",
    8: "burst_suppression",
    9: "seizure",
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature collection
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_features(
    encoder,
    sae,
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
    loader: DataLoader,
    target_layer: int,
    max_windows: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect mean-pooled encoder embeddings and SAE z-features for every window.

    Returns
    -------
    enc_feats  (N, embed_dim)   — mean-pooled raw layer activations
    sae_feats  (N, n_features)  — mean-pooled sparse z-activations
    labels     (N,)             — integer label per window (0–9)
    """
    act_mean = act_mean.to(DEVICE)
    act_std  = act_std.to(DEVICE)
    sae.to(DEVICE)

    hookable = encoder.get_hookable_layers()
    enc_list, sae_list, lbl_list = [], [], []
    n_seen = 0

    from tqdm import tqdm
    for batch in tqdm(loader, desc="Collecting features", leave=False):
        x, y, _ = batch
        x = x.to(DEVICE)
        B = x.shape[0]

        # Capture layer activations via hook
        captured: dict[int, torch.Tensor] = {}

        def _hook(m, inp, out, _k=target_layer):
            captured[_k] = out

        handle = hookable[target_layer].register_forward_hook(_hook)
        encoder.encode(x)
        handle.remove()

        acts = captured[target_layer]          # (B, S, E)
        S, E = acts.shape[1], acts.shape[2]

        # Encoder path — mean-pool raw activations over tokens
        enc_pooled = acts.mean(dim=1).cpu()    # (B, E)

        # SAE path — normalise → encode → mean-pool
        acts_flat = acts.reshape(B * S, E)
        acts_norm = (acts_flat - act_mean) / (act_std + 1e-8)
        z_flat    = sae.encode(acts_norm)      # (B*S, n_features)
        z_pooled  = z_flat.reshape(B, S, -1).mean(dim=1).cpu()  # (B, n_features)

        enc_list.append(enc_pooled)
        sae_list.append(z_pooled)
        lbl_list.append(y.cpu())

        n_seen += B
        if max_windows is not None and n_seen >= max_windows:
            break

    return (
        torch.cat(enc_list).numpy().astype(np.float32),
        torch.cat(sae_list).numpy().astype(np.float32),
        torch.cat(lbl_list).long().numpy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Probes
# ─────────────────────────────────────────────────────────────────────────────

def _fit_lr(X_tr, y_tr, X_te, y_te, seed: int) -> dict:
    """Fit logistic regression with StandardScaler; return AUROC + balanced acc."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs", random_state=seed,
    )
    clf.fit(X_tr_s, y_tr)
    proba = clf.predict_proba(X_te_s)[:, 1]
    preds = clf.predict(X_te_s)
    return {
        "auc": float(roc_auc_score(y_te, proba)),
        "ba":  float(balanced_accuracy_score(y_te, preds)),
    }


def binary_probes(
    enc_tr, sae_tr, lbl_tr,
    enc_te, sae_te, lbl_te,
    n_seeds: int,
) -> dict:
    results = {}
    for cls, name in V4_LABEL_NAMES.items():
        if cls == 0:
            continue
        tr_mask = (lbl_tr == cls) | (lbl_tr == 0)
        te_mask = (lbl_te == cls) | (lbl_te == 0)
        n_pos_tr = int((lbl_tr[tr_mask] == cls).sum())
        n_pos_te = int((lbl_te[te_mask] == cls).sum())

        if n_pos_tr < 5 or n_pos_te < 3:
            print(f"  [{name:25s}] skipped  train_pos={n_pos_tr}  test_pos={n_pos_te}")
            continue

        y_tr = (lbl_tr[tr_mask] == cls).astype(int)
        y_te = (lbl_te[te_mask] == cls).astype(int)

        enc_aucs, sae_aucs = [], []
        for seed in range(n_seeds):
            enc_aucs.append(_fit_lr(enc_tr[tr_mask], y_tr,
                                    enc_te[te_mask], y_te, seed)["auc"])
            sae_aucs.append(_fit_lr(sae_tr[tr_mask], y_tr,
                                    sae_te[te_mask], y_te, seed)["auc"])

        enc_m, enc_s = float(np.mean(enc_aucs)), float(np.std(enc_aucs))
        sae_m, sae_s = float(np.mean(sae_aucs)), float(np.std(sae_aucs))
        delta = sae_m - enc_m

        print(f"  [{name:25s}]  "
              f"enc={enc_m:.3f}±{enc_s:.3f}  "
              f"sae={sae_m:.3f}±{sae_s:.3f}  "
              f"Δ={delta:+.3f}  "
              f"(n_pos_te={n_pos_te})")

        results[name] = {
            "label_class":  cls,
            "n_pos_train":  n_pos_tr,
            "n_pos_test":   n_pos_te,
            "enc_auc_mean": round(enc_m, 4),
            "enc_auc_std":  round(enc_s, 4),
            "sae_auc_mean": round(sae_m, 4),
            "sae_auc_std":  round(sae_s, 4),
            "delta_auc":    round(delta, 4),
        }
    return results


def confusion_plots(
    enc_tr, sae_tr, lbl_tr,
    enc_te, sae_te, lbl_te,
    target_layer: int,
    out_dir: Path = OUT_DIR,
    suffix: str = "",
) -> None:
    """Multi-class probe on abnormal-only subset; save confusion matrix PNGs."""
    abn_tr = lbl_tr != 0
    abn_te = lbl_te != 0

    # Only keep classes with ≥3 test examples
    present = sorted(c for c in range(1, 10)
                     if int((lbl_te[abn_te] == c).sum()) >= 3)
    if len(present) < 2:
        print("  Confusion matrix skipped — fewer than 2 classes with ≥3 test examples")
        return

    valid_tr = abn_tr & np.isin(lbl_tr, present)
    valid_te = abn_te & np.isin(lbl_te, present)
    print(f"  Multi-class: {len(present)} classes  "
          f"train={valid_tr.sum()}  test={valid_te.sum()}")

    short = {c: V4_LABEL_NAMES[c].replace("_", "\n") for c in present}
    tick_labels = [short[c] for c in present]

    for rep_name, X_tr, X_te in [
        ("encoder", enc_tr, enc_te),
        ("sae",     sae_tr, sae_te),
    ]:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[valid_tr])
        X_te_s = scaler.transform(X_te[valid_te])
        clf = LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs", random_state=0,
        )
        clf.fit(X_tr_s, lbl_tr[valid_tr])
        preds = clf.predict(X_te_s)
        y_te_cls = lbl_te[valid_te]

        cm = confusion_matrix(y_te_cls, preds, labels=present)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1e-9)

        fig, ax = plt.subplots(figsize=(max(6, len(present)), max(5, len(present) - 1)))
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Recall (row-normalised)")
        ax.set_xticks(range(len(present)))
        ax.set_yticks(range(len(present)))
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Abnormal-only confusion  |  {rep_name}  |  layer {target_layer}")
        for i in range(len(present)):
            for j in range(len(present)):
                v = cm_norm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if v > 0.55 else "black")
        fig.tight_layout()
        out = out_dir / f"confusion_{rep_name}_layer{target_layer}{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer",        type=int,   default=2,
                        help="Target encoder layer (default: 2)")
    parser.add_argument("--expansion",    type=float, default=1.0)
    parser.add_argument("--k",            type=int,   default=8)
    parser.add_argument("--test-size",    type=float, default=0.2,
                        help="Fraction of subjects held out for test (default: 0.2)")
    parser.add_argument("--split-seed",   type=int,   default=99,
                        help="GroupShuffleSplit seed (default: 99 ≠ TCAV's 42)")
    parser.add_argument("--seeds",        type=int,   default=3,
                        help="LR seeds for variance estimate (default: 3)")
    parser.add_argument("--max-windows",  type=int,   default=None,
                        help="Cap total windows collected (for quick testing)")
    parser.add_argument("--weights-path", default=None,
                        help="Override encoder weights path (default: sleepfm1.ckpt)")
    parser.add_argument("--sae-dir",      default=None,
                        help="Override SAE directory (default: results/features/sleepfm_finetuned)")
    parser.add_argument("--encoder-name", default="sleepfm",
                        help="Encoder name for load_encoder() (default: sleepfm). "
                             "Use 'sleepfm_granular' for the granular checkpoint.")
    parser.add_argument("--suffix",       default="",
                        help="Filename suffix for output files (e.g. '_granular')")
    args = parser.parse_args()

    # Apply overrides
    weights_path = Path(args.weights_path) if args.weights_path else WEIGHTS_PATH
    sae_dir      = Path(args.sae_dir)      if args.sae_dir      else SAE_DIR
    out_dir      = OUT_DIR
    suffix       = args.suffix

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Encoder: {args.encoder_name}  weights={weights_path.name}")
    print(f"SAE dir: {sae_dir.relative_to(ROOT)}")

    # ── Load encoder ──────────────────────────────────────────────────────────
    from sae4eeg.encoders import load_encoder
    encoder = load_encoder(args.encoder_name, weights_path=weights_path)
    encoder.to(DEVICE).eval()
    print(f"Encoder loaded")

    # ── Load SAE ─────────────────────────────────────────────────────────────
    from sae4eeg.sae import SparseAutoencoder
    # Support both naming conventions: sae_sleepfm_... and sae_sleepfm_granular_...
    enc_tag = args.encoder_name if args.encoder_name != "sleepfm" else "sleepfm"
    sae_path = sae_dir / f"sae_{enc_tag}_exp{args.expansion}_k{args.k}_layer{args.layer}.pt"
    if not sae_path.exists():
        # Fallback to legacy sleepfm naming
        sae_path = sae_dir / f"sae_sleepfm_exp{args.expansion}_k{args.k}_layer{args.layer}.pt"
    sae_ckpt = torch.load(str(sae_path), map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(
        input_dim=int(sae_ckpt["embed_dim"]),
        expansion=float(sae_ckpt["expansion"]),
        k=int(sae_ckpt["k"]),
        mode="topk",
    )
    sae.load_state_dict(sae_ckpt["sae_state_dict"])
    sae.eval()
    act_mean = sae_ckpt["act_mean"]
    act_std  = sae_ckpt["act_std"]
    n_features = sae.encoder.weight.shape[0]
    print(f"SAE loaded: {n_features} features  layer={args.layer}")

    # ── Load full V4 dataset ─────────────────────────────────────────────────
    from sae4eeg.dataset import H5PYDatasetLabeled, V4ResampleTransform
    full_ds = H5PYDatasetLabeled(str(V4_DATA_PATH), transform=V4ResampleTransform())
    subjects_all = full_ds.subjects.numpy()
    n_subjects   = len(np.unique(subjects_all))
    print(f"V4 dataset: {len(full_ds)} windows  {n_subjects} subjects")

    full_loader = DataLoader(
        full_ds, batch_size=32, shuffle=False, num_workers=4,
    )

    # ── Collect all features in one pass ─────────────────────────────────────
    print(f"\nCollecting features (layer {args.layer}) …")
    enc_all, sae_all, lbl_all = collect_features(
        encoder, sae, act_mean, act_std,
        full_loader, args.layer, args.max_windows,
    )
    n_windows = len(lbl_all)
    print(f"Collected {n_windows} windows")
    print(f"Label distribution: " +
          "  ".join(f"{V4_LABEL_NAMES[c]}={int((lbl_all==c).sum())}"
                   for c in range(10) if (lbl_all==c).any()))

    # Align subjects_all with collected windows (in case max_windows truncated)
    subjects_sub = subjects_all[:n_windows]

    # ── Subject-level train/test split ───────────────────────────────────────
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.split_seed,
    )
    all_idx = np.arange(n_windows)
    train_idx, test_idx = next(splitter.split(all_idx, groups=subjects_sub))

    n_tr_subj = len(np.unique(subjects_sub[train_idx]))
    n_te_subj = len(np.unique(subjects_sub[test_idx]))
    print(f"\nSplit (seed={args.split_seed}, test_size={args.test_size}): "
          f"train={len(train_idx)} windows / {n_tr_subj} subjects  "
          f"test={len(test_idx)} windows / {n_te_subj} subjects")

    enc_tr, sae_tr, lbl_tr = enc_all[train_idx], sae_all[train_idx], lbl_all[train_idx]
    enc_te, sae_te, lbl_te = enc_all[test_idx],  sae_all[test_idx],  lbl_all[test_idx]

    # ── Binary probes ─────────────────────────────────────────────────────────
    print(f"\n── Binary probes (pos=class, neg=normal)  seeds={args.seeds} ──")
    bin_results = binary_probes(
        enc_tr, sae_tr, lbl_tr,
        enc_te, sae_te, lbl_te,
        n_seeds=args.seeds,
    )

    out_json = out_dir / f"binary_auroc{suffix}.json"
    with open(out_json, "w") as f:
        json.dump(bin_results, f, indent=2)
    print(f"\nSaved → {out_json.relative_to(ROOT)}")

    # ── Confusion matrices ────────────────────────────────────────────────────
    print(f"\n── Multi-class confusion (abnormal-only, layer {args.layer}) ──")
    confusion_plots(
        enc_tr, sae_tr, lbl_tr,
        enc_te, sae_te, lbl_te,
        target_layer=args.layer,
        out_dir=out_dir,
        suffix=suffix,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
