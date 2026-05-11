"""Demographic isolation analysis for SAE features.

Explores how well SAE features separate age groups and sex.

Outputs (saved to results/demographics_analysis/<experiment>/):
  1. enrichment_heatmaps.png  — feature × age-group and feature × sex heatmaps
  2. spectral_by_age.png      — differential spectra for top age-enriched features
  3. spectral_by_sex.png      — differential spectra for top sex-enriched features
  4. umap_demographics.png    — existing token UMAP coloured by age and sex
  5. probe_results.png        — linear probe accuracy + confusion matrices

Usage:
  uv run tools/explore_demographics.py --experiment sleepfm_finetuned_layer2
  uv run tools/explore_demographics.py --experiment sleepfm_finetuned_layer2 --skip-probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ── Age ordering ──────────────────────────────────────────────────────────────
def _age_key(v: str) -> int:
    try:
        return int(v.split("-")[0].replace("+", ""))
    except ValueError:
        return 999

AGE_ORDER = ["0-3", "4-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
SEX_ORDER = ["female", "male"]

BAND_COLORS = {
    "delta":     "#5C85D6",
    "theta":     "#85C17E",
    "alpha":     "#E8C84A",
    "low-beta":  "#E07B39",
    "high-beta": "#C0392B",
    "gamma":     "#9B59B6",
}


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(experiment: str) -> dict:
    p = ROOT / "results" / "experiments" / experiment / "app_cache.pt"
    if not p.exists():
        raise FileNotFoundError(f"No app_cache.pt for {experiment}. Run build_app_cache.py first.")
    return torch.load(str(p), map_location="cpu", weights_only=False)


def load_meta(experiment: str) -> dict:
    p = ROOT / "results" / "experiments" / experiment / "metadata.json"
    return json.loads(p.read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Data prep
# ─────────────────────────────────────────────────────────────────────────────

def build_enrichment_matrix(
    feature_meta_enrichment: list[dict],
    field: str,
    categories: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ratio_matrix, pval_matrix) each shape (n_features, n_cats)."""
    n_feat = len(feature_meta_enrichment)
    n_cats = len(categories)
    ratios = np.zeros((n_feat, n_cats), dtype=np.float32)
    pvals  = np.ones( (n_feat, n_cats), dtype=np.float32)
    cat_idx = {c: i for i, c in enumerate(categories)}
    for fi, fenr in enumerate(feature_meta_enrichment):
        for cat, ratio, pval in fenr.get(field, []):
            if cat in cat_idx:
                ratios[fi, cat_idx[cat]] = ratio
                pvals[ fi, cat_idx[cat]] = pval
    return ratios, pvals


def sort_features_by_peak(ratios: np.ndarray, categories: list[str]) -> np.ndarray:
    """Sort features: primary key = which category they peak on, secondary = ratio."""
    peak_cat = np.argmax(ratios, axis=1)
    peak_val = ratios[np.arange(len(ratios)), peak_cat]
    order = np.lexsort((- peak_val, peak_cat))
    return order


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Enrichment heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def plot_enrichment_heatmaps(cache: dict, out_dir: Path) -> None:
    enr = cache["feature_meta_enrichment"]
    n_feat = len(enr)

    age_cats = [c for c in AGE_ORDER if any(
        c in [x[0] for x in fe.get("age_group", [])] for fe in enr
    )]
    sex_cats = [c for c in SEX_ORDER if any(
        c in [x[0] for x in fe.get("gender", [])] for fe in enr
    )]

    age_rat, age_p = build_enrichment_matrix(enr, "age_group", age_cats)
    sex_rat, sex_p = build_enrichment_matrix(enr, "gender",    sex_cats)

    age_order = sort_features_by_peak(age_rat, age_cats)
    sex_order = sort_features_by_peak(sex_rat, sex_cats)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10),
                             gridspec_kw={"width_ratios": [len(age_cats), len(sex_cats)]})
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("#111111")

    vmax = 4.0
    cmap = plt.get_cmap("RdBu_r")

    for ax, ratios, pvals, order, cats, title in [
        (axes[0], age_rat, age_p, age_order, age_cats, "Age group enrichment"),
        (axes[1], sex_rat, sex_p, sex_order, sex_cats, "Sex enrichment"),
    ]:
        mat = ratios[order]
        p_m = pvals[order]
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=vmax)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm,
                       interpolation="nearest")
        # Mark significant cells
        for fi in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                if p_m[fi, ci] < 0.05:
                    ax.plot(ci, fi, ".", color="white", markersize=2, alpha=0.6)

        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, color="white", fontsize=9, rotation=35, ha="right")
        ax.set_yticks([])
        ax.set_ylabel(f"Features (n={n_feat}, sorted by peak)", color="white", fontsize=9)
        ax.set_title(title, color="white", fontsize=12, pad=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Enrichment ×  (1 = global baseline)", color="white", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
        ax.axvline(-0.5, color="#333", linewidth=0.5)

    plt.tight_layout()
    out = out_dir / "enrichment_heatmaps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 & 3 — Top spectral signatures per demographic
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectral_by_field(
    cache: dict,
    field: str,
    categories: list[str],
    title: str,
    out_path: Path,
    top_n: int = 3,
) -> None:
    enr     = cache["feature_meta_enrichment"]
    amp_diff = cache["feature_amp_diff"].numpy()   # (n_feat, n_bins)
    amp_base = cache["feature_amp_baseline"].numpy()
    freqs    = cache["feature_freqs"]
    band_names = cache["feature_band_names"]
    band_deltas = cache["feature_band_deltas"].numpy()

    cats = [c for c in categories if any(
        c in [x[0] for x in fe.get(field, [])] for fe in enr
    )]
    n_cats = len(cats)
    if n_cats == 0:
        return

    ratios, pvals = build_enrichment_matrix(enr, field, cats)
    n_feat = len(enr)

    fig, axes = plt.subplots(
        n_cats, top_n,
        figsize=(4 * top_n, 3 * n_cats),
        squeeze=False,
    )
    fig.patch.set_facecolor("#111111")

    for ci, cat in enumerate(cats):
        # rank features by enrichment for this category
        col = ratios[:, ci]
        sig = pvals[:, ci] < 0.05
        ranked = np.argsort(-col)
        top_feats = [fi for fi in ranked if sig[fi]][:top_n]
        # pad with non-significant if needed
        if len(top_feats) < top_n:
            top_feats += [fi for fi in ranked if fi not in top_feats][: top_n - len(top_feats)]

        for ti, fi in enumerate(top_feats):
            ax = axes[ci][ti]
            ax.set_facecolor("#111111")
            base = np.expm1(amp_base)
            active = np.expm1(amp_base + amp_diff[fi])
            ax.plot(freqs, base,   color="#555555", linewidth=1, linestyle="--", label="Baseline")
            ax.plot(freqs, active, color="#E53935", linewidth=1.5,             label="Feature ON")
            ax.fill_between(freqs, base, active,
                            where=active > base, alpha=0.2, color="#E53935")
            ax.fill_between(freqs, base, active,
                            where=active < base, alpha=0.2, color="#4fc3f7")

            # Band shading
            CLINICAL_BANDS = {
                "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
                "low-beta": (13.0, 20.0), "high-beta": (20.0, 30.0), "gamma": (30.0, 45.0),
            }
            for bn, (f0, f1) in CLINICAL_BANDS.items():
                ax.axvspan(f0, f1, alpha=0.04, color=BAND_COLORS.get(bn, "#888"),
                           label=None)

            sig_str = "★" if pvals[fi, ci] < 0.05 else ""
            ax.set_title(f"F{fi}  ×{col[fi]:.2f}{sig_str}  p={pvals[fi,ci]:.1e}",
                         color="white", fontsize=8, pad=3)
            ax.tick_params(colors="white", labelsize=7)
            ax.spines[:].set_color("#444")
            ax.set_xlim(freqs[0], freqs[-1])
            if ti == 0:
                ax.set_ylabel(cat, color="white", fontsize=9, fontweight="bold")
            else:
                ax.set_yticklabels([])
            if ci == n_cats - 1:
                ax.set_xlabel("Hz", color="white", fontsize=8)

    fig.suptitle(title, color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — UMAP coloured by age and sex
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap_demographics(cache: dict, out_dir: Path) -> None:
    coords = cache["codebook_umap_coords"]   # (N, 2)
    meta   = cache["codebook_umap_meta"]
    age_arr = np.array(meta["age_group"])
    sex_arr = np.array(meta["gender"])
    xy = coords if isinstance(coords, np.ndarray) else coords.numpy()

    # subsample for speed if very large
    rng = np.random.default_rng(42)
    n = len(xy)
    idx = rng.choice(n, min(n, 8000), replace=False)
    xy, age_arr, sex_arr = xy[idx], age_arr[idx], sex_arr[idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#111111")

    # ── Age panel ──
    ax = axes[0]
    ax.set_facecolor("#111111")
    known_ages = [a for a in AGE_ORDER if a in age_arr]
    age_cmap = cm.get_cmap("plasma", len(known_ages))
    age_color_map = {v: age_cmap(i / max(len(known_ages) - 1, 1)) for i, v in enumerate(known_ages)}
    age_color_map["unknown"] = (0.4, 0.4, 0.4, 0.3)

    for age in known_ages + ["unknown"]:
        m = age_arr == age
        if m.sum() == 0:
            continue
        c = age_color_map[age]
        alpha = 0.15 if age == "unknown" else 0.5
        ax.scatter(xy[m, 0], xy[m, 1], c=[c], s=2, alpha=alpha,
                   label=age, rasterized=True)

    ax.set_title("UMAP — Age group", color="white", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_color("#333")
    leg = ax.legend(title="Age", fontsize=7, title_fontsize=8,
                    loc="lower right", markerscale=4,
                    facecolor="#222", labelcolor="white")
    leg.get_title().set_color("white")

    # ── Sex panel ──
    ax = axes[1]
    ax.set_facecolor("#111111")
    sex_palette = {"female": "#fd8d3c", "male": "#6baed6"}
    for sex in SEX_ORDER:
        m = sex_arr == sex
        if m.sum() == 0:
            continue
        ax.scatter(xy[m, 0], xy[m, 1], c=sex_palette[sex], s=2, alpha=0.5,
                   label=sex, rasterized=True)

    ax.set_title("UMAP — Sex", color="white", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_color("#333")
    leg = ax.legend(title="Sex", fontsize=8, title_fontsize=8,
                    loc="lower right", markerscale=4,
                    facecolor="#222", labelcolor="white")
    leg.get_title().set_color("white")

    plt.tight_layout()
    out = out_dir / "umap_demographics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Linear probe (requires model forward pass)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sae_features(meta_json: dict, max_tokens: int = 30_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run encoder + SAE on validation set. Returns (z_all, age_labels, sex_labels)."""
    from sae4eeg.encoders import load_encoder
    from sae4eeg.sae import SparseAutoencoder
    from sae4eeg.dataset import get_dataloaders, StandardizeLabel

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATHS = {
        "sleepfm":           ROOT / "data" / "D4-v3-preprocessed-v2",
        "sleepfm_finetuned": ROOT / "data" / "D4-v3-preprocessed-v2",
    }
    encoder_name = meta_json["encoder"]
    data_path = DATA_PATHS.get(encoder_name, ROOT / "data" / "D4-v3-preprocessed-v2")

    print(f"  Loading encoder: {encoder_name}")
    kwargs = {}
    if "weights_path" in meta_json:
        kwargs["weights_path"] = str(ROOT / meta_json["weights_path"])
    encoder = load_encoder(encoder_name, **kwargs)
    encoder.to(DEVICE).eval()

    sae_ckpt = torch.load(ROOT / meta_json["sae_checkpoint"],
                          weights_only=False, map_location="cpu")
    embed_dim = sae_ckpt.get("embed_dim", meta_json["embed_dim"])
    sae = SparseAutoencoder(embed_dim, expansion=meta_json["expansion"],
                            mode="topk", k=meta_json["k"])
    sae.load_state_dict(sae_ckpt["sae_state_dict"])
    act_mean = sae_ckpt["act_mean"].to(DEVICE)
    act_std  = sae_ckpt["act_std"].to(DEVICE)
    sae.to(DEVICE).eval()

    gen = get_dataloaders(train_path=str(data_path), transformer=StandardizeLabel())
    _, _, val_loader, _ = next(gen)

    from sae4eeg.sae import ActivationExtractor
    target_layer = meta_json["target_layer"]
    extractor    = ActivationExtractor(encoder.model, encoder.get_hookable_layers())

    import h5py as _h5
    from tqdm import tqdm

    tokens_per_window = None
    embed_list = []
    n_collected = 0

    dataset = val_loader.dataset
    file_indices  = dataset.index_map["file_indices"].numpy()
    local_indices = dataset.index_map["local_indices"].numpy()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="  Collecting")):
            x = batch[0].to(DEVICE)
            extractor.clear()
            encoder.model(x)
            acts = extractor.get_activations()[target_layer]  # (B, T, D) on cpu
            acts = acts.to(DEVICE)
            B, T, D = acts.shape
            if tokens_per_window is None:
                tokens_per_window = T
            acts_flat = acts.reshape(B * T, D)
            x_norm = (acts_flat - act_mean) / act_std.clamp(min=1e-8)
            z = sae.encode(x_norm).cpu()  # (B*T, n_features)
            embed_list.append(z)
            n_collected += B * T
            if n_collected >= max_tokens:
                break

    z_all = torch.cat(embed_list, dim=0)[:max_tokens].numpy()
    n_windows = (len(z_all) + tokens_per_window - 1) // tokens_per_window
    n_windows = min(n_windows, len(dataset))

    # Load metadata for the collected windows
    age_win = np.full(n_windows, "unknown", dtype=object)
    sex_win = np.full(n_windows, "unknown", dtype=object)
    fis = file_indices[:n_windows]
    lis = local_indices[:n_windows]
    for file_val in np.unique(fis):
        positions    = np.where(fis == file_val)[0]
        sort_args    = np.argsort(lis[positions])
        sorted_pos   = positions[sort_args]
        sorted_local = lis[sorted_pos].tolist()
        with _h5.File(dataset.paths[int(file_val)], "r") as f:
            for field, arr in [("age_group", age_win), ("gender", sex_win)]:
                try:
                    raw = f["metadata"][field][sorted_local]
                    for pos, v in zip(sorted_pos, raw):
                        arr[pos] = v.decode("utf-8").strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip()
                except Exception:
                    pass

    age_tok = np.repeat(age_win, tokens_per_window)[:len(z_all)]
    sex_tok = np.repeat(sex_win, tokens_per_window)[:len(z_all)]
    extractor.remove_hooks()
    print(f"  Collected {len(z_all)} tokens")
    return z_all, age_tok, sex_tok


def plot_probe_results(z_all: np.ndarray, age_tok: np.ndarray, sex_tok: np.ndarray,
                       out_dir: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import confusion_matrix, balanced_accuracy_score
    from sklearn.preprocessing import LabelEncoder

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#111111")

    for ax in axes:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")

    results = {}
    for field, labels_raw, ax_bar, ax_cm, cats_order in [
        ("Age group", age_tok, axes[0], axes[1], AGE_ORDER),
        ("Sex",       sex_tok, axes[2], None,   SEX_ORDER),
    ]:
        # filter unknowns
        mask   = np.array([str(v) not in ("unknown", "") for v in labels_raw])
        z_filt = z_all[mask]
        l_filt = np.array([str(v) for v in labels_raw[mask]])
        if len(np.unique(l_filt)) < 2:
            print(f"  [warn] Not enough classes for {field} probe")
            continue

        le = LabelEncoder()
        y  = le.fit_transform(l_filt)
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        accs = []
        cms  = []
        for train_idx, test_idx in sss.split(z_filt, y):
            clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                     random_state=42)
            clf.fit(z_filt[train_idx], y[train_idx])
            y_pred = clf.predict(z_filt[test_idx])
            accs.append(balanced_accuracy_score(y[test_idx], y_pred))
            cms.append(confusion_matrix(y[test_idx], y_pred,
                                        labels=range(len(le.classes_))))

        mean_cm  = np.mean(cms, axis=0)
        mean_acc = np.mean(accs)
        chance   = 1 / len(le.classes_)
        results[field] = {"acc": mean_acc, "chance": chance, "classes": le.classes_.tolist()}
        print(f"  {field} probe: balanced acc = {mean_acc:.3f}  (chance = {chance:.3f})")

        # normalise CM by row
        row_sums = mean_cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm  = mean_cm / row_sums

        # order classes
        ordered = [c for c in cats_order if c in le.classes_]
        ordered += [c for c in le.classes_ if c not in ordered]
        order_idx = [le.transform([c])[0] for c in ordered]
        cm_ord = cm_norm[np.ix_(order_idx, order_idx)]

        ax_cm_use = ax_cm if ax_cm is not None else axes[2]
        im = ax_cm_use.imshow(cm_ord, cmap="Blues", vmin=0, vmax=1,
                              aspect="auto", interpolation="nearest")
        ax_cm_use.set_xticks(range(len(ordered)))
        ax_cm_use.set_xticklabels(ordered, rotation=40, ha="right", fontsize=8, color="white")
        ax_cm_use.set_yticks(range(len(ordered)))
        ax_cm_use.set_yticklabels(ordered, fontsize=8, color="white")
        for i in range(len(ordered)):
            for j in range(len(ordered)):
                ax_cm_use.text(j, i, f"{cm_ord[i,j]:.2f}", ha="center", va="center",
                               fontsize=7, color="white" if cm_ord[i,j] < 0.5 else "black")
        ax_cm_use.set_title(
            f"{field} confusion matrix\nbal. acc = {mean_acc:.3f}  (chance = {chance:.3f})",
            color="white", fontsize=10,
        )
        cb = fig.colorbar(im, ax=ax_cm_use, fraction=0.04, pad=0.02)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    # Bar chart: accuracy vs chance for both fields
    ax = axes[0]
    fields = list(results.keys())
    accs   = [results[f]["acc"]    for f in fields]
    chances = [results[f]["chance"] for f in fields]
    x = np.arange(len(fields))
    ax.bar(x - 0.2, accs,    0.35, label="SAE probe",    color="#E53935", alpha=0.85)
    ax.bar(x + 0.2, chances, 0.35, label="Chance",       color="#555555", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(fields, color="white", fontsize=10)
    ax.set_ylabel("Balanced accuracy", color="white", fontsize=9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelcolor="white")
    ax.set_title("Linear probe accuracy\n(5-fold, LR on SAE z-features)",
                 color="white", fontsize=10)
    leg = ax.legend(facecolor="#222", labelcolor="white", fontsize=9)

    plt.tight_layout()
    out = out_dir / "probe_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Band power profile per age group (mean enrichment × band delta)
# ─────────────────────────────────────────────────────────────────────────────

def plot_band_profile_by_demo(cache: dict, out_dir: Path) -> None:
    enr        = cache["feature_meta_enrichment"]
    band_deltas = cache["feature_band_deltas"].numpy()   # (n_feat, n_bands)
    band_names  = cache["feature_band_names"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#111111")

    for ax, field, cats_order, title in [
        (axes[0], "age_group", AGE_ORDER, "Age group — mean band Δ for top-enriched features"),
        (axes[1], "gender",    SEX_ORDER, "Sex — mean band Δ for top-enriched features"),
    ]:
        ax.set_facecolor("#111111")
        cats = [c for c in cats_order if any(
            c in [x[0] for x in fe.get(field, [])] for fe in enr
        )]
        ratios, _ = build_enrichment_matrix(enr, field, cats)

        # For each category: compute enrichment-weighted average of band_deltas
        profiles = np.zeros((len(cats), len(band_names)))
        for ci, cat in enumerate(cats):
            weights = ratios[:, ci].clip(min=0)
            if weights.sum() > 0:
                profiles[ci] = (weights[:, None] * band_deltas).sum(0) / weights.sum()

        x = np.arange(len(band_names))
        width = 0.8 / max(len(cats), 1)
        if field == "age_group":
            cmap_fn = cm.get_cmap("plasma", len(cats))
            cat_colors = [cmap_fn(i / max(len(cats) - 1, 1)) for i in range(len(cats))]
        else:
            sex_pal = {"female": "#fd8d3c", "male": "#6baed6"}
            cat_colors = [sex_pal.get(c, "#888") for c in cats]

        for ci, (cat, color) in enumerate(zip(cats, cat_colors)):
            offset = (ci - len(cats) / 2 + 0.5) * width
            ax.bar(x + offset, profiles[ci], width * 0.9,
                   color=color, alpha=0.85, label=cat)

        ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(band_names, color="white", fontsize=9, rotation=20)
        ax.yaxis.set_tick_params(labelcolor="white")
        ax.set_ylabel("Enrichment-weighted Δ amplitude", color="white", fontsize=9)
        ax.set_title(title, color="white", fontsize=10)
        ax.spines[:].set_color("#444")
        leg = ax.legend(fontsize=7, facecolor="#222", labelcolor="white",
                        ncol=2 if len(cats) > 4 else 1)

    plt.tight_layout()
    out = out_dir / "band_profile_by_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Top-3 features per demographic group (the isolation plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_features_per_group(cache: dict, out_dir: Path, top_n: int = 3) -> None:
    """For each age group and sex value: show the top-N features and their
    full enrichment profile across all categories — makes selectivity obvious."""

    enr        = cache["feature_meta_enrichment"]
    band_deltas = cache["feature_band_deltas"].numpy()   # (n_feat, n_bands)
    band_names  = cache["feature_band_names"]
    n_feat      = len(enr)

    def _make_panel(
        ax: plt.Axes,
        field: str,
        all_cats: list[str],
        focal_cat: str,
        feature_idx: int,
        ratio_val: float,
        pval: float,
        cat_color_map: dict[str, str],
    ) -> None:
        """Draw enrichment profile for one feature, highlighting the focal category."""
        ratios_row, pvals_row = build_enrichment_matrix([enr[feature_idx]], field, all_cats)
        vals = ratios_row[0]
        ps   = pvals_row[0]

        colors = [
            cat_color_map[cat] if cat == focal_cat
            else (*mcolors.to_rgb(cat_color_map[cat]), 0.2)
            for cat in all_cats
        ]

        ax.bar(range(len(all_cats)), vals, color=colors, width=0.7, zorder=2)
        ax.axhline(1.0, color="#555", linewidth=0.8, linestyle="--", zorder=1)

        for i, (v, p) in enumerate(zip(vals, ps)):
            if p < 0.05:
                ax.text(i, v + 0.05, "★", ha="center", va="bottom",
                        fontsize=7, color="white", zorder=3)

        ax.set_xticks(range(len(all_cats)))
        ax.set_xticklabels(all_cats, fontsize=6.5, color="white", rotation=40, ha="right")
        ax.set_ylim(0, max(vals.max() * 1.3, 2.2))
        ax.yaxis.set_tick_params(labelsize=7, labelcolor="white")
        ax.set_facecolor("#111111")
        ax.spines[:].set_color("#333")

        sig = "★" if pval < 0.05 else ""
        ax.set_title(f"F{feature_idx}  ×{ratio_val:.2f}{sig}", color="white", fontsize=9, pad=3)

    # ── Build consistent color maps ───────────────────────────────────────────
    _age_present = [c for c in AGE_ORDER if any(
        c in [x[0] for x in fe.get("age_group", [])] for fe in enr
    )]
    _age_grad = plt.colormaps["plasma"].resampled(max(len(_age_present), 2))
    _age_color_map = {
        c: mcolors.to_hex(_age_grad(i / max(len(_age_present) - 1, 1)))
        for i, c in enumerate(_age_present)
    }
    _sex_color_map = {"female": "#fd8d3c", "male": "#6baed6"}

    field_color_maps = {
        "age_group": _age_color_map,
        "gender":    _sex_color_map,
    }

    for field, all_cats, field_label, fname in [
        ("age_group",        AGE_ORDER, "Age group",        "top_features_age.png"),
        ("gender",           SEX_ORDER, "Sex",              "top_features_sex.png"),
        ("indication_group", None,      "Indication group", "top_features_indication.png"),
        ("medication_group", None,      "Medication group", "top_features_medication.png"),
    ]:
        # Determine present categories (use canonical order if given, else sort by freq)
        if all_cats is not None:
            present = [c for c in all_cats if any(
                c in [x[0] for x in fe.get(field, [])] for fe in enr
            )]
        else:
            _all_seen = {}
            for fe in enr:
                for cat, ratio, _ in fe.get(field, []):
                    _all_seen[cat] = _all_seen.get(cat, 0) + ratio
            present = sorted(_all_seen, key=lambda c: -_all_seen[c])
        if not present:
            continue

        # Build color map for this field if not pre-defined
        if field not in field_color_maps:
            _grad = plt.colormaps["tab20"].resampled(max(len(present), 2))
            field_color_maps[field] = {
                c: mcolors.to_hex(_grad(i / max(len(present) - 1, 1)))
                for i, c in enumerate(present)
            }
        cat_color_map = field_color_maps[field]
        ratios, pvals = build_enrichment_matrix(enr, field, present)
        n_groups = len(present)

        fig, axes = plt.subplots(
            n_groups, top_n,
            figsize=(top_n * 2.8, n_groups * 2.4),
            squeeze=False,
        )
        fig.patch.set_facecolor("#111111")
        fig.suptitle(
            f"Top-{top_n} SAE features per {field_label}\n"
            f"(bars show enrichment across all {field_label.lower()} groups — "
            f"focal group in brackets)",
            color="white", fontsize=11, y=1.01,
        )

        for gi, focal_cat in enumerate(present):
            ci = present.index(focal_cat)
            col_ratios = ratios[:, ci]
            col_pvals  = pvals[:, ci]
            # rank by enrichment, prefer significant
            ranked = np.argsort(-col_ratios)
            top_feats = ranked[:top_n]

            for ti, fi in enumerate(top_feats):
                ax = axes[gi][ti]
                _make_panel(
                    ax, field, present, focal_cat,
                    int(fi),
                    float(col_ratios[fi]), float(col_pvals[fi]),
                    cat_color_map,
                )
                if ti == 0:
                    ax.set_ylabel(focal_cat, color="white", fontsize=9, fontweight="bold")
                if gi == 0:
                    ax.set_xlabel(f"Rank {ti + 1}", color="#aaaaaa", fontsize=8,
                                  labelpad=2)
                    ax.xaxis.set_label_position("top")

        plt.tight_layout()
        out = out_dir / fname
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111111")
        plt.close(fig)
        print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="sleepfm_finetuned_layer2")
    parser.add_argument("--skip-probe", action="store_true",
                        help="Skip linear probe (no model forward pass needed)")
    parser.add_argument("--max-tokens", type=int, default=30_000)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "demographics_analysis" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Demographics analysis: {args.experiment}")
    print(f"  Output: {out_dir}")
    print(f"{'='*64}\n")

    cache    = load_cache(args.experiment)
    meta_json = load_meta(args.experiment)

    print("[1/5] Enrichment heatmaps ...")
    plot_enrichment_heatmaps(cache, out_dir)

    print("[2/5] Spectral signatures by age group ...")
    plot_spectral_by_field(
        cache, "age_group", AGE_ORDER,
        "Top-enriched feature spectra per age group",
        out_dir / "spectral_by_age.png",
    )

    print("[3/5] Spectral signatures by sex ...")
    plot_spectral_by_field(
        cache, "gender", SEX_ORDER,
        "Top-enriched feature spectra per sex",
        out_dir / "spectral_by_sex.png",
    )

    print("[4/5] UMAP coloured by demographics ...")
    plot_umap_demographics(cache, out_dir)

    print("[5/5] Band power profile by demographic ...")
    plot_band_profile_by_demo(cache, out_dir)

    print("[6/6] Top-3 features per demographic group ...")
    plot_top_features_per_group(cache, out_dir)

    if not args.skip_probe:
        print("[6/6] Linear probe (loading model + data) ...")
        z_all, age_tok, sex_tok = compute_sae_features(meta_json, args.max_tokens)
        plot_probe_results(z_all, age_tok, sex_tok, out_dir)
    else:
        print("[6/6] Skipping linear probe (--skip-probe)")

    print(f"\n✅  Done — figures in {out_dir}/")


if __name__ == "__main__":
    main()
