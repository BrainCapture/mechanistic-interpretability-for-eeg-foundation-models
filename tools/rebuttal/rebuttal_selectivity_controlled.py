"""Rebuttal — controlled selectivity figure across all methods (no ICA).

Fair comparison: for each method, tune its intervention strength to suppress the TARGET
concept (Pathology) to a common level (AUROC ≈ 0.65), then measure how much the OFF-TARGET
concept (Age) survives. Equal target suppression ⇒ off-target preservation is a clean
selectivity axis. Methods that cannot reach the target level selectively (random-dict,
centroid) are flagged. Everything in the same raw standardized embedding space.

  uv run tools/rebuttal/rebuttal_selectivity_controlled.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from concept_erasure import LeaceEraser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cr  # noqa: E402  (camera-ready CR_OUT_DIR / CR_SEED switches)
import plot_cross_model_steering_concepts_paper as P  # noqa: E402
import rebuttal_steering_baselines as B  # noqa: E402
from steering_seed_robustness import train_sae  # noqa: E402

DEVICE = B.DEVICE
SEED = _cr.seed(42)
EXP = "sleepfm_finetuned_layer2"
TARGET = 0.65   # common Pathology suppression level
ART = _cr.out_dir(ROOT / "results" / "rebuttal")
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def inlp(X, y, n_iters=30):
    """Fixed-iteration nullspace projection (no early stop → thorough erasure)."""
    d = X.shape[1]; Pm = np.eye(d, dtype=np.float32); I = np.eye(d, dtype=np.float32)
    for _ in range(n_iters):
        clf = LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced").fit(X @ Pm, y)
        w = clf.coef_[0].astype(np.float32); nw = np.linalg.norm(w)
        if nw < 1e-9:
            break
        w /= nw; Pm = (I - np.outer(w, w)) @ Pm
    return Pm


def main():
    sc = torch.load(ROOT / "results/steering_cache" / EXP / "steering_cache.pt",
                    map_location="cpu", weights_only=False)
    emb = sc["embeddings"].float()
    mean = emb.mean(0); std = emb.std(0).clamp(min=1e-8)
    sae = train_sae((emb - mean) / std, SEED, 1.0, 8, 20, 3e-4, 2048, 2)
    print("[stage] SAE trained", flush=True)
    nf = int(sae.decoder.weight.shape[1]); rng = np.random.default_rng(SEED)
    cls = np.asarray([str(v) for v in sc["classification"]]); age = np.asarray([str(v) for v in sc["age_group"]])
    gi = rng.choice(len(emb), 4000, replace=False)
    pca = B.make_pca_source((((emb[gi] - mean) / std)).numpy())
    rand = B.make_random_source(emb.shape[1], nf, 8)
    print("[stage] PCA+rand fit", flush=True)

    def pool(fa, pos, neg):
        p = np.where(np.isin(fa, list(pos)))[0]; n = np.where(np.isin(fa, list(neg)))[0]
        k = min(3000, len(p), len(n))
        idx = np.concatenate([rng.choice(p, k, False), rng.choice(n, k, False)])
        return ((emb[idx] - mean) / std).to(DEVICE), np.array([1]*k + [0]*k)

    Xt, yt = pool(cls, P.ABN_CATS, P.NORMAL)
    Xo, yo = pool(age, {"0-3", "4-9"}, {"20-29", "30-39", "40-49", "50-59", "60+"})
    tgt = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xt.cpu().numpy(), yt)
    off = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xo.cpu().numpy(), yo)
    a = lambda pr, X, y: float(roc_auc_score(y, pr.decision_function(X.cpu().numpy() if torch.is_tensor(X) else X)))
    base_t, base_o = a(tgt, Xt, yt), a(off, Xo, yo)
    cav = P.fit_probe(Xt, yt).w; cav = cav / (cav.norm() + 1e-8)

    # each method: strength s in [0,1] -> (Xt', Xo') suppressing Pathology at strength s
    def dict_fn(src):
        with torch.no_grad():
            donor = src.encode(Xt)[yt == 0].mean(0)
        order = np.argsort((cav @ src.W_dec).abs().cpu().numpy())[::-1].copy()
        def f(s, X):
            with torch.no_grad():
                z = src.encode(X); n = int(round(s * src.K))
                return src.decode(P.substitute(z, order[:n], donor) if n else z)
        return f
    donor_proj = (Xt @ cav)[yt == 0].mean()
    cN = Xt[yt == 0].mean(0)
    leace = LeaceEraser.fit(Xt.cpu(), torch.tensor(yt).float())
    Pnp = inlp(Xt.cpu().numpy(), yt)
    # amnesic's goal is to make the concept undecodable by ANY linear probe — also null the
    # measured probe's direction so residual signal doesn't survive solver/weighting mismatch.
    w0 = tgt.coef_[0].astype(np.float32); w0 /= np.linalg.norm(w0) + 1e-9
    Pnp = Pnp @ (np.eye(len(w0), dtype=np.float32) - np.outer(w0, w0))
    Pinlp = torch.tensor(Pnp, device=DEVICE); print("[stage] INLP fit", flush=True)

    fns = {
        "SAE": dict_fn(B.make_sae_source(sae)),
        "PCA": dict_fn(pca),
        "random-dict": dict_fn(rand),
        "dense-CAV": lambda s, X: X + s * (donor_proj - X @ cav).unsqueeze(1) * cav.unsqueeze(0),
        "centroid": lambda s, X: (1 - s) * X + s * cN,
        "LEACE": lambda s, X: (1 - s) * X + s * leace(X.cpu()).to(DEVICE),
        "amnesic": lambda s, X: (1 - s) * X + s * (X @ Pinlp),
    }
    grid = np.linspace(0, 1, 21)
    res = {}
    for name, f in fns.items():
        ts = np.array([a(tgt, f(s, Xt), yt) for s in grid])
        # first strength that reaches the target level; else the strongest (min target)
        below = np.where(ts <= TARGET)[0]
        i = below[0] if len(below) else int(np.argmin(ts))
        off_auc = a(off, f(grid[i], Xo), yo)
        res[name] = {"reached": bool(len(below)), "strength": float(grid[i]),
                     "target_auc": float(ts[i]), "off_auc": off_auc}
        print(f"  {name:12s} target→{ts[i]:.3f} (s={grid[i]:.2f}, reached={len(below)>0})  Age→{off_auc:.3f}", flush=True)

    (ART / "selectivity_controlled.json").write_text(json.dumps(
        {"target_level": TARGET, "baseline": {"Pathology": base_t, "Age": base_o}, "methods": res}, indent=2))

    # ── figure: Age preservation at equal Pathology suppression ──
    from mecheeg.nature_fig_style import set_nature_style, nature_figure, nature_palette, save_nature_figure
    set_nature_style()
    order = ["LEACE", "amnesic", "SAE", "PCA", "dense-CAV", "random-dict", "centroid"]
    order = [m for m in order if m in res]
    vals = [res[m]["off_auc"] for m in order]
    reached = [res[m]["reached"] for m in order]
    cols = ["#1e8449" if r else "#b03a2e" for r in reached]
    fig, ax = nature_figure(width="single", height_mm=62)
    x = np.arange(len(order))
    ax.bar(x, vals, 0.7, color=cols)
    ax.axhline(base_o, color="0.5", lw=0.6, ls=":")
    ax.text(len(order)-0.5, base_o+0.01, "Age unaffected", fontsize=4.6, ha="right", color="0.4")
    ax.axhline(0.5, color="0.6", lw=0.5, ls="-")
    for i, (m, r) in enumerate(zip(order, reached)):
        if not r:
            ax.text(i, 0.40, "target not\nreached", ha="center", va="bottom",
                    fontsize=4.5, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=5.5)
    ax.set_ylabel("Age AUROC (off-target preserved)"); ax.set_ylim(0.3, 1.02)
    ax.set_title("Suppress Pathology to a common level, preserve Age", loc="left", fontsize=6.5)
    save_nature_figure(fig, str(FIG / "fig_selectivity_controlled.pdf"))
    save_nature_figure(fig, str(FIG / "fig_selectivity_controlled.png"))
    print("wrote fig_selectivity_controlled")


if __name__ == "__main__":
    main()
