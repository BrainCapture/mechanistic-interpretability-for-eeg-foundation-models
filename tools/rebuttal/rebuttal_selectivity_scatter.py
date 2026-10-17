"""Rebuttal — one unified selectivity figure across ALL methods (steerers + erasers).

Steering (Δ̃) and erasure (LEACE/amnesic) use different metrics, so we compare them on the
axis that is meaningful for both: suppress the TARGET concept (Pathology) and measure how much
the OFF-TARGET concept (Age) survives. Everything is measured in the same raw standardized
embedding space (probes fit on clean embeddings), so the 8 methods are directly comparable.

  x = Pathology AUROC after intervention (lower = target more suppressed)
  y = Age AUROC after the SAME intervention (higher = off-target better preserved = more selective)

  uv run tools/rebuttal/rebuttal_selectivity_scatter.py
"""
from __future__ import annotations

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
import plot_cross_model_steering_concepts_paper as P  # noqa: E402
import rebuttal_steering_baselines as B  # noqa: E402
from steering_seed_robustness import train_sae  # noqa: E402

DEVICE = B.DEVICE
SEED = 42
EXP = "sleepfm_finetuned_layer2"
CLAMP_FRAC = 0.5
FIG = ROOT / "results" / "rebuttal" / "figures"


def inlp(X, y, n_iters=35, tol=0.52):
    d = X.shape[1]; Pm = np.eye(d, dtype=np.float32); I = np.eye(d, dtype=np.float32)
    for _ in range(n_iters):
        Xp = X @ Pm
        clf = LogisticRegression(max_iter=300, solver="liblinear", class_weight="balanced").fit(Xp, y)
        if clf.score(Xp, y) < tol:
            break
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
    am, as_ = mean.to(DEVICE), std.to(DEVICE)
    sae = train_sae((emb - mean) / std, SEED, 1.0, 8, 20, 3e-4, 2048, 2)
    print("[stage] SAE trained", flush=True)
    nf = int(sae.decoder.weight.shape[1]); ncl = int(CLAMP_FRAC * nf)
    rng = np.random.default_rng(SEED)

    cls = np.asarray([str(v) for v in sc["classification"]])
    age = np.asarray([str(v) for v in sc["age_group"]])
    # global fit for PCA/ICA/random (small sample; FastICA capped for speed)
    gi = rng.choice(len(emb), min(4000, len(emb)), replace=False)
    Xg = (((emb[gi] - mean) / std)).numpy()
    pca = B.make_pca_source(Xg); print("[stage] PCA fit", flush=True)
    from sklearn.decomposition import FastICA
    _ica = FastICA(n_components=Xg.shape[1], random_state=SEED, max_iter=150, tol=1e-2,
                   whiten="unit-variance").fit(Xg)
    comp = torch.as_tensor(_ica.components_, dtype=torch.float32, device=DEVICE)
    mix = torch.as_tensor(_ica.mixing_, dtype=torch.float32, device=DEVICE)
    im = torch.as_tensor(_ica.mean_, dtype=torch.float32, device=DEVICE)
    ica = B.DictSource("ICA", lambda X: (X - im) @ comp.T, lambda z: z @ mix.T + im, mix, k=None)
    print("[stage] ICA fit", flush=True)
    rand = B.make_random_source(emb.shape[1], nf, 8)

    def pool(field_arr, pos, neg):
        p = np.where(np.isin(field_arr, list(pos)))[0]; n = np.where(np.isin(field_arr, list(neg)))[0]
        k = min(3000, len(p), len(n))
        idx = np.concatenate([rng.choice(p, k, False), rng.choice(n, k, False)])
        y = np.array([1] * k + [0] * k)
        Xstd = ((emb[idx] - mean) / std).to(DEVICE)
        return Xstd, y

    Xt, yt = pool(cls, P.ABN_CATS, P.NORMAL)                                   # Pathology (target)
    Xo, yo = pool(age, {"0-3", "4-9"}, {"20-29", "30-39", "40-49", "50-59", "60+"})  # Age (off)

    # probes on clean (raw standardized) embeddings
    tgt_probe = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xt.cpu().numpy(), yt)
    off_probe = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xo.cpu().numpy(), yo)
    auc = lambda probe, X, y: float(roc_auc_score(y, probe.decision_function(X.cpu().numpy() if torch.is_tensor(X) else X)))
    base_t, base_o = auc(tgt_probe, Xt, yt), auc(off_probe, Xo, yo)

    # Pathology direction / donor for the target concept
    cav = P.fit_probe(Xt, yt).w; cav = cav / (cav.norm() + 1e-8)

    def dict_intervene(src, X):
        with torch.no_grad():
            z = src.encode(X); donor = src.encode(Xt)[yt == 0].mean(0)
            order = np.argsort((cav @ src.W_dec).abs().cpu().numpy())[::-1][:ncl].copy()
            return src.decode(P.substitute(z, order, donor))

    def cav_intervene(X):
        donor = (Xt @ cav)[yt == 0].mean()
        return X + (donor - X @ cav).unsqueeze(1) * cav.unsqueeze(0)

    def centroid_intervene(X):
        c = Xt[yt == 0].mean(0)
        return 0.0 * X + c  # full push to Normal centroid (wrecking-ball limit)

    leace = LeaceEraser.fit(Xt.cpu(), torch.tensor(yt).float()); print("[stage] LEACE fit", flush=True)
    Pinlp = inlp(Xt.cpu().numpy(), yt); print("[stage] INLP fit", flush=True)

    methods = {}
    for name, src in [("SAE", B.make_sae_source(sae)), ("PCA", pca), ("ICA", ica), ("RandDict", rand)]:
        methods[name] = (auc(tgt_probe, dict_intervene(src, Xt), yt), auc(off_probe, dict_intervene(src, Xo), yo))
    methods["dense-CAV"] = (auc(tgt_probe, cav_intervene(Xt), yt), auc(off_probe, cav_intervene(Xo), yo))
    methods["centroid"] = (auc(tgt_probe, centroid_intervene(Xt), yt), auc(off_probe, centroid_intervene(Xo), yo))
    methods["LEACE"] = (auc(tgt_probe, leace(Xt.cpu()).to(DEVICE), yt), auc(off_probe, leace(Xo.cpu()).to(DEVICE), yo))
    methods["amnesic"] = (float(roc_auc_score(yt, tgt_probe.decision_function(Xt.cpu().numpy() @ Pinlp))),
                          float(roc_auc_score(yo, off_probe.decision_function(Xo.cpu().numpy() @ Pinlp))))

    print(f"baseline: Pathology {base_t:.3f}, Age {base_o:.3f}")
    for m, (t, o) in methods.items():
        print(f"  {m:10s}  Pathology→{t:.3f}  Age→{o:.3f}")

    # ── figure ──
    from mecheeg.nature_fig_style import set_nature_style, nature_figure, nature_palette, save_nature_figure
    set_nature_style(); pal = nature_palette(8)
    fig, ax = nature_figure(width="single", height_mm=70)
    cats = {"SAE": pal[0], "PCA": pal[1], "ICA": pal[2], "RandDict": "0.6",
            "dense-CAV": pal[3], "centroid": "0.6", "LEACE": pal[4], "amnesic": pal[5]}
    for m, (t, o) in methods.items():
        ax.scatter(t, o, s=32, color=cats[m], zorder=3, edgecolor="white", lw=0.4)
        ax.annotate(m, (t, o), fontsize=4.8, xytext=(3, 3), textcoords="offset points")
    ax.scatter(base_t, base_o, marker="*", s=60, color="k", zorder=4)
    ax.annotate("no intervention", (base_t, base_o), fontsize=4.8, xytext=(3, -6), textcoords="offset points")
    ax.axhline(base_o, color="0.8", lw=0.5, ls=":"); ax.axvline(0.5, color="0.8", lw=0.5, ls=":")
    ax.set_xlabel("Pathology AUROC after (← target suppressed)")
    ax.set_ylabel("Age AUROC after (off-target preserved →)")
    ax.set_title("Selectivity: suppress Pathology, preserve Age (SleepFM L2)", loc="left", fontsize=6.3)
    ax.annotate("selective\n(erase target,\nkeep off-target)", (0.53, base_o - 0.02), fontsize=4.6, color="0.4")
    FIG.mkdir(parents=True, exist_ok=True)
    save_nature_figure(fig, str(FIG / "fig_selectivity_all.pdf"))
    save_nature_figure(fig, str(FIG / "fig_selectivity_all.png"))
    print("wrote fig_selectivity_all")


if __name__ == "__main__":
    main()
