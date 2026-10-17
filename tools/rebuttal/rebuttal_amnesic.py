"""Rebuttal #1 — amnesic probing (INLP) baseline (HdqW Q3 named it explicitly).

Iterative Nullspace Projection (Ravfogel et al. 2020): repeatedly fit a linear probe for
the target concept and project the representation onto its nullspace until the concept is
no longer linearly decodable. We use it as a second purpose-built erasure baseline
alongside LEACE, and compare off-target preservation to SAE steering — the SAE is not an
erasure method, so this contextualizes (not undercuts) its interpretability contribution.

  uv run tools/rebuttal/rebuttal_amnesic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cr  # noqa: E402  (camera-ready CR_OUT_DIR / CR_SEED switches)
import plot_cross_model_steering_concepts_paper as P  # noqa: E402

EXP = "sleepfm_finetuned_layer2"
OUT = _cr.out_dir(ROOT / "results" / "rebuttal")
CONCEPTS = {
    "Pathology": ("classification", P.ABN_CATS, P.NORMAL),
    "Age": ("age_group", {"0-3", "4-9"}, {"20-29", "30-39", "40-49", "50-59", "60+"}),
    "Sex": ("gender", {"male"}, {"female"}),
    "Med-Psych": ("medication_group", {"Psychiatric"}, {"No Current Medication"}),
    "Med-ASM": ("medication_group", {"ASM"}, {"No Current Medication"}),
}


def inlp_projection(X, y, n_iters=35, acc_tol=0.52):
    """Accumulated nullspace projection P s.t. concept y is un-decodable from X @ P.
    Runs thoroughly (no early subsample) so strongly-encoded concepts are actually erased."""
    d = X.shape[1]
    Pmat = np.eye(d, dtype=np.float32); I = np.eye(d, dtype=np.float32)
    for _ in range(n_iters):
        Xp = X @ Pmat
        clf = LogisticRegression(max_iter=300, solver="liblinear", class_weight="balanced").fit(Xp, y)
        if clf.score(Xp, y) < acc_tol:   # concept no longer linearly decodable
            break
        w = clf.coef_[0].astype(np.float32); nw = np.linalg.norm(w)
        if nw < 1e-9:
            break
        w = w / nw
        Pmat = (I - np.outer(w, w)) @ Pmat
    return Pmat


def main():
    sc = torch.load(ROOT / "results/steering_cache" / EXP / "steering_cache.pt",
                    map_location="cpu", weights_only=False)
    emb = sc["embeddings"].float()
    mean = emb.mean(0); std = emb.std(0).clamp(min=1e-8)
    rng = np.random.default_rng(42)

    def asarr(f):
        return np.asarray([str(v) for v in sc[f]])

    C = {}
    for name, (field, pos, neg) in CONCEPTS.items():
        fa = asarr(field)
        pm = np.where(np.isin(fa, list(pos)))[0]; nm = np.where(np.isin(fa, list(neg)))[0]
        k = min(3000, len(pm), len(nm))
        idx = np.concatenate([rng.choice(pm, k, False), rng.choice(nm, k, False)])
        y = np.array([1] * k + [0] * k)
        X = ((emb[idx] - mean) / std).numpy().astype(np.float32)
        tr, te = train_test_split(np.arange(2 * k), test_size=0.3, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr], y[tr])
        C[name] = dict(X=X, y=y, tr=tr, te=te, clf=clf)

    names = list(C)
    base = {m: float(roc_auc_score(C[m]["y"][C[m]["te"]], C[m]["clf"].decision_function(C[m]["X"][C[m]["te"]])))
            for m in names}

    mat = {}
    for s in names:
        Pmat = inlp_projection(C[s]["X"][C[s]["tr"]], C[s]["y"][C[s]["tr"]])
        mat[s] = {}
        for m in names:
            Xe = C[m]["X"] @ Pmat
            mat[s][m] = float(roc_auc_score(C[m]["y"][C[m]["te"]], C[m]["clf"].decision_function(Xe[C[m]["te"]])))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "amnesic_matrix.json").write_text(json.dumps({"baseline": base, "matrix": mat}, indent=2))

    print("=== Amnesic (INLP) erasure collateral: AUROC of MEASURED after erasing STEERED ===")
    print("steer\\measure   " + "".join(f"{m[:9]:>10s}" for m in names))
    for s in names:
        print(f"{s:14s}  " + "".join(f"{mat[s][m]:>10.3f}" for m in names))
    print("baseline        " + "".join(f"{base[m]:>10.3f}" for m in names))

    # 3-way: SAE steering vs LEACE vs amnesic, off-target preservation when target suppressed
    sae = json.load(open(OUT / "collateral_matrix.json"))["matrix"]
    lea = json.load(open(OUT / "leace_matrix.json"))["matrix"]
    print("\n=== target→ / off-target-avg preserved: SAE / LEACE / amnesic ===")
    for s in names:
        oth = [m for m in names if m != s]
        row = {"SAE": sae[s], "LEACE": lea[s], "amnesic": mat[s]}
        tgt = " / ".join(f"{row[k][s]:.3f}" for k in row)
        off = " / ".join(f"{np.mean([row[k][m] for m in oth]):.3f}" for k in row)
        print(f"  erase {s:10s}: target {tgt} | off-avg {off}")
    print(f"\n[#1] wrote {OUT/'amnesic_matrix.json'}")


if __name__ == "__main__":
    main()
