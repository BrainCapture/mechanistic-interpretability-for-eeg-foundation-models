"""Rebuttal #2 — LEACE erasure baseline (HdqW named it explicitly).

LEACE (Belrose et al. 2023) is a closed-form linear concept eraser designed for minimal
collateral. We use it as a *strong* selectivity baseline: erase a target concept and measure
how much every concept's linear decodability survives — directly comparable to the SAE
collateral matrix (#1). If SAE steering achieves selectivity comparable to a purpose-built
erasure, that validates the SAE intervention; where LEACE is cleaner we say so.

  uv run tools/rebuttal/rebuttal_leace.py
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


def main():
    sc = torch.load(ROOT / "results/steering_cache" / EXP / "steering_cache.pt",
                    map_location="cpu", weights_only=False)
    emb = sc["embeddings"].float()
    mean = emb.mean(0); std = emb.std(0).clamp(min=1e-8)
    rng = np.random.default_rng(42)

    def asarr(f):
        return np.asarray([str(v) for v in sc[f]])

    # per-concept balanced pool (standardized), train/test split, probe on clean
    C = {}
    for name, (field, pos, neg) in CONCEPTS.items():
        fa = asarr(field)
        pm = np.where(np.isin(fa, list(pos)))[0]; nm = np.where(np.isin(fa, list(neg)))[0]
        k = min(3000, len(pm), len(nm))
        idx = np.concatenate([rng.choice(pm, k, replace=False), rng.choice(nm, k, replace=False)])
        y = np.array([1] * k + [0] * k)
        X = ((emb[idx] - mean) / std)
        tr, te = train_test_split(np.arange(2 * k), test_size=0.3, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr].numpy(), y[tr])
        C[name] = dict(X=X, y=y, tr=tr, te=te, clf=clf)

    names = list(C)
    base = {m: float(roc_auc_score(C[m]["y"][C[m]["te"]],
                                   C[m]["clf"].decision_function(C[m]["X"][C[m]["te"]].numpy())))
            for m in names}

    # erase target concept c (fit LEACE on c's pool), measure every concept on erased space
    mat = {}
    for s in names:
        Xs = C[s]["X"]; ys = torch.as_tensor(C[s]["y"]).float()
        eraser = LeaceEraser.fit(Xs, ys)
        mat[s] = {}
        for m in names:
            Xe = eraser(C[m]["X"]).numpy()
            auc = float(roc_auc_score(C[m]["y"][C[m]["te"]], C[m]["clf"].decision_function(Xe[C[m]["te"]])))
            mat[s][m] = auc

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leace_matrix.json").write_text(json.dumps({"baseline": base, "matrix": mat}, indent=2))

    print("=== LEACE erasure collateral: AUROC of MEASURED after erasing STEERED ===")
    print("steer\\measure   " + "".join(f"{m[:9]:>10s}" for m in names))
    for s in names:
        print(f"{s:14s}  " + "".join(f"{mat[s][m]:>10.3f}" for m in names))
    print("baseline        " + "".join(f"{base[m]:>10.3f}" for m in names))

    # compare selectivity vs SAE collateral (#1): off-target preservation when target suppressed
    try:
        sae = json.load(open(OUT / "collateral_matrix.json"))
        print("\n=== Selectivity: off-target AUROC preserved (mean over others), SAE vs LEACE ===")
        for s in names:
            others = [m for m in names if m != s]
            sae_off = np.mean([sae["matrix"][s][m] for m in others])
            lea_off = np.mean([mat[s][m] for m in others])
            sae_tgt = sae["matrix"][s][s]; lea_tgt = mat[s][s]
            print(f"  erase {s:10s}: target→ SAE {sae_tgt:.3f} / LEACE {lea_tgt:.3f} | "
                  f"off-target avg SAE {sae_off:.3f} / LEACE {lea_off:.3f}")
    except FileNotFoundError:
        pass
    print(f"\n[#2] wrote {OUT/'leace_matrix.json'}")


if __name__ == "__main__":
    main()
