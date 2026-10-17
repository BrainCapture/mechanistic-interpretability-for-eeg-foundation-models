"""Rebuttal #1 — multi-off-target collateral matrix.

rkJV's standing weakness: selectivity is only ever tested against ONE off-target at a
time, so other clinical/demographic info might still be disrupted. This measures, when we
steer concept c (clamp its TCAV-ranked features toward the c-negative donor), the AUROC of
EVERY concept's probe — a full collateral matrix. Diagonal = the steered target (should
drop); off-diagonal = collateral on the other concepts (high = selective, low = damaged).

Uses a fresh SAE (the on-disk L2 SAE is the O5 anomaly) on the finetuned BrainCapture cache.

  uv run tools/rebuttal/rebuttal_collateral_matrix.py --clamp-frac 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cr  # noqa: E402  (camera-ready CR_OUT_DIR / CR_SEED switches)
import plot_cross_model_steering_concepts_paper as P  # noqa: E402
from steering_seed_robustness import train_sae  # noqa: E402

DEVICE = P.DEVICE
SEED = _cr.seed(42)
EXP = "sleepfm_finetuned_layer2"
OUT = _cr.out_dir(ROOT / "results" / "rebuttal")

# concept: (field, pos, neg, short label)
CONCEPTS = {
    "Pathology": ("classification", P.ABN_CATS, P.NORMAL),
    "Age": ("age_group", {"0-3", "4-9"}, {"20-29", "30-39", "40-49", "50-59", "60+"}),
    "Sex": ("gender", {"male"}, {"female"}),
    "Med-Psych": ("medication_group", {"Psychiatric"}, {"No Current Medication"}),
    "Med-ASM": ("medication_group", {"ASM"}, {"No Current Medication"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clamp-frac", type=float, default=0.25)
    args = ap.parse_args()

    sc = torch.load(ROOT / "results/steering_cache" / EXP / "steering_cache.pt",
                    map_location="cpu", weights_only=False)
    emb = sc["embeddings"].float()
    mean = emb.mean(0); std = emb.std(0).clamp(min=1e-8)
    am, as_ = mean.to(DEVICE), std.to(DEVICE)
    sae = train_sae((emb - mean) / std, SEED, 1.0, 8, 20, 3e-4, 2048, 2)
    nf = int(sae.decoder.weight.shape[1])
    n_clamp = int(round(args.clamp_frac * nf))
    print(f"[collateral] fresh SAE nf={nf}, clamp top {n_clamp} features ({args.clamp_frac:.0%})", flush=True)

    def asarr(field):
        return np.asarray([str(v) for v in sc[field]])

    # per-concept: balanced eval pool + clean SAE code + probe (fit on clean decoded) + ranking + donor
    rng = np.random.default_rng(SEED)
    C = {}
    for name, (field, pos, neg) in CONCEPTS.items():
        fa = asarr(field)
        pm = np.isin(fa, list(pos)); nm = np.isin(fa, list(neg))
        idx, y, n = P.balanced_pair(emb, pm, nm, P.N_PROBE_PER_GROUP, rng)
        if idx is None:
            print(f"  [skip] {name} ({n}/group)"); continue
        Xraw = (torch.as_tensor(np.asarray(emb[idx])).float().to(DEVICE) - am) / (as_ + 1e-8)
        with torch.no_grad():
            z = sae.encode(Xraw); clean = sae.decode(z)
        Xtr, ytr, iv, yv = P.split_train_holdout(clean, y, SEED)
        probe = P.fit_probe(Xtr, ytr)
        align = P.build_tcav_alignment(Xraw, y, sae.decoder.weight.detach())
        ranking = np.argsort(align)[::-1].copy()
        donor = z[y == 0].mean(0)
        C[name] = dict(z=z, iv=iv, yv=yv, probe=probe, ranking=ranking, donor=donor)

    names = list(C)
    # baseline AUROC (no steering)
    base = {}
    for m in names:
        d = C[m]
        with torch.no_grad():
            Xv = d["z"]
            Xdec = sae.decode(Xv).index_select(0, torch.as_tensor(d["iv"], device=DEVICE))
            base[m] = float(roc_auc_score(d["yv"], (Xdec @ d["probe"].w + d["probe"].b).cpu().numpy()))

    # matrix[steer][measure] = AUROC of `measure` after steering `steer`
    mat = {}
    for s in names:
        mat[s] = {}
        feats = C[s]["ranking"][:n_clamp]; donor = C[s]["donor"]
        for m in names:
            d = C[m]
            with torch.no_grad():
                zc = P.substitute(d["z"], feats, donor)      # clamp steer-concept features
                Xdec = sae.decode(zc).index_select(0, torch.as_tensor(d["iv"], device=DEVICE))
                auc = float(roc_auc_score(d["yv"], (Xdec @ d["probe"].w + d["probe"].b).cpu().numpy()))
            mat[s][m] = auc

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "collateral_matrix.json").write_text(json.dumps(
        {"experiment": EXP, "clamp_frac": args.clamp_frac, "baseline": base, "matrix": mat}, indent=2))

    print("\n=== Collateral matrix: AUROC of MEASURED concept after steering STEERED concept ===")
    print("(diagonal = steered target; off-diagonal = collateral; baseline in [])")
    hdr = "steer\\measure   " + "".join(f"{m[:9]:>10s}" for m in names)
    print(hdr)
    for s in names:
        row = f"{s:14s}  " + "".join(f"{mat[s][m]:>10.3f}" for m in names)
        print(row)
    print("baseline        " + "".join(f"{base[m]:>10.3f}" for m in names))
    print(f"\n[#1] wrote {OUT/'collateral_matrix.json'}")


if __name__ == "__main__":
    main()
