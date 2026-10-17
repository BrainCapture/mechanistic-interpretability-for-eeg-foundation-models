"""Rebuttal — does age–pathology entanglement reproduce on public TUAB data?

Headline replication for AC point 4 / rkJV Q3. Uses the TUAB-finetuned SleepFM L2
steering cache (build_tuab_steering_cache.py) and the IDENTICAL selectivity machinery
as the BrainCapture analysis (tools/rebuttal/rebuttal_steering_baselines.py), with fresh SAE
seeds. Reports, for Pathology (off-target = Age) and Age (off-target = Pathology):
  Δ̃ = excess selectivity ; if steering the target also collapses the off-target
  (Δ̃ ≈ 0 despite high AUROC₀), the two axes are ENTANGLED — the paper's finding.

TUAB is adult-only, so Age = young(<40) vs old(≥60), not the paper's child-vs-adult.

  uv run tools/rebuttal/rebuttal_tuab_entanglement.py --sae-seeds 8 --b-perm 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))

import rebuttal_steering_baselines as B  # noqa: E402  (eval_dict, make_*_source, build_pools)
import plot_cross_model_steering_concepts_paper as P  # noqa: E402
from steering_seed_robustness import train_sae  # noqa: E402

DEVICE = B.DEVICE
SEED = 42

TUAB_CONCEPTS = [
    {"name": "Pathology", "tgt_field": "classification", "tgt_pos": {"Abnormal"}, "tgt_neg": {"Normal"},
     "off_field": "age_group", "off_pos": {"young"}, "off_neg": {"old"},
     "filter_field": None, "filter_set": None},
    {"name": "Age (young/old)", "tgt_field": "age_group", "tgt_pos": {"young"}, "tgt_neg": {"old"},
     "off_field": "classification", "off_pos": {"Abnormal"}, "off_neg": {"Normal"},
     "filter_field": None, "filter_set": None},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-seeds", type=int, default=8)
    ap.add_argument("--b-perm", type=int, default=200)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--expansion", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--cache-tag", default="sleepfm_tuab_layer2",
                    help="steering-cache experiment tag (…_pretrained_… for the AC#1 frozen encoder)")
    args = ap.parse_args()

    cache = ROOT / "results" / "steering_cache" / args.cache_tag / "steering_cache.pt"
    out = ROOT / "results" / "rebuttal" / f"{args.cache_tag}_entanglement.json"
    sc = torch.load(cache, map_location="cpu", weights_only=False)
    emb = sc["embeddings"].float()
    mean = emb.mean(0); std = emb.std(0).clamp(min=1e-8)
    Xn = (emb - mean) / std
    am, as_ = mean.to(DEVICE), std.to(DEVICE)
    nf = int(args.expansion * emb.shape[1])
    print(f"[tuab-entangle] {emb.shape[0]} tokens × {emb.shape[1]}  fresh SAEs={args.sae_seeds} "
          f"(E={args.expansion}, k={args.k}, nf={nf})", flush=True)

    saes = []
    for s in range(args.sae_seeds):
        saes.append(train_sae(Xn, s, args.expansion, args.k, args.epochs, 3e-4, 2048, 2))
        print(f"  trained SAE seed {s}", flush=True)

    # baselines fit once on a global standardized sample
    rng = np.random.default_rng(SEED)
    gi = rng.choice(len(emb), min(B.N_GLOBAL_FIT, len(emb)), replace=False)
    Xg = ((emb[gi].to(DEVICE) - am) / (as_ + 1e-8)).cpu().numpy()
    baselines = [B.make_pca_source(Xg), B.make_ica_source(Xg), B.make_random_source(emb.shape[1], nf, args.k)]

    results = {}
    for spec in TUAB_CONCEPTS:
        pools = B.build_pools(sc, spec, am, as_, np.random.default_rng(SEED))
        if pools is None:
            print(f"[skip] {spec['name']}"); continue
        Xt, Xo, y_t, y_o, n_t, n_o = pools
        print(f"\n=== {spec['name']}  (target n={n_t}/grp, off n={n_o}/grp) ===", flush=True)
        rows = []
        sd = []
        for s, sae_s in enumerate(saes):
            r = B.eval_dict(B.make_sae_source(sae_s), Xt, Xo, y_t, y_o, args.b_perm, np.random.default_rng(SEED + 7))
            r["method"] = f"SAE[seed{s}]"; rows.append(r); sd.append(r["delta"])
            print(f"  {r['method']:12s} AUROC0={r['auroc0']:.3f} Δ̃={r['delta']:+.4f} p={r['p']:.3f}", flush=True)
        sd = np.array(sd)
        nsel = sum(1 for r in rows if r["p"] < 0.05 and r["delta"] >= 0.05)
        verdict = "SELECTIVE" if nsel >= len(sd) * 0.6 else "ENTANGLED"
        print(f"  → SAE Δ̃ {sd.mean():+.3f}±{sd.std(ddof=1):.3f}  {int((sd>0).sum())}/{len(sd)} pos  "
              f"{nsel}/{len(sd)} selective  ⇒ {verdict}", flush=True)
        for src in baselines:
            r = B.eval_dict(src, Xt, Xo, y_t, y_o, args.b_perm, np.random.default_rng(SEED + 7))
            rows.append(r); print(f"  {r['method']:12s} AUROC0={r['auroc0']:.3f} Δ̃={r['delta']:+.4f} p={r['p']:.3f}", flush=True)
        for fn in (B.eval_cav, B.eval_centroid):
            r = fn(Xt, Xo, y_t, y_o, args.b_perm, np.random.default_rng(SEED + 7))
            rows.append(r); print(f"  {r['method']:12s} AUROC0={r['auroc0']:.3f} Δ̃={r['delta']:+.4f} p={r['p']:.3f}", flush=True)
        results[spec["name"]] = {"n_target": n_t, "n_off": n_o,
                                 "sae_delta_mean": float(sd.mean()), "sae_delta_std": float(sd.std(ddof=1)),
                                 "sae_pos": int((sd > 0).sum()), "sae_selective": nsel, "verdict": verdict,
                                 "rows": rows}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": "TUAB v3.0.1", "encoder": args.cache_tag,
                               "sae_seeds": args.sae_seeds, "results": results}, indent=2))
    print(f"\n[tuab-entangle] wrote {out}")


if __name__ == "__main__":
    main()
