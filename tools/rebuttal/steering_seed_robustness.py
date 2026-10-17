"""SAE-seed robustness of the steering-selectivity metric Δ̃ (rebuttal O5 follow-up).

Retrains the operating-point SAE N times (identical `SAETrainer` procedure, different
seed) on a given experiment's layer activations, and re-measures Δ̃ + permutation p-value
for each. Everything downstream of the SAE is held fixed by the shared RNG in
`steering_significance.analyse_cell`, so the ONLY varied factor is the SAE seed. This
quantifies how much the steering-selectivity conclusion depends on the SAE training run.

Generalised across architectures (SleepFM / LaBraM / REVE). Submitted (v4-cache) Δ̃
values for the chosen experiment are loaded automatically for comparison.

Usage:
  # SleepFM operating layer (embed_dim inferred = 128)
  uv run tools/rebuttal/steering_seed_robustness.py --experiment sleepfm_finetuned_layer2 \
      --layer-idx 2 --seeds 0 1 2 3 4 5 6 7 --concepts Pathology Age
  # LaBraM operating layer (7.2M tokens → cap training set for CPU speed)
  uv run tools/rebuttal/steering_seed_robustness.py --experiment labram_layer11 --layer-idx 11 \
      --max-train-tokens 500000 --seeds 0 1 2 3 4 5 6 7 --concepts Pathology Age
  # REVE operating layer
  uv run tools/rebuttal/steering_seed_robustness.py --experiment reve_qjbe08_layer8 --layer-idx 8 \
      --seeds 0 1 2 3 4 5 6 7 --concepts Pathology Age

Output: results/steering_significance/seed_robustness_{experiment}.json  (resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))

import plot_cross_model_steering_concepts_paper as P  # noqa: E402
P.ROOT = ROOT
import steering_significance as S  # noqa: E402  (reuses analyse_cell)
from mecheeg.sae import SAETrainer  # noqa: E402

DEVICE = P.DEVICE
V4_CACHE = ROOT / "paper/concept_steering_figures/cross_model_steering_concepts_data_v4.npz"


def submitted_delta(exp, concepts):
    """Excess Δ̃ (area − rand_area_mean) from the v4 cache for the given cells."""
    out = {}
    if not V4_CACHE.exists():
        return out
    d = np.load(V4_CACHE, allow_pickle=True)
    for k in d.files:
        if not k.startswith("entry_"):
            continue
        e = json.loads(str(d[k]))
        if e["experiment"] == exp and e["concept"] in concepts:
            out[e["concept"]] = round(float(e["area"]) - float(e.get("rand_area_mean", 0.0)), 4)
    return out


def load_cache(exp):
    sc = torch.load(ROOT / "results/steering_cache" / exp / "steering_cache.pt",
                    map_location="cpu", weights_only=False)
    return sc


def train_sae(Xn, seed, expansion, k, epochs, lr, batch_size, layer_idx):
    torch.manual_seed(seed)
    np.random.seed(seed)
    trainer = SAETrainer(embed_dim=Xn.shape[1], expansion=expansion, mode="topk", k=k,
                         lr=lr, batch_size=batch_size, epochs=epochs, device=str(DEVICE))
    sae = trainer._train_single_sae(Xn.clone(), layer_idx)   # same procedure as the paper
    sae.to(DEVICE).eval()
    return sae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="sleepfm_finetuned_layer2")
    ap.add_argument("--layer-idx", type=int, default=2)
    ap.add_argument("--expansion", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-train-tokens", type=int, default=None,
                    help="cap SAE-training tokens (random subsample) for CPU speed")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--concepts", nargs="+", default=["Pathology", "Age"])
    ap.add_argument("--b-perm", type=int, default=200)
    ap.add_argument("--b-boot", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-json", default=None,
                    help="output path (default results/steering_significance/seed_robustness_{exp}.json)")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = [0]
        args.b_perm, args.b_boot, args.epochs = 20, 50, 5

    out = (Path(args.out_json) if args.out_json else
           ROOT / "results" / "steering_significance" / f"seed_robustness_{args.experiment}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = submitted_delta(args.experiment, args.concepts)
    print(f"Experiment {args.experiment} (layer {args.layer_idx}, E={args.expansion}, k={args.k})")
    print(f"Submitted v4 Δ̃: {sub}", flush=True)

    print("Loading steering cache…", flush=True)
    sc = load_cache(args.experiment)
    emb_full = torch.as_tensor(np.asarray(sc["embeddings"])).float()
    print(f"  {emb_full.shape[0]} tokens × {emb_full.shape[1]} dims", flush=True)

    # SAE-training activations (optionally subsampled), with normalisation from that set.
    if args.max_train_tokens and emb_full.shape[0] > args.max_train_tokens:
        g = torch.Generator().manual_seed(12345)
        idx = torch.randperm(emb_full.shape[0], generator=g)[:args.max_train_tokens]
        emb_tr = emb_full[idx]
        print(f"  training SAEs on a {emb_tr.shape[0]}-token subsample", flush=True)
    else:
        emb_tr = emb_full
    mean = emb_tr.mean(dim=0)
    std = emb_tr.std(dim=0).clamp(min=1e-8)
    Xn = (emb_tr - mean) / std
    am = mean.to(DEVICE); as_ = std.to(DEVICE)
    del emb_full

    specs = [s for s in P.CONCEPTS if s["name"] in args.concepts]

    rows = []
    have = set()
    if out.exists() and not args.force:
        rows = json.loads(out.read_text()).get("cells", [])
        have = {(r["seed"], r["concept"]) for r in rows}
        print(f"Resuming: {len(rows)} existing cells", flush=True)

    for seed in args.seeds:
        todo = [s for s in specs if (seed, s["name"]) not in have]
        if not todo:
            print(f"=== seed {seed}: cached, skip ==="); continue
        t0 = time.time()
        print(f"\n=== seed {seed}: training SAE ({args.epochs} epochs) ===", flush=True)
        sae = train_sae(Xn, seed, args.expansion, args.k, args.epochs, args.lr,
                        args.batch_size, args.layer_idx)
        nf = int(sae.decoder.weight.shape[1])
        layer_state = (sc, sae, am, as_, None, nf)
        for spec in todo:
            r = S.analyse_cell(spec, layer_state, args.b_perm, args.b_boot)
            if r is None:
                print(f"  [skip] {spec['name']}: insufficient samples"); continue
            r["seed"] = seed
            rows.append(r); have.add((seed, spec["name"]))
            print(f"  seed {seed}  {spec['name']:12s}  Δ̃={r['delta_tilde']:+.4f} "
                  f"CI[{r['delta_tilde_ci'][0]:+.3f},{r['delta_tilde_ci'][1]:+.3f}] "
                  f"AUROC₀={r['auroc0']:.3f} p={r['p_value']:.4f} → {r['regime']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        out.write_text(json.dumps({"experiment": args.experiment, "layer": args.layer_idx,
                                   "expansion": args.expansion, "k": args.k,
                                   "max_train_tokens": args.max_train_tokens,
                                   "submitted_delta": sub, "epochs": args.epochs,
                                   "b_perm": args.b_perm, "b_boot": args.b_boot,
                                   "cells": rows}, indent=2))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n===== SEED-ROBUSTNESS SUMMARY — {args.experiment} =====")
    for cname in args.concepts:
        ds = np.array([r["delta_tilde"] for r in rows if r["concept"] == cname])
        sig = sum(1 for r in rows if r["concept"] == cname and r["significant"])
        if not len(ds):
            continue
        print(f"\n{cname}:  n={len(ds)} seeds  |  submitted Δ̃={sub.get(cname, 'n/a')}")
        print(f"  Δ̃: mean={ds.mean():+.4f} sd={ds.std(ddof=1):.4f} "
              f"median={np.median(ds):+.4f} range=[{ds.min():+.3f},{ds.max():+.3f}]")
        print(f"  direction Δ̃>0: {(ds>0).sum()}/{len(ds)}  |  "
              f"selectively-steerable: {sig}/{len(ds)} ({100*sig/len(ds):.0f}%)")


if __name__ == "__main__":
    main()
