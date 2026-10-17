"""Rebuttal — the FULL framework (steering + entanglement) on epileptiform events (rkJV Q4).

Encoding (separability) is only the prerequisite. Here we run the paper's steering machinery
on the specific TUEV events: for each epileptiform type we clamp its top TCAV-ranked SAE
features toward the background donor and measure (i) steerability — the drop in that event's
own AUROC — and (ii) cross-event entanglement — the collateral on the other event types.
A diagonal collapse with preserved off-diagonal = the event is selectively steerable.

  uv run tools/rebuttal/rebuttal_tuev_steering.py --clamp-frac 0.25
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cr  # noqa: E402  (camera-ready CR_OUT_DIR / CR_SEED switches)
import finetune_sleepfm_v1_binary as F  # noqa: E402
import plot_cross_model_steering_concepts_paper as P  # noqa: E402
from mecheeg.sae import ActivationExtractor, SAETrainer  # noqa: E402

DEVICE = P.DEVICE
CLIP, STD = 5e-5, 1e-5
CKPT = ROOT / "checkpoints" / "finetuned" / "sleepfm_tuab" / "finetuned.ckpt"
DATA = ROOT / "data" / "tuev-sleepfm-128"
LAB = {0: "SPSW", 1: "GPED", 2: "PLED"}          # epileptiform events; bckg = 5
BCKG = 5
OUT = _cr.out_dir(ROOT / "results" / "rebuttal")
SEED = _cr.seed(42)


def extract():
    model = F.build_model()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"])
    model.eval().to(DEVICE)
    enc = model.encoder
    ex = ActivationExtractor(enc, layers=list(enc.transformer_encoder.layers))
    embs, labs = [], []
    for sp in sorted(glob.glob(str(DATA / "*.hdf5"))):
        with h5py.File(sp, "r") as h:
            data = h["data"][:]; y = h["labels"][:]
        x = np.clip(data, -CLIP, CLIP) / STD
        for i in range(0, len(x), 32):
            xb = torch.from_numpy(x[i:i + 32]).float().to(DEVICE)
            with torch.no_grad():
                ex.clear(); _ = enc(xb)
                embs.append(ex.get_activations()[2].mean(1).cpu().numpy())
        labs.append(y)
    return np.concatenate(embs), np.concatenate(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clamp-frac", type=float, default=0.25)
    ap.add_argument("--n-rand", type=int, default=20)
    args = ap.parse_args()

    X, y = extract()
    Xt = torch.from_numpy(X).float()
    mean = Xt.mean(0); std = Xt.std(0).clamp(min=1e-8)
    am, as_ = mean.to(DEVICE), std.to(DEVICE)
    torch.manual_seed(SEED); np.random.seed(SEED)  # camera-ready: SAE was unseeded in the rebuttal run
    sae = SAETrainer(embed_dim=X.shape[1], expansion=1.0, mode="topk", k=8, lr=3e-4,
                     batch_size=1024, epochs=20, device=str(DEVICE))._train_single_sae(
        ((Xt - mean) / std).clone(), 2)
    sae.to(DEVICE).eval()
    nf = int(sae.decoder.weight.shape[1]); n_clamp = int(round(args.clamp_frac * nf))
    print(f"[tuev-steer] {X.shape[0]} windows, SAE nf={nf}, clamp {n_clamp} feats", flush=True)

    rng = np.random.default_rng(SEED)
    bck = np.where(y == BCKG)[0]
    C = {}
    for c, name in LAB.items():
        pos = np.where(y == c)[0]
        k = min(len(pos), len(bck))
        idx = np.concatenate([rng.choice(pos, k, False), rng.choice(bck, k, False)])
        yb = np.array([1] * k + [0] * k)
        Xr = (torch.as_tensor(X[idx]).float().to(DEVICE) - am) / (as_ + 1e-8)
        with torch.no_grad():
            z = sae.encode(Xr); clean = sae.decode(z)
        Xtr, ytr, iv, yv = P.split_train_holdout(clean, yb, SEED)
        probe = P.fit_probe(Xtr, ytr)
        align = P.build_tcav_alignment(Xr, yb, sae.decoder.weight.detach())
        C[name] = dict(z=z, iv=iv, yv=yv, probe=probe,
                       ranking=np.argsort(align)[::-1].copy(), donor=z[yb == 0].mean(0))

    names = list(C)
    base = {}
    for m in names:
        d = C[m]
        with torch.no_grad():
            Xd = sae.decode(d["z"]).index_select(0, torch.as_tensor(d["iv"], device=DEVICE))
        base[m] = float(roc_auc_score(d["yv"], (Xd @ d["probe"].w + d["probe"].b).cpu().numpy()))

    mat = {}
    for s in names:
        feats = C[s]["ranking"][:n_clamp]; donor = C[s]["donor"]; mat[s] = {}
        for m in names:
            d = C[m]
            with torch.no_grad():
                zc = P.substitute(d["z"], feats, donor)
                Xd = sae.decode(zc).index_select(0, torch.as_tensor(d["iv"], device=DEVICE))
                mat[s][m] = float(roc_auc_score(d["yv"], (Xd @ d["probe"].w + d["probe"].b).cpu().numpy()))

    # Random-clamp null (camera-ready): same number of features, uniformly random sets.
    rng_r = np.random.default_rng(SEED + 1)
    mat_rand = {s: {m: [] for m in names} for s in names}
    for s in names:
        donor = C[s]["donor"]
        for _ in range(args.n_rand):
            feats = rng_r.choice(nf, n_clamp, replace=False)
            for m in names:
                d = C[m]
                with torch.no_grad():
                    zc = P.substitute(d["z"], feats, donor)
                    Xd = sae.decode(zc).index_select(0, torch.as_tensor(d["iv"], device=DEVICE))
                    mat_rand[s][m].append(float(roc_auc_score(
                        d["yv"], (Xd @ d["probe"].w + d["probe"].b).cpu().numpy())))
    # Selectivity index per steered event: self AUROC drop minus mean off-target drop.
    # "excess" = TCAV-ranked index minus the random-clamp expectation (>0 = more selective).
    sel = {}
    for s in names:
        oth = [m for m in names if m != s]
        def idx(M):
            return float((base[s] - M[s]) - np.mean([base[m] - M[m] for m in oth]))
        rand_idx = [idx({m: mat_rand[s][m][r] for m in names}) for r in range(args.n_rand)]
        sel[s] = {"tcav": idx(mat[s]), "rand_mean": float(np.mean(rand_idx)),
                  "rand_sd": float(np.std(rand_idx, ddof=1)),
                  "excess": idx(mat[s]) - float(np.mean(rand_idx))}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tuev_steering.json").write_text(json.dumps(
        {"clamp_frac": args.clamp_frac, "seed": SEED, "n_rand": args.n_rand, "baseline": base,
         "matrix": mat, "matrix_rand_mean": {s: {m: float(np.mean(v)) for m, v in r.items()}
                                             for s, r in mat_rand.items()},
         "selectivity": sel}, indent=2))
    print("\n=== TUEV event steering: AUROC of MEASURED event after steering STEERED event → bckg ===")
    print("steer\\measure  " + "".join(f"{m:>8s}" for m in names))
    for s in names:
        print(f"{s:12s}  " + "".join(f"{mat[s][m]:>8.3f}" for m in names))
    print("baseline      " + "".join(f"{base[m]:>8.3f}" for m in names))
    for s in names:
        oth = [m for m in names if m != s]
        print(f"  steer {s}: self {base[s]:.2f}→{mat[s][s]:.2f} (steerable) | "
              f"off-target avg {np.mean([mat[s][m] for m in oth]):.2f} (baseline "
              f"{np.mean([base[m] for m in oth]):.2f})")
    print(f"\n[Q4-framework] wrote {OUT/'tuev_steering.json'}")


if __name__ == "__main__":
    main()
