"""Rebuttal (HdqW Q5) — native-head vs. probe steering, per encoder.

For each encoder we clamp TCAV-ranked SAE features and read the abnormality signal
two ways over the same windows: (i) the **real finetuned classifier head** (native),
and (ii) a **linear probe on decoded embeddings** (the paper's surrogate). Overlaying
them shows the probe faithfully tracks the deployed model — validating the paper's
steering read-out — and that concept-ranked clamping collapses abnormality far faster
than random.

  SleepFM (TUAB, L2=last)  : native + probe   (in-repo Linear head)
  LaBraM  (L11=last)       : native + probe   (in-repo Linear head)
  REVE    (L8=mid)         : probe only       (native head expects the final layer —
                                               propagating from a mid layer is the O6 concession)

  uv run tools/rebuttal/rebuttal_native_vs_probe.py --n-windows 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cr  # noqa: E402  (camera-ready CR_OUT_DIR / CR_SEED switches)

import plot_cross_model_steering_concepts_paper as P  # noqa: E402
P.ROOT = ROOT
from steering_seed_robustness import train_sae  # noqa: E402
from mecheeg.sae import SparseAutoencoder  # noqa: E402

DEVICE = P.DEVICE
SEED = _cr.seed(42)
OUT = _cr.out_dir(ROOT / "results" / "rebuttal")

CONFIGS = [
    {"name": "SleepFM (TUAB)", "exp": "sleepfm_tuab_layer2", "tpw": 60, "custom": True,
     "head": ROOT / "checkpoints/finetuned/sleepfm_tuab/finetuned.ckpt",
     "abn": {"Abnormal"}, "norm": {"Normal"}},
    {"name": "REVE", "exp": "reve_qjbe08_layer8", "tpw": 1254, "custom": False, "head": None,
     "abn": P.ABN_CATS, "norm": P.NORMAL},
    {"name": "LaBraM", "exp": "labram_layer11", "tpw": 1140, "custom": False,
     "head": ROOT / "checkpoints/finetuned/labram_binary/finetuned.ckpt",
     "abn": P.ABN_CATS, "norm": P.NORMAL},
]


def load_cell(cfg):
    if cfg["custom"]:
        sc = torch.load(ROOT / "results/steering_cache" / cfg["exp"] / "steering_cache.pt",
                        map_location="cpu", weights_only=False)
        emb = sc["embeddings"].float()
        mean = emb.mean(0); std = emb.std(0).clamp(min=1e-8)
        sae = train_sae((emb - mean) / std, SEED, 1.0, 8, 20, 3e-4, 2048, 2)
        am, as_ = mean.to(DEVICE), std.to(DEVICE)
        nf = int(sae.decoder.weight.shape[1])
        return sc, sae, am, as_, nf
    sc, sae, am, as_, _e, nf = P.load_layer(cfg["exp"])
    return sc, sae, am, as_, nf


def load_head(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    return sd["head.weight"].float().to(DEVICE), sd["head.bias"].float().to(DEVICE)


def build_windows(sc, cfg, n_per, rng):
    tpw = cfg["tpw"]
    n_win = int(sc["n_tokens"]) // tpw
    cls_tok = np.asarray([str(v) for v in sc["classification"]])
    win_cls = cls_tok[np.arange(n_win) * tpw]
    abn = np.where(np.isin(win_cls, list(cfg["abn"])))[0]
    nrm = np.where(np.isin(win_cls, list(cfg["norm"])))[0]
    n = min(n_per, len(abn), len(nrm))
    abn = rng.choice(abn, n, replace=False); nrm = rng.choice(nrm, n, replace=False)
    return np.concatenate([abn, nrm]), np.array([1] * n + [0] * n), n


def gather(sc, win_idx, tpw):
    emb = sc["embeddings"]; D = emb.shape[1]
    out = torch.empty((len(win_idx), tpw, D), dtype=torch.float32)
    for i, w in enumerate(win_idx):
        s = int(w) * tpw
        out[i] = torch.as_tensor(np.asarray(emb[s:s + tpw])).float()
    return out


def ranking_and_donor(sc, cfg, sae, am, as_, rng):
    emb = sc["embeddings"]
    cls = np.asarray([str(v) for v in sc["classification"]])
    pos = np.isin(cls, list(cfg["abn"])); neg = np.isin(cls, list(cfg["norm"]))
    idx, y, n = P.balanced_pair(emb, pos, neg, P.N_PROBE_PER_GROUP, rng)
    Xt = (torch.as_tensor(np.asarray(emb[idx])).float().to(DEVICE) - am) / (as_ + 1e-8)
    with torch.no_grad():
        zt = sae.encode(Xt)
    donor = zt[y == 0].mean(0)
    align = P.build_tcav_alignment(Xt, y, sae.decoder.weight.detach())
    return np.argsort(align)[::-1].copy(), donor


@torch.no_grad()
def pooled_at(sae, am, as_, win_tokens, ranking, donor, n):
    W, Tk, D = win_tokens.shape
    Xn = (win_tokens.reshape(W * Tk, D).to(DEVICE) - am) / (as_ + 1e-8)
    z = sae.encode(Xn)
    if n > 0:
        z = P.substitute(z, ranking[:n], donor)
    Xraw = sae.decode(z) * (as_ + 1e-8) + am
    return Xraw.reshape(W, Tk, D).mean(1)   # (W, D) window-pooled


def run(cfg, n_windows, n_rand):
    print(f"\n=== {cfg['name']} ({cfg['exp']}) ===", flush=True)
    sc, sae, am, as_, nf = load_cell(cfg)
    rng = np.random.default_rng(SEED)
    win_idx, y, n = build_windows(sc, cfg, n_windows, rng)
    print(f"  {n} Abnormal + {n} Normal windows; nf={nf}", flush=True)
    wt = gather(sc, win_idx, cfg["tpw"])
    ranking, donor = ranking_and_donor(sc, cfg, sae, am, as_, np.random.default_rng(SEED))
    ns = sorted(set(int(round(f * nf)) for f in np.linspace(0, 1, P.N_STEPS + 1)))
    fracs = (np.array(ns) / nf).tolist()

    head = load_head(cfg["head"]) if cfg["head"] else None
    # train/test split of windows for the probe (native head needs no fitting)
    tr, te = train_test_split(np.arange(2 * n), test_size=0.4, random_state=SEED, stratify=y)

    native, probe = [], []
    # fit probe on the CLEAN (n=0) pooled embeddings, train split
    pooled0 = pooled_at(sae, am, as_, wt, ranking, donor, 0).cpu().numpy()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(pooled0[tr], y[tr])
    for nn in ns:
        pooled = pooled_at(sae, am, as_, wt, ranking, donor, nn)
        pnp = pooled.cpu().numpy()
        probe.append(float(roc_auc_score(y[te], clf.decision_function(pnp[te]))))
        if head is not None:
            logit = (pooled @ head[0].t() + head[1]).squeeze(-1).cpu().numpy()
            native.append(float(roc_auc_score(y, logit)))
    # random-ranking native baseline
    rand = None
    if head is not None:
        rng_r = np.random.default_rng(SEED + 1); curves = []
        for _ in range(n_rand):
            perm = rng_r.permutation(nf); c = []
            for nn in ns:
                pooled = pooled_at(sae, am, as_, wt, perm, donor, nn)
                c.append(float(roc_auc_score(y, (pooled @ head[0].t() + head[1]).squeeze(-1).cpu().numpy())))
            curves.append(c)
        rand = np.mean(curves, 0).tolist()
    return {"name": cfg["name"], "fracs": fracs, "native": native or None,
            "probe": probe, "native_rand": rand, "has_native": head is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=300)
    ap.add_argument("--n-rand", type=int, default=3)
    args = ap.parse_args()
    cells = [run(c, args.n_windows, args.n_rand) for c in CONFIGS]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "native_vs_probe.json").write_text(json.dumps({"cells": cells}, indent=2))
    for c in cells:
        tag = "native+probe" if c["has_native"] else "probe-only"
        p0 = c["probe"][0]; pend = min(c["probe"])
        print(f"  {c['name']:16s} [{tag}] probe AUROC {p0:.3f}→{pend:.3f}"
              + (f"  native {c['native'][0]:.3f}→{min(c['native']):.3f}" if c["has_native"] else ""))
    print(f"\n[Q5] wrote {OUT/'native_vs_probe.json'}")


if __name__ == "__main__":
    main()
