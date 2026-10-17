"""Rebuttal T2 — does the SAE beat simpler direction sources for *selective* steering?

Answers AC point 2 / rkJV Q2 / HdqW Q3: compare TopK-SAE concept steering against
dense CAV, PCA, ICA, a random sparse dictionary, and centroid-based steering, using
the IDENTICAL selectivity metric (Δ̃ = ∫(off_auc − tgt_auc) d x) and random-baseline
protocol as the paper (reused verbatim from
tools/paper_figures/plot_cross_model_steering_concepts_paper.py).

For every (concept, method) we report:
  AUROC₀  (target encoding at no intervention)
  area    (raw selectivity area under the method's steer sweep)
  rand    (mean±std area of the method's own random-ranking / random-direction null)
  Δ̃      (excess selectivity = area − rand_mean)
  p       (permutation: fraction of null areas ≥ observed)

Method families and what their "sweep axis" is:
  SAE / PCA / ICA / random-dict : DICTIONARY methods. Clamp the top-n concept-aligned
      code coordinates to the target-negative donor, decode, sweep n = fraction of atoms.
      Null = random atom rankings.
  dense-CAV                     : single concept direction. Push each embedding's CAV
      projection to the donor (Normal) mean; sweep α∈[0,1]. Null = random unit directions.
  centroid                      : move each embedding toward the target-negative class
      centroid; sweep α∈[0,1]. Null = movement toward random other centroids.

All methods operate in the SAE's standardized embedding space and fit their probes on
their own clean reconstruction, so Δ̃ is comparable across methods.

Usage:
  uv run tools/rebuttal/rebuttal_steering_baselines.py --experiment sleepfm_finetuned_layer2
  uv run tools/rebuttal/rebuttal_steering_baselines.py --experiment sleepfm_finetuned_layer2 \
      --concepts Pathology Age --b-perm 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))

import plot_cross_model_steering_concepts_paper as P  # noqa: E402

P.ROOT = ROOT
DEVICE = P.DEVICE
SEED = P.RNG_SEED
N_GLOBAL_FIT = 12000  # tokens sampled to fit PCA/ICA/random dict per layer

OUT_DIR = ROOT / "results" / "rebuttal"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dictionary adapters: expose encode(X)->z, decode(z)->X, W_dec (d×K) ──────────
class DictSource:
    """Wraps a linear dictionary so it plugs into the SAE clamp/decode sweep."""

    def __init__(self, name, encode_fn, decode_fn, W_dec, k=None):
        self.name = name
        self._encode = encode_fn
        self._decode = decode_fn
        self.W_dec = W_dec           # torch (d, K) on DEVICE — atom directions for ranking
        self.K = int(W_dec.shape[1])
        self.k = k                   # top-k sparsity (random dict); None = dense code

    def encode(self, X):
        z = self._encode(X)
        if self.k is not None and self.k < z.shape[1]:
            thresh = torch.topk(z.abs(), self.k, dim=1).values[:, -1:]
            z = torch.where(z.abs() >= thresh, z, torch.zeros_like(z))
        return z

    def decode(self, z):
        return self._decode(z)


def make_sae_source(sae):
    return DictSource("SAE", sae.encode, sae.decode, sae.decoder.weight.detach(), k=None)


def make_pca_source(Xfit):
    p = PCA(n_components=Xfit.shape[1], svd_solver="full").fit(Xfit)
    comp = torch.as_tensor(p.components_, dtype=torch.float32, device=DEVICE)   # (K, d)
    mean = torch.as_tensor(p.mean_, dtype=torch.float32, device=DEVICE)
    enc = lambda X: (X - mean) @ comp.T
    dec = lambda z: z @ comp + mean
    return DictSource("PCA", enc, dec, comp.T.contiguous(), k=None)


def make_ica_source(Xfit, seed=SEED):
    ica = FastICA(n_components=Xfit.shape[1], random_state=seed, max_iter=400, whiten="unit-variance")
    ica.fit(Xfit)
    comp = torch.as_tensor(ica.components_, dtype=torch.float32, device=DEVICE)  # (K, d)
    mix = torch.as_tensor(ica.mixing_, dtype=torch.float32, device=DEVICE)       # (d, K)
    mean = torch.as_tensor(ica.mean_, dtype=torch.float32, device=DEVICE)
    enc = lambda X: (X - mean) @ comp.T
    dec = lambda z: z @ mix.T + mean
    return DictSource("ICA", enc, dec, mix, k=None)


def make_random_source(d, K, k, seed=SEED):
    g = torch.Generator(device="cpu").manual_seed(seed)
    D = torch.randn(d, K, generator=g)
    D = (D / (D.norm(dim=0, keepdim=True) + 1e-8)).to(DEVICE)   # unit-norm atoms
    enc = lambda X: X @ D          # analysis = D^T x  (top-k applied in DictSource.encode)
    dec = lambda z: z @ D.T
    return DictSource(f"RandDict(k={k})", enc, dec, D, k=k)


# ── Metric helpers (mirror the paper) ────────────────────────────────────────────
def _area(tgt_auc, off_auc, xs):
    return float(np.trapezoid(off_auc - tgt_auc, xs))


def _fit_probes(clean_t, clean_o, y_t, y_o):
    Xt_tr, yt_tr, idx_v_t, y_v_t = P.split_train_holdout(clean_t, y_t, SEED)
    Xo_tr, yo_tr, idx_v_o, y_v_o = P.split_train_holdout(clean_o, y_o, SEED)
    return (P.fit_probe(Xt_tr, yt_tr), P.fit_probe(Xo_tr, yo_tr),
            idx_v_t, y_v_t, idx_v_o, y_v_o)


def _auc_at(probe, X, idx_v, y_v):
    Xv = X.index_select(0, torch.as_tensor(idx_v, device=DEVICE, dtype=torch.long))
    s = (Xv @ probe.w + probe.b).cpu().numpy()
    return float(roc_auc_score(y_v, s))


# ── Dictionary-method sweep + permutation null ────────────────────────────────────
def eval_dict(src, Xt_std, Xo_std, y_t, y_o, b_perm, rng):
    with torch.no_grad():
        zt = src.encode(Xt_std); zo = src.encode(Xo_std)
        clean_t = src.decode(zt); clean_o = src.decode(zo)
    donor = zt[y_t == 0].mean(dim=0)
    tgt_probe, off_probe, iv_t, yv_t, iv_o, yv_o = _fit_probes(clean_t, clean_o, y_t, y_o)

    # concept ranking: |cav · atom|
    cav = P.fit_probe(Xt_std, y_t).w
    cav = cav / (cav.norm() + 1e-8)
    align = (cav @ src.W_dec).abs().cpu().numpy()
    ranking = np.argsort(align)[::-1].copy()

    K = src.K
    ns = sorted(set(int(round(f * K)) for f in np.linspace(0, 1, P.N_STEPS + 1)))
    xs = np.array(ns) / K

    def sweep(order):
        t_au, o_au = [], []
        for n in ns:
            with torch.no_grad():
                if n == 0:
                    Xt_d, Xo_d = clean_t, clean_o
                else:
                    feats = order[:n]
                    Xt_d = src.decode(P.substitute(zt, feats, donor))
                    Xo_d = src.decode(P.substitute(zo, feats, donor))
            t_au.append(_auc_at(tgt_probe, Xt_d, iv_t, yv_t))
            o_au.append(_auc_at(off_probe, Xo_d, iv_o, yv_o))
        return np.array(t_au), np.array(o_au)

    t_au, o_au = sweep(ranking)
    obs = _area(t_au, o_au, xs)
    auroc0 = float(t_au[0])
    null = np.array([_area(*sweep(rng.permutation(K)), xs) for _ in range(b_perm)])
    p = float((1 + np.sum(null >= obs)) / (b_perm + 1))
    return dict(method=src.name, auroc0=auroc0, area=obs,
                rand_mean=float(null.mean()), rand_std=float(null.std(ddof=1)),
                delta=obs - float(null.mean()), p=p, axis="atom-fraction", K=K)


# ── Raw-space methods: dense CAV and centroid ─────────────────────────────────────
def eval_cav(Xt_std, Xo_std, y_t, y_o, b_perm, rng):
    tgt_probe, off_probe, iv_t, yv_t, iv_o, yv_o = _fit_probes(Xt_std, Xo_std, y_t, y_o)
    cav = P.fit_probe(Xt_std, y_t).w
    cav = cav / (cav.norm() + 1e-8)

    def push(X, direction):
        # move each row's projection on `direction` to the target-negative donor mean
        proj = X @ direction
        donor_proj = proj[y_t == 0].mean() if X is Xt_std else (Xo_std @ direction)[y_o == 0].mean()
        xs = np.linspace(0, 1, P.N_STEPS + 1)
        return [X + (a * (donor_proj - proj)).unsqueeze(1) * direction.unsqueeze(0) for a in xs], xs

    def sweep(direction):
        Xt_list, xs = push(Xt_std, direction); Xo_list, _ = push(Xo_std, direction)
        t_au = np.array([_auc_at(tgt_probe, Xt_list[i], iv_t, yv_t) for i in range(len(xs))])
        o_au = np.array([_auc_at(off_probe, Xo_list[i], iv_o, yv_o) for i in range(len(xs))])
        return t_au, o_au, xs

    t_au, o_au, xs = sweep(cav)
    obs = _area(t_au, o_au, xs); auroc0 = float(t_au[0])
    d = Xt_std.shape[1]
    null = []
    for _ in range(b_perm):
        r = torch.randn(d, device=DEVICE); r = r / (r.norm() + 1e-8)
        tn, on, _ = sweep(r); null.append(_area(tn, on, xs))
    null = np.array(null)
    p = float((1 + np.sum(null >= obs)) / (b_perm + 1))
    return dict(method="dense-CAV", auroc0=auroc0, area=obs,
                rand_mean=float(null.mean()), rand_std=float(null.std(ddof=1)),
                delta=obs - float(null.mean()), p=p, axis="alpha", K=1)


def eval_centroid(Xt_std, Xo_std, y_t, y_o, b_perm, rng):
    tgt_probe, off_probe, iv_t, yv_t, iv_o, yv_o = _fit_probes(Xt_std, Xo_std, y_t, y_o)
    c_neg_t = Xt_std[y_t == 0].mean(dim=0)
    xs = np.linspace(0, 1, P.N_STEPS + 1)

    def sweep(target_point):
        t_au = np.array([_auc_at(tgt_probe, (1 - a) * Xt_std + a * target_point, iv_t, yv_t) for a in xs])
        o_au = np.array([_auc_at(off_probe, (1 - a) * Xo_std + a * target_point, iv_o, yv_o) for a in xs])
        return t_au, o_au

    t_au, o_au = sweep(c_neg_t)
    obs = _area(t_au, o_au, xs); auroc0 = float(t_au[0])
    # null: move toward a random point of the same norm as the donor centroid
    norm = c_neg_t.norm()
    null = []
    for _ in range(b_perm):
        r = torch.randn_like(c_neg_t); r = r / (r.norm() + 1e-8) * norm
        tn, on = sweep(r); null.append(_area(tn, on, xs))
    null = np.array(null)
    p = float((1 + np.sum(null >= obs)) / (b_perm + 1))
    return dict(method="centroid", auroc0=auroc0, area=obs,
                rand_mean=float(null.mean()), rand_std=float(null.std(ddof=1)),
                delta=obs - float(null.mean()), p=p, axis="alpha", K=1)


def build_pools(sc, spec, am, as_, rng):
    def asarr(field):
        return np.asarray([str(v) for v in sc[field]])
    filt = (np.ones(len(sc["embeddings"]), dtype=bool) if spec["filter_field"] is None
            else np.isin(asarr(spec["filter_field"]), list(spec["filter_set"])))
    tf, off = asarr(spec["tgt_field"]), asarr(spec["off_field"])
    tpos = filt & np.isin(tf, list(spec["tgt_pos"])); tneg = filt & np.isin(tf, list(spec["tgt_neg"]))
    opos = filt & np.isin(off, list(spec["off_pos"])); oneg = filt & np.isin(off, list(spec["off_neg"]))
    emb = sc["embeddings"]
    idx_t, y_t, n_t = P.balanced_pair(emb, tpos, tneg, P.N_PROBE_PER_GROUP, rng)
    idx_o, y_o, n_o = P.balanced_pair(emb, opos, oneg, P.N_PROBE_PER_GROUP, rng)
    if idx_t is None or idx_o is None:
        return None
    std = lambda idx: (torch.as_tensor(np.asarray(emb[idx])).float().to(DEVICE) - am) / (as_ + 1e-8)
    return std(idx_t), std(idx_o), y_t, y_o, n_t, n_o


def _fmt(r, t0):
    return (f"  {r['method']:14s} AUROC0={r['auroc0']:.3f} area={r['area']:+.4f} "
            f"rand={r['rand_mean']:+.4f}±{r['rand_std']:.3f} Δ̃={r['delta']:+.4f} "
            f"p={r['p']:.3f}  ({time.time()-t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="sleepfm_finetuned_layer2")
    ap.add_argument("--concepts", nargs="+", default=None)
    ap.add_argument("--b-perm", type=int, default=200)
    ap.add_argument("--sae-seeds", type=int, default=0,
                    help="if >0, retrain N fresh SAEs (O5-consistent) and report the SAE arm "
                         "as a seed distribution; the on-disk L2 SAE is the O5 outlier so this "
                         "is the FAIR comparison. 0 = use the on-disk SAE.")
    ap.add_argument("--layer-idx", type=int, default=2)
    ap.add_argument("--expansion", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    sc, on_disk_sae, am0, as0, _enr, nf = P.load_layer(args.experiment)
    specs = [s for s in P.CONCEPTS if args.concepts is None or s["name"] in args.concepts]

    fresh_saes = []
    if args.sae_seeds > 0:
        # Fresh-SAE normalization (matches tools/steering_seed_robustness.train_sae).
        from steering_seed_robustness import train_sae  # noqa: E402
        emb = torch.as_tensor(np.asarray(sc["embeddings"])).float()
        mean = emb.mean(dim=0); std = emb.std(dim=0).clamp(min=1e-8)
        Xn = (emb - mean) / std
        am, as_ = mean.to(DEVICE), std.to(DEVICE)
        nf = int(args.expansion * emb.shape[1])
        print(f"[seeds] retraining {args.sae_seeds} fresh SAEs (E={args.expansion}, k={args.k}, "
              f"nf={nf}) on {emb.shape[0]}×{emb.shape[1]}", flush=True)
        for s in range(args.sae_seeds):
            t0 = time.time()
            fresh_saes.append(train_sae(Xn, s, args.expansion, args.k, args.epochs, 3e-4, 2048, args.layer_idx))
            print(f"  trained SAE seed {s}  ({time.time()-t0:.1f}s)", flush=True)
    else:
        am, as_ = am0, as0

    d = int(am.shape[0])
    rng = np.random.default_rng(SEED)
    emb_raw = sc["embeddings"]
    gi = rng.choice(len(emb_raw), min(N_GLOBAL_FIT, len(emb_raw)), replace=False)
    Xg = ((torch.as_tensor(np.asarray(emb_raw[gi])).float().to(DEVICE) - am) / (as_ + 1e-8)).cpu().numpy()
    print(f"[fit] PCA/ICA/random baselines on {Xg.shape[0]}×{d}", flush=True)
    baselines = [make_pca_source(Xg), make_ica_source(Xg), make_random_source(d, nf, args.k)]

    results = {}
    for spec in specs:
        pools = build_pools(sc, spec, am, as_, np.random.default_rng(SEED))
        if pools is None:
            print(f"[skip] {spec['name']}: insufficient samples"); continue
        Xt, Xo, y_t, y_o, n_t, n_o = pools
        print(f"\n=== {spec['name']}  (n={n_t}/group) ===", flush=True)
        rows = []

        # SAE arm — fresh seeds (fair) or single on-disk SAE.
        if args.sae_seeds > 0:
            sae_deltas = []
            for s, sae_s in enumerate(fresh_saes):
                t0 = time.time()
                r = eval_dict(make_sae_source(sae_s), Xt, Xo, y_t, y_o, args.b_perm,
                              np.random.default_rng(SEED + 7))
                r["method"] = f"SAE[seed{s}]"; rows.append(r); sae_deltas.append(r["delta"])
                print(_fmt(r, t0), flush=True)
            sd = np.array(sae_deltas)
            print(f"  SAE fresh-seed Δ̃: mean {sd.mean():+.4f} ± {sd.std(ddof=1):.4f} "
                  f"median {np.median(sd):+.4f}  ({int((sd > 0).sum())}/{len(sd)} positive)", flush=True)
        else:
            t0 = time.time()
            r = eval_dict(make_sae_source(on_disk_sae), Xt, Xo, y_t, y_o, args.b_perm,
                          np.random.default_rng(SEED + 7))
            r["method"] = "SAE[on-disk]"; rows.append(r); print(_fmt(r, t0), flush=True)

        for src in baselines:
            t0 = time.time()
            r = eval_dict(src, Xt, Xo, y_t, y_o, args.b_perm, np.random.default_rng(SEED + 7))
            rows.append(r); print(_fmt(r, t0), flush=True)
        for fn in (eval_cav, eval_centroid):
            t0 = time.time()
            r = fn(Xt, Xo, y_t, y_o, args.b_perm, np.random.default_rng(SEED + 7))
            rows.append(r); print(_fmt(r, t0), flush=True)
        results[spec["name"]] = {"n_per_group": n_t, "rows": rows}

    tag = f"_seeds{args.sae_seeds}" if args.sae_seeds > 0 else ""
    out = OUT_DIR / f"steering_baselines_{args.experiment}{tag}.json"
    out.write_text(json.dumps({"experiment": args.experiment, "b_perm": args.b_perm,
                               "sae_seeds": args.sae_seeds, "results": results}, indent=2))
    print(f"\n[T2] wrote {out}")


if __name__ == "__main__":
    main()
