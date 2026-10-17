"""Significance layer for the cross-model steering selectivity metric (Δ̃).

Addresses rebuttal pre-mortem O5: the headline "excess selectivity" Δ̃ metric
(§4.4, fig:steerability) ships with no significance test, no confidence interval,
and no explicit regime thresholds. This script adds all three, recomputed directly
from the existing steering caches (results/steering_cache/{exp}/steering_cache.pt)
using the *identical* sweep/probe machinery as
tools/paper_figures/plot_cross_model_steering_concepts_paper.py, so the observed
Δ̃ reconciles with the paper's bar heights.

For each (experiment, concept) cell it produces:
  • observed Δ̃  (area = ∫(off_auc − tgt_auc) d frac under TCAV-ranked clamping)
  • permutation p-value  — null = random feature rankings; p = (1+#{null≥obs})/(B+1)
  • bootstrap 95% CI on Δ̃  — resample the probe hold-out set (efficient: per-example
    step scores are computed once for the observed ranking, then resampled)
  • AUROC₀ (encoding strength) with bootstrap 95% CI
  • regime label  ∈ {non-encoded, encoded-but-entangled, selectively-steerable}

Regime thresholds (explicit, defined once here):
  ENC_MODERATE = 0.70   # AUROC₀ below this ⇒ concept not linearly encoded
  DELTA_MIN    = 0.05   # minimum Δ̃ magnitude to call steering "selective"
  ALPHA        = 0.05   # permutation-test significance level
  non-encoded          : AUROC₀ <  ENC_MODERATE
  selectively-steerable: AUROC₀ >= ENC_MODERATE  AND  p < ALPHA  AND  ci_low > 0  AND  Δ̃ >= DELTA_MIN
  encoded-but-entangled: AUROC₀ >= ENC_MODERATE  AND  not selective (clamping hits off-target as hard as target)

Usage:
  uv run tools/rebuttal/steering_significance.py --experiments sleepfm_finetuned_layer2
  uv run tools/rebuttal/steering_significance.py --experiments all --b-perm 500 --b-boot 1000
  uv run tools/rebuttal/steering_significance.py --smoke   # 1 cell, tiny B, prints timing

Outputs: results/steering_significance/{experiment}.json  (one file per experiment,
resumable — existing per-concept entries are skipped unless --force).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))

# Reuse the EXACT machinery behind the paper figure so numbers reconcile.
import plot_cross_model_steering_concepts_paper as P  # noqa: E402

# P computes its own ROOT from __file__ = tools/paper_figures/… → resolves to
# tools/ (one level too shallow after the paper_figures/ move). Point its path
# helpers (load_layer) at the real repo root.
P.ROOT = ROOT

DEVICE = P.DEVICE
RNG_SEED = P.RNG_SEED

# ── Regime thresholds (explicit) ──────────────────────────────────────────────
ENC_MODERATE = 0.70
DELTA_MIN = 0.05
ALPHA = 0.05

OUT_DIR = ROOT / "results" / "steering_significance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The 12 experiments shown in the paper figure.
FIGURE_EXPERIMENTS = (
    [f"sleepfm_finetuned_layer{L}" for L in P._SLEEPFM_LAYERS]
    + [f"labram_layer{L}" for L in P._LABRAM_LAYERS]
    + [f"reve_qjbe08_layer{L}" for L in P._REVE_LAYERS]
)


def _sweep_scores(sae, zt, zo, donor_mean, ranking, ns, idx_v_t, idx_v_o,
                  tgt_probe, off_probe):
    """Like P.sweep_one_ranking but also returns per-example hold-out scores at
    each step (needed for cheap bootstrap). Returns (tgt_auc, off_auc,
    tgt_scores[n_steps, n_holdout_t], off_scores[n_steps, n_holdout_o])."""
    idx_v_t_t = torch.as_tensor(idx_v_t, device=DEVICE, dtype=torch.long)
    idx_v_o_t = torch.as_tensor(idx_v_o, device=DEVICE, dtype=torch.long)
    tgt_auc, off_auc, tgt_sc, off_sc = [], [], [], []
    for n in ns:
        with torch.no_grad():
            if n == 0:
                Xt_dec = sae.decode(zt); Xo_dec = sae.decode(zo)
            else:
                feats = ranking[:n]
                ztc = P.substitute(zt, feats, donor_mean)
                zoc = P.substitute(zo, feats, donor_mean)
                Xt_dec = sae.decode(ztc); Xo_dec = sae.decode(zoc)
            Xt_v = Xt_dec.index_select(0, idx_v_t_t)
            Xo_v = Xo_dec.index_select(0, idx_v_o_t)
            st = (Xt_v @ tgt_probe.w + tgt_probe.b).cpu().numpy()
            so = (Xo_v @ off_probe.w + off_probe.b).cpu().numpy()
        tgt_sc.append(st); off_sc.append(so)
    tgt_sc = np.asarray(tgt_sc); off_sc = np.asarray(off_sc)
    return tgt_sc, off_sc


def _auc_curve(scores, y):
    return np.array([roc_auc_score(y, scores[i]) for i in range(scores.shape[0])])


def _area(tgt_auc, off_auc, fracs):
    return float(np.trapezoid(off_auc - tgt_auc, fracs))


def analyse_cell(spec, layer_state, b_perm, b_boot, rng_seed=RNG_SEED):
    """Recompute one (concept, layer) cell with significance + CI. Mirrors
    P.run_layer_concept's setup exactly, then adds the statistics layer."""
    sc, sae, am, as_, _enr, nf = layer_state
    rng = np.random.default_rng(rng_seed)

    def asarr(field):
        return np.asarray([str(v) for v in sc[field]])

    if spec["filter_field"] is not None:
        filt = np.isin(asarr(spec["filter_field"]), list(spec["filter_set"]))
    else:
        filt = np.ones(len(sc["embeddings"]), dtype=bool)

    tgt_field = asarr(spec["tgt_field"]); off_field = asarr(spec["off_field"])
    tgt_pos = filt & np.isin(tgt_field, list(spec["tgt_pos"]))
    tgt_neg = filt & np.isin(tgt_field, list(spec["tgt_neg"]))
    off_pos = filt & np.isin(off_field, list(spec["off_pos"]))
    off_neg = filt & np.isin(off_field, list(spec["off_neg"]))

    emb = sc["embeddings"]
    idx_t, y_t, n_t = P.balanced_pair(emb, tgt_pos, tgt_neg, P.N_PROBE_PER_GROUP, rng)
    idx_o, y_o, n_o = P.balanced_pair(emb, off_pos, off_neg, P.N_PROBE_PER_GROUP, rng)
    if idx_t is None or idx_o is None:
        return None

    Xt_raw = (torch.as_tensor(np.asarray(emb[idx_t])).float().to(DEVICE) - am) / (as_ + 1e-8)
    Xo_raw = (torch.as_tensor(np.asarray(emb[idx_o])).float().to(DEVICE) - am) / (as_ + 1e-8)
    with torch.no_grad():
        zt = sae.encode(Xt_raw); zo = sae.encode(Xo_raw)
    donor_mean = zt[(y_t == 0)].mean(dim=0)

    W_dec = sae.decoder.weight.detach()
    align = P.build_tcav_alignment(Xt_raw, y_t, W_dec)
    tcav_ranking = np.argsort(align)[::-1].copy()

    ns = sorted(set(int(round(f * nf)) for f in np.linspace(0, 1, P.N_STEPS + 1)))
    fracs = np.array(ns) / nf

    with torch.no_grad():
        Xt_dec_clean = sae.decode(zt); Xo_dec_clean = sae.decode(zo)
    Xt_tr, y_t_tr, idx_v_t, y_v_t = P.split_train_holdout(Xt_dec_clean, y_t, rng_seed)
    Xo_tr, y_o_tr, idx_v_o, y_v_o = P.split_train_holdout(Xo_dec_clean, y_o, rng_seed)
    tgt_probe = P.fit_probe(Xt_tr, y_t_tr)
    off_probe = P.fit_probe(Xo_tr, y_o_tr)

    # Observed TCAV ranking — keep per-example scores for the bootstrap.
    tgt_sc, off_sc = _sweep_scores(sae, zt, zo, donor_mean, tcav_ranking, ns,
                                   idx_v_t, idx_v_o, tgt_probe, off_probe)
    tgt_auc = _auc_curve(tgt_sc, y_v_t); off_auc = _auc_curve(off_sc, y_v_o)
    obs_area = _area(tgt_auc, off_auc, fracs)
    auroc0 = float(tgt_auc[0])

    # ── Permutation null: random feature rankings ────────────────────────────
    rng_p = np.random.default_rng(rng_seed + 12345)
    null_areas = np.empty(b_perm)
    null_t_sum = np.zeros(len(ns)); null_o_sum = np.zeros(len(ns))
    for b in range(b_perm):
        perm = rng_p.permutation(nf)
        t_au, o_au = P.sweep_one_ranking(sae, zt, zo, donor_mean, perm, ns,
                                         tgt_probe, off_probe, idx_v_t, y_v_t,
                                         idx_v_o, y_v_o)
        null_areas[b] = _area(t_au, o_au, fracs)
        null_t_sum += t_au; null_o_sum += o_au
    # One-sided: is TCAV-guided clamping MORE target-selective than a random ranking?
    p_value = float((1 + np.sum(null_areas >= obs_area)) / (b_perm + 1))
    null_mean = float(null_areas.mean()); null_std = float(null_areas.std(ddof=1))

    # ── Bootstrap CI on Δ̃ and AUROC₀ (resample hold-out, reuse cached scores) ─
    rng_b = np.random.default_rng(rng_seed + 999)
    nvt = len(y_v_t); nvo = len(y_v_o)
    boot_area = np.empty(b_boot); boot_auroc0 = np.empty(b_boot)
    for b in range(b_boot):
        bt = rng_b.integers(0, nvt, nvt)
        bo = rng_b.integers(0, nvo, nvo)
        yt_b = y_v_t[bt]; yo_b = y_v_o[bo]
        # skip degenerate resamples (single class) by nudging — rare with balanced sets
        if yt_b.min() == yt_b.max() or yo_b.min() == yo_b.max():
            boot_area[b] = obs_area; boot_auroc0[b] = auroc0; continue
        t_au = np.array([roc_auc_score(yt_b, tgt_sc[i][bt]) for i in range(tgt_sc.shape[0])])
        o_au = np.array([roc_auc_score(yo_b, off_sc[i][bo]) for i in range(off_sc.shape[0])])
        boot_area[b] = _area(t_au, o_au, fracs)
        boot_auroc0[b] = float(t_au[0])
    ci_low, ci_high = (float(np.percentile(boot_area, 2.5)),
                       float(np.percentile(boot_area, 97.5)))
    a0_low, a0_high = (float(np.percentile(boot_auroc0, 2.5)),
                       float(np.percentile(boot_auroc0, 97.5)))

    # ── Regime classification ────────────────────────────────────────────────
    significant = (p_value < ALPHA) and (ci_low > 0) and (obs_area >= DELTA_MIN)
    if auroc0 < ENC_MODERATE:
        regime = "non-encoded"
    elif significant:
        regime = "selectively-steerable"
    else:
        regime = "encoded-but-entangled"

    return {
        "concept": spec["name"], "n_target_per_group": n_t, "n_off_per_group": n_o,
        "n_features": int(nf), "n_holdout_target": int(nvt), "n_holdout_off": int(nvo),
        "auroc0": auroc0, "auroc0_ci": [a0_low, a0_high],
        "delta_tilde": obs_area, "delta_tilde_ci": [ci_low, ci_high],
        "perm_null_mean": null_mean, "perm_null_std": null_std,
        "p_value": p_value, "b_perm": int(b_perm), "b_boot": int(b_boot),
        "significant": bool(significant), "regime": regime,
        "thresholds": {"enc_moderate": ENC_MODERATE, "delta_min": DELTA_MIN, "alpha": ALPHA},
        # Curves (camera-ready Fig 5): observed sweep + mean of the permutation-null sweeps.
        "excess_delta": obs_area - null_mean,
        "fracs": fracs.tolist(), "tgt_auc": tgt_auc.tolist(), "off_auc": off_auc.tolist(),
        "null_tgt_auc_mean": (null_t_sum / max(b_perm, 1)).tolist(),
        "null_off_auc_mean": (null_o_sum / max(b_perm, 1)).tolist(),
    }


def run_experiment(exp, b_perm, b_boot, force=False, concepts=None):
    out_path = OUT_DIR / f"{exp}.json"
    existing = {}
    if out_path.exists() and not force:
        existing = {e["concept"]: e for e in json.loads(out_path.read_text())["cells"]}

    sc_path = ROOT / "results/steering_cache" / exp / "steering_cache.pt"
    if not sc_path.exists():
        print(f"[skip — no steering cache] {exp}  ({sc_path.relative_to(ROOT)})")
        return None
    print(f"\n=== {exp} ===", flush=True)
    layer_state = P.load_layer(exp)

    specs = [s for s in P.CONCEPTS if concepts is None or s["name"] in concepts]
    cells = list(existing.values())
    done = set(existing)
    for spec in specs:
        if spec["name"] in done:
            print(f"  [cached] {spec['name']}"); continue
        t0 = time.time()
        r = analyse_cell(spec, layer_state, b_perm, b_boot)
        if r is None:
            print(f"  [skip] {spec['name']}: insufficient samples"); continue
        cells.append(r)
        print(f"  {spec['name']:26s} Δ̃={r['delta_tilde']:+.4f} "
              f"CI[{r['delta_tilde_ci'][0]:+.3f},{r['delta_tilde_ci'][1]:+.3f}] "
              f"AUROC₀={r['auroc0']:.3f} p={r['p_value']:.4f} "
              f"→ {r['regime']}  ({time.time()-t0:.1f}s)", flush=True)
        out_path.write_text(json.dumps(
            {"experiment": exp, "b_perm": b_perm, "b_boot": b_boot, "cells": cells}, indent=2))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=["sleepfm_finetuned_layer2"],
                    help="experiment names, or 'all' for the 12 figure cells")
    ap.add_argument("--concepts", nargs="+", default=None)
    ap.add_argument("--b-perm", type=int, default=500)
    ap.add_argument("--b-boot", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="1 cell, tiny B, prints timing then exits")
    args = ap.parse_args()

    if args.smoke:
        print("SMOKE TEST: Pathology / sleepfm_finetuned_layer2, B_perm=20, B_boot=100")
        ls = P.load_layer("sleepfm_finetuned_layer2")
        spec = next(s for s in P.CONCEPTS if s["name"] == "Pathology")
        t0 = time.time()
        r = analyse_cell(spec, ls, b_perm=20, b_boot=100)
        print(json.dumps(r, indent=2))
        print(f"\none cell (B_perm=20) took {time.time()-t0:.1f}s → "
              f"full B_perm=500 ≈ {(time.time()-t0)*25:.0f}s/cell")
        return

    exps = FIGURE_EXPERIMENTS if args.experiments == ["all"] else args.experiments
    for exp in exps:
        run_experiment(exp, args.b_perm, args.b_boot, args.force, args.concepts)


if __name__ == "__main__":
    main()
