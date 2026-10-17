"""O6 — native-head steering propagation (LaBraM operating layer).

Rebuttal pre-emptive fix #3 / pre-mortem O6: the paper's steering selectivity
(Δ̃) is read out with *linear probes fit on decoded embeddings*, never through
the deployed classifier head. A reviewer can object that we steered a surrogate
instrument, not the model. This tool closes that gap on LaBraM's operating layer
(block 11 = the classification layer) by pushing the *clamped, decoded*
representation through the **real finetuned head** and reporting the native
abnormality AUROC as concept-ranked features are clamped.

Design (mirrors tools/paper_figures/plot_cross_model_steering_concepts_paper.py
exactly for ranking + donor + clamp; only the *readout* changes):

  * Ranking + donor centroid: reuse P.balanced_pair / P.build_tcav_alignment on a
    token-level balanced pool for the steered concept C (identical to the paper).
  * Readout: a fixed, balanced set of Abnormal vs Normal *windows*. For each clamp
    fraction f we clamp the top-f C-ranked SAE features (toward C's donor centroid)
    on every token of those windows, decode, DE-NORMALISE back to raw encoder space,
    mean-pool the window's tokens, and apply the real head -> native abnormality logit.
  * Because the LaBraM native head IS the abnormality classifier, one native curve
    per steered concept suffices:
        - Pathology steering  -> abnormality AUROC should COLLAPSE (target steered on
          the real model, not just a probe).
        - Age steering        -> abnormality AUROC should be SPARED (selectivity holds
          on the real model).

Faithfulness caveats (disclosed, not hidden):
  * SleepFM's real head (extended_classifier->clf, 768-dim multimodal) is not in this
    repo and is incompatible with the single-modality L2 cache -> SleepFM native-head
    propagation is NOT attempted here; it rests on the Table 2 faithfulness bound
    (SAE substitution preserves the native head's AUROC within 0.005 at every SleepFM
    layer). This tool is LaBraM-only by design.
  * The labram_layer11 cache stores post-per-token-fc_norm tokens (model.encode path),
    whereas the classifier applies fc_norm AFTER mean-pooling. We therefore read out
    head(mean_pool(denorm(decoded))) — the real head weights on the model's real
    fc_norm'd features — approximating the pool/fc_norm ordering. ROC-AUC is
    scale-invariant, so the direction of the effect is preserved; the absolute logit
    scale is not claimed to match the deployed model bit-for-bit.

Usage:
  uv run tools/rebuttal/steering_native_head.py --smoke
  uv run tools/rebuttal/steering_native_head.py --concepts Pathology Age --n-windows 300
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))

import plot_cross_model_steering_concepts_paper as P  # noqa: E402

# P.ROOT is computed relative to tools/paper_figures/ and resolves to tools/;
# point it back at the repo root so P.load_layer finds results/ + checkpoints.
P.ROOT = ROOT

DEVICE = P.DEVICE
EXP = "labram_layer11"
TOKENS_PER_WINDOW = 1140  # 19 channels x 60 time patches (LaBraM base)
HEAD_CKPT = ROOT / "checkpoints" / "finetuned" / "labram_binary" / "finetuned.ckpt"
OUT = ROOT / "results" / "steering_significance" / "native_head_labram_layer11.json"
FIG = ROOT / "results" / "steering_significance" / "native_head_labram_layer11"


# ── Native head ───────────────────────────────────────────────────────────────
def load_native_head():
    """Real finetuned LaBraM head: Linear(200->1). fc_norm is already baked into
    the cached (post-fc_norm) token space, so we do not re-apply it."""
    ck = torch.load(HEAD_CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    w = sd["head.weight"].float().to(DEVICE)   # (1, 200)
    b = sd["head.bias"].float().to(DEVICE)     # (1,)
    return w, b


@torch.no_grad()
def native_logit(tokens_raw_pooled, head_w, head_b):
    """tokens_raw_pooled: (W, 200) de-normalised, mean-pooled per window."""
    return (tokens_raw_pooled @ head_w.t() + head_b).squeeze(-1)


# ── Concept ranking + donor (identical to the paper) ────────────────────────────
def concept_ranking_and_donor(spec, sc, sae, am, as_, rng):
    """Token-level TCAV ranking + target-negative donor centroid, exactly as
    P.run_layer_concept computes them."""
    emb = sc["embeddings"]

    def asarr(field):
        return np.asarray([str(v) for v in sc[field]])

    tgt_field = asarr(spec["tgt_field"])
    pos_mask = np.isin(tgt_field, list(spec["tgt_pos"]))
    neg_mask = np.isin(tgt_field, list(spec["tgt_neg"]))
    idx_t, y_t, n_t = P.balanced_pair(emb, pos_mask, neg_mask, P.N_PROBE_PER_GROUP, rng)
    if idx_t is None:
        return None, None, n_t
    Xt_raw = (torch.as_tensor(np.asarray(emb[idx_t])).float().to(DEVICE) - am) / (as_ + 1e-8)
    with torch.no_grad():
        zt = sae.encode(Xt_raw)
    donor_mean = zt[y_t == 0].mean(dim=0)                    # target-negative centroid
    W_dec = sae.decoder.weight.detach()
    w = P.build_tcav_alignment(Xt_raw, y_t, W_dec)
    sorted_features = np.argsort(w)[::-1].copy()
    return sorted_features, donor_mean, n_t


# ── Fixed balanced Abnormal/Normal readout WINDOWS ──────────────────────────────
def build_readout_windows(sc, n_windows_per_group, rng):
    n_tok = int(sc["n_tokens"])
    n_windows = n_tok // TOKENS_PER_WINDOW
    cls_tok = np.asarray([str(v) for v in sc["classification"]])
    # Per-window label = label of the window's first token (all tokens share it).
    win_cls = cls_tok[np.arange(n_windows) * TOKENS_PER_WINDOW]
    abn = np.where(np.isin(win_cls, list(P.ABN_CATS)))[0]
    nrm = np.where(np.isin(win_cls, list(P.NORMAL)))[0]
    n = min(n_windows_per_group, len(abn), len(nrm))
    abn = rng.choice(abn, n, replace=False)
    nrm = rng.choice(nrm, n, replace=False)
    win_idx = np.concatenate([abn, nrm])
    y = np.array([1] * n + [0] * n)
    return win_idx, y, n, n_windows


def gather_window_tokens(sc, win_idx):
    """(W, TOKENS_PER_WINDOW, D) raw token embeddings for the given windows."""
    emb = sc["embeddings"]
    W = len(win_idx)
    D = emb.shape[1]
    out = torch.empty((W, TOKENS_PER_WINDOW, D), dtype=torch.float32)
    for i, w in enumerate(win_idx):
        s = int(w) * TOKENS_PER_WINDOW
        out[i] = torch.as_tensor(np.asarray(emb[s:s + TOKENS_PER_WINDOW])).float()
    return out


# ── Native-head sweep for one concept ranking ───────────────────────────────────
@torch.no_grad()
def native_sweep(sae, am, as_, head_w, head_b, win_tokens, y_read,
                 sorted_features, donor_mean, ns):
    """Return native abnormality AUROC at each clamp count in `ns`."""
    W, Tk, D = win_tokens.shape
    flat = win_tokens.reshape(W * Tk, D).to(DEVICE)
    Xn = (flat - am) / (as_ + 1e-8)
    z_base = sae.encode(Xn)                                   # encode once
    aucs = []
    for n in ns:
        z = z_base if n == 0 else P.substitute(z_base, sorted_features[:n], donor_mean)
        Xdec = sae.decode(z)                                 # normalised decoded
        Xraw = Xdec * (as_ + 1e-8) + am                      # de-normalise -> post-fc_norm space
        pooled = Xraw.reshape(W, Tk, D).mean(dim=1)          # per-window mean-pool
        logit = native_logit(pooled, head_w, head_b).cpu().numpy()
        aucs.append(float(roc_auc_score(y_read, logit)))
    return np.array(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="+", default=["Pathology", "Age"])
    ap.add_argument("--n-windows", type=int, default=300,
                    help="windows per group (Abnormal / Normal) for the native readout")
    ap.add_argument("--n-rand", type=int, default=3, help="random-ranking baseline draws")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_windows, args.n_rand = 60, 1

    print(f"Loading steering cache + SAE for {EXP} …", flush=True)
    sc, sae, am, as_, _enr, nf = P.load_layer(EXP)
    head_w, head_b = load_native_head()
    print(f"  {int(sc['n_tokens']):,} tokens x {sc['embed_dim']} dims · "
          f"SAE n_features={nf} · head {tuple(head_w.shape)}", flush=True)

    rng = np.random.default_rng(args.seed)
    win_idx, y_read, n_per, n_windows_total = build_readout_windows(sc, args.n_windows, rng)
    print(f"  readout windows: {n_per} Abnormal + {n_per} Normal "
          f"(of {n_windows_total:,} total)", flush=True)
    win_tokens = gather_window_tokens(sc, win_idx)

    ns = sorted(set(int(round(f * nf)) for f in np.linspace(0, 1, P.N_STEPS + 1)))
    fracs = (np.array(ns) / nf).tolist()

    cells = []
    for cname in args.concepts:
        spec = next((s for s in P.CONCEPTS if s["name"] == cname), None)
        if spec is None:
            print(f"  [skip] unknown concept {cname}"); continue
        t0 = time.time()
        rng_c = np.random.default_rng(args.seed)
        sorted_features, donor_mean, n_t = concept_ranking_and_donor(spec, sc, sae, am, as_, rng_c)
        if sorted_features is None:
            print(f"  [skip] {cname}: too few tokens ({n_t}/group)"); continue

        tcav_auc = native_sweep(sae, am, as_, head_w, head_b, win_tokens, y_read,
                                sorted_features, donor_mean, ns)
        # Random-ranking baseline
        rand_curves = []
        rng_r = np.random.default_rng(args.seed + 1)
        for _ in range(args.n_rand):
            perm = rng_r.permutation(nf)
            rand_curves.append(native_sweep(sae, am, as_, head_w, head_b, win_tokens,
                                            y_read, perm, donor_mean, ns))
        rand_mean = np.mean(rand_curves, axis=0)

        frac_arr = np.array(fracs)
        # At full clamp the representation IS the donor centroid -> AUROC=0.5 for
        # any ranking, so endpoint drops are uninformative. The native steering
        # effect is the integrated gap between random and TCAV-ranked collapse:
        # positive => targeted clamping drives the real head down faster than random.
        steer_area = float(np.trapezoid(rand_mean - tcav_auc, frac_arr))

        def auc_at(f):
            return float(np.interp(f, frac_arr, tcav_auc))

        cells.append({
            "concept": cname,
            "native_auc": tcav_auc.tolist(),
            "native_auc_rand_mean": rand_mean.tolist(),
            "fracs": fracs,
            "auc0": float(tcav_auc[0]),
            "auc_at_0.10": auc_at(0.10),
            "auc_at_0.25": auc_at(0.25),
            "auc_at_0.50": auc_at(0.50),
            "steer_area_vs_random": steer_area,
        })
        print(f"  {cname:10s} native AUROC0={tcav_auc[0]:.3f} | @10%={auc_at(0.10):.3f} "
              f"@25%={auc_at(0.25):.3f} @50%={auc_at(0.50):.3f} | "
              f"steer-area vs random={steer_area:+.3f}  [{time.time()-t0:.0f}s]", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": EXP, "readout": "native LaBraM head (Linear 200->1)",
        "n_windows_per_group": n_per, "n_features": nf, "seed": args.seed,
        "caveat": "post-fc_norm cache; head(mean_pool(denorm(decode))) approximates "
                  "the classifier's fc_norm(mean_pool(.)) ordering (ROC scale-invariant). "
                  "SleepFM native head not applicable (external multimodal 768-d classifier).",
        "cells": cells,
    }, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")

    if not args.no_plot and cells:
        make_plot(cells)


def make_plot(cells):
    from mecheeg.nature_fig_style import (
        set_nature_style, nature_figure, nature_palette, save_nature_figure,
    )
    set_nature_style()
    pal = nature_palette()
    fig, axes = nature_figure(width="double", height_mm=62, ncols=len(cells))
    if len(cells) == 1:
        axes = [axes]
    for ax, cell in zip(np.atleast_1d(axes), cells):
        fr = cell["fracs"]
        ax.plot(fr, cell["native_auc"], "o-", color=pal[0], markersize=3,
                label="TCAV-ranked clamp")
        ax.plot(fr, cell["native_auc_rand_mean"], "s--", color=pal[2], markersize=2.5,
                label="random clamp")
        ax.axhline(0.5, color="0.6", lw=0.5, ls=":")
        ax.set_ylim(0.45, 1.0)
        ax.set_xlabel("fraction of features clamped")
        ax.set_title(cell["concept"] + " steering")
    np.atleast_1d(axes)[0].set_ylabel("native head\nabnormality AUROC")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=5, loc="lower left")
    save_nature_figure(fig, str(FIG) + ".pdf")
    save_nature_figure(fig, str(FIG) + ".png")
    print(f"Wrote {FIG.relative_to(ROOT)}.{{pdf,png}}")


if __name__ == "__main__":
    main()
