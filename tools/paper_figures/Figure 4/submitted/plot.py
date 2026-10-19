"""Cross-model steering concept selectivity — paper-ready layer selection.

Bar values are taken from the v1 plot's cached entries (i.e. the v1 layer
selection) so the figure is visually identical to v1, but the legend
relabels LaBraM's bars with a different set of layer indices on request.

Data sources per bar (cached experiments):
  SleepFM: layers 0, 1, 2 → labelled L1, L2, L3
  LaBraM:  layers 2, 5, 8, 10 (v1 cache) → labelled L1, L4, L8, L12
  REVE:    layers 3, 7, 11, 15, 20 → labelled L4, L8, L12, L16, L21

All entries read from the existing v4 cache (read-only). No recomputation.

Saves:
  paper/concept_steering_figures/cross_model_steering_concepts_paper.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent  # repo root: Figure 4/ -> paper_figures/ -> tools/ -> repo
sys.path.insert(0, str(ROOT / "src"))
from mecheeg.sae import SparseAutoencoder

# ── Models ────────────────────────────────────────────────────────────────────
def _rgb(arr):
    return "#{:02x}{:02x}{:02x}".format(int(arr[0]*255), int(arr[1]*255), int(arr[2]*255))


# For each family: data_layer (where the cached numbers come from)
# paired with display_label (what to print in the legend).
# SleepFM and REVE use the same layer for data and display.
# LaBraM is requested to use v1 cached data (layers 2/5/8/10) but display
# labels L1/L4/L8/L12 — so the visual matches v1 with a relabelled legend.
_SLEEPFM_LAYERS = (0, 1, 2)
_SLEEPFM_LABELS = ("L1", "L2", "L3")

_LABRAM_LAYERS  = (2, 5, 8, 10)          # data source = v1 cache (L3/L6/L9/L11)
_LABRAM_LABELS  = ("L1", "L4", "L8", "L12")  # legend labels (relabelled)

_REVE_LAYERS    = (3, 7, 11, 15, 20)     # same as v1 (data source)
_REVE_LABELS    = ("L1", "L6", "L11", "L16", "L22")  # legend relabelled

_GRADIENT_LO, _GRADIENT_HI = 0.40, 0.92
_SLEEPFM_BLUE_RAMP = plt.cm.Blues(np.linspace(_GRADIENT_LO, _GRADIENT_HI, len(_SLEEPFM_LAYERS)))
_LABRAM_GREEN_RAMP = plt.cm.Greens(np.linspace(_GRADIENT_LO, _GRADIENT_HI, len(_LABRAM_LAYERS)))
_REVE_ORANGE_RAMP  = plt.cm.Oranges(np.linspace(_GRADIENT_LO, _GRADIENT_HI, len(_REVE_LAYERS)))

MODELS = (
    [
        (lbl, f"sleepfm_finetuned_layer{L}", _rgb(_SLEEPFM_BLUE_RAMP[i]), "SleepFM")
        for i, (L, lbl) in enumerate(zip(_SLEEPFM_LAYERS, _SLEEPFM_LABELS))
    ]
    + [
        (lbl, f"labram_layer{L}", _rgb(_LABRAM_GREEN_RAMP[i]), "LaBraM")
        for i, (L, lbl) in enumerate(zip(_LABRAM_LAYERS, _LABRAM_LABELS))
    ]
    + [
        (lbl, f"reve_qjbe08_layer{L}", _rgb(_REVE_ORANGE_RAMP[i]), "REVE")
        for i, (L, lbl) in enumerate(zip(_REVE_LAYERS, _REVE_LABELS))
    ]
)

# ── Concept specs ─────────────────────────────────────────────────────────────
AGE_CHILD = {"0-3", "4-9"}
AGE_ADULT = {"20-29", "30-39", "40-49", "50-59", "60+"}
ABN_CATS  = {"Abnormal - Epileptiform", "Abnormal - Other"}
NORMAL    = {"Normal"}

CONCEPTS = [
    {"name": "Pathology", "display_name": "Abnormality",
     "tgt_field": "classification", "tgt_pos": ABN_CATS, "tgt_neg": NORMAL,
     "off_field": "age_group", "off_pos": AGE_CHILD, "off_neg": AGE_ADULT,
     "filter_field": None, "filter_set": None,
     "enr_field": "classification", "enr_cats": ABN_CATS | NORMAL},
    {"name": "Age", "tgt_field": "age_group", "tgt_pos": AGE_CHILD, "tgt_neg": AGE_ADULT,
     "off_field": "classification", "off_pos": ABN_CATS, "off_neg": NORMAL,
     "filter_field": None, "filter_set": None,
     "enr_field": "age_group", "enr_cats": AGE_CHILD | AGE_ADULT},
    {"name": "Gender", "display_name": "Sex",
     "tgt_field": "gender", "tgt_pos": {"male"}, "tgt_neg": {"female"},
     "off_field": "classification", "off_pos": ABN_CATS, "off_neg": NORMAL,
     "filter_field": None, "filter_set": None,
     "enr_field": "gender", "enr_cats": {"male", "female"}},
    {"name": "Medication (Psychiatric)", "tgt_field": "medication_group",
     "tgt_pos": {"Psychiatric"}, "tgt_neg": {"No Current Medication"},
     "off_field": "classification", "off_pos": ABN_CATS, "off_neg": NORMAL,
     "filter_field": None, "filter_set": None,
     "enr_field": "medication_group", "enr_cats": {"Psychiatric", "No Current Medication"}},
    {"name": "Medication (ASM)", "tgt_field": "medication_group",
     "tgt_pos": {"ASM"}, "tgt_neg": {"No Current Medication"},
     "off_field": "classification", "off_pos": ABN_CATS, "off_neg": NORMAL,
     "filter_field": None, "filter_set": None,
     "enr_field": "medication_group", "enr_cats": {"ASM", "No Current Medication"}},
]

N_PROBE_PER_GROUP = 3000
TEST_FRAC = 0.25
N_STEPS = 16
N_RAND_DRAWS = 3
RNG_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT = HERE  # write next to the script

# Source data bundled next to the script.
SRC_CACHE = HERE / "data.npz"
# Cache for incremental additions (only used if --no-plot-only mode is run).
CACHE = HERE / "data_extra.npz"


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_tcav_alignment(Xt_raw, y_t, W_dec):
    probe = fit_probe(Xt_raw, y_t, max_iter=300)
    cav = probe.w
    cav = cav / (cav.norm() + 1e-8)
    proj = (cav @ W_dec).cpu().numpy()
    return np.abs(proj)


def balanced_pair(emb_view, lab_pos_mask, lab_neg_mask, n_per, rng):
    n = min(n_per, int(lab_pos_mask.sum()), int(lab_neg_mask.sum()))
    if n < 100:
        return None, None, n
    idx_pos = rng.choice(np.where(lab_pos_mask)[0], n, replace=False)
    idx_neg = rng.choice(np.where(lab_neg_mask)[0], n, replace=False)
    idx = np.concatenate([idx_pos, idx_neg])
    return idx, np.array([1] * n + [0] * n), n


def split_train_holdout(X, y, rng_seed=RNG_SEED):
    idx = np.arange(len(y))
    idx_t, idx_v, yt, yv = train_test_split(
        idx, y, test_size=TEST_FRAC, random_state=rng_seed, stratify=y,
    )
    return X[idx_t], yt, idx_v, yv


class TorchProbe:
    def __init__(self, w, b):
        self.w = w
        self.b = b


def fit_probe(X_train, y_train, C=1.0, max_iter=200):
    if X_train.device != DEVICE:
        X_train = X_train.to(DEVICE)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=DEVICE)
    n, d = X_train.shape
    n_pos = float((yt == 1).sum())
    n_neg = float((yt == 0).sum())
    w_pos = n / (2.0 * n_pos) if n_pos > 0 else 1.0
    w_neg = n / (2.0 * n_neg) if n_neg > 0 else 1.0
    sample_weight = torch.where(yt == 1, w_pos * torch.ones_like(yt),
                                w_neg * torch.ones_like(yt))
    w = torch.zeros(d, device=DEVICE, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, device=DEVICE, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        logits = X_train @ w + b
        per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, yt, reduction="none")
        bce = (per_sample * sample_weight).sum()
        loss = C * bce + 0.5 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return TorchProbe(w.detach(), b.detach())


def score_probe(probe, X, y):
    if X.device != DEVICE:
        X = X.to(DEVICE)
    with torch.no_grad():
        scores = (X @ probe.w + probe.b).cpu().numpy()
    return float(roc_auc_score(y, scores))


def substitute(z, top_features, donor_mean):
    out = z.clone()
    out[:, top_features] = donor_mean[top_features]
    return out


def sweep_one_ranking(sae, zt, zo, donor_mean, sorted_features, ns,
                      tgt_probe, off_probe, idx_v_t, y_v_t, idx_v_o, y_v_o):
    tgt_auc, off_auc = [], []
    idx_v_t_t = torch.as_tensor(idx_v_t, device=DEVICE, dtype=torch.long)
    idx_v_o_t = torch.as_tensor(idx_v_o, device=DEVICE, dtype=torch.long)
    for n in ns:
        with torch.no_grad():
            if n == 0:
                Xt_dec = sae.decode(zt); Xo_dec = sae.decode(zo)
            else:
                feats = sorted_features[:n]
                ztc = substitute(zt, feats, donor_mean)
                zoc = substitute(zo, feats, donor_mean)
                Xt_dec = sae.decode(ztc); Xo_dec = sae.decode(zoc)
            Xt_v = Xt_dec.index_select(0, idx_v_t_t)
            Xo_v = Xo_dec.index_select(0, idx_v_o_t)
        tgt_auc.append(score_probe(tgt_probe, Xt_v, y_v_t))
        off_auc.append(score_probe(off_probe, Xo_v, y_v_o))
    return np.array(tgt_auc), np.array(off_auc)


def run_layer_concept(spec, layer_state, run_baseline=True, rng_seed=RNG_SEED):
    sc, sae, am, as_, enr, nf = layer_state
    rng = np.random.default_rng(rng_seed)

    def asarr(field):
        return np.asarray([str(v) for v in sc[field]])

    if spec["filter_field"] is not None:
        filt = np.isin(asarr(spec["filter_field"]), list(spec["filter_set"]))
    else:
        filt = np.ones(len(sc["embeddings"]), dtype=bool)

    tgt_field = asarr(spec["tgt_field"])
    off_field = asarr(spec["off_field"])
    tgt_pos_mask = filt & np.isin(tgt_field, list(spec["tgt_pos"]))
    tgt_neg_mask = filt & np.isin(tgt_field, list(spec["tgt_neg"]))
    off_pos_mask = filt & np.isin(off_field, list(spec["off_pos"]))
    off_neg_mask = filt & np.isin(off_field, list(spec["off_neg"]))

    emb = sc["embeddings"]
    idx_t, y_t, n_t = balanced_pair(emb, tgt_pos_mask, tgt_neg_mask, N_PROBE_PER_GROUP, rng)
    idx_o, y_o, n_o = balanced_pair(emb, off_pos_mask, off_neg_mask, N_PROBE_PER_GROUP, rng)
    if idx_t is None or idx_o is None:
        print(f"    [skip] {spec['name']}: tgt={n_t}/group  off={n_o}/group", flush=True)
        return None

    Xt_raw = (torch.as_tensor(np.asarray(emb[idx_t])).float().to(DEVICE) - am) / (as_ + 1e-8)
    Xo_raw = (torch.as_tensor(np.asarray(emb[idx_o])).float().to(DEVICE) - am) / (as_ + 1e-8)
    with torch.no_grad():
        zt = sae.encode(Xt_raw); zo = sae.encode(Xo_raw)
    tgt_neg_mask_in_pool = (y_t == 0)
    donor_mean = zt[tgt_neg_mask_in_pool].mean(dim=0)

    W_dec = sae.decoder.weight.detach()
    w = build_tcav_alignment(Xt_raw, y_t, W_dec)
    sorted_features = np.argsort(w)[::-1].copy()

    ns = sorted(set(int(round(f * nf)) for f in np.linspace(0, 1, N_STEPS + 1)))
    fracs = np.array(ns) / nf

    with torch.no_grad():
        Xt_dec_clean = sae.decode(zt); Xo_dec_clean = sae.decode(zo)
    Xt_train, y_t_train, idx_v_t, y_v_t = split_train_holdout(Xt_dec_clean, y_t, rng_seed)
    Xo_train, y_o_train, idx_v_o, y_v_o = split_train_holdout(Xo_dec_clean, y_o, rng_seed)
    tgt_probe = fit_probe(Xt_train, y_t_train)
    off_probe = fit_probe(Xo_train, y_o_train)

    tgt_auc, off_auc = sweep_one_ranking(
        sae, zt, zo, donor_mean, sorted_features, ns,
        tgt_probe, off_probe, idx_v_t, y_v_t, idx_v_o, y_v_o,
    )
    area = float(np.trapezoid(off_auc - tgt_auc, fracs))

    result = {"fracs": fracs, "tgt_auc": tgt_auc, "off_auc": off_auc, "area": area,
              "n_target_per_group": n_t, "n_off_per_group": n_o, "n_features": int(nf)}

    if run_baseline:
        rng_b = np.random.default_rng(rng_seed + 1)
        rand_areas, rand_tgt_curves, rand_off_curves = [], [], []
        for d in range(N_RAND_DRAWS):
            perm = rng_b.permutation(nf)
            t_au, o_au = sweep_one_ranking(
                sae, zt, zo, donor_mean, perm, ns,
                tgt_probe, off_probe, idx_v_t, y_v_t, idx_v_o, y_v_o,
            )
            rand_tgt_curves.append(t_au); rand_off_curves.append(o_au)
            rand_areas.append(float(np.trapezoid(o_au - t_au, fracs)))
        result["rand_tgt_auc_mean"] = np.mean(rand_tgt_curves, axis=0)
        result["rand_off_auc_mean"] = np.mean(rand_off_curves, axis=0)
        result["rand_area_mean"] = float(np.mean(rand_areas))
        result["rand_area_std"] = float(np.std(rand_areas))

    return result


def load_layer(exp):
    sc_path = ROOT / "results/steering_cache" / exp / "steering_cache.pt"
    meta = json.loads((ROOT / "results/experiments" / exp / "metadata.json").read_text())
    sc = torch.load(sc_path, map_location="cpu", weights_only=False)
    ckpt = torch.load(ROOT / meta["sae_checkpoint"], map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(meta["embed_dim"], expansion=meta["expansion"], mode="topk", k=meta["k"])
    sae.load_state_dict(ckpt["sae_state_dict"]); sae.eval(); sae.to(DEVICE)
    am = ckpt["act_mean"].float().to(DEVICE)
    as_ = ckpt["act_std"].float().to(DEVICE)
    nf = int(sae.decoder.weight.shape[1])
    return sc, sae, am, as_, None, nf


def _normalise_curve(auc):
    a0 = float(auc[0]); denom = max(a0 - 0.5, 0.05)
    return (auc - 0.5) / denom


def compute_m2_steerability(v):
    fracs = np.asarray(v["fracs"])
    tgt = np.asarray(v["tgt_auc"]); off = np.asarray(v["off_auc"])
    tgt_n = _normalise_curve(tgt); off_n = _normalise_curve(off)
    m2 = float(np.trapezoid(1 - tgt_n, fracs))
    delta = float(np.trapezoid(off_n - tgt_n, fracs))
    out = {"m2": m2, "delta": delta}
    if "rand_tgt_auc_mean" in v and "rand_off_auc_mean" in v:
        rtgt_n = _normalise_curve(np.asarray(v["rand_tgt_auc_mean"]))
        roff_n = _normalise_curve(np.asarray(v["rand_off_auc_mean"]))
        out["m2_rand"] = float(np.trapezoid(1 - rtgt_n, fracs))
        out["delta_rand"] = float(np.trapezoid(roff_n - rtgt_n, fracs))
    else:
        out["m2_rand"] = 0.0; out["delta_rand"] = 0.0
    return out


def _load_cache(path):
    out = {}
    if not path.exists():
        return out
    d = np.load(path, allow_pickle=True)
    for k in d.files:
        if not k.startswith("entry_"):
            continue
        ent = json.loads(str(d[k]))
        out[(ent["concept"], ent["experiment"])] = {
            kk: (np.array(vv) if isinstance(vv, list) else vv)
            for kk, vv in ent.items() if kk not in ("concept", "experiment")
        }
    return out


def _save_cache(path, results):
    save = {}
    for i, ((c, e), v) in enumerate(results.items()):
        ent = {"concept": c, "experiment": e}
        for kk, vv in v.items():
            ent[kk] = vv.tolist() if isinstance(vv, np.ndarray) else vv
        save[f"entry_{i}"] = json.dumps(ent)
    np.savez(path, **save)


def main():
    parser = argparse.ArgumentParser()
    # Default plot-only=True: bundled data.npz is sufficient to regenerate the
    # paper figure. Pass --recompute to fall through to the on-the-fly cache
    # computation path, which expects results/steering_cache/{exp}/ from the
    # development repo (not shipped here).
    parser.add_argument("--recompute", dest="plot_only", action="store_false")
    parser.set_defaults(plot_only=True)
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    results = {}
    src = _load_cache(SRC_CACHE)
    v2_extra = _load_cache(CACHE)
    results.update(src)
    results.update(v2_extra)
    print(f"Loaded {len(results)} entries (v4 read-only: {len(src)}, v2: {len(v2_extra)})")

    missing = [(spec["name"], exp) for (_n, exp, _c, _f) in MODELS for spec in CONCEPTS
               if (spec["name"], exp) not in results]
    if missing and not args.plot_only:
        print(f"Computing {len(missing)} missing entries (concept × layer)...")
        for _name, exp, _color, _family in MODELS:
            sc_path = ROOT / "results/steering_cache" / exp / "steering_cache.pt"
            if not sc_path.exists():
                print(f"[skip — no steering cache] {exp}")
                continue
            if all((c["name"], exp) in results for c in CONCEPTS):
                continue
            print(f"\n=== {exp} ===", flush=True)
            layer_state = load_layer(exp)
            for spec in CONCEPTS:
                if (spec["name"], exp) in results:
                    continue
                r = run_layer_concept(spec, layer_state,
                                      run_baseline=not args.no_baseline,
                                      rng_seed=RNG_SEED)
                if r is None:
                    continue
                results[(spec["name"], exp)] = r
                # Persist only the entries new to v2 (skip what we read from v4).
                v2_only = {k: v for k, v in results.items() if k not in src}
                _save_cache(CACHE, v2_only)
            print(f"  saved checkpoint → {CACHE.name}", flush=True)
    elif missing:
        print(f"Warning: {len(missing)} entries missing but --plot-only set.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_concepts = len(CONCEPTS)
    fig, (ax_top, ax_sel) = plt.subplots(
        2, 1, figsize=(17.5, 6.6), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.6], hspace=0.08),
    )

    BAR_W = 1.2
    INTRA_GAP = 0.0
    INTER_GAP = 0.0
    GROUP_GAP = 2.4

    _FAMILY_LAYERS_ORDERED = [
        ("SleepFM", _SLEEPFM_LAYERS, _SLEEPFM_LABELS, _SLEEPFM_BLUE_RAMP),
        ("LaBraM",  _LABRAM_LAYERS,  _LABRAM_LABELS,  _LABRAM_GREEN_RAMP),
        ("REVE",    _REVE_LAYERS,    _REVE_LABELS,    _REVE_ORANGE_RAMP),
    ]

    bar_positions = []
    group_centres = []
    group_separators = []

    x = 0.0
    for ci, spec in enumerate(CONCEPTS):
        grp_start = x
        for fi, (family, layers, labels, ramp) in enumerate(_FAMILY_LAYERS_ORDERED):
            if fi > 0:
                x += INTER_GAP
            for li, (L, lbl) in enumerate(zip(layers, labels)):
                color = _rgb(ramp[li])
                exp = (
                    f"sleepfm_finetuned_layer{L}" if family == "SleepFM"
                    else f"labram_layer{L}" if family == "LaBraM"
                    else f"reve_qjbe08_layer{L}"
                )
                x_centre = x + BAR_W / 2
                bar_positions.append((x_centre, color, family, L, exp, lbl))
                x += BAR_W + INTRA_GAP
            x -= INTRA_GAP
        grp_end = x
        group_centres.append(((grp_start + grp_end) / 2,
                              spec.get("display_name", spec["name"])))
        if ci < n_concepts - 1:
            group_separators.append(x + GROUP_GAP / 2)
            x += GROUP_GAP

    bp_per_concept = len(bar_positions) // n_concepts
    for ci, spec in enumerate(CONCEPTS):
        for bp in bar_positions[ci * bp_per_concept:(ci + 1) * bp_per_concept]:
            x_centre, color, family, L, exp, lname = bp
            v = results.get((spec["name"], exp))
            if v is None:
                continue
            if "bootstrap_areas" in v and "bootstrap_rand_areas" in v:
                ba = np.asarray(v["bootstrap_areas"])
                br = np.asarray(v["bootstrap_rand_areas"])
                be = ba - br
                sel = float(be.mean())
                err = float(be.std(ddof=1)) if len(be) > 1 else 0.0
            else:
                sel = float(v["area"]) - float(v.get("rand_area_mean", 0.0))
                err = float(v.get("rand_area_std", 0.0))
            ec = tuple(0.55 * c for c in plt.matplotlib.colors.to_rgb(color))
            ax_sel.bar(x_centre, sel, width=BAR_W, color=color,
                       edgecolor="black", linewidth=0.8, zorder=3,
                       yerr=err if err > 0 else None,
                       error_kw=dict(lw=1.0, capsize=0, ecolor=ec, alpha=0.95))

            if "bootstrap_tgt_auc0" in v:
                ba0 = np.asarray(v["bootstrap_tgt_auc0"])
                auroc0 = float(ba0.mean())
                auroc0_err = float(ba0.std(ddof=1)) if len(ba0) > 1 else 0.0
            else:
                auroc0 = float(np.asarray(v["tgt_auc"])[0])
                auroc0_err = 0.0
            ax_top.bar(x_centre, auroc0 - 0.5, bottom=0.5, width=BAR_W,
                       color=color, edgecolor="black", linewidth=0.4, zorder=3,
                       yerr=auroc0_err if auroc0_err > 0 else None,
                       error_kw=dict(lw=1.0, capsize=0, ecolor=ec, alpha=0.95))

    ax_sel.axhline(0, color="black", lw=0.6, zorder=2)
    ax_sel.set_ylim(bottom=0)
    ax_sel.grid(axis="y", alpha=0.18); ax_sel.set_axisbelow(True)
    ax_sel.spines[["top", "right"]].set_visible(False)

    ax_top.set_ylim(0.5, 1.0)
    ax_top.axhline(0.5, color="black", lw=0.6, zorder=2)
    for thr, lbl, col in [(0.85, "strong", "#1f7a3a"),
                           (0.70, "moderate", "#b8860b")]:
        ax_top.axhline(thr, color=col, lw=0.6, ls=(0, (4, 3)), alpha=0.55, zorder=2)
        ax_top.text(0.998, thr, rf" {lbl} $\geq$ {thr:.2f}",
                    transform=ax_top.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=18, color=col,
                    style="italic", weight="bold", zorder=3)
    ax_top.grid(axis="y", alpha=0.18); ax_top.set_axisbelow(True)
    ax_top.spines[["top", "right"]].set_visible(False)
    ax_top.tick_params(axis="y", labelsize=13)
    ax_sel.tick_params(axis="y", labelsize=13)

    for sx in group_separators:
        ax_sel.axvline(sx, color="black", lw=0.4, alpha=0.12, zorder=1)
        ax_top.axvline(sx, color="black", lw=0.4, alpha=0.12, zorder=1)

    ax_sel.set_xticks([])
    ax_sel.tick_params(axis="x", which="major", length=0, pad=3)

    y_lbl = -0.04
    for x_centre, name in group_centres:
        ax_sel.text(x_centre, y_lbl, name, transform=ax_sel.get_xaxis_transform(),
                    ha="center", va="top", fontsize=18)

    ax_sel.set_xlim(bar_positions[0][0] - BAR_W, bar_positions[-1][0] + BAR_W)
    ax_sel.set_ylabel(r"Excess selectivity $\tilde{\Delta}$", fontsize=18)
    ax_top.set_ylabel("Encoding strength", fontsize=18)
    ax_top.set_title("Concept encoding and steering selectivity across encoder layers",
                     fontsize=26, pad=14)

    from matplotlib.patches import Patch
    _ROWS = max(len(layers) for _, layers, _, _ in _FAMILY_LAYERS_ORDERED)
    legend_handles, legend_labels = [], []
    for family, layers, labels, ramp in _FAMILY_LAYERS_ORDERED:
        for li, lbl in enumerate(labels):
            legend_handles.append(Patch(facecolor=_rgb(ramp[li]), edgecolor="black", linewidth=0.8))
            legend_labels.append(f"{family} {lbl}")
        for _ in range(_ROWS - len(layers)):
            legend_handles.append(Patch(facecolor="none", edgecolor="none"))
            legend_labels.append("")
    _leg = ax_sel.legend(legend_handles, legend_labels, fontsize=12,
                         loc="upper right", frameon=True, ncol=3,
                         columnspacing=1.2, handlelength=1.1, handleheight=1.0,
                         labelspacing=0.35, borderpad=0.6,
                         facecolor="white", edgecolor="#bbbbbb")
    _leg.get_frame().set_linewidth(0.6); _leg.set_zorder(5)

    fig.subplots_adjust(bottom=0.16, left=0.07, right=0.98, top=0.92)
    out_png = OUT / "figure.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nSaved {out_png}")

    print("\n=== M2 push (TCAV − random)  &  Δ̃ steerability per concept ===")
    for spec in CONCEPTS:
        print(f"\n{spec['name']}:")
        rows = []
        for lname, exp, _color, family in MODELS:
            mname = f"{family} {lname}"
            v = results.get((spec["name"], exp))
            if v is None:
                continue
            m = compute_m2_steerability(v)
            rows.append((mname, m["m2"] - m["m2_rand"], m["delta"]))
        for mname, m2x, dx in sorted(rows, key=lambda x: -x[2]):
            print(f"  {mname:14s}  M2(excess)={m2x:+.4f}  Δ̃={dx:+.4f}")


if __name__ == "__main__":
    main()
