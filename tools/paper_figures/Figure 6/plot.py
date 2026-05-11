"""Three-panel "perfect example" steering figure for paper concepts.

For a single concept, produces a 1×3 figure:

    Panel A — Target:  source spectrum + target centroid + bootstrap CI band
              (no steering — shows the gap that steering must close)
    Panel B — Moderate: source + steered (small n) + target band
    Panel C — Perfect:  source + steered (n* — optimum M1) + target band

Concepts supported (out of the box):
    --concept classification   (Adult Abnormal → Normal)
    --concept age              (0–3 → 50+, abnormal EEG)
    --concept asm              (ASM → No medication, abnormal EEG)

Style mirrors ``plot_age_steering_clean_ramp.py``:
* tight 95 % bootstrap CI on the target *mean* (not ±1 SD across tokens)
* viridis ramp colour for the steered line
* probe-based M2 / M3 / S inline in the legend

Usage::

    uv run tools/plot_perfect_steering_three_panel.py --concept classification
    uv run tools/plot_perfect_steering_three_panel.py --concept age
    uv run tools/plot_perfect_steering_three_panel.py --concept asm

Sweeping n to find n*::

    uv run tools/plot_perfect_steering_three_panel.py --concept asm --sweep \
        --n-grid 50 100 200 400 800 1024

Override picks::

    uv run tools/plot_perfect_steering_three_panel.py --concept age \
        --n-moderate 100 --n-perfect 461
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from sae4eeg.sae import SparseAutoencoder
from sae4eeg.xae import XAETrainer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent  # repo root: Figure 6/ -> paper_figures/ -> tools/ -> repo

ABN_CATS = {"Abnormal - Epileptiform", "Abnormal - Other"}

BANDS = {
    "δ": (0.5, 4, "#c0392b"),
    "θ": (4, 8, "#d35400"),
    "α": (8, 13, "#9a7d0a"),
    "β": (13, 30, "#1e8449"),
    "γ": (30, 45, "#1a5276"),
}

SRC_COL = "#c0392b"
TGT_COL = "#1e8449"


@dataclass
class ConceptCfg:
    key: str
    field: str
    src_vals: set
    tgt_vals: set
    src_lbl: str
    tgt_lbl: str
    title: str
    cls_filter: str | None = None         # 'normal' | 'abnormal' | None
    age_filter: set = dc_field(default_factory=set)
    default_n_moderate: int = 100
    default_n_perfect: int = 461
    default_n_grid: tuple = (25, 50, 100, 200, 400, 600, 800, 1024)


CONCEPTS: dict[str, ConceptCfg] = {
    "classification": ConceptCfg(
        key="classification",
        field="classification",
        src_vals={"Abnormal - Epileptiform", "Abnormal - Other"},
        tgt_vals={"Normal"},
        src_lbl="Abnormal",
        tgt_lbl="Normal",
        title="Representative example of steering the pathological amplitude spectrum from abnormal to normal (adult EEG)",
        cls_filter=None,
        age_filter={"20-29", "30-39", "40-49", "50-59", "60+"},
        default_n_moderate=50,
        default_n_perfect=175,
    ),
    "age": ConceptCfg(
        key="age",
        field="age_group",
        src_vals={"0-3"},
        tgt_vals={"50-59", "60+"},
        src_lbl="Age 0–3",
        tgt_lbl="Age 50+",
        title="Age: 0–3 → 50+ · Abnormal EEG",
        cls_filter="abnormal",
        default_n_moderate=200,
        default_n_perfect=400,
    ),
    "asm": ConceptCfg(
        key="asm",
        field="medication_group",
        src_vals={"ASM"},
        tgt_vals={"No Current Medication"},
        src_lbl="ASM",
        tgt_lbl="No medication",
        title="Medication: ASM → No medication · Abnormal EEG",
        cls_filter="abnormal",
        default_n_moderate=100,
        default_n_perfect=600,
    ),
    "psych": ConceptCfg(
        key="psych",
        field="medication_group",
        src_vals={"Psychiatric"},
        tgt_vals={"No Current Medication"},
        src_lbl="Psychiatric meds",
        tgt_lbl="No medication",
        title="Medication: Psychiatric → No medication · Normal EEG",
        cls_filter="normal",
        default_n_moderate=20,
        default_n_perfect=50,
    ),
}


def selectivity(m2: float, m3: float) -> float:
    return float(1.0 - np.sqrt((1.0 - m2) ** 2 + (1.0 - m3) ** 2) / np.sqrt(2))


def bootstrap_mean_ci(amps: np.ndarray, n_boot: int = 2000,
                      ci: float = 95.0, seed: int = 0,
                      chunk: int = 50,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Token-level bootstrap CI on the mean (chunked to avoid OOM)."""
    rng = np.random.default_rng(seed)
    N, F = amps.shape
    boots = np.empty((n_boot, F), dtype=np.float32)
    for s in range(0, n_boot, chunk):
        e = min(s + chunk, n_boot)
        idx = rng.integers(0, N, size=(e - s, N))
        boots[s:e] = amps[idx].mean(axis=1)
    lo = np.percentile(boots, (100 - ci) / 2, axis=0)
    hi = np.percentile(boots, 100 - (100 - ci) / 2, axis=0)
    return amps.mean(0), lo, hi


def subject_balanced_mean(amps: np.ndarray, subj_ids: np.ndarray) -> np.ndarray:
    """Mean spectrum that weights each subject equally (avg of subject-means)."""
    uniq = np.unique(subj_ids)
    return np.stack([amps[subj_ids == s].mean(0) for s in uniq], axis=0).mean(0)


def subject_bootstrap_mean_ci(amps: np.ndarray, subj_ids: np.ndarray,
                              n_boot: int = 2000, ci: float = 95.0,
                              seed: int = 0
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subject-level bootstrap CI on the population-mean spectrum.

    Resamples *subjects* with replacement (cluster bootstrap) — the right
    statistical level when tokens are nested within subjects. This gives
    a visibly wider, honest band that represents uncertainty in the
    target-population estimate, not within-target diversity.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(subj_ids)
    n_subj = len(uniq)
    F = amps.shape[1]

    # Pre-compute subject-mean spectra (n_subj, F)
    subj_means = np.stack([amps[subj_ids == s].mean(0) for s in uniq], axis=0)

    boots = np.empty((n_boot, F), dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n_subj, size=n_subj)
        boots[b] = subj_means[idx].mean(0)
    lo = np.percentile(boots, (100 - ci) / 2, axis=0)
    hi = np.percentile(boots, 100 - (100 - ci) / 2, axis=0)
    # Use the mean-of-subject-means as the point estimate so it sits at the
    # centre of the bootstrap distribution. The token-weighted mean
    # (amps.mean(0)) is biased toward prolific subjects and would drift
    # outside the subject-resampled CI band.
    return subj_means.mean(0), lo, hi


def rank_by_enrichment(enr, field: str, src_vals: set, tgt_vals: set, nf: int) -> np.ndarray:
    def weights(cats):
        w = np.zeros(nf)
        for i, fe in enumerate(enr):
            for c, r, p in fe.get(field, []):
                if c in cats and p < 0.05:
                    w[i] += (r - 1) * -np.log10(max(p, 1e-300))
        return w
    ws = weights(src_vals)
    wt = weights(tgt_vals)
    union = np.where((ws > 0) | (wt > 0))[0]
    if len(union) == 0:
        return np.array([], dtype=int)
    return union[np.argsort(np.maximum(ws, wt)[union])[::-1]]


def rank_by_probe(z_all: torch.Tensor, src_idx: np.ndarray, tgt_idx: np.ndarray,
                  nf: int, n_per: int = 1500, seed: int = 42) -> np.ndarray:
    """Rank features by |probe coef|, keeping only those above 5% of the max."""
    rng = np.random.default_rng(seed)
    if len(src_idx) > n_per:
        src_idx = rng.choice(src_idx, n_per, replace=False)
    if len(tgt_idx) > n_per:
        tgt_idx = rng.choice(tgt_idx, n_per, replace=False)
    X = np.vstack([z_all[src_idx].numpy(), z_all[tgt_idx].numpy()])
    y = np.array([0] * len(src_idx) + [1] * len(tgt_idx))
    probe = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs").fit(X, y)
    coef = np.abs(probe.coef_.squeeze())
    thresh = max(coef.max() * 0.05, 1e-6)
    sig = np.where(coef > thresh)[0]
    return np.ascontiguousarray(sig[np.argsort(coef[sig])[::-1]])


def cls_mask(cls_lbl: np.ndarray, filt: str | None) -> np.ndarray:
    if filt == "abnormal":
        return np.isin(cls_lbl, list(ABN_CATS))
    if filt == "normal":
        return cls_lbl == "Normal"
    return np.ones(len(cls_lbl), dtype=bool)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--concept", choices=list(CONCEPTS), required=True)
    p.add_argument("--experiment", default="sleepfm_finetuned_exp8_k64_layer2")
    p.add_argument("--n-moderate", type=int, default=None,
                   help="Override moderate-steering n (else uses concept default).")
    p.add_argument("--n-perfect", type=int, default=None,
                   help="Override 'perfect' n (else uses concept default — or sweep min).")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep --n-grid and pick the M1-min n as 'perfect'.")
    p.add_argument("--n-grid", type=int, nargs="+", default=None,
                   help="Sweep grid (used with --sweep).")
    p.add_argument("--n-subsample", type=int, default=2000,
                   help="Source-token subsample for steering eval (default 2000).")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--ci", type=float, default=95.0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "figures")
    p.add_argument("--stem", default=None)
    args = p.parse_args()

    cc = CONCEPTS[args.concept]
    EXP_DIR = ROOT / "results" / "experiments" / args.experiment
    SC_PATH = ROOT / "results" / "steering_cache" / args.experiment / "steering_cache.pt"

    print(f"[load] {args.experiment}")
    meta = json.loads((EXP_DIR / "metadata.json").read_text())
    cache = torch.load(EXP_DIR / "app_cache.pt", map_location="cpu", weights_only=False)
    sae_ckpt = torch.load(meta["sae_checkpoint"], map_location="cpu", weights_only=False)

    sae = SparseAutoencoder(meta["embed_dim"], expansion=meta["expansion"],
                            mode="topk", k=meta["k"])
    sae.load_state_dict(sae_ckpt["sae_state_dict"])
    sae.eval()
    act_mean = sae_ckpt["act_mean"].float()
    act_std = sae_ckpt["act_std"].float()

    xae_trainer = XAETrainer(embed_dim=meta["embed_dim"])
    xae_trainer.load(meta["xae_checkpoint"])
    xae_trainer.xae.cpu().eval()

    enr = cache["feature_meta_enrichment"]
    nf = len(enr)
    freqs = np.array(cache["feature_freqs"])

    sc = torch.load(SC_PATH, map_location="cpu", weights_only=False)
    pemb = sc["embeddings"].numpy()
    age_lbl = np.array([str(v) for v in sc["age_group"]])
    cls_lbl = np.array([str(v) for v in sc["classification"]])
    med_lbl = np.array([str(v) for v in sc["medication_group"]])
    subj_lbl = np.array([str(v) for v in sc["subject_id"]])
    field_arrays = {
        "age_group": age_lbl,
        "classification": cls_lbl,
        "medication_group": med_lbl,
    }

    emb_all = torch.tensor(pemb.astype(np.float32))
    x_norm = (emb_all - act_mean) / (act_std + 1e-8)
    print("[encode] all tokens through SAE …")
    BATCH = 512
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x_norm), BATCH):
            chunks.append(sae.encode(x_norm[i:i + BATCH]))
    z_all = torch.cat(chunks, dim=0)

    # Build source / target masks
    cm = cls_mask(cls_lbl, cc.cls_filter)
    field_arr = field_arrays[cc.field]
    m_src = np.isin(field_arr, list(cc.src_vals)) & cm
    m_tgt = np.isin(field_arr, list(cc.tgt_vals)) & cm
    if cc.age_filter:
        m_src &= np.isin(age_lbl, list(cc.age_filter))
        m_tgt &= np.isin(age_lbl, list(cc.age_filter))
    n_src = int(m_src.sum())
    n_tgt = int(m_tgt.sum())
    n_src_subj = len(set(subj_lbl[m_src]))
    n_tgt_subj = len(set(subj_lbl[m_tgt]))
    print(f"  src n={n_src} ({n_src_subj} subj)  tgt n={n_tgt} ({n_tgt_subj} subj)")
    if n_src < 50 or n_tgt < 50:
        raise RuntimeError(f"Insufficient tokens: src={n_src} tgt={n_tgt}")

    # Feature ranking
    feat_order = rank_by_enrichment(enr, cc.field, cc.src_vals, cc.tgt_vals, nf)
    if len(feat_order) < 50:
        print(f"  [fallback] only {len(feat_order)} enriched feats — using probe rank")
        feat_order = rank_by_probe(z_all, np.where(m_src)[0], np.where(m_tgt)[0], nf)
    print(f"  [rank] {len(feat_order)} concept-enriched/significant features (of {nf})")

    rng = np.random.default_rng(42)
    src_idx_all = np.where(m_src)[0]
    if len(src_idx_all) > args.n_subsample:
        idx_src = rng.choice(src_idx_all, args.n_subsample, replace=False)
    else:
        idx_src = src_idx_all

    z_src_batch = z_all[idx_src]
    z_donor = z_all[m_tgt].mean(dim=0)

    def decode_amp(z: torch.Tensor, batch: int = 2048) -> np.ndarray:
        # decode_direction returns log1p-scaled amplitude (xae.py:1630);
        # apply expm1 to get linear μV.
        out = []
        with torch.no_grad():
            for i in range(0, len(z), batch):
                x_dec = sae.decode(z[i:i + batch])
                emb_r = x_dec * (act_std + 1e-8) + act_mean
                log_amp, _, _ = xae_trainer.decode_direction(
                    emb_r.cpu(), denormalise=True)
                out.append(torch.expm1(log_amp.clamp(min=0)).numpy())
        return np.concatenate(out, axis=0)

    def substitute(z_batch: torch.Tensor, feat_idx: np.ndarray) -> torch.Tensor:
        feat_idx = np.ascontiguousarray(feat_idx)
        z = z_batch.clone()
        z[:, feat_idx] = z_donor[feat_idx].unsqueeze(0)
        return z

    # Decode target population (full set) for subject-level bootstrap CI
    print(f"[decode] target spectra ({n_tgt:,} tokens) …")
    amp_tgt_full = decode_amp(z_all[m_tgt])
    tgt_subj = subj_lbl[m_tgt]
    tgt_mean, tgt_lo, tgt_hi = subject_bootstrap_mean_ci(
        amp_tgt_full, tgt_subj, n_boot=args.bootstrap, ci=args.ci)
    band_hw = (tgt_hi - tgt_lo).mean() / 2
    print(f"         target-mean ±{int(args.ci)}% subject-bootstrap CI "
          f"half-width ≈ {band_hw:.4f} μV  ({n_tgt_subj} subj)")

    amp_src = decode_amp(z_src_batch)
    src_subj = subj_lbl[idx_src]
    src_mean = subject_balanced_mean(amp_src, src_subj)
    d_src = float(np.linalg.norm(src_mean - tgt_mean))
    print(f"         d(src→tgt) = {d_src:.4f} μV (M1 baseline)")

    # Probe M2 / M3 (concept + class preservation)
    rng_p = np.random.default_rng(42)
    src_pool = np.where(m_src)[0]
    tgt_pool = np.where(m_tgt)[0]
    n_pp = min(800, len(src_pool), len(tgt_pool))
    if len(src_pool) > n_pp:
        src_pool = rng_p.choice(src_pool, n_pp, replace=False)
    if len(tgt_pool) > n_pp:
        tgt_pool = rng_p.choice(tgt_pool, n_pp, replace=False)
    Xm2 = np.vstack([z_all[src_pool].numpy(), z_all[tgt_pool].numpy()])
    ym2 = np.array([0] * len(src_pool) + [1] * len(tgt_pool))
    probe_m2 = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs").fit(Xm2, ym2)

    # M3: classification preservation (Abnormal vs Normal); skipped for the
    # classification concept itself (M2 already covers the relevant axis).
    has_m3 = cc.field != "classification"
    if has_m3:
        c_pos = np.where(np.isin(cls_lbl, list(ABN_CATS)))[0]
        c_neg = np.where(cls_lbl == "Normal")[0]
        if len(c_pos) > 800:
            c_pos = rng_p.choice(c_pos, 800, replace=False)
        if len(c_neg) > 800:
            c_neg = rng_p.choice(c_neg, 800, replace=False)
        Xm3 = np.vstack([z_all[c_pos].numpy(), z_all[c_neg].numpy()])
        ym3 = np.array([1] * len(c_pos) + [0] * len(c_neg))
        probe_m3 = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs").fit(Xm3, ym3)
        m3_col = 1 if cc.cls_filter == "abnormal" else 0

    # Choose ns
    n_max = len(feat_order)
    if args.sweep:
        if args.n_grid is None:
            args.n_grid = list(cc.default_n_grid)
        n_grid = sorted(set(int(n) for n in args.n_grid if 0 < n <= n_max))
        print(f"[sweep] n ∈ {n_grid}")
        sweep_rows = []
        for n in n_grid:
            z_mod = substitute(z_src_batch, feat_order[:n])
            amp = decode_amp(z_mod)
            amp_mu = subject_balanced_mean(amp, src_subj)
            m1 = float(np.linalg.norm(amp_mu - tgt_mean) / max(d_src, 1e-12))
            m2 = float(probe_m2.predict_proba(z_mod.numpy())[:, 1].mean())
            if has_m3:
                m3 = float(probe_m3.predict_proba(z_mod.numpy())[:, m3_col].mean())
            else:
                m3 = float("nan")
            sweep_rows.append((n, m1, m2, m3))
            S = selectivity(m2, m3) if has_m3 else m2
            print(f"  n={n:4d}  M1={m1:.3f}  M2={m2:.3f}  "
                  f"M3={m3:.3f}  S={S:.3f}")
        # n_perfect = argmin M1
        n_perfect = min(sweep_rows, key=lambda r: r[1])[0]
        # n_moderate = closest to half the M1 reduction
        m1_min = min(r[1] for r in sweep_rows)
        m1_target = (1.0 + m1_min) / 2.0
        n_moderate = min(sweep_rows, key=lambda r: abs(r[1] - m1_target))[0]
        print(f"[sweep] n_perfect = {n_perfect}  n_moderate = {n_moderate}")
    else:
        n_moderate = args.n_moderate or cc.default_n_moderate
        n_perfect = args.n_perfect or cc.default_n_perfect
    n_moderate = max(1, min(n_moderate, n_max))
    n_perfect = max(1, min(n_perfect, n_max))

    # Compute the two real steering decoded spectra
    panels = {}    # n -> dict
    for n in (0, n_moderate, n_perfect):
        if n == 0:
            amp = amp_src.copy()
            z_mod = z_src_batch
        else:
            z_mod = substitute(z_src_batch, feat_order[:n])
            amp = decode_amp(z_mod)
        amp_mu = subject_balanced_mean(amp, src_subj)
        m1 = float(np.linalg.norm(amp_mu - tgt_mean) / max(d_src, 1e-12))
        m2 = float(probe_m2.predict_proba(z_mod.numpy())[:, 1].mean())
        if has_m3:
            m3 = float(probe_m3.predict_proba(z_mod.numpy())[:, m3_col].mean())
            S = selectivity(m2, m3)
        else:
            m3 = float("nan")
            S = m2
        panels[n] = dict(amp=amp, mean=amp_mu, m1=m1, m2=m2, m3=m3, S=S)
        print(f"  n={n:4d}  M1={m1:.3f}  M2={m2:.3f}  M3={m3:.3f}  S={S:.3f}")

    # ── Plot 1×3 panels ───────────────────────────────────────────────────
    # NeurIPS paper style: STIX serif + cm mathtext (matches
    # plot_cross_model_steering_concepts.py). All metric/Greek symbols use
    # mathtext so they render with the body font in the compiled PDF.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 14,
    })
    band_tex = {"δ": r"$\delta$", "θ": r"$\theta$", "α": r"$\alpha$",
                "β": r"$\beta$", "γ": r"$\gamma$"}

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), sharey=True,
                             facecolor="white")
    # Both steered curves are blue: moderate is a medium blue, full is the
    # darker viridis_r(0.85). They share a hue family but are clearly
    # distinguishable, and both contrast with the green target/red source.
    step_colors = {n_moderate: "#3b6ea5",
                   n_perfect: matplotlib.colormaps["viridis_r"](0.85)}

    y_max = max(
        src_mean.max(),
        tgt_hi.max(),
        *[panels[n]["mean"].max() for n in panels],
    ) * 1.12

    panel_specs = [
        ("Baseline", None),
        (rf"Moderate steering  ($n={n_moderate}$)", n_moderate),
        (rf"Full steering  ($n={n_perfect}$)", n_perfect),
    ]

    f_lo, f_hi = 0.5, 45.0
    for ax, (title, n_step) in zip(axes, panel_specs):
        ax.set_facecolor("white")
        # Frequency-band shading + Greek letter at top
        for band_lbl, (f0, f1, col) in BANDS.items():
            f0v, f1v = max(f0, f_lo), min(f1, f_hi)
            if f1v <= f0v:
                continue
            ax.axvspan(f0v, f1v, alpha=0.09, color=col, zorder=0)
            ax.text((f0v + f1v) / 2, y_max * 0.965, band_tex[band_lbl],
                    ha="center", va="top", fontsize=13, color=col)

        # Target reference (band + dotted mean)
        ax.fill_between(freqs, tgt_lo, tgt_hi, alpha=0.55, color=TGT_COL,
                        zorder=1, linewidth=0)
        ax.plot(freqs, tgt_mean, color=TGT_COL, lw=1.4, ls=":",
                label=f"{cc.tgt_lbl} target", zorder=4)

        # Source mean
        ax.plot(freqs, src_mean, color=SRC_COL, lw=2.0,
                label=f"{cc.src_lbl} source", zorder=5)

        if n_step is not None:
            d = panels[n_step]
            col = step_colors[n_step]
            ax.plot(freqs, d["mean"], color=col, lw=2.0,
                    label="Steered", zorder=6)

        ax.set_xlim(f_lo, f_hi)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(title, pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        for sp_ in ["left", "bottom"]:
            ax.spines[sp_].set_color("#888888")
        ax.yaxis.grid(True, alpha=0.25, linestyle="--", color="#bbbbbb")
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r"Amplitude ($\mu$V)")

    # Single shared legend along the bottom — avoids 3× repetition.
    import matplotlib.patches as _mp
    handles = [
        plt.Line2D([], [], color=SRC_COL, lw=2.0,
                   label=f"{cc.src_lbl} source"),
        plt.Line2D([], [], color=TGT_COL, lw=1.4, ls=":",
                   label=f"{cc.tgt_lbl} target"),
        _mp.Patch(facecolor=TGT_COL, alpha=0.55,
                  label=f"{int(args.ci)}% subject-bootstrap CI"),
        plt.Line2D([], [], color=step_colors[n_moderate], lw=2.0,
                   label=rf"Moderate ($n={n_moderate}$)"),
        plt.Line2D([], [], color=step_colors[n_perfect], lw=2.0,
                   label=rf"Full ($n={n_perfect}$)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=5,
               frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.8, columnspacing=1.8)

    fig.suptitle(cc.title, fontsize=14, y=1.00)
    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(bottom=0.24, top=0.84)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"perfect_steering_{cc.key}"
    out = args.out_dir / stem
    fig.savefig(str(out) + ".png", dpi=180, bbox_inches="tight",
                facecolor="white")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved → {out}.{{png,pdf}}")

    # Save sidecar JSON for reproducibility
    side = {
        "concept": cc.key,
        "experiment": args.experiment,
        "n_features_total": int(nf),
        "n_features_concept": int(len(feat_order)),
        "n_src_tokens": int(n_src),
        "n_src_subjects": int(n_src_subj),
        "n_tgt_tokens": int(n_tgt),
        "n_tgt_subjects": int(n_tgt_subj),
        "d_src_to_tgt_uv": float(d_src),
        "target_ci_halfwidth_uv": float(band_hw),
        "n_moderate": int(n_moderate),
        "n_perfect": int(n_perfect),
        "panels": {
            str(n): {k: (None if isinstance(v, float) and np.isnan(v) else float(v))
                     for k, v in d.items() if k in ("m1", "m2", "m3", "S")}
            for n, d in panels.items()
        },
    }
    (Path(str(out) + ".json")).write_text(json.dumps(side, indent=2))
    print(f"Saved → {out}.json")


if __name__ == "__main__":
    main()
