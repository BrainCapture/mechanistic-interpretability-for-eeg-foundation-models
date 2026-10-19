"""Population spectral steering over independently seeded SAEs (camera-ready, App. G).

Identical to rebuttal_population_spectral_multi.py except that the SAE is retrained per
seed (same procedure as steering_seed_robustness.train_sae) and the per-feature metadata
enrichment that selects the clamped features is recomputed for that SAE with the same
subject-level Mann-Whitney U + Benjamini-Hochberg procedure build_app_cache uses.

  uv run tools/rebuttal/rebuttal_population_spectral_seeds.py --seeds 0 1 2 3 4

Output: results/camera_ready/population_spectral_seeds.json (+ per-seed rows)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import false_discovery_control, mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_perfect_steering_three_panel as PS  # noqa: E402
import rebuttal_population_spectral_multi as M  # noqa: E402
from mecheeg.spectral_decoder import SpectralDecoderTrainer  # noqa: E402
from steering_seed_robustness import train_sae  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = ROOT / "results" / "camera_ready" / "population_spectral_seeds.json"
MIN_SUBJ = 5

# Controlled-regime cells only (the paper's definition); SAE operating point per model.
CELLS = [
    ("SleepFM", "sleepfm_finetuned_exp8_k64_layer2", 8.0, 64, 2, None),
    ("LaBraM", "labram_layer11", 1.0, 8, 11, 1_000_000),
    ("REVE", "reve_qjbe08_layer8", 1.0, 8, 8, 1_000_000),
]
CONCEPTS = ["classification", "age"]


def enrichment(fire, subj, field_vals, nf):
    """Subject-level MWU + BH enrichment for one field (mirrors build_app_cache)."""
    uniq, inv = np.unique(subj, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    inv_t = torch.as_tensor(inv)
    fr_t = torch.zeros(len(uniq), nf, dtype=torch.float64)
    for a in range(0, len(fire), 1_000_000):
        fr_t.index_add_(0, inv_t[a:a + 1_000_000], torch.as_tensor(fire[a:a + 1_000_000]).double())
    fr = (fr_t / torch.as_tensor(counts)[:, None]).numpy()
    # subject category = majority vote over its tokens (sort-based grouping)
    vals, vinv = np.unique(field_vals, return_inverse=True)
    tab = np.zeros((len(uniq), len(vals)), dtype=np.int64)
    np.add.at(tab, (inv, vinv), 1)
    cat = vals[tab.argmax(1)].astype(object)
    cats = sorted(set(c for c in cat if c not in ("", "unknown")))
    enr = [dict() for _ in range(nf)]
    ratios, raw = [], []
    known = np.isin(cat, cats)
    glob = fr[known].mean(0)
    for c in cats:
        inn = cat == c; out = (~inn) & known
        if inn.sum() < MIN_SUBJ:
            continue
        for fi in range(nf):
            try:
                _, p = mannwhitneyu(fr[inn, fi], fr[out, fi], alternative="two-sided")
            except ValueError:
                p = 1.0
            raw.append(float(np.clip(p, 0, 1)) if np.isfinite(p) else 1.0)
            ratios.append((c, fi, float(fr[inn, fi].mean()) / max(float(glob[fi]), 1e-8)))
    if raw:
        for (c, fi, r), pa in zip(ratios, false_discovery_control(np.array(raw), method="bh")):
            enr[fi].setdefault("_field", []).append((c, r, float(pa)))
    return enr


def run(model, exp, E, k, layer, cap, seed, rng):
    meta = json.loads((ROOT / "results/experiments" / exp / "metadata.json").read_text())
    sc = torch.load(ROOT / "results/steering_cache" / exp / "steering_cache.pt",
                    map_location="cpu", weights_only=False)
    emb = torch.as_tensor(np.asarray(sc["embeddings"])).float()
    tr = emb
    if cap and len(emb) > cap:
        tr = emb[torch.randperm(len(emb), generator=torch.Generator().manual_seed(12345))[:cap]]
    am, as_ = tr.mean(0), tr.std(0).clamp(min=1e-8)
    sae = train_sae((tr - am) / as_, seed, E, k, 20, 3e-4, 2048, layer).cpu().eval()
    del tr
    with torch.no_grad():
        z_all = torch.cat([sae.encode((emb[i:i + 8192] - am) / (as_ + 1e-8)) for i in range(0, len(emb), 8192)])
    nf = z_all.shape[1]
    fire = (z_all > 0).numpy()
    sdt = SpectralDecoderTrainer(embed_dim=meta["embed_dim"])
    sdt.load(meta.get("spectral_decoder_checkpoint") or meta["xae_checkpoint"])
    sdt.spectral_decoder.cpu().eval()
    bm = {}
    for b in M.BANDS:
        m = None
        for part in M.BAND_PARTS.get(b, [b]):
            x = np.asarray(sdt.spectral.get_band_mask(part), dtype=bool)
            m = x if m is None else (m | x)
        if m.sum():
            bm[b] = m
    arr = {f: np.array([str(v) for v in sc[f]]) for f in ("classification", "age_group", "medication_group", "subject_id")}
    out = {}
    for concept in CONCEPTS:
        cc = M.CONCEPT_CFG[(concept, "controlled")]
        fv = arr[cc.field]
        enr = ([] if cc.field == "classification" else
               [{cc.field: e.get("_field", [])} for e in enrichment(fire, arr["subject_id"], fv, nf)])
        cm = PS.cls_mask(arr["classification"], cc.cls_filter)
        af = np.isin(arr["age_group"], list(cc.age_filter)) if cc.age_filter else np.ones(len(fv), bool)
        m_src = np.isin(fv, list(cc.src_vals)) & af & cm
        m_tgt = np.isin(fv, list(cc.tgt_vals)) & af & cm
        if m_src.sum() < 100 or m_tgt.sum() < 100:
            continue
        # As in the published run: the app cache carries no enrichment for the
        # classification field, so abnormality always uses the probe ranking; age uses
        # subject-level enrichment (falling back to the probe if < 50 features).
        order = (np.array([], dtype=int) if cc.field == "classification"
                 else PS.rank_by_enrichment(enr, cc.field, cc.src_vals, cc.tgt_vals, nf))
        if len(order) < 50:
            order = PS.rank_by_probe(z_all, np.where(m_src)[0], np.where(m_tgt)[0], nf)
        donor = z_all[m_tgt].mean(0)
        rows = []
        for s in sorted(set(arr["subject_id"][m_src])):
            idx = np.where(m_src & (arr["subject_id"] == s))[0]
            if len(idx) < M.MIN_TOK:
                continue
            if len(idx) > M.MAX_TOK_SUBJ:
                idx = rng.choice(idx, M.MAX_TOK_SUBJ, replace=False)
            z0 = z_all[idx]; z1 = z0.clone(); z1[:, order] = donor[order].unsqueeze(0)
            with torch.no_grad():
                a0, _, _ = sdt.decode_direction((sae.decode(z0) * (as_ + 1e-8) + am), denormalise=True)
                a1, _, _ = sdt.decode_direction((sae.decode(z1) * (as_ + 1e-8) + am), denormalise=True)
            a0 = torch.expm1(a0.clamp(min=0)).numpy(); a1 = torch.expm1(a1.clamp(min=0)).numpy()
            rows.append({b: float(a1[:, m].mean() - a0[:, m].mean()) for b, m in bm.items()})
        if len(rows) >= 5:
            out[concept] = {"n_subjects": len(rows), "n_clamp": int(len(order)),
                            "bands": {b: float(np.mean([r[b] for r in rows])) for b in bm},
                            "frac_neg_delta": float(np.mean([r["delta"] < 0 for r in rows]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=[c[0] for c in CELLS])
    args = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for model, exp, E, k, layer, cap in CELLS:
        if model not in args.models:
            continue
        for seed in args.seeds:
            key = f"{model}|seed{seed}"
            if key in res:
                continue
            print(f"=== {model} ({exp}, E={E}, k={k}) seed {seed} ===", flush=True)
            res[key] = run(model, exp, E, k, layer, cap, seed, np.random.default_rng(42))
            for c, r in res[key].items():
                print(f"  {c}: δ {r['bands']['delta']:+.1f} µV θ {r['bands']['theta']:+.1f} "
                      f"α {r['bands']['alpha']:+.1f} (n={r['n_subjects']}, clamp {r['n_clamp']})", flush=True)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(res, indent=2))
    # summary across seeds
    summ = {}
    for model, *_ in CELLS:
        for c in CONCEPTS:
            vals = [res[k][c] for k in res if k.startswith(model + "|") and c in res[k]]
            if vals:
                summ[f"{model} / {c}"] = {
                    "n_seeds": len(vals), "n_subjects": vals[0]["n_subjects"],
                    "n_clamp": [v["n_clamp"] for v in vals],
                    **{b: {"mean": float(np.mean([v["bands"][b] for v in vals])),
                           "sd": float(np.std([v["bands"][b] for v in vals], ddof=1)) if len(vals) > 1 else 0.0}
                       for b in vals[0]["bands"]}}
    res["_summary"] = summ
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
