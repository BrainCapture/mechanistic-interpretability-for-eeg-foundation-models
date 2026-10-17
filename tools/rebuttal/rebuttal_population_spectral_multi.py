"""Rebuttal #2 — population spectral steering ACROSS models and concepts (HdqW Q4).

Extends the single-cell (SleepFM / abnormality) population spectral result to a
models × concepts grid, aggregating the per-band amplitude shift across subjects with a
subject-level bootstrap CI. Handles per-channel REVE/LaBraM decoders by averaging band
amplitude over all of a subject's tokens (= channel-averaged), per the paper's convention.

  uv run tools/rebuttal/rebuttal_population_spectral_multi.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))
import plot_perfect_steering_three_panel as PS  # noqa: E402
from mecheeg.sae import SparseAutoencoder  # noqa: E402
from mecheeg.spectral_decoder import SpectralDecoderTrainer  # noqa: E402

OUT = ROOT / "results" / "rebuttal"
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
# β is reported collapsed (13–30 Hz) = union of the decoder's low-beta and high-beta bins.
BAND_PARTS = {"beta": ["low-beta", "high-beta"]}
MIN_TOK, MAX_TOK_SUBJ, BOOT = 10, 300, 1000

# Two filter regimes per concept:
#   "controlled" — the paper's confound-controlled definition (abnormality measured on adults
#                  only; age steered within abnormal recordings only).
#   "all"        — the same steering with the confound filter dropped, giving ~4-6x more
#                  subjects. Reported alongside as a robustness check, NOT as the headline:
#                  without the filters the age-pathology confound is back in the pool.
def _relaxed(key):
    cc = PS.CONCEPTS[key]
    if key == "classification":
        return replace(cc, age_filter=set())      # all ages, not just adults
    if key == "age":
        return replace(cc, cls_filter=None)       # all recordings, not just abnormal
    return cc


CONCEPT_CFG = {
    ("classification", "controlled"): PS.CONCEPTS["classification"],
    ("classification", "all"): _relaxed("classification"),
    ("age", "controlled"): PS.CONCEPTS["age"],
    ("age", "all"): _relaxed("age"),
}

CELLS = [
    ("SleepFM", "sleepfm_finetuned_exp8_k64_layer2", "classification", "controlled"),
    ("SleepFM", "sleepfm_finetuned_exp8_k64_layer2", "age", "controlled"),
    ("SleepFM", "sleepfm_finetuned_exp8_k64_layer2", "classification", "all"),
    ("SleepFM", "sleepfm_finetuned_exp8_k64_layer2", "age", "all"),
    ("REVE", "reve_qjbe08_layer8", "classification", "controlled"),
    ("REVE", "reve_qjbe08_layer8", "age", "controlled"),
    ("LaBraM", "labram_layer11", "classification", "controlled"),
    ("LaBraM", "labram_layer11", "age", "controlled"),
    ("LaBraM", "labram_layer11", "classification", "all"),
    ("LaBraM", "labram_layer11", "age", "all"),
]


def run_cell(exp, concept, regime, rng):
    cc = CONCEPT_CFG[(concept, regime)]
    E = ROOT / "results" / "experiments" / exp
    meta = json.loads((E / "metadata.json").read_text())
    cache = torch.load(E / "app_cache.pt", map_location="cpu", weights_only=False)
    ck = torch.load(meta["sae_checkpoint"], map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(meta["embed_dim"], expansion=meta["expansion"], mode="topk", k=meta["k"])
    sae.load_state_dict(ck["sae_state_dict"]); sae.eval()
    am = ck["act_mean"].float(); as_ = ck["act_std"].float()
    sdt = SpectralDecoderTrainer(embed_dim=meta["embed_dim"])
    sdt.load(meta.get("spectral_decoder_checkpoint") or meta["xae_checkpoint"])
    sdt.spectral_decoder.cpu().eval()
    enr = cache["feature_meta_enrichment"]; nf = len(enr)
    band_masks = {}
    for b in BANDS:
        parts = BAND_PARTS.get(b, [b])
        m = np.zeros_like(np.asarray(sdt.spectral.get_band_mask(parts[0])), dtype=bool)
        for p in parts:
            m |= np.asarray(sdt.spectral.get_band_mask(p), dtype=bool)
        band_masks[b] = m
    band_masks = {b: m for b, m in band_masks.items() if np.asarray(m).sum() > 0}

    sc = torch.load(ROOT / "results/steering_cache" / exp / "steering_cache.pt",
                    map_location="cpu", weights_only=False)
    cls = np.array([str(v) for v in sc["classification"]])
    age = np.array([str(v) for v in sc["age_group"]])
    med = np.array([str(v) for v in sc["medication_group"]])
    subj = np.array([str(v) for v in sc["subject_id"]])
    emb = torch.tensor(sc["embeddings"].numpy().astype(np.float32))
    fmap = {"classification": cls, "age_group": age, "medication_group": med}
    field_arr = fmap[cc.field]

    with torch.no_grad():
        z_all = torch.cat([sae.encode(((emb[i:i+512]-am)/(as_+1e-8))) for i in range(0, len(emb), 512)], 0)

    cm = PS.cls_mask(cls, cc.cls_filter)
    af = np.isin(age, list(cc.age_filter)) if cc.age_filter else np.ones(len(cls), bool)
    m_src = np.isin(field_arr, list(cc.src_vals)) & af & cm
    m_tgt = np.isin(field_arr, list(cc.tgt_vals)) & af & cm
    if m_src.sum() < 100 or m_tgt.sum() < 100:
        return None
    feat_order = PS.rank_by_enrichment(enr, cc.field, cc.src_vals, cc.tgt_vals, nf)
    if len(feat_order) < 50:
        feat_order = PS.rank_by_probe(z_all, np.where(m_src)[0], np.where(m_tgt)[0], nf)
    z_donor = z_all[m_tgt].mean(0)

    def delta(z, batch=2048):
        out = []
        with torch.no_grad():
            for i in range(0, len(z), batch):
                x = sae.decode(z[i:i+batch]) * (as_+1e-8) + am
                amp, _, _ = sdt.decode_direction(x.cpu(), denormalise=True)
                out.append(torch.expm1(amp.clamp(min=0)).numpy())
        return np.concatenate(out, 0)

    rows = []
    for s in sorted(set(subj[m_src])):
        idx = np.where(m_src & (subj == s))[0]
        if len(idx) < MIN_TOK:
            continue
        if len(idx) > MAX_TOK_SUBJ:
            idx = rng.choice(idx, MAX_TOK_SUBJ, replace=False)
        z0 = z_all[idx]; z1 = z0.clone(); z1[:, feat_order] = z_donor[feat_order].unsqueeze(0)
        a0, a1 = delta(z0), delta(z1)
        rows.append({b: float(a1[:, band_masks[b]].mean() - a0[:, band_masks[b]].mean()) for b in band_masks})
    if len(rows) < 5:
        return None
    summ = {}
    for b in band_masks:
        v = np.array([r[b] for r in rows])
        boot = np.array([rng.choice(v, len(v), True).mean() for _ in range(BOOT)])
        summ[b] = {"mean": float(v.mean()), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                   "frac_neg": float((v < 0).mean())}
    return {"n_subjects": len(rows), "n_clamp": len(feat_order), "bands": summ}


def main():
    rng = np.random.default_rng(42)
    results = {}
    for model, exp, concept, regime in CELLS:
        key = f"{model} / {concept}" + ("" if regime == "controlled" else f" / {regime}")
        print(f"=== {key} ({exp}) ===", flush=True)
        try:
            r = run_cell(exp, concept, regime, rng)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {e}"); r = None
        if r is None:
            print("  [skip] insufficient data"); continue
        r["regime"] = regime
        results[key] = r
        d = r["bands"]
        print("  " + "  ".join(f"{b} {d[b]['mean']:+.1f}μV ({100*d[b]['frac_neg']:.0f}%↓)" for b in d)
              + f"  (n={r['n_subjects']})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "population_spectral_multi.json").write_text(json.dumps(results, indent=2))
    print(f"\n[#2] wrote {OUT/'population_spectral_multi.json'}")


if __name__ == "__main__":
    main()
