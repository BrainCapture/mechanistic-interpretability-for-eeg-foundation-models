"""Rebuttal T3 — population abnormal→normal spectral steering (mean ± CI over subjects).

Answers HdqW Q4 / AC#3: replace the single Fig.6 worked example with the aggregate
spectral shift (δ/θ suppression, α restoration) across ALL abnormal adult subjects,
with a subject-level bootstrap CI. Reuses the exact loading/decode machinery of
tools/paper_figures/plot_perfect_steering_three_panel.py.

  uv run tools/rebuttal/rebuttal_population_spectral.py --experiment sleepfm_finetuned_exp8_k64_layer2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))

import plot_perfect_steering_three_panel as PS  # noqa: E402  (loading + helpers)
from mecheeg.sae import SparseAutoencoder  # noqa: E402
from mecheeg.spectral_decoder import SpectralDecoderTrainer  # noqa: E402

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
OUT = ROOT / "results" / "rebuttal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="sleepfm_finetuned_exp8_k64_layer2")
    ap.add_argument("--min-tokens", type=int, default=10, help="min tokens/subject to include")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    cc = PS.CONCEPTS["classification"]  # src=Abnormal, tgt=Normal, age_filter=adults
    EXP = ROOT / "results" / "experiments" / args.experiment
    meta = json.loads((EXP / "metadata.json").read_text())
    cache = torch.load(EXP / "app_cache.pt", map_location="cpu", weights_only=False)
    sae_ckpt = torch.load(meta["sae_checkpoint"], map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(meta["embed_dim"], expansion=meta["expansion"], mode="topk", k=meta["k"])
    sae.load_state_dict(sae_ckpt["sae_state_dict"]); sae.eval()
    act_mean = sae_ckpt["act_mean"].float(); act_std = sae_ckpt["act_std"].float()
    sd_ckpt = meta.get("spectral_decoder_checkpoint") or meta["xae_checkpoint"]
    sdt = SpectralDecoderTrainer(embed_dim=meta["embed_dim"])
    sdt.load(sd_ckpt); sdt.spectral_decoder.cpu().eval()
    enr = cache["feature_meta_enrichment"]; nf = len(enr)

    sc = torch.load(ROOT / "results" / "steering_cache" / args.experiment / "steering_cache.pt",
                    map_location="cpu", weights_only=False)
    age = np.array([str(v) for v in sc["age_group"]])
    cls = np.array([str(v) for v in sc["classification"]])
    subj = np.array([str(v) for v in sc["subject_id"]])
    emb = torch.tensor(sc["embeddings"].numpy().astype(np.float32))
    x_norm = (emb - act_mean) / (act_std + 1e-8)

    with torch.no_grad():
        z_all = torch.cat([sae.encode(x_norm[i:i + 512]) for i in range(0, len(x_norm), 512)], 0)

    cm = PS.cls_mask(cls, cc.cls_filter)
    m_src = np.isin(age, list(cc.age_filter)) & np.isin(cls, list(cc.src_vals)) & cm
    m_tgt = np.isin(age, list(cc.age_filter)) & np.isin(cls, list(cc.tgt_vals)) & cm
    print(f"[pop] abnormal-adult tokens={int(m_src.sum())} ({len(set(subj[m_src]))} subj); "
          f"normal-adult tokens={int(m_tgt.sum())}", flush=True)

    feat_order = PS.rank_by_enrichment(enr, cc.field, cc.src_vals, cc.tgt_vals, nf)
    if len(feat_order) < 50:
        feat_order = PS.rank_by_probe(z_all, np.where(m_src)[0], np.where(m_tgt)[0], nf)
    z_donor = z_all[m_tgt].mean(dim=0)
    n_clamp = len(feat_order)
    print(f"[pop] clamping all {n_clamp} concept-enriched features", flush=True)

    def decode_amp(z, batch=2048):
        out = []
        with torch.no_grad():
            for i in range(0, len(z), batch):
                x_dec = sae.decode(z[i:i + batch])
                emb_r = x_dec * (act_std + 1e-8) + act_mean
                log_amp, _, _ = sdt.decode_direction(emb_r.cpu(), denormalise=True)
                out.append(torch.expm1(log_amp.clamp(min=0)).numpy())
        return np.concatenate(out, 0)

    band_masks = {b: sdt.spectral.get_band_mask(b) for b in BANDS}
    band_masks = {b: m for b, m in band_masks.items() if np.asarray(m).sum() > 0}
    bands = list(band_masks)
    print(f"[pop] bands with non-empty masks: {bands}", flush=True)

    # per-subject band deltas (steered − source), in μV
    rows = []
    abn_subs = sorted(set(subj[m_src]))
    for s in abn_subs:
        idx = np.where(m_src & (subj == s))[0]
        if len(idx) < args.min_tokens:
            continue
        z0 = z_all[idx]
        z1 = z0.clone(); z1[:, feat_order] = z_donor[feat_order].unsqueeze(0)
        a0 = decode_amp(z0); a1 = decode_amp(z1)
        d = {b: float(a1[:, band_masks[b]].mean() - a0[:, band_masks[b]].mean()) for b in bands}
        d["subject"] = s; rows.append(d)
    print(f"[pop] {len(rows)} abnormal subjects with ≥{args.min_tokens} tokens", flush=True)

    # bootstrap over subjects
    rng = np.random.default_rng(42)
    arr = {b: np.array([r[b] for r in rows]) for b in bands}
    summary = {}
    for b in bands:
        v = arr[b]
        boot = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(args.bootstrap)])
        summary[b] = {"mean": float(v.mean()),
                      "ci_low": float(np.percentile(boot, 2.5)),
                      "ci_high": float(np.percentile(boot, 97.5)),
                      "frac_subjects_negative": float((v < 0).mean())}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "population_spectral.json").write_text(json.dumps(
        {"experiment": args.experiment, "n_subjects": len(rows), "n_clamp": n_clamp,
         "bands": summary}, indent=2))

    print("\n=== Population abnormal→normal spectral shift (μV, mean [95% CI]) ===")
    for b in bands:
        s = summary[b]
        arrow = "↓" if s["mean"] < 0 else "↑"
        print(f"  {b:6s} {s['mean']:+.4f} [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] {arrow}  "
              f"({100*s['frac_subjects_negative']:.0f}% subj negative)")
    print(f"\n[T3] wrote {OUT/'population_spectral.json'}")


if __name__ == "__main__":
    main()
