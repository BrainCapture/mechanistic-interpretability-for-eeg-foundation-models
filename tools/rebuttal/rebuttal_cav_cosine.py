"""Rebuttal #7 — CAV-cosine: is age–pathology entanglement induced by finetuning?

A seed-free, direct entanglement metric that does NOT depend on SAE retraining: fit a
linear CAV per concept in the (standardized) encoder embedding space, and measure the
cosine between the abnormality axis and the age axis. If |cos(CAV_abn, CAV_age)| rises
with finetuning, the two axes literally rotate toward each other — entanglement is
induced by the supervised objective (backs the pretrain-vs-finetune claim, AC#1).

  uv run tools/rebuttal/rebuttal_cav_cosine.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))
import plot_cross_model_steering_concepts_paper as P  # noqa: E402

OUT = ROOT / "results" / "rebuttal"

# (name, field, positive-set, negative-set)
CONCEPTS_TUAB = [
    ("abnormality", "classification", {"Abnormal"}, {"Normal"}),
    ("age", "age_group", {"young"}, {"old"}),
]
CONCEPTS_BC = [
    ("abnormality", "classification", P.ABN_CATS, P.NORMAL),
    ("age", "age_group", {"0-3", "4-9"}, {"20-29", "30-39", "40-49", "50-59", "60+"}),
    ("gender", "gender", {"male"}, {"female"}),
    ("medication", "medication_group", {"Psychiatric"}, {"No Current Medication"}),
]

CACHES = [
    ("SleepFM TUAB — pretrained", "sleepfm_tuab_pretrained_layer2", CONCEPTS_TUAB),
    ("SleepFM TUAB — finetuned", "sleepfm_tuab_layer2", CONCEPTS_TUAB),
    ("SleepFM BrainCapture — finetuned", "sleepfm_finetuned_layer2", CONCEPTS_BC),
    ("SleepFM BrainCapture — pretrained", "sleepfm_bc_pretrained_layer2", CONCEPTS_BC),
]


def fit_cav_on(emb, idx, y, am, as_):
    X = (torch.as_tensor(np.asarray(emb[idx])).float().to(P.DEVICE) - am) / (as_ + 1e-8)
    w = P.fit_probe(X, y).w.cpu()
    return (w / (w.norm() + 1e-8)).numpy()


def main(n_boot=30):
    results = {}
    for label, exp, concepts in CACHES:
        path = ROOT / "results/steering_cache" / exp / "steering_cache.pt"
        if not path.exists():
            print(f"[skip] {label} (no cache {exp})"); continue
        sc = torch.load(path, map_location="cpu", weights_only=False)
        emb = sc["embeddings"].float()
        am = emb.mean(0).to(P.DEVICE); as_ = emb.std(0).clamp(min=1e-8).to(P.DEVICE)
        rng = np.random.default_rng(42)
        # pools per concept
        pools = {}
        for name, field, pos, neg in concepts:
            fa = np.asarray([str(v) for v in sc[field]])
            p = np.where(np.isin(fa, list(pos)))[0]; n = np.where(np.isin(fa, list(neg)))[0]
            k = min(3000, len(p), len(n))
            if k >= 100:
                pools[name] = (p, n, k)
        # bootstrap: resample each concept's pool, refit CAVs, record |cos| for abn·age
        boots = {"abnormality·age": []}
        for _ in range(n_boot):
            cavs = {}
            for name, (p, n, k) in pools.items():
                idx = np.concatenate([rng.choice(p, k, replace=True), rng.choice(n, k, replace=True)])
                y = np.array([1] * k + [0] * k)
                cavs[name] = fit_cav_on(emb, idx, y, am, as_)
            if "abnormality" in cavs and "age" in cavs:
                boots["abnormality·age"].append(abs(float(np.dot(cavs["abnormality"], cavs["age"]))))
        v = np.array(boots["abnormality·age"])
        results[label] = {"abn_age_cos_mean": float(v.mean()), "abn_age_cos_std": float(v.std(ddof=1)),
                          "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "n_boot": n_boot}
        print(f"{label:40s}  |cos(abn,age)| = {v.mean():.3f} ± {v.std(ddof=1):.3f} "
              f"[{np.percentile(v,2.5):.3f}, {np.percentile(v,97.5):.3f}]", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cav_cosine.json").write_text(json.dumps(results, indent=2))
    print(f"\n[#7] wrote {OUT/'cav_cosine.json'}")


if __name__ == "__main__":
    main()
