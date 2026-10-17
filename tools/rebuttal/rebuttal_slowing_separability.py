"""Rebuttal #3 — slowing separability on public TUH Slowing (rkJV Q4: focal slowing).

The paper's binary label collapses slowing with other abnormality; rkJV asked whether the
framework can address *focal slowing* specifically. We test, on public TUH Slowing (TUSL,
60 s / 128 Hz / 19-ch), whether the TUAB-finetuned SleepFM representation + SAE separate
the corpus's slowing class from normal (and seizure/background as references), with a
leakage-free subject split.

Classes: 0=normal, 1=bckg, 2=seizure, 3=slowing.

  uv run tools/rebuttal/rebuttal_slowing_separability.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cr  # noqa: E402  (camera-ready CR_OUT_DIR / CR_SEED switches)
import finetune_sleepfm_v1_binary as F  # noqa: E402
from mecheeg.sae import ActivationExtractor, SAETrainer  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP, STD = 5e-5, 1e-5
CKPT = ROOT / "checkpoints" / "finetuned" / "sleepfm_tuab" / "finetuned.ckpt"
DATA = ROOT / "data" / "tuh-slowing"
NAMES = {0: "normal", 1: "bckg", 2: "seizure", 3: "slowing"}
OUT = _cr.out_dir(ROOT / "results" / "rebuttal")
SEED = _cr.seed(42)
N_SPLITS = 20


def extract():
    model = F.build_model()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"])
    model.eval().to(DEVICE)
    enc = model.encoder
    ex = ActivationExtractor(enc, layers=list(enc.transformer_encoder.layers))
    embs, labs, subs = [], [], []
    for sp in sorted(glob.glob(str(DATA / "*.hdf5"))):
        with h5py.File(sp, "r") as h:
            data = h["data"][:]; y = h["labels"][:]; s = h["subjects"][:]
        x = np.clip(data, -CLIP, CLIP) / STD
        for i in range(0, len(x), 32):
            xb = torch.from_numpy(x[i:i + 32]).float().to(DEVICE)
            with torch.no_grad():
                ex.clear(); _ = enc(xb)
                act = ex.get_activations()[2]
            embs.append(act.mean(1).cpu().numpy())
        labs.append(y); subs.append(s)
    return np.concatenate(embs), np.concatenate(labs), np.concatenate(subs)


_TCRIT = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
          9: 2.306, 10: 2.262, 12: 2.201, 15: 2.145, 20: 2.093, 25: 2.064, 30: 2.045}


def _tcrit(n):
    """Two-sided 95% t critical value for n-1 dof (nearest tabulated, ≥30 → 1.96)."""
    if n >= 30:
        return 1.96
    return _TCRIT[min(k for k in _TCRIT if k >= n)]


def probe(X, y_bin, groups, n_splits=N_SPLITS):
    """Linear-probe AUROC over repeated leakage-free subject splits.

    Returns (mean, ci_lo, ci_hi, n_used); interval is a 95% t-interval over splits,
    reflecting subject-sampling variability (classes here are ~90 windows).
    """
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
    aucs = []
    for tr, te in gss.split(X, y_bin, groups):
        if y_bin[tr].sum() < 5 or y_bin[te].sum() < 5 or y_bin[te].sum() == len(te):
            continue  # degenerate split (a whole class landed on one side)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr], y_bin[tr])
        aucs.append(roc_auc_score(y_bin[te], clf.decision_function(X[te])))
    a = np.asarray(aucs)
    m = float(a.mean())
    if len(a) < 2:
        return m, m, m, len(a)
    h = _tcrit(len(a)) * float(a.std(ddof=1)) / np.sqrt(len(a))
    return m, max(0.0, m - h), min(1.0, m + h), len(a)


def main():
    print("[slowing] extracting L2 embeddings (TUAB-finetuned SleepFM)…", flush=True)
    X, y, subj = extract()
    print(f"  {X.shape[0]} windows × {X.shape[1]};  classes " +
          str({NAMES[k]: int(v) for k, v in zip(*np.unique(y, return_counts=True))}), flush=True)

    # SAE features (fresh SAE on these embeddings)
    Xt = torch.from_numpy(X).float()
    mean = Xt.mean(0); std = Xt.std(0).clamp(min=1e-8); Xn = (Xt - mean) / std
    torch.manual_seed(SEED); np.random.seed(SEED)  # camera-ready: SAE was unseeded in the rebuttal run
    tr = SAETrainer(embed_dim=X.shape[1], expansion=1.0, mode="topk", k=8, lr=3e-4,
                    batch_size=1024, epochs=20, device=str(DEVICE))
    sae = tr._train_single_sae(Xn.clone(), 2); sae.to(DEVICE).eval()
    with torch.no_grad():
        Z = torch.cat([sae.encode(Xn[i:i + 4096].to(DEVICE)).cpu() for i in range(0, len(Xn), 4096)]).numpy()

    normal = (y == 0)
    results = {}
    for c in (3, 2, 1):  # slowing, seizure, bckg vs normal
        mask = (y == c) | normal
        yb = (y[mask] == c).astype(int)
        e_m, e_lo, e_hi, n_used = probe(X[mask], yb, subj[mask])
        s_m, s_lo, s_hi, _ = probe(Z[mask], yb, subj[mask])
        results[NAMES[c]] = {"n_pos": int((y == c).sum()), "n_splits": n_used,
                             "auroc_embedding": round(e_m, 3), "ci_embedding": [round(e_lo, 3), round(e_hi, 3)],
                             "auroc_sae": round(s_m, 3), "ci_sae": [round(s_lo, 3), round(s_hi, 3)]}
        print(f"  {NAMES[c]:8s} vs normal (n={int((y == c).sum()):3d}, {n_used} splits): "
              f"embedding {e_m:.3f} [{e_lo:.3f},{e_hi:.3f}]  "
              f"SAE {s_m:.3f} [{s_lo:.3f},{s_hi:.3f}]", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "slowing_separability.json").write_text(json.dumps(
        {"dataset": "TUH Slowing (TUSL)", "encoder": "sleepfm_tuab L2 (mean-pooled)",
         "n_windows": int(X.shape[0]),
         "ci": f"mean ± 95% t-interval over {N_SPLITS} leakage-free subject splits",
         "results": results}, indent=2))
    print(f"\n[#3] wrote {OUT/'slowing_separability.json'}")


if __name__ == "__main__":
    main()
