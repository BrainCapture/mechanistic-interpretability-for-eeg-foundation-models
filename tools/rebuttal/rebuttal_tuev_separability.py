"""Rebuttal — can the framework separate specific epileptiform phenomena? (rkJV Q4)

The paper's epileptiform null (binary normal/abnormal collapses spike/sharp-wave +
slowing) is a *labeling* limitation, not a representational one. Here we test, on public
TUEV event data, whether the SleepFM representation + SAE separate specific epileptiform
event types (SPSW, GPED, PLED) from background — with a leakage-free subject split.

Extracts L2 (mean-pooled per 10 s window) embeddings from the TUAB-finetuned SleepFM,
then reports: (a) linear-probe AUROC for each epileptiform class vs background on the
encoder embedding, and (b) the same on SAE features (SAE preserves the morphology signal).

  uv run tools/rebuttal/rebuttal_tuev_separability.py
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
DATA = ROOT / "data" / "tuev-sleepfm-128"
NAMES = {0: "spsw", 1: "gped", 2: "pled", 3: "eyem", 4: "artf", 5: "bckg"}
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
        for i in range(0, len(x), 64):
            xb = torch.from_numpy(x[i:i + 64]).float().to(DEVICE)
            with torch.no_grad():
                ex.clear(); _ = enc(xb)
                act = ex.get_activations()[2]      # (B, S, E)
            embs.append(act.mean(1).cpu().numpy())  # mean-pool tokens → (B, E)
        labs.append(y); subs.append(s)
    return np.concatenate(embs), np.concatenate(labs), np.concatenate(subs)


_TCRIT = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
          9: 2.306, 10: 2.262, 12: 2.201, 15: 2.145, 20: 2.093, 25: 2.064, 30: 2.045}


def _tcrit(n):
    """Two-sided 95% t critical value for n-1 dof (nearest tabulated, ≥30 → 1.96)."""
    if n >= 30:
        return 1.96
    return _TCRIT[min(k for k in _TCRIT if k >= n)]


def probe_auroc(X, y_bin, groups, n_splits=N_SPLITS):
    """Linear-probe AUROC over repeated leakage-free subject splits.

    Returns (mean, ci_lo, ci_hi, n_used). The interval is a 95% t-interval over the
    per-split AUROCs, i.e. it reflects subject-sampling variability — the dominant
    source of uncertainty at these class sizes (SPSW n=74).
    """
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
    aucs = []
    for tr, te in gss.split(X, y_bin, groups):
        if y_bin[tr].sum() < 5 or y_bin[te].sum() < 5 or y_bin[te].sum() == len(te):
            continue  # degenerate split (a whole class landed on one side)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X[tr], y_bin[tr])
        aucs.append(roc_auc_score(y_bin[te], clf.decision_function(X[te])))
    a = np.asarray(aucs)
    m = float(a.mean())
    if len(a) < 2:
        return m, m, m, len(a)
    h = _tcrit(len(a)) * float(a.std(ddof=1)) / np.sqrt(len(a))
    return m, max(0.0, m - h), min(1.0, m + h), len(a)


def main():
    print("[tuev] extracting L2 embeddings (TUAB-finetuned SleepFM)…", flush=True)
    X, y, subj = extract()
    print(f"  {X.shape[0]} windows × {X.shape[1]}", flush=True)

    # standardize + SAE features (fresh SAE on these embeddings)
    Xt = torch.from_numpy(X).float()
    mean = Xt.mean(0); std = Xt.std(0).clamp(min=1e-8)
    Xn = (Xt - mean) / std
    torch.manual_seed(SEED); np.random.seed(SEED)  # camera-ready: SAE was unseeded in the rebuttal run
    trainer = SAETrainer(embed_dim=X.shape[1], expansion=1.0, mode="topk", k=8,
                         lr=3e-4, batch_size=2048, epochs=20, device=str(DEVICE))
    sae = trainer._train_single_sae(Xn.clone(), 2); sae.to(DEVICE).eval()
    with torch.no_grad():
        Z = torch.cat([sae.encode(Xn[i:i + 4096].to(DEVICE)).cpu()
                       for i in range(0, len(Xn), 4096)]).numpy()

    bck = (y == 5)
    results = {}
    for c in (0, 1, 2):  # spsw, gped, pled
        mask = (y == c) | bck
        if (y == c).sum() < 20:
            note = "sparse"
        else:
            note = ""
        yb = (y[mask] == c).astype(int)
        e_m, e_lo, e_hi, n_used = probe_auroc(X[mask], yb, subj[mask])
        s_m, s_lo, s_hi, _ = probe_auroc(Z[mask], yb, subj[mask])
        results[NAMES[c]] = {"n_pos": int((y == c).sum()), "n_splits": n_used,
                             "auroc_embedding": round(e_m, 3), "ci_embedding": [round(e_lo, 3), round(e_hi, 3)],
                             "auroc_sae": round(s_m, 3), "ci_sae": [round(s_lo, 3), round(s_hi, 3)],
                             "note": note}
        print(f"  {NAMES[c]:5s} vs bckg (n_pos={int((y==c).sum()):4d}, {n_used} splits): "
              f"embedding {e_m:.3f} [{e_lo:.3f},{e_hi:.3f}]  "
              f"SAE {s_m:.3f} [{s_lo:.3f},{s_hi:.3f}] {note}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tuev_separability.json").write_text(json.dumps(
        {"dataset": "TUEV", "encoder": "sleepfm_tuab L2 (mean-pooled)",
         "n_windows": int(X.shape[0]),
         "ci": f"mean ± 95% t-interval over {N_SPLITS} leakage-free subject splits",
         "results": results}, indent=2))
    print(f"\n[tuev] wrote {OUT/'tuev_separability.json'}")


if __name__ == "__main__":
    main()
