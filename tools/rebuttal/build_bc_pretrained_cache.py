"""Rebuttal #3 — L2 steering cache from the FROZEN PRETRAINED SleepFM on BrainCapture.

Companion to the finetuned BrainCapture cache (results/steering_cache/sleepfm_finetuned_layer2)
so the pretrain-vs-finetune comparison (AC#1) can be made on the paper's OWN cohort, not
only public TUAB. Runs the pretrained SleepFM (no finetuning) over the D4 data, extracts
layer-2 per-token embeddings + BrainCapture metadata.

  uv run tools/rebuttal/build_bc_pretrained_cache.py --max-tokens 434220
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mecheeg.encoders import load_encoder  # noqa: E402
from mecheeg.sae import ActivationExtractor  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP, STD = 5e-5, 1e-5
TARGET_LAYER = 2
DATA_DIR = ROOT / "data" / "D4-v3-preprocessed-v2"
PRETRAIN = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"
OUT = ROOT / "results" / "steering_cache" / "sleepfm_bc_pretrained_layer2"
META = ["age_group", "classification", "gender", "medication_group"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=434220)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    backend = load_encoder("sleepfm", weights_path=PRETRAIN)
    enc = backend.model.to(DEVICE).eval()
    extractor = ActivationExtractor(enc, layers=list(enc.transformer_encoder.layers))

    shards = sorted(glob.glob(str(DATA_DIR / "*.hdf5")))
    emb_chunks = []
    meta = {k: [] for k in META + ["subject_id"]}
    n_tok = 0; tpw = None
    for sp in shards:
        with h5py.File(sp, "r") as h:
            data = h["data"][:]                                    # (N,27,7680)
            subj = h["subjects"][:]
            md = {k: [x.decode() if isinstance(x, bytes) else str(x) for x in h["metadata"][k][:]]
                  for k in META}
        x = np.clip(data, -CLIP, CLIP) / STD
        for i in range(0, len(x), args.batch_size):
            xb = torch.from_numpy(x[i:i + args.batch_size]).float().to(DEVICE)
            with torch.no_grad():
                extractor.clear(); _ = enc(xb)
                act = extractor.get_activations()[TARGET_LAYER]    # (B,S,E)
            B, S, E = act.shape; tpw = S
            emb_chunks.append(act.reshape(B * S, E).cpu().numpy().astype(np.float32))
            for j in range(B):
                gi = i + j
                for k in META:
                    meta[k].extend([md[k][gi]] * S)
                meta["subject_id"].extend([str(int(subj[gi]))] * S)
            n_tok += B * S
        print(f"  {Path(sp).name}: {n_tok} tokens", flush=True)
        if n_tok >= args.max_tokens:
            break

    embeddings = np.concatenate(emb_chunks)
    if len(embeddings) > args.max_tokens:
        idx = np.sort(np.random.default_rng(42).choice(len(embeddings), args.max_tokens, replace=False))
        embeddings = embeddings[idx]
        for k in meta:
            meta[k] = [meta[k][i] for i in idx]

    payload = {
        "experiment": "sleepfm_bc_pretrained_layer2",
        "embeddings": torch.from_numpy(embeddings), "embeddings_final": None,
        "embed_dim": embeddings.shape[1], "n_tokens": len(embeddings),
        "tokens_per_window": tpw, "target_layer": TARGET_LAYER, "final_layer": TARGET_LAYER,
        "subject_id": np.array(meta["subject_id"], dtype=object),
        "data_path": str(DATA_DIR),
    }
    for k in META:
        payload[k] = np.array(meta[k], dtype=object)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT / "steering_cache.pt")
    import collections
    print(f"\n[bc-pretrained] {len(embeddings)} tokens × {embeddings.shape[1]}  tpw={tpw}")
    print("  classification:", dict(collections.Counter(meta["classification"])))
    print(f"  → {OUT/'steering_cache.pt'}")


if __name__ == "__main__":
    main()
