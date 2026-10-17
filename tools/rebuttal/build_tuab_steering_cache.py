"""Rebuttal — build an L2 steering cache from the TUAB-finetuned SleepFM encoder.

Extracts layer-2 per-token embeddings + per-token metadata (classification, age,
gender, subject) from the finetuned checkpoint, in the same schema the steering /
selectivity machinery consumes (results/steering_cache/{exp}/steering_cache.pt).
Feeds the age-pathology entanglement replication on public TUAB data.

  uv run tools/rebuttal/build_tuab_steering_cache.py --max-tokens 800000
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
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import finetune_sleepfm_v1_binary as F  # noqa: E402
from mecheeg.sae import ActivationExtractor  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP, STD = 5e-5, 1e-5  # StandardizeLabel constants
TARGET_LAYER = 2
DATA_DIR = ROOT / "data" / "tuab-sleepfm-128"
CKPT = ROOT / "checkpoints" / "finetuned" / "sleepfm_tuab" / "finetuned.ckpt"


def load_encoder_for(pretrained: bool):
    """finetuned = TUAB-finetuned head+encoder; pretrained = frozen pretrained SleepFM (AC#1)."""
    model = F.build_model()   # build_model already loads PRETRAIN weights into the encoder
    if not pretrained:
        state = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model.encoder  # SetTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=800000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pretrained", action="store_true",
                    help="use frozen PRETRAINED SleepFM (no finetuning) — for the AC#1 comparison")
    args = ap.parse_args()

    tag = "sleepfm_tuab_pretrained_layer2" if args.pretrained else "sleepfm_tuab_layer2"
    out_dir = ROOT / "results" / "steering_cache" / tag
    enc = load_encoder_for(args.pretrained)
    extractor = ActivationExtractor(enc, layers=list(enc.transformer_encoder.layers))

    shards = sorted(glob.glob(str(DATA_DIR / "*.hdf5")))
    emb_chunks, meta = [], {k: [] for k in ("classification", "age", "gender", "subject_id")}
    n_tok = 0
    tpw = None
    rng = np.random.default_rng(42)

    for sp in shards:
        with h5py.File(sp, "r") as h:
            data = h["data"][:]                     # (n,19,7680) Volts
            subj = h["subjects"][:]
            age = h["metadata"]["age"][:]
            gender = [g.decode() for g in h["metadata"]["gender"][:]]
            cls = [c.decode() for c in h["metadata"]["classification"][:]]
        x = np.clip(data, -CLIP, CLIP) / STD
        for i in range(0, len(x), args.batch_size):
            xb = torch.from_numpy(x[i:i + args.batch_size]).float().to(DEVICE)
            with torch.no_grad():
                extractor.clear()
                _ = enc(xb)
                act = extractor.get_activations()[TARGET_LAYER]   # (B, S, E)
            B, S, E = act.shape
            tpw = S
            emb_chunks.append(act.reshape(B * S, E).cpu().numpy().astype(np.float32))
            sl = slice(i, i + B)
            for j in range(B):
                gi = i + j
                meta["classification"].extend([cls[gi]] * S)
                meta["age"].extend([int(age[gi])] * S)
                meta["gender"].extend([gender[gi]] * S)
                meta["subject_id"].extend([str(int(subj[gi]))] * S)
            n_tok += B * S
        print(f"  {Path(sp).name}: {n_tok} tokens", flush=True)
        if n_tok >= args.max_tokens:
            break

    embeddings = np.concatenate(emb_chunks)
    # subsample to max_tokens (keep alignment)
    if len(embeddings) > args.max_tokens:
        idx = rng.choice(len(embeddings), args.max_tokens, replace=False)
        idx.sort()
        embeddings = embeddings[idx]
        for k in meta:
            meta[k] = [meta[k][i] for i in idx]

    # age_group for the TUAB adult contrast: young (<40) / old (>=60), else 'mid'/'unknown'
    def ag(a):
        if a < 0 or a > 120:
            return "unknown"
        if a < 40:
            return "young"
        if a >= 60:
            return "old"
        return "mid"

    age_group = np.array([ag(a) for a in meta["age"]], dtype=object)

    payload = {
        "experiment": tag,
        "embeddings": torch.from_numpy(embeddings),
        "embeddings_final": None,
        "embed_dim": embeddings.shape[1],
        "n_tokens": len(embeddings),
        "tokens_per_window": tpw,
        "target_layer": TARGET_LAYER,
        "final_layer": TARGET_LAYER,
        "classification": np.array(meta["classification"], dtype=object),
        "age": np.array(meta["age"], dtype=np.int32),
        "age_group": age_group,
        "gender": np.array(meta["gender"], dtype=object),
        "subject_id": np.array(meta["subject_id"], dtype=object),
        "data_path": str(DATA_DIR),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / "steering_cache.pt")
    print(f"\n[cache] {len(embeddings)} tokens × {embeddings.shape[1]}  tpw={tpw}")
    import collections
    print("  classification:", dict(collections.Counter(meta["classification"])))
    print("  age_group:", dict(collections.Counter(age_group.tolist())))
    print(f"  → {out_dir/'steering_cache.pt'}")


if __name__ == "__main__":
    main()
