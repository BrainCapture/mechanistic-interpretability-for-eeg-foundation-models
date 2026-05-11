"""build_taxonomy_cache.py — Lightweight per-(L,E) taxonomy cache.

Produces only what the monosemanticity taxonomy classifier needs:

    feature_meta_enrichment  list[dict]  — per-feature × per-field BH-corrected
                                            (cat, ratio, p_adj) tuples
    feature_stats            list[dict]  — per-feature {fire_rate_pct}
    meta                     dict        — {n_features, k, target_layer, encoder, expansion}

Skips everything else build_app_cache.py builds (SpectralDecoder spectral decoding, codebook
UMAP, morphology probe, dashboards, co-occurrence, etc.). Result: ~3-5 min per
cell on L4 instead of ~12-15 min for the full app_cache pipeline. Saves to::

    results/experiments/<exp>/taxonomy_cache.pt

Usage::

    uv run tools/build_taxonomy_cache.py --experiment reve_qjbe08_exp4_layer16
    uv run tools/build_taxonomy_cache.py --experiment ... --enr-max-tokens 1000000

Layer-sweep mode (collects encoder activations once at layer L, runs SAE-encode
+ enrichment for every E in --expansions, sharing the activation buffer)::

    uv run tools/build_taxonomy_cache.py --encoder reve --tag qjbe08 \\
        --layer 16 --expansions 1,4,16 --weights-path checkpoints/finetuned/reve_qjbe08.ckpt
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.stats import false_discovery_control as _bh
from scipy.stats import mannwhitneyu as _mwu
from torch.utils.data import DataLoader
from tqdm import tqdm

from mecheeg.dataset import H5PYDatasetLabeled, StandardizeLabel, V4ResampleTransform
from mecheeg.encoders import EncoderBackend, load_encoder
from mecheeg.sae import ActivationExtractor, SparseAutoencoder

ROOT   = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mirror the encoder→data + checkpoint maps from train_sae_layers.py.
MODEL_PATH = ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"
_V2_DIR = ROOT / "checkpoints" / "pretrained" / "SleepFM v2 Models"
_V2_CHECKPOINTS: dict[str, Path] = {}  # filled if needed
_LABRAM_CHECKPOINTS = {"labram": ROOT / "checkpoints" / "finetuned" / "labram_binary" / "finetuned.ckpt"}
_GRANULAR_CHECKPOINTS = {"sleepfm_granular": ROOT / "checkpoints" / "granular" / "sleepfm_granular.ckpt"}
_ENCODER_DATA = {
    "sleepfm":          ROOT / "data" / "D4-v3-preprocessed-v2",
    "sleepfm_granular": ROOT / "data" / "D4-v4-preprocessed-10s",
    "reve":             ROOT / "data" / "D4-v3-preprocessed-v1",
    "labram":           ROOT / "data" / "D4-v3-preprocessed-v1",
}

METADATA_FIELDS = ("age_group", "gender", "indication_group", "medication_group")
MIN_SUBJ        = 5
P_THRESH        = 0.05  # only used by downstream classifier; we save raw p_adj


# ── Encoder helper ──────────────────────────────────────────────────────────-

def _load_encoder(encoder_name: str, weights_path):
    if weights_path is not None:
        resolved = weights_path
    elif encoder_name == "sleepfm":
        resolved = MODEL_PATH
    elif encoder_name in _GRANULAR_CHECKPOINTS:
        resolved = _GRANULAR_CHECKPOINTS[encoder_name]
    elif encoder_name in _LABRAM_CHECKPOINTS:
        ckpt = _LABRAM_CHECKPOINTS[encoder_name]
        resolved = ckpt if ckpt.exists() else (ROOT / "checkpoints" / "pretrained" / "labram" / "labram-base.pth")
    else:
        resolved = None
    kwargs = {"weights_path": resolved} if resolved is not None else {}
    enc = load_encoder(encoder_name, **kwargs)
    enc.to(DEVICE).eval()
    return enc


# ── Activation collection (single layer L, shared across E) ─────────────────-

@torch.no_grad()
def _collect_activations(
    encoder, dataset: H5PYDatasetLabeled, target_layer: int,
    *, max_tokens: int, batch_size: int, num_workers: int,
) -> tuple[torch.Tensor, dict]:
    """Run the encoder over the (subject-interleaved) dataset; return CPU
    activation tensor (N_tokens, embed_dim) and per-token metadata dict.

    Subject-interleaved order is used so capping at max_tokens covers subjects
    uniformly instead of exhausting long recordings first.
    """
    is_backend = isinstance(encoder, EncoderBackend)
    reve_mode  = is_backend and hasattr(encoder, "channel_names")

    n_layers = len(encoder.get_hookable_layers()) if is_backend else None
    extractor = None
    if is_backend and (n_layers is None or target_layer != n_layers - 1):
        extractor = ActivationExtractor(
            encoder.model, layers=[encoder.get_hookable_layers()[target_layer]],
        )

    # Subject-interleaved permutation
    subj_arr = dataset.index_map["subjects"]
    if torch.is_tensor(subj_arr):
        subj_arr = subj_arr.numpy()
    uniq_subj = np.unique(subj_arr)
    subj_wins: dict = defaultdict(list)
    for wi, s in enumerate(subj_arr):
        subj_wins[s].append(wi)
    max_wins = max(len(v) for v in subj_wins.values())
    perm: list[int] = []
    for r in range(max_wins):
        for s in uniq_subj:
            if r < len(subj_wins[s]):
                perm.append(subj_wins[s][r])
    perm_t = torch.tensor(perm, dtype=torch.long)

    orig_index_map = {
        k: (v.clone() if torch.is_tensor(v) else v.copy())
        for k, v in dataset.index_map.items()
    }
    for k, v in dataset.index_map.items():
        if torch.is_tensor(v):
            dataset.index_map[k] = v[perm_t]
        elif isinstance(v, np.ndarray):
            dataset.index_map[k] = v[perm]

    try:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

        embed_chunks: list[torch.Tensor] = []
        total = 0
        tokens_per_window = 1
        pbar = tqdm(loader, desc=f"L{target_layer} collect", leave=False)
        for batch in pbar:
            x = batch[0].to(DEVICE)
            B, C, T = x.shape

            if is_backend and extractor is not None:
                extractor.clear()
                _ = encoder.encode(x)
                acts = extractor.get_activations()[0]
                if acts.dim() == 4:
                    Bb, Cc, Ss, Ee = acts.shape
                    acts = acts.reshape(Bb, Cc * Ss, Ee)
            elif is_backend:
                acts = encoder.encode(x)
            else:
                tmp_ex = ActivationExtractor(encoder)
                tmp_ex.clear()
                _ = encoder(x)
                acts = tmp_ex.get_activations()[target_layer]
                tmp_ex.remove_hooks()

            n_new = acts.shape[0] * acts.shape[1]
            embed_chunks.append(acts.reshape(n_new, -1).cpu())
            if total == 0:
                tokens_per_window = n_new // B
            total += n_new
            pbar.set_postfix(tokens=f"{total:,}")
            if total >= max_tokens:
                break
        pbar.close()
        if extractor is not None:
            extractor.remove_hooks()

        embeddings = torch.cat(embed_chunks)[:max_tokens]
        n_tokens   = len(embeddings)

        # Per-token metadata (subject_id + the 4 enrichment fields). The
        # metadata fields live in the HDF5 files under metadata/<field>, NOT
        # in dataset.index_map — read them out per file like
        # build_app_cache._collect_token_metadata does.
        token_meta = _read_window_metadata_from_h5(
            dataset, n_tokens, tokens_per_window, METADATA_FIELDS,
        )
    finally:
        for k, v in orig_index_map.items():
            dataset.index_map[k] = v

    return embeddings, token_meta


def _read_window_metadata_from_h5(
    dataset: H5PYDatasetLabeled, n_tokens: int, tokens_per_window: int,
    fields: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Mirror tools/build_app_cache.py:_collect_token_metadata — load
    per-window metadata from HDF5 files (metadata/<field>) and tile to
    per-token arrays. dataset.index_map must already be in the order the
    DataLoader walked it (i.e. subject-interleaved permutation already
    applied here)."""
    import h5py

    n_windows = min(
        (n_tokens + tokens_per_window - 1) // tokens_per_window,
        len(dataset),
    )
    file_indices  = dataset.index_map["file_indices"][:n_windows]
    local_indices = dataset.index_map["local_indices"][:n_windows]
    if torch.is_tensor(file_indices):
        file_indices = file_indices.numpy()
    if torch.is_tensor(local_indices):
        local_indices = local_indices.numpy()

    window_meta = {f: np.full(n_windows, "", dtype=object) for f in fields}

    for file_val in np.unique(file_indices):
        positions  = np.where(file_indices == file_val)[0]
        sort_args  = np.argsort(local_indices[positions])
        sorted_pos = positions[sort_args]
        sorted_loc = local_indices[sorted_pos].tolist()

        with h5py.File(dataset.paths[int(file_val)], "r") as f:
            for field in fields:
                try:
                    raw = f["metadata"][field][sorted_loc]
                    for pos, v in zip(sorted_pos, raw):
                        if isinstance(v, (bytes, np.bytes_)):
                            v = v.decode("utf-8", errors="replace")
                        window_meta[field][pos] = str(v).strip()
                except Exception as exc:
                    print(f"  [warn] metadata/{field}: {exc}")

    token_meta: dict[str, np.ndarray] = {}
    for field in fields:
        arr = window_meta[field]
        expanded = np.repeat(arr, tokens_per_window)[:n_tokens]
        token_meta[field] = expanded

    subj = dataset.index_map["subjects"][:n_windows]
    if torch.is_tensor(subj):
        subj = subj.numpy()
    token_meta["subject_id"] = np.repeat(
        subj.astype(str), tokens_per_window
    )[:n_tokens]

    print(f"  metadata: subject_id={len(np.unique(token_meta['subject_id']))}  "
          + "  ".join(
              f"{f}={len(np.unique(token_meta[f]))}u"
              for f in fields if f in token_meta
          ))
    return token_meta


# ── SAE forward + enrichment ────────────────────────────────────────────────-

@torch.no_grad()
def _sae_firing_pattern(
    embeddings: torch.Tensor, sae: SparseAutoencoder,
    act_mean: torch.Tensor, act_std: torch.Tensor,
    *, batch: int = 4096,
) -> np.ndarray:
    """Return (N_tokens, n_features) bool — feature firings under TopK SAE."""
    sae.to(DEVICE).eval()
    am = act_mean.to(DEVICE)
    as_ = act_std.to(DEVICE).clamp(min=1e-8)
    chunks = []
    for s in range(0, len(embeddings), batch):
        c = embeddings[s:s + batch].to(DEVICE)
        z = sae.encode((c - am) / as_)
        chunks.append((z > 0).cpu().numpy())
    sae.cpu()
    return np.concatenate(chunks, axis=0)


def _enrichment(
    fire_np: np.ndarray, token_meta: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    """Subject-level Mann-Whitney + BH per field. Returns
    (feature_meta_enrichment, feature_stats).

    MWU is vectorized across features via scipy's `axis=` arg — gives ~5-30×
    speedup at high E (32k features) vs per-feature looped scipy calls.
    Validated equivalent to the loop version on synthetic data: max p_adj
    diff 0, max ratio diff 6e-15.
    """
    n_features = fire_np.shape[1]
    feature_meta_enrichment: list[dict] = [{} for _ in range(n_features)]

    subj_arr  = token_meta["subject_id"]
    uniq_subj = np.unique(subj_arr)
    if len(uniq_subj) < 2 * MIN_SUBJ:
        # Probably nothing useful.
        feature_stats = [{"fire_rate_pct": 100.0 * fire_np[:, i].mean()}
                         for i in range(n_features)]
        return feature_meta_enrichment, feature_stats

    # Subject-level mean firing rate (S, n_features)
    subj_fr = np.stack([
        fire_np[subj_arr == s].mean(axis=0) for s in uniq_subj
    ])

    def _age_sort(v: str) -> int:
        try:
            return int(v.split("-")[0].replace("+", ""))
        except ValueError:
            return 999

    for field in METADATA_FIELDS:
        tok_arr = token_meta.get(field)
        if tok_arr is None:
            continue
        # Subject → category (majority vote across the subject's tokens)
        subj_cat = {}
        for s in uniq_subj:
            mask = subj_arr == s
            vals, cnts = np.unique(tok_arr[mask], return_counts=True)
            subj_cat[s] = str(vals[np.argmax(cnts)])
        subj_cat_arr = np.array([subj_cat[s] for s in uniq_subj])

        valid_mask = ~np.isin(subj_cat_arr, ("", "unknown"))
        if field == "age_group":
            cats = sorted(set(subj_cat_arr[valid_mask].tolist()), key=_age_sort)
        else:
            cats = sorted(set(subj_cat_arr[valid_mask].tolist()))

        all_ratios: list[tuple[str, int, float]] = []
        all_raw_p: list[float] = []

        global_mask = np.isin(subj_cat_arr, cats)
        global_fr = subj_fr[global_mask].mean(axis=0)
        global_fr_safe = np.maximum(global_fr, 1e-8)

        for cat in cats:
            in_mask  = subj_cat_arr == cat
            out_mask = (~in_mask) & global_mask
            if in_mask.sum() < MIN_SUBJ:
                continue
            fr_in    = subj_fr[in_mask]
            fr_out   = subj_fr[out_mask]

            # Vectorized MWU across all features at once
            try:
                res = _mwu(fr_in, fr_out, alternative="two-sided", axis=0)
                p_vec = np.asarray(res.pvalue, dtype=float)
            except ValueError:
                p_vec = np.ones(n_features)
            # Constant-input features yield NaN — treat as not significant.
            p_vec = np.where(np.isnan(p_vec), 1.0, p_vec)

            ratio_vec = fr_in.mean(axis=0) / global_fr_safe

            for fi in range(n_features):
                all_ratios.append((cat, fi, float(ratio_vec[fi])))
                all_raw_p.append(float(p_vec[fi]))

        if not all_raw_p:
            continue
        adj_p = _bh(np.array(all_raw_p), method="bh")
        for (cat, fi, ratio), p_adj in zip(all_ratios, adj_p):
            feature_meta_enrichment[fi].setdefault(field, []).append(
                (cat, ratio, float(p_adj))
            )

    feature_stats = [
        {"fire_rate_pct": float(100.0 * fire_np[:, i].mean())}
        for i in range(n_features)
    ]
    return feature_meta_enrichment, feature_stats


# ── Save ────────────────────────────────────────────────────────────────────-

def _save_cache(
    out_path: Path, *,
    feature_meta_enrichment: list[dict],
    feature_stats: list[dict],
    n_features: int, k: int, target_layer: int, encoder: str, expansion: float,
    n_tokens: int, n_subjects: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "feature_meta_enrichment": feature_meta_enrichment,
        "feature_stats":           feature_stats,
        "meta": {
            "n_features":   n_features,
            "k":            k,
            "target_layer": target_layer,
            "encoder":      encoder,
            "expansion":    float(expansion),
            "n_tokens":     int(n_tokens),
            "n_subjects":   int(n_subjects),
        },
    }
    torch.save(cache, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  ✅ Saved {out_path}  ({size_mb:.1f} MB)")


# ── CLI: single experiment vs. layer-sweep ──────────────────────────────────-

def _run_single_experiment(args) -> None:
    """Build taxonomy cache for an existing experiment dir (metadata.json present)."""
    exp_dir = ROOT / "results" / "experiments" / args.experiment
    meta = json.loads((exp_dir / "metadata.json").read_text())
    encoder_name = meta["encoder"]
    target_layer = int(meta["target_layer"])
    weights_path = ROOT / meta["weights_path"] if meta.get("weights_path") else None

    print(f"\n=== Building taxonomy cache: {args.experiment} ===")
    enc = _load_encoder(encoder_name, weights_path)

    data_path = _ENCODER_DATA[encoder_name]
    transform = V4ResampleTransform() if "D4-v4" in str(data_path) else StandardizeLabel()
    full_ds = H5PYDatasetLabeled(str(data_path), transform=transform)
    print(f"  Dataset: {len(full_ds)} windows, "
          f"{len(np.unique(full_ds.subjects))} subjects")

    enr_max = args.enr_max_tokens or 2_000_000
    print(f"\n[Collecting up to {enr_max:,} tokens at layer {target_layer}, "
          f"batch={args.batch_size}, workers={args.num_workers}]")
    t0 = time.time()
    embeddings, token_meta = _collect_activations(
        enc, full_ds, target_layer,
        max_tokens=enr_max, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"  Collected {len(embeddings):,} tokens in {(time.time()-t0)/60:.1f} min")

    sae_ckpt = torch.load(ROOT / meta["sae_checkpoint"], map_location="cpu",
                          weights_only=False)
    sae = SparseAutoencoder(meta["embed_dim"], expansion=meta["expansion"],
                            mode="topk", k=meta["k"])
    sae.load_state_dict(sae_ckpt["sae_state_dict"])
    act_mean = sae_ckpt["act_mean"].float()
    act_std  = sae_ckpt["act_std"].float()
    n_features = sae.encoder.weight.shape[0]

    print(f"\n[SAE encode (n_features={n_features}) → enrichment]")
    t0 = time.time()
    fire_np = _sae_firing_pattern(embeddings, sae, act_mean, act_std)
    print(f"  Firing pattern computed in {(time.time()-t0)/60:.1f} min  "
          f"({fire_np.nbytes / 1e6:.0f} MB)")

    t0 = time.time()
    fme, fs = _enrichment(fire_np, token_meta)
    print(f"  BH-corrected enrichment computed in {(time.time()-t0)/60:.1f} min")

    n_subj = len(np.unique(token_meta["subject_id"]))
    out = exp_dir / "taxonomy_cache.pt"
    _save_cache(
        out,
        feature_meta_enrichment=fme, feature_stats=fs,
        n_features=n_features, k=meta["k"], target_layer=target_layer,
        encoder=encoder_name, expansion=meta["expansion"],
        n_tokens=len(embeddings), n_subjects=n_subj,
    )


def _run_layer_sweep(args) -> None:
    """Collect activations once at layer L, build taxonomy_cache for each E."""
    Es = [float(x.strip()) for x in args.expansions.split(",")]
    print(f"=== Layer-sweep mode  encoder={args.encoder}  layer={args.layer}  "
          f"expansions={Es}  k={args.k} ===")

    enc = _load_encoder(args.encoder, args.weights_path)
    data_path = _ENCODER_DATA[args.encoder]
    transform = V4ResampleTransform() if "D4-v4" in str(data_path) else StandardizeLabel()
    full_ds = H5PYDatasetLabeled(str(data_path), transform=transform)
    print(f"  Dataset: {len(full_ds)} windows, "
          f"{len(np.unique(full_ds.subjects))} subjects")

    enr_max = args.enr_max_tokens or 2_000_000
    print(f"\n[Collecting up to {enr_max:,} tokens at layer {args.layer}, "
          f"batch={args.batch_size}, workers={args.num_workers}]")
    t0 = time.time()
    embeddings, token_meta = _collect_activations(
        enc, full_ds, args.layer,
        max_tokens=enr_max, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"  Collected {len(embeddings):,} tokens in {(time.time()-t0)/60:.1f} min")
    n_subj = len(np.unique(token_meta["subject_id"]))

    run_name = f"{args.encoder}_{args.tag}" if args.tag else args.encoder
    sae_dir  = ROOT / "results" / "features" / run_name

    for E in Es:
        sae_path = sae_dir / f"sae_{args.encoder}_exp{E}_k{args.k}_layer{args.layer}.pt"
        if not sae_path.exists():
            print(f"\n[skip] SAE checkpoint missing: {sae_path.name}  "
                  f"(train it with tools/train_sae_expansions.py)")
            continue

        # Resolve experiment dir from naming convention.
        # LaBraM uses {run_name}_layer{L}_exp{E}; everything else uses
        # {run_name}_exp{E}_layer{L}.
        if args.encoder == "labram":
            exp_name = (
                f"{run_name}_layer{args.layer}" if E == 1.0
                else f"{run_name}_layer{args.layer}_exp{int(E)}"
            )
        else:
            exp_name = (
                f"{run_name}_layer{args.layer}" if E == 1.0
                else f"{run_name}_exp{int(E)}_layer{args.layer}"
            )
        exp_dir  = ROOT / "results" / "experiments" / exp_name
        if not exp_dir.exists():
            print(f"  [skip] experiment dir missing: {exp_dir.name}  "
                  f"(create metadata.json first)")
            continue

        print(f"\n--- E={E} ({sae_path.name}) ---")
        sae_ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
        sae = SparseAutoencoder(enc.embed_dim, expansion=E, mode="topk", k=args.k)
        sae.load_state_dict(sae_ckpt["sae_state_dict"])
        act_mean = sae_ckpt["act_mean"].float()
        act_std  = sae_ckpt["act_std"].float()
        n_features = sae.encoder.weight.shape[0]

        t0 = time.time()
        fire_np = _sae_firing_pattern(embeddings, sae, act_mean, act_std)
        print(f"  SAE firing pattern: {(time.time()-t0)/60:.1f} min  "
              f"({fire_np.nbytes / 1e6:.0f} MB)")

        t0 = time.time()
        fme, fs = _enrichment(fire_np, token_meta)
        print(f"  Enrichment: {(time.time()-t0)/60:.1f} min")

        out = exp_dir / "taxonomy_cache.pt"
        _save_cache(
            out,
            feature_meta_enrichment=fme, feature_stats=fs,
            n_features=n_features, k=args.k, target_layer=args.layer,
            encoder=args.encoder, expansion=E,
            n_tokens=len(embeddings), n_subjects=n_subj,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default=None,
                   help="Single-experiment mode: name of a results/experiments/<name> dir")
    p.add_argument("--encoder", default=None, help="Layer-sweep mode: encoder name")
    p.add_argument("--tag",     default=None, help="Layer-sweep mode: tag (e.g. qjbe08)")
    p.add_argument("--layer",   type=int, default=None,
                   help="Layer-sweep mode: target layer index")
    p.add_argument("--expansions", default=None,
                   help="Layer-sweep mode: comma-separated E values, e.g. 1,4,16")
    p.add_argument("--weights-path", default=None,
                   help="Optional encoder weights override")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--enr-max-tokens", type=int, default=2_000_000)
    p.add_argument("--batch-size", type=int, default=128,
                   help="Encoder dataloader batch size (L4 fits 128 comfortably)")
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    if args.experiment is not None:
        _run_single_experiment(args)
    elif args.encoder is not None and args.layer is not None and args.expansions is not None:
        _run_layer_sweep(args)
    else:
        p.error("Provide either --experiment, OR (--encoder + --layer + --expansions)")


if __name__ == "__main__":
    main()
