"""
build_layer_umap_cache.py — Build joint UMAP cache for Layer Explorer tab
=========================================================================

Collects token activations at ALL transformer layers in a single forward pass,
along with labels and subject IDs.  Runs a JOINT UMAP on all layers concatenated
(N*L points, same N tokens at every layer) so the coordinate space is consistent
and tokens are traceable as they move through the network.

Cache is saved to:
  results/layer_umap/{encoder}/umap_cache.pt

Usage:
  uv run tools/build_layer_umap_cache.py --encoder sleepfm_v2.1
  uv run tools/build_layer_umap_cache.py --encoder sleepfm --max-tokens 5000
  uv run tools/build_layer_umap_cache.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import DataLoader

from sae4eeg.dataset import H5PYDatasetLabeled, StandardizeLabel, V4ResampleTransform, get_dataloaders
from sae4eeg.encoders import load_encoder, MODEL_CARDS
from sae4eeg.sae import ActivationExtractor

METADATA_FIELDS = [
    "age_group", "gender", "classification",
    "indication_group", "medication_group", "recording_date", "clinic",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_V2_DIR = ROOT / "checkpoints" / "pretrained" / "SleepFM v2 Models"

ENCODER_CONFIGS = {
    "sleepfm_v2.0": dict(
        load_as="sleepfm_v2.0",
        weights_path=_V2_DIR / "settransformer_exp0_cl_cnn_sgd_fp32_128d_640p_lr0.001_20260307_113442" / "best.pt",
        data_path=ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path=ROOT / "results" / "xae" / "sleepfm_v2.0" / "codebook" / "codebook.pt",
    ),
    "sleepfm": dict(
        load_as="sleepfm",
        weights_path=ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt",
        data_path=ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path=ROOT / "results" / "xae" / "sleepfm" / "codebook" / "codebook.pt",
    ),
    "sleepfm_v2.1": dict(
        load_as="sleepfm_v2.1",
        weights_path=_V2_DIR / "settransformer_exp1_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_114250" / "best.pt",
        data_path=ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path=ROOT / "results" / "xae" / "sleepfm_v2.1" / "codebook" / "codebook.pt",
    ),
    "sleepfm_v2.3": dict(
        load_as="sleepfm_v2.3",
        weights_path=_V2_DIR / "settransformer_exp2_cl_mae_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_113651" / "best.pt",
        data_path=ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path=ROOT / "results" / "xae" / "sleepfm_v2.3" / "codebook" / "codebook.pt",
    ),
    "sleepfm_v2.4": dict(
        load_as="sleepfm_v2.4",
        weights_path=_V2_DIR / "settransformer_exp4_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260307_210846" / "best.pt",
        data_path=ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path=ROOT / "results" / "xae" / "sleepfm_v2.4" / "codebook" / "codebook.pt",
    ),
    "sleepfm_v2.5": dict(
        load_as="sleepfm_v2.5",
        weights_path=_V2_DIR / "settransformer_exp5_cl_cnn_adamw_bf16_128d_640p_lr0.0003_20260308_111156" / "best.pt",
        data_path=ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path=ROOT / "results" / "xae" / "sleepfm_v2.5" / "codebook" / "codebook.pt",
    ),
    # sleepfm_finetuned / sleepfm_pretrained share the SleepFM v1.1 architecture
    # but use different checkpoint weights; load as "sleepfm" to reuse the backend.
    "sleepfm_finetuned": dict(
        load_as="sleepfm",
        weights_path=ROOT / "checkpoints" / "finetuned" / "sleepfm1.ckpt",
        data_path=ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path=ROOT / "results" / "xae" / "sleepfm_finetuned" / "codebook" / "codebook.pt",
    ),
    "sleepfm_pretrained": dict(
        load_as="sleepfm",
        weights_path=ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt",
        data_path=ROOT / "data" / "D4-v3-preprocessed-v2",
        codebook_path=ROOT / "results" / "xae" / "sleepfm_pretrained" / "codebook" / "codebook.pt",
    ),
    "sleepfm_granular": dict(
        load_as="sleepfm_granular",
        weights_path=ROOT / "checkpoints" / "granular" / "sleepfm_granular.ckpt",
        data_path=ROOT / "data" / "D4-v4-preprocessed-10s",
        codebook_path=ROOT / "results" / "xae" / "sleepfm_granular" / "codebook" / "codebook.pt",
    ),
    "reve_qjbe08": dict(
        load_as="reve",
        weights_path=ROOT / "checkpoints" / "finetuned" / "reve_qjbe08.ckpt",
        data_path=ROOT / "data" / "D4-v3-preprocessed-v1",
        codebook_path=ROOT / "results" / "xae" / "reve_qjbe08" / "codebook" / "codebook.pt",
    ),
    "labram": dict(
        load_as="labram",
        weights_path=ROOT / "checkpoints" / "finetuned" / "labram_binary" / "finetuned.ckpt",
        data_path=ROOT / "data" / "D4-v3-preprocessed-v1",
        codebook_path=ROOT / "results" / "xae" / "labram" / "codebook" / "codebook.pt",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build joint UMAP cache for Layer Explorer")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--encoder",
        choices=list(ENCODER_CONFIGS.keys()),
        help="Single encoder to process",
    )
    group.add_argument("--all", action="store_true", help="Process all configured encoders")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=20_000,
        help="Hard cap on total tokens collected (default: 20000)",
    )
    p.add_argument(
        "--max-per-subject",
        type=int,
        default=1,
        help=(
            "Max windows to take from any single subject (default: 1). "
            "Set 0 to disable and collect sequentially."
        ),
    )
    p.add_argument(
        "--tokens-per-window",
        type=int,
        default=1,
        help=(
            "How many tokens to sample from each selected window (default: 1). "
            "Tokens are drawn evenly-spaced across the window."
        ),
    )
    p.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer subset, e.g. 0,2,4 (default: all layers)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Activation + label/subject collection
# ─────────────────────────────────────────────────────────────────────────────

def _collect_with_metadata(
    encoder,
    val_loader,
    all_layer_ids: List[int],
    max_tokens: int,
    max_windows_per_subject: Optional[int] = None,
    tokens_per_window: int = 1,
    dataset=None,
) -> Tuple[Dict[int, torch.Tensor], np.ndarray, np.ndarray, int, List[int]]:
    """
    Single-pass collection of activations, labels, and subject IDs.

    When max_windows_per_subject is set, windows are selected window-by-window
    and skipped once a subject has contributed that many windows.  Within each
    selected window, tokens_per_window evenly-spaced tokens are sampled rather
    than taking all S tokens, keeping total token count manageable.

    Returns
    -------
    layer_acts : dict[int -> Tensor (N, E)]
    labels_arr : np.ndarray (N,) int  — 0=Normal, 1=Abnormal, -1=unknown
    subjects_arr : np.ndarray (N,) int — subject IDs (-1 if unavailable)
    tokens_per_window : int — number of tokens per HDF5 window (for metadata expansion)
    collected_window_indices : list[int] — dataset indices of included windows (for metadata lookup)
    """
    from sae4eeg.encoders import EncoderBackend as _EB

    if isinstance(encoder, _EB):
        model = encoder.model
        all_layers = encoder.get_hookable_layers()
        target_sorted = sorted(all_layer_ids)
        hook_layers = [all_layers[i] for i in target_sorted if i < len(all_layers)]
        idx_map: Dict[int, int] = {
            ei: li for ei, li in enumerate(target_sorted) if li < len(all_layers)
        }
        call_fn = lambda x: encoder.encode(x)  # noqa: E731
    else:
        model = encoder
        hook_layers = None
        idx_map = {}
        call_fn = lambda x: model(x)  # noqa: E731

    model.eval()
    if isinstance(encoder, _EB):
        encoder.to(DEVICE)
    else:
        model.to(DEVICE)

    extractor = ActivationExtractor(model, layers=hook_layers)

    # Capture pre-transformer tokens (CNN tokenizer output) via forward_pre_hook
    # on the first transformer layer.  Stored as layer index -1.
    pre_transformer_buffer: List[torch.Tensor] = []

    def _pre_hook(module, args):  # noqa: ANN001
        act = args[0].detach().cpu()  # (B, S, E)
        pre_transformer_buffer.append(act)

    first_transformer_layer = all_layers[0] if isinstance(encoder, _EB) else None
    pre_hook_handle = (
        first_transformer_layer.register_forward_pre_hook(_pre_hook)
        if first_transformer_layer is not None
        else None
    )

    # Pre-extract subject strings from the dataset's index_map for reliable per-subject
    # quota tracking. batch[2] is channel names, NOT subject IDs.
    _ds = dataset if dataset is not None else getattr(val_loader, "dataset", None)
    if _ds is not None and hasattr(_ds, "index_map") and "subjects" in _ds.index_map:
        _all_subj = _ds.index_map["subjects"]
        if torch.is_tensor(_all_subj):
            _all_subj = _all_subj.numpy()
        _all_subj_strs: Optional[np.ndarray] = _all_subj.astype(str)
    else:
        _all_subj_strs = None

    buffers: Dict[int, List[torch.Tensor]] = {}
    labels_list: List[torch.Tensor] = []
    subjects_raw: List[str] = []   # accumulated as strings, converted at the end
    collected_window_indices: List[int] = []  # dataset index of each included window
    tokens_collected = 0
    full_tokens_per_window: Optional[int] = None  # S: inferred from first batch
    tok_offsets: np.ndarray = np.array([0])        # within-window token positions (set on first batch)
    subject_window_counts: Dict[str, int] = {}  # for max_windows_per_subject enforcement
    dataset_window_idx = 0  # running index into dataset (batch_size * batch_idx + wi)

    n_batches = len(val_loader)
    for batch_idx, batch in enumerate(val_loader):
        # Batch format: (x, labels, subjects, file_idxs, time_slices)
        # or shorter tuples depending on dataset
        x = batch[0].to(DEVICE) if isinstance(batch, (list, tuple)) else batch.to(DEVICE)
        B = x.shape[0]

        # Extract optional metadata
        raw_labels = batch[1] if (isinstance(batch, (list, tuple)) and len(batch) > 1) else None
        raw_subjects = batch[2] if (isinstance(batch, (list, tuple)) and len(batch) > 2) else None

        extractor.clear()
        with torch.no_grad():
            _ = call_fn(x)

        acts_this_batch = extractor.get_activations()

        # Infer full tokens per window (S) from the first batch
        if full_tokens_per_window is None and acts_this_batch:
            first_act = next(iter(acts_this_batch.values()))
            _, S_full, _ = first_act.shape
            full_tokens_per_window = S_full

        S = full_tokens_per_window if full_tokens_per_window is not None else 1

        # ── Build row-index mask (per-window, then flattened to per-token) ─────
        # Use dataset.index_map subjects (not batch[2], which is channel names).
        if _all_subj_strs is not None:
            end = min(dataset_window_idx + B, len(_all_subj_strs))
            subj_list = list(_all_subj_strs[dataset_window_idx:end])
            while len(subj_list) < B:
                subj_list.append("")
        elif raw_subjects is not None:
            subj_list = [str(s) for s in list(raw_subjects)[:B]]
        else:
            subj_list = [""] * B

        if max_windows_per_subject is not None:
            included: List[int] = []
            for wi, sid in enumerate(subj_list[:B]):
                if subject_window_counts.get(sid, 0) < max_windows_per_subject:
                    included.append(wi)
                    subject_window_counts[sid] = subject_window_counts.get(sid, 0) + 1
            if not included:
                dataset_window_idx += B
                pre_transformer_buffer.clear()
                continue
        else:
            included = list(range(B))

            # Within each included window, sample tokens_per_window evenly-spaced tokens
        tok_offsets = np.round(np.linspace(0, S - 1, min(tokens_per_window, S))).astype(int)
        row_indices = np.concatenate([wi * S + tok_offsets for wi in included])
        collected_window_indices.extend(dataset_window_idx + wi for wi in included)

        # Respect global token budget
        remaining = max_tokens - tokens_collected
        row_indices = row_indices[:remaining]
        n_take = len(row_indices)
        if n_take == 0:
            pre_transformer_buffer.clear()
            break

        # ── Labels ────────────────────────────────────────────────────────────
        if raw_labels is not None:
            if not isinstance(raw_labels, torch.Tensor):
                raw_labels = torch.tensor(raw_labels)
            lbl_int = raw_labels.long().cpu()
            lbl_expanded = lbl_int.unsqueeze(1).expand(B, S).reshape(B * S)
            labels_list.append(lbl_expanded[row_indices])
        else:
            labels_list.append(torch.full((n_take,), -1, dtype=torch.long))

        # ── Subjects ──────────────────────────────────────────────────────────
        # Use dataset index_map subjects (reliable), not batch[2] (channel names).
        if _all_subj_strs is not None:
            end = min(dataset_window_idx + B, len(_all_subj_strs))
            batch_subj = list(_all_subj_strs[dataset_window_idx:end])
            while len(batch_subj) < B:
                batch_subj.append("")
            subj_expanded = []
            for sid in batch_subj:
                subj_expanded.extend([str(sid)] * S)
            for ri in row_indices:
                subjects_raw.append(subj_expanded[ri])
        else:
            subjects_raw.extend([""] * n_take)

        # ── Activations ───────────────────────────────────────────────────────
        for enum_idx, act in acts_this_batch.items():
            layer_idx = idx_map[enum_idx] if idx_map else enum_idx
            if all_layer_ids and layer_idx not in all_layer_ids:
                continue
            flat = act.reshape(B * S, -1)[row_indices].cpu()
            buffers.setdefault(layer_idx, []).append(flat)

        # Pre-transformer (layer -1): one entry per batch added by the pre_hook
        if pre_transformer_buffer:
            pre_act = pre_transformer_buffer.pop()  # (B, S, E)
            flat_pre = pre_act.reshape(B * S, -1)[row_indices]
            buffers.setdefault(-1, []).append(flat_pre)
            pre_transformer_buffer.clear()

        tokens_collected += n_take
        dataset_window_idx += B
        n_subjects_seen = len(subject_window_counts) if max_windows_per_subject else "?"
        print(
            f"    batch {batch_idx + 1}/{n_batches}  "
            f"({tokens_collected}/{max_tokens} tokens"
            + (f", {n_subjects_seen} subjects" if max_windows_per_subject else "")
            + ")",
            flush=True,
        )

        if tokens_collected >= max_tokens:
            print(f"    reached max_tokens={max_tokens}, stopping early", flush=True)
            break

    extractor.remove_hooks()
    if pre_hook_handle is not None:
        pre_hook_handle.remove()

    layer_acts = {k: torch.cat(v, dim=0) for k, v in buffers.items()}
    labels_arr = torch.cat(labels_list, dim=0).numpy().astype(np.int32) if labels_list else np.full(tokens_collected, -1, dtype=np.int32)

    # Convert string subject IDs to integer indices
    unique_subjects = sorted(set(subjects_raw))
    subj_to_idx = {s: i for i, s in enumerate(unique_subjects)}
    subjects_arr = np.array([subj_to_idx.get(s, -1) for s in subjects_raw[:tokens_collected]], dtype=np.int32)

    return layer_acts, labels_arr, subjects_arr, full_tokens_per_window or 1, collected_window_indices, tok_offsets


# ─────────────────────────────────────────────────────────────────────────────
# Codebook band assignment (same as layer_umap.py)
# ─────────────────────────────────────────────────────────────────────────────

def _assign_codebook_labels(acts: torch.Tensor, codebook: dict) -> np.ndarray:
    """Assign each token to its nearest codebook centroid (cosine similarity)."""
    centroids = codebook["centroids_emb"]
    if torch.is_tensor(centroids):
        centroids = centroids.float().numpy()
    acts_np = acts.float().numpy()
    acts_norm = acts_np / (np.linalg.norm(acts_np, axis=1, keepdims=True) + 1e-8)
    cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    sim = acts_norm @ cent_norm.T  # (N, K)
    return sim.argmax(axis=1)  # (N,)


def _band_labels_from_codebook(codebook: dict, cluster_ids: np.ndarray) -> np.ndarray:
    """Map cluster IDs → dominant band name string per token."""
    band_per_cluster = codebook["cluster_band_label"]  # list[str], length K
    return np.array([band_per_cluster[int(c)] for c in cluster_ids], dtype=object)


# ─────────────────────────────────────────────────────────────────────────────
# Joint UMAP
# ─────────────────────────────────────────────────────────────────────────────

def _run_joint_umap(
    layer_acts: Dict[int, torch.Tensor],
    sorted_layers: List[int],
) -> Dict[int, np.ndarray]:
    """
    Stack all layers' activations, run UMAP once, then split back.

    The joint embedding means the same N tokens appear L times in the UMAP
    fit, ensuring a consistent coordinate space across layers.

    Returns
    -------
    dict[layer_idx -> (N, 2) float32]
    """
    import umap as umap_module

    acts_list = [layer_acts[L].float().numpy() for L in sorted_layers]
    N = acts_list[0].shape[0]

    # Verify all layers have the same N
    for i, a in enumerate(acts_list):
        if a.shape[0] != N:
            print(f"  Warning: layer {sorted_layers[i]} has {a.shape[0]} tokens, expected {N}. Truncating.")
            acts_list[i] = a[:N]

    all_acts = np.vstack(acts_list)  # (N*L, E)
    print(f"  Running UMAP on {all_acts.shape[0]:,} points ({N} tokens × {len(sorted_layers)} layers)…", flush=True)

    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.3,
        random_state=42,
        metric="euclidean",
        low_memory=True,
    )
    all_xy = reducer.fit_transform(all_acts).astype(np.float32)  # (N*L, 2)

    xy_per_layer: Dict[int, np.ndarray] = {}
    for i, L in enumerate(sorted_layers):
        xy_per_layer[L] = all_xy[i * N : (i + 1) * N]

    return xy_per_layer


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 metadata collection (mirrors build_app_cache._collect_token_metadata)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_token_metadata(val_dataset, fields, n_tokens, tokens_per_window, collected_window_indices: Optional[List[int]] = None, tokens_per_window_subsample: int = 1):
    """Load per-window HDF5 metadata and expand to per-token arrays.

    collected_window_indices: dataset indices of the windows that were actually
    collected (in collection order). If None, falls back to first n_windows.
    """
    import h5py as _h5

    if collected_window_indices is not None:
        n_windows = len(collected_window_indices)
        idx_tensor = torch.tensor(collected_window_indices, dtype=torch.long)
        file_indices  = val_dataset.index_map["file_indices"][idx_tensor].numpy()
        local_indices = val_dataset.index_map["local_indices"][idx_tensor].numpy()
    else:
        n_windows = min(
            (n_tokens + tokens_per_window - 1) // tokens_per_window,
            len(val_dataset),
        )
        file_indices  = val_dataset.index_map["file_indices"][:n_windows].numpy()
        local_indices = val_dataset.index_map["local_indices"][:n_windows].numpy()

    window_meta = {f: np.full(n_windows, "", dtype=object) for f in fields}
    for file_val in np.unique(file_indices):
        positions    = np.where(file_indices == file_val)[0]
        sort_args    = np.argsort(local_indices[positions])
        sorted_pos   = positions[sort_args]
        sorted_local = local_indices[sorted_pos].tolist()
        with _h5.File(val_dataset.paths[int(file_val)], "r") as f:
            for field in fields:
                try:
                    raw = f["metadata"][field][sorted_local]
                    for pos, v in zip(sorted_pos, raw):
                        if isinstance(v, (bytes, np.bytes_)):
                            v = v.decode("utf-8", errors="replace")
                        window_meta[field][pos] = str(v).strip()
                except Exception:
                    pass

    # Expand per-window metadata to per-token using the subsampled token count
    expand_by = tokens_per_window_subsample if tokens_per_window_subsample > 0 else tokens_per_window
    token_meta: dict[str, np.ndarray] = {}
    for field in fields:
        arr = window_meta[field]
        token_meta[field] = np.repeat(arr, expand_by)[:n_tokens]

    if collected_window_indices is not None:
        subj = val_dataset.index_map["subjects"][idx_tensor]
    else:
        subj = val_dataset.index_map["subjects"][:n_windows]
    if torch.is_tensor(subj):
        subj = subj.numpy()
    token_meta["subject_id"] = np.repeat(subj.astype(str), expand_by)[:n_tokens]

    return token_meta


# ─────────────────────────────────────────────────────────────────────────────
# Main per-encoder processing
# ─────────────────────────────────────────────────────────────────────────────

def process_encoder(encoder_key: str, max_tokens: int, max_windows_per_subject: Optional[int] = None, tokens_per_window: int = 1, layer_subset: Optional[List[int]] = None) -> None:
    cfg = ENCODER_CONFIGS[encoder_key]

    # Resolve display_name — fall back gracefully if not in MODEL_CARDS
    display_name = MODEL_CARDS.get(encoder_key, {}).get("display_name", encoder_key)

    codebook_path = cfg["codebook_path"]

    print(f"\n{'=' * 60}")
    print(f"  {display_name}  ({encoder_key})")
    print(f"{'=' * 60}")

    if not codebook_path.exists():
        print(f"  SKIP — codebook not found: {codebook_path}")
        print(f"  Run: uv run tools/build_codebook.py --encoder {encoder_key}")
        return

    out_dir = ROOT / "results" / "layer_umap" / encoder_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "umap_cache.pt"

    # ── Load encoder ──────────────────────────────────────────────────────────
    # Use `load_as` to handle aliases like sleepfm_finetuned → "sleepfm" backend.
    print("  Loading encoder…", flush=True)
    load_as = cfg.get("load_as", encoder_key)
    encoder = load_encoder(load_as, weights_path=str(cfg["weights_path"]))
    encoder.to(DEVICE).eval()

    n_layers = len(encoder.get_hookable_layers())
    all_layer_ids = list(range(n_layers)) if layer_subset is None else layer_subset
    all_layer_ids = [L for L in all_layer_ids if L < n_layers]
    print(f"  {n_layers} total layers, processing: {all_layer_ids} + layer -1 (tokenizer)")

    # ── Load codebook ─────────────────────────────────────────────────────────
    print("  Loading codebook…", flush=True)
    codebook = torch.load(codebook_path, map_location="cpu", weights_only=False)
    print(f"  Codebook: {codebook['n_clusters']} clusters")

    # ── Load full dataset (all subjects, no split — visualization only) ────────
    print("  Loading full dataset…", flush=True)
    _transform = V4ResampleTransform() if "D4-v4" in str(cfg["data_path"]) else StandardizeLabel()
    full_ds = H5PYDatasetLabeled(str(cfg["data_path"]), transform=_transform)
    # shuffle=False so that _collect_token_metadata can read index_map[:n_windows] in order
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=False, num_workers=4)
    print(f"  Dataset: {len(full_ds):,} windows, {len(set(full_ds.subjects.tolist())):,} subjects")

    # ── Collect activations + metadata ───────────────────────────────────────
    _mwps = max_windows_per_subject if max_windows_per_subject else None
    print(
        f"  Collecting activations (max {max_tokens:,} tokens"
        + (f", max {_mwps} window(s)/subject, {tokens_per_window} token(s)/window" if _mwps else "")
        + ")…",
        flush=True,
    )
    layer_acts, labels_arr, subjects_arr, full_S, collected_window_indices, tok_offsets = _collect_with_metadata(
        encoder, full_loader, all_layer_ids, max_tokens,
        max_windows_per_subject=_mwps, tokens_per_window=tokens_per_window, dataset=full_ds
    )
    N_collected = len(labels_arr)

    print("  Collecting HDF5 metadata…", flush=True)
    try:
        token_meta = _collect_token_metadata(
            full_loader.dataset, METADATA_FIELDS, N_collected, tokens_per_window,
            collected_window_indices=collected_window_indices,
            tokens_per_window_subsample=tokens_per_window)
    except Exception as e:
        print(f"  [warn] metadata collection failed: {e}")
        token_meta = {}

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not layer_acts:
        print("  No activations collected. Aborting.")
        return

    sorted_layers = sorted(layer_acts.keys())
    N = layer_acts[sorted_layers[0]].shape[0]
    print(f"  Collected {N:,} tokens at {len(sorted_layers)} layers.")

    # Ensure labels/subjects are the same length as tokens
    if len(labels_arr) != N:
        print(f"  Warning: label array length {len(labels_arr)} != {N} tokens. Padding with -1.")
        labels_padded = np.full(N, -1, dtype=np.int32)
        labels_padded[: len(labels_arr)] = labels_arr
        labels_arr = labels_padded

    # Prefer subject IDs from token_meta (from HDF5 index_map) over batch-derived ones,
    # since the dataset's __getitem__ returns channel names in batch[2], not subject IDs.
    if "subject_id" in token_meta and len(token_meta["subject_id"]) == N:
        raw_subj_strs = token_meta["subject_id"]
        unique_subj = sorted(set(raw_subj_strs))
        subj_to_idx = {s: i for i, s in enumerate(unique_subj)}
        subjects_arr = np.array([subj_to_idx.get(s, -1) for s in raw_subj_strs], dtype=np.int32)
    elif len(subjects_arr) != N:
        subjects_padded = np.full(N, -1, dtype=np.int32)
        subjects_padded[: len(subjects_arr)] = subjects_arr
        subjects_arr = subjects_padded

    # ── Band labels from codebook (use last transformer layer's activations) ────
    # Codebook was built on encoder output, so we use the final positive layer.
    # Layer -1 (tokenizer) is not in the same space as the codebook centroids.
    print("  Assigning codebook band labels…", flush=True)
    positive_layers = [L for L in sorted_layers if L >= 0]
    ref_layer = positive_layers[-1] if positive_layers else sorted_layers[-1]
    ref_acts = layer_acts[ref_layer]  # Use last transformer layer for band assignment
    cluster_ids = _assign_codebook_labels(ref_acts, codebook)
    band_arr = _band_labels_from_codebook(codebook, cluster_ids)  # (N,) dtype=object

    # ── File/local indices for EEG retrieval ──────────────────────────────────
    _idx_t = torch.tensor(collected_window_indices, dtype=torch.long)
    _fi_win = full_ds.index_map["file_indices"][_idx_t].numpy()   # (n_windows,)
    _li_win = full_ds.index_map["local_indices"][_idx_t].numpy()  # (n_windows,)
    _tpw = len(tok_offsets)
    file_indices_arr  = np.repeat(_fi_win,  _tpw)[:N]  # (N,) — HDF5 file index
    local_indices_arr = np.repeat(_li_win,  _tpw)[:N]  # (N,) — local row in file
    token_positions_arr = np.tile(tok_offsets, len(collected_window_indices))[:N]  # (N,) — token pos in window

    # ── Joint UMAP ────────────────────────────────────────────────────────────
    xy_per_layer = _run_joint_umap(layer_acts, sorted_layers)
    print("  UMAP complete.")

    # ── KMeans on tokenizer activations (layer -1, K=4..12) ──────────────────
    kmeans_cache: dict = {}
    if -1 in layer_acts:
        from sklearn.cluster import MiniBatchKMeans
        tok_acts_np = layer_acts[-1].float().numpy()
        print("  Running KMeans (K=4..12) on tokenizer activations…", flush=True)
        for k in range(4, 13):
            km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=1024)
            km.fit(tok_acts_np)
            kmeans_cache[k] = {
                "labels": km.labels_.astype(np.int32),
                "centroids": km.cluster_centers_.astype(np.float32),
                "inertia": float(km.inertia_),
            }
        print(f"  KMeans done.")

    # ── Summary statistics ────────────────────────────────────────────────────
    print(f"\n  Band distribution ({N} tokens):")
    unique_bands, counts = np.unique(band_arr, return_counts=True)
    for cnt, band_name in sorted(zip(counts, unique_bands), reverse=True):
        print(f"    {cnt:>6,}  {band_name}  ({100 * cnt / N:.1f}%)")

    label_names = {0: "Normal", 1: "Abnormal", -1: "Unknown"}
    print(f"\n  Label distribution:")
    for lv in sorted(set(labels_arr.tolist())):
        cnt = int((labels_arr == lv).sum())
        print(f"    {cnt:>6,}  {label_names.get(lv, str(lv))}  ({100 * cnt / N:.1f}%)")

    n_subjects = int((subjects_arr >= 0).sum())
    n_unique_subjects = len(set(subjects_arr[subjects_arr >= 0].tolist()))
    print(f"\n  Subject IDs: {n_unique_subjects} unique subjects ({n_subjects:,} tokens with valid ID)")

    # ── Save cache ────────────────────────────────────────────────────────────
    cache = {
        "encoder": encoder_key,
        "display_name": display_name,
        "layers": sorted_layers,
        "n_tokens": N,
        "xy": xy_per_layer,              # dict[int -> (N, 2) float32]
        "band": band_arr,                # (N,) dtype=object
        "label": labels_arr,             # (N,) int32
        "subject": subjects_arr,         # (N,) int32
        "token_meta": token_meta,        # dict[field -> (N,) dtype=object]
        "file_indices": file_indices_arr,     # (N,) int — HDF5 file index per token
        "local_indices": local_indices_arr,   # (N,) int — local row within HDF5 file
        "token_positions": token_positions_arr,  # (N,) int — token offset (in tokens) within window
        "data_path": str(cfg["data_path"]),  # for HDF5 path resolution in app
        "kmeans": kmeans_cache,          # dict[K -> {labels, centroids, inertia}]
    }
    torch.save(cache, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n  Saved cache: {out_path}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    layer_subset: Optional[List[int]] = None
    if args.layers:
        layer_subset = sorted(int(x) for x in args.layers.split(","))

    mwps = args.max_per_subject if args.max_per_subject > 0 else None
    tpw = args.tokens_per_window
    if args.all:
        for enc_key in ENCODER_CONFIGS:
            process_encoder(enc_key, args.max_tokens, mwps, tpw, layer_subset)
    else:
        process_encoder(args.encoder, args.max_tokens, mwps, tpw, layer_subset)


if __name__ == "__main__":
    main()
