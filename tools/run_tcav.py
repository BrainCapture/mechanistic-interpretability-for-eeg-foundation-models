"""Run TCAV (Testing with Concept Activation Vectors) for a named experiment.

Three TCAV variants are computed (see src/sae4eeg/tcav.py for full details):

  A. Weight-space alignment score  — NOT proper Kim et al.
     Fraction of K fold-CAVs where W_enc[i]·v_C > 0.  Purely weight-space,
     no examples used.  Discrete {0, 0.1, …, 1.0} for K=10 folds.
     Stored as: tcav_scores  (C, n_features)

  B. Per-feature Kim et al. TCAV  — proper but collapses for TopK SAE
     firing_rate_i × I(W_enc[i]·v_C > 0).  Example-dependent via firing
     rate; sign gate reduces to binary 0 / firing_rate.
     Derived from: firing_rates + alignments (no separate storage needed)

  C. Model-level Kim et al. TCAV  — full Kim et al. formulation
     Downstream linear probe h trained on SAE z-features.  Per-example:
       S_k(x) = Σ_i [ w_probe_i × (W_enc[i]·v_C) × I(z_i(x)>0) ]
     TCAV_k = fraction of concept examples where S_k(x) > 0, 0.5 = chance.
     Stored as: model_tcav_scores (C,), model_tcav_p_values (C,),
                probe_accuracies (C,)

Prerequisites
-------------
  - Trained SAE checkpoint (explore_features.py or train_sae_layers.py)
  - Pre-built app_cache.pt (build_app_cache.py)

Usage
-----
    uv run tools/run_tcav.py --experiment reve_qjbe08_layer21
    uv run tools/run_tcav.py --experiment reve_qjbe08_layer21 \\
        --max-tokens 50000 --neg-per-concept 2000

Output
------
    results/tcav/<experiment>/tcav_cache.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT   = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_ENCODER_DATA = {
    "sleepfm":            "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.0":       "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.1":       "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.3":       "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.4":       "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.5":       "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.6":       "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.7":       "data/D4-v3-preprocessed-v2",
    "sleepfm_finetuned":  "data/D4-v3-preprocessed-v2",
    "sleepfm_pretrained": "data/D4-v3-preprocessed-v2",
    "sleepfm_granular":   "data/D4-v4-preprocessed-10s",
    "reve":               "data/D4-v3-preprocessed-v1",
    "labram":             "data/D4-v3-preprocessed-v1",
}


# ─────────────────────────────────────────────────────────────────────────────
# Activation collection (raw + labels in one pass)
# ─────────────────────────────────────────────────────────────────────────────

_META_CONCEPTS = [
    # (concept_name, field, pos_values, neg_values or None → all non-pos)
    ("child",    "age_group",        ["0-3", "4-9"],
                                     ["40-49", "50-59", "60+"]),
    ("asm",      "medication_group", ["ASM"],                None),
    ("seizure",  "indication_group",
                 ["Consciousness (loss of)", "Drop (attacks/spells)"],
                 ["Non-Symptom/Follow-up"]),
]

_META_FIELDS = sorted({f for _, f, *_ in _META_CONCEPTS})

_V4_LABEL_NAMES = {
    1: "diffuse_slowing",
    2: "focal_slowing",
    3: "focal_sharp_waves",
    4: "focal_spike_wave",
    5: "gen_spike_wave",
    6: "gen_polyspike_wave",
    7: "gen_sharp_waves",
    8: "burst_suppression",
    9: "seizure",
}


def _collect_token_metadata(
    val_dataset,
    fields: list[str],
    n_tokens: int,
    tokens_per_window: int,
) -> dict[str, np.ndarray]:
    """Read per-token metadata from HDF5 files aligned with val_loader iteration order."""
    import h5py as _h5

    n_windows = min(
        (n_tokens + tokens_per_window - 1) // tokens_per_window,
        len(val_dataset),
    )
    file_indices  = val_dataset.index_map["file_indices"][:n_windows].numpy()
    local_indices = val_dataset.index_map["local_indices"][:n_windows].numpy()
    window_meta   = {f: np.full(n_windows, "", dtype=object) for f in fields}

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
                except Exception as exc:
                    print(f"  [warn] metadata/{field}: {exc}")

    return {
        field: np.repeat(window_meta[field], tokens_per_window)[:n_tokens]
        for field in fields
    }


def _collect(
    encoder,
    val_loader,
    target_layer: int,
    max_tokens: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Collect raw (unnormalised) encoder activations and per-token labels.

    Returns
    -------
    acts              : (N, embed_dim) raw activations on CPU
    labels            : (N,) int labels or None if unavailable
    tokens_per_window : S — number of tokens per input window
    """
    from sae4eeg.encoders import EncoderBackend
    from sae4eeg.sae import ActivationExtractor

    is_backend = isinstance(encoder, EncoderBackend)
    act_list: list[torch.Tensor] = []
    lbl_list: list[torch.Tensor] = []
    total = 0
    tokens_per_window = 1

    for batch in tqdm(val_loader, desc="Collecting activations", leave=False):
        x = batch[0].to(device)
        y = batch[1] if len(batch) > 1 else None
        B = x.shape[0]

        with torch.no_grad():
            if is_backend:
                hookable = encoder.get_hookable_layers()
                captured: dict[int, torch.Tensor] = {}

                def _make_hook(idx: int):
                    def _h(module, inp, out):
                        captured[idx] = out
                    return _h

                handle = hookable[target_layer].register_forward_hook(
                    _make_hook(target_layer)
                )
                encoder.encode(x)
                handle.remove()
                layer_act = captured[target_layer]  # (B, S, E)
            else:
                extractor = ActivationExtractor(encoder)
                extractor.clear()
                encoder(x)
                layer_act = extractor.get_activations()[target_layer]
                extractor.remove_hooks()

        B2, S, E = layer_act.shape
        tokens_per_window = S
        flat = layer_act.reshape(B2 * S, E).cpu()
        act_list.append(flat)

        if y is not None:
            tok_y = y.unsqueeze(1).expand(-1, S).reshape(B2 * S)
            lbl_list.append(tok_y.cpu())

        total += B2 * S
        if total >= max_tokens:
            break

    acts   = torch.cat(act_list)[:max_tokens]
    labels = torch.cat(lbl_list)[:max_tokens] if lbl_list else None
    return acts, labels, tokens_per_window


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TCAV cache for a SAE4EEG experiment"
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Experiment name under results/experiments/  e.g. reve_qjbe08_layer21",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50_000,
        help="Max validation tokens to collect (default: 50000)",
    )
    parser.add_argument(
        "--neg-per-concept", type=int, default=2_000,
        help="Random negative tokens per concept for CAV training (default: 2000)",
    )
    parser.add_argument(
        "--n-splits", type=int, default=10,
        help="Cross-validation folds for CAV training (default: 10)",
    )
    parser.add_argument(
        "--n-random", type=int, default=50,
        help="Random CAVs for TCAV null distribution (default: 50)",
    )
    args = parser.parse_args()

    exp_dir    = ROOT / "results" / "experiments" / args.experiment
    meta_path  = exp_dir / "metadata.json"
    cache_path = exp_dir / "app_cache.pt"

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"app_cache.pt not found: {cache_path}\n"
            "Run build_app_cache.py for this experiment first."
        )

    meta         = json.loads(meta_path.read_text())
    encoder_name = meta.get("encoder", "sleepfm")
    target_layer = meta.get("target_layer", 2)
    expansion    = meta.get("expansion", 1.0)
    k_val        = meta.get("k", 8)

    print(f"[TCAV] {args.experiment}")
    print(f"       encoder={encoder_name}  layer={target_layer}")

    # ── 1. Load SAE ──────────────────────────────────────────────────────────
    from sae4eeg.sae import SparseAutoencoder

    sae_path = ROOT / meta["sae_checkpoint"]
    sae_ckpt = torch.load(str(sae_path), map_location="cpu", weights_only=False)
    embed_dim = sae_ckpt.get("embed_dim", meta.get("embed_dim", 128))
    expansion = sae_ckpt.get("expansion", expansion)
    k_val     = sae_ckpt.get("k", k_val)
    act_mean  = sae_ckpt["act_mean"]   # (E,) cpu
    act_std   = sae_ckpt["act_std"]    # (E,) cpu

    sae = SparseAutoencoder(embed_dim, expansion=expansion, mode="topk", k=k_val)
    sae.load_state_dict(sae_ckpt["sae_state_dict"])
    sae.eval()

    W_dec      = sae.decoder.weight.T.detach().numpy()   # (n_features, embed_dim)
    W_enc      = sae.encoder.weight.detach().numpy()     # (n_features, embed_dim)
    n_features = W_dec.shape[0]
    print(f"[TCAV] SAE loaded: {n_features} features  embed_dim={embed_dim}")

    # ── 2. Codebook from app_cache ───────────────────────────────────────────
    print("[TCAV] Loading app cache …")
    app_cache    = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    band_labels  = app_cache["codebook_band_labels"]      # list[str]  len K
    centroids_t  = app_cache["codebook_centroids_emb"]    # (K, E)
    K            = int(app_cache["codebook_n_clusters"])
    if torch.is_tensor(centroids_t):
        centroids_np = centroids_t.numpy().astype(np.float32)
    else:
        centroids_np = np.array(centroids_t, dtype=np.float32)
    print(f"[TCAV] Codebook: {K} clusters")

    # ── 3. Encoder + dataset ─────────────────────────────────────────────────
    from sae4eeg.dataset  import get_dataloaders, StandardizeLabel
    from sae4eeg.encoders import load_encoder

    weights_path = meta.get("weights_path") or meta.get("encoder_weights")
    if encoder_name == "sleepfm":
        ckpt    = (ROOT / weights_path) if weights_path else ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"
        encoder = load_encoder("sleepfm", weights_path=ckpt)
    else:
        kw      = {"weights_path": ROOT / weights_path} if weights_path else {}
        encoder = load_encoder(encoder_name, **kw)
    encoder = encoder.to(DEVICE).eval()

    data_path = meta.get("data_path", _ENCODER_DATA.get(encoder_name, "data/D4-v3-preprocessed-v2"))
    data_root = ROOT / data_path
    if "D4-v4" in str(data_path):
        from sae4eeg.dataset import V4ResampleTransform
        _transform = V4ResampleTransform()
    else:
        _transform = StandardizeLabel()
    gen = get_dataloaders(
        train_path=str(data_root),
        transformer=_transform,
        batch_size=8,
    )
    _, _, val_loader, _ = next(gen)

    # ── 4. Collect raw activations + labels ──────────────────────────────────
    print(f"[TCAV] Collecting ≤{args.max_tokens:,} validation tokens …")
    raw_acts, token_labels, tokens_per_window = _collect(
        encoder, val_loader, target_layer, args.max_tokens, DEVICE
    )
    N = raw_acts.shape[0]
    print(f"[TCAV] Collected: {raw_acts.shape}  labels={'yes' if token_labels is not None else 'no'}")

    # ── 4b. Collect per-token metadata from HDF5 ─────────────────────────────
    print(f"[TCAV] Reading token metadata (tokens_per_window={tokens_per_window}) …")
    token_meta = _collect_token_metadata(
        val_loader.dataset, _META_FIELDS, N, tokens_per_window
    )

    # ── 5. Normalise activations (same stats as SAE training) ────────────────
    norm_acts = (raw_acts - act_mean) / (act_std + 1e-8)   # (N, E)
    norm_np   = norm_acts.numpy()

    # Normalise codebook centroids the same way so nearest-centroid is valid
    centroids_norm = (centroids_np - act_mean.numpy()) / (act_std.numpy() + 1e-8)
    centroids_t_norm = torch.tensor(centroids_norm, dtype=torch.float32)

    # ── 6. Assign tokens to nearest codebook cluster → concept band label ────
    print("[TCAV] Assigning tokens to codebook clusters …")
    batch_sz = 8192
    assigns: list[torch.Tensor] = []
    for i in range(0, N, batch_sz):
        chunk = norm_acts[i : i + batch_sz].float()
        dists = torch.cdist(chunk, centroids_t_norm)   # (B, K)
        assigns.append(dists.argmin(dim=1))
    cluster_ids       = torch.cat(assigns).numpy()            # (N,)
    token_band_labels = np.array([band_labels[c] for c in cluster_ids])

    unique_bands = sorted(set(band_labels))
    counts = {b: int((token_band_labels == b).sum()) for b in unique_bands}
    print("[TCAV] Band token counts: " + "  ".join(f"{b}={n}" for b, n in counts.items()))

    # ── 7. Train CAVs ────────────────────────────────────────────────────────
    from sae4eeg.tcav import (
        cav_sae_alignment, compute_model_tcav_score, compute_tcav_scores,
        concept_firing_rates, firing_rate_enrichment, get_sae_features,
        train_cav, train_probe,
    )

    concept_names: list[str] = list(unique_bands)
    has_labels = (
        token_labels is not None
        and token_labels.unique().numel() >= 2
    )
    if has_labels:
        concept_names.append("abnormal")

    # Register metadata concepts that have enough tokens
    meta_concept_masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cname, field, pos_vals, neg_vals in _META_CONCEPTS:
        field_arr = token_meta.get(field, np.array([]))
        pos_mask  = np.isin(field_arr, pos_vals)
        if neg_vals is None:
            neg_mask = ~pos_mask
        else:
            neg_mask = np.isin(field_arr, neg_vals)
        if pos_mask.sum() >= 10 and neg_mask.sum() >= 10:
            concept_names.append(cname)
            meta_concept_masks[cname] = (pos_mask, neg_mask)
        else:
            print(f"  [meta/{cname}] skipped — pos={pos_mask.sum()}  neg={neg_mask.sum()}")

    # ASM controlled: ASM patients vs non-ASM *abnormal* patients.
    # Controls for pathology so the CAV captures medication effect rather than epilepsy.
    if has_labels and "asm" in meta_concept_masks:
        asm_pos_mask, _ = meta_concept_masks["asm"]
        lbl_np_bin = token_labels.numpy().astype(bool)  # type: ignore[union-attr]
        asm_ctrl_neg = (~asm_pos_mask) & lbl_np_bin
        if asm_pos_mask.sum() >= 10 and asm_ctrl_neg.sum() >= 10:
            concept_names.append("asm_controlled")
            meta_concept_masks["asm_controlled"] = (asm_pos_mask, asm_ctrl_neg)
            print(f"  [meta/asm_controlled] pos={asm_pos_mask.sum()}  neg={asm_ctrl_neg.sum()}")
        else:
            print(f"  [meta/asm_controlled] skipped — pos={asm_pos_mask.sum()}  neg={asm_ctrl_neg.sum()}")

    # Register V4 label-class concepts (pos: labels==i, neg: labels==0 for i in 1..9)
    label_concept_map: dict[str, int] = {}
    if token_labels is not None:
        unique_lbl_ints = sorted(int(v) for v in token_labels.unique().tolist())
        if len(unique_lbl_ints) > 2:  # multi-class means V4 (binary V3 has only 0,1)
            lbl_np = token_labels.numpy()
            neg_pool_lbl = lbl_np == 0
            for cls in unique_lbl_ints:
                if cls == 0:
                    continue
                cname = _V4_LABEL_NAMES.get(cls, f"label_{cls}")
                pos_mask_cls = lbl_np == cls
                if pos_mask_cls.sum() >= 10 and neg_pool_lbl.sum() >= 10:
                    concept_names.append(cname)
                    label_concept_map[cname] = cls
                else:
                    print(f"  [label/{cname}] skipped — pos={pos_mask_cls.sum()}  neg={neg_pool_lbl.sum()}")

    rng = np.random.default_rng(42)
    cav_vecs: list[np.ndarray]          = []
    cav_accs: list[float]               = []
    alignments: list[np.ndarray]        = []
    firing_rates: list[np.ndarray]      = []
    firing_rates_neg: list[np.ndarray]  = []
    delta_rates: list[np.ndarray]       = []
    p_values: list[np.ndarray]          = []
    tcav_scores_list: list[np.ndarray]  = []
    tcav_p_values_list: list[np.ndarray]= []
    model_tcav_scores: list[float]      = []     # Variant C: one float per concept
    model_tcav_p_values: list[float]    = []     # Variant C: one float per concept
    probe_accuracies: list[float]       = []     # Variant C: probe CV accuracy
    concept_sizes: list[int]            = []

    print(f"[TCAV] Training {len(concept_names)} CAVs …")
    for concept in concept_names:
        if concept == "abnormal":
            pos_mask = token_labels.numpy().astype(bool)  # type: ignore[union-attr]
            neg_mask_full = ~pos_mask
        elif concept in meta_concept_masks:
            pos_mask, neg_mask_meta = meta_concept_masks[concept]
            neg_mask_full = neg_mask_meta
        elif concept in label_concept_map:
            cls = label_concept_map[concept]
            lbl_np = token_labels.numpy()  # type: ignore[union-attr]
            pos_mask = lbl_np == cls
            neg_mask_full = lbl_np == 0
        else:
            pos_mask = token_band_labels == concept
            neg_mask_full = ~pos_mask

        pos_idx  = np.where(pos_mask)[0]
        neg_pool = np.where(neg_mask_full)[0]
        n_pos    = len(pos_idx)
        n_neg    = min(args.neg_per_concept, len(neg_pool))

        nan_arr = np.full(n_features, float("nan"), dtype=np.float32)
        if n_pos < 10 or n_neg < 10:
            print(f"  [{concept:12s}]  skipped — pos={n_pos}  neg={n_neg}")
            cav_vecs.append(np.zeros(embed_dim, dtype=np.float32))
            cav_accs.append(float("nan"))
            alignments.append(nan_arr.copy())
            firing_rates.append(nan_arr.copy())
            firing_rates_neg.append(nan_arr.copy())
            delta_rates.append(nan_arr.copy())
            p_values.append(np.ones(n_features, dtype=np.float32))
            tcav_scores_list.append(np.full(n_features, 0.5, dtype=np.float32))
            tcav_p_values_list.append(np.ones(n_features, dtype=np.float32))
            model_tcav_scores.append(float("nan"))
            model_tcav_p_values.append(float("nan"))
            probe_accuracies.append(float("nan"))
            concept_sizes.append(n_pos)
            continue

        neg_idx    = rng.choice(neg_pool, n_neg, replace=False)
        pos_acts_n = norm_np[pos_idx]
        neg_acts_n = norm_np[neg_idx]

        cav, acc, fold_cavs = train_cav(pos_acts_n, neg_acts_n, n_splits=args.n_splits)
        aln      = cav_sae_alignment(cav, W_dec)
        rate_pos = concept_firing_rates(raw_acts[pos_idx], sae, act_mean, act_std)
        rate_neg = concept_firing_rates(raw_acts[neg_idx], sae, act_mean, act_std)

        p_adj, delta = firing_rate_enrichment(rate_pos, rate_neg, n_pos, n_neg)

        # ── Variant A: weight-space TCAV (fold-CAV directional derivatives) ──
        print(f"  [{concept:12s}]  acc={acc:.3f}  n_pos={n_pos:6d}  "
              f"computing TCAV variants (n_random={args.n_random}) …", flush=True)
        tcav_sc, tcav_pv = compute_tcav_scores(
            fold_cavs, W_enc, pos_acts_n, neg_acts_n,
            n_random=args.n_random, random_state=42,
        )

        # ── Variant C: model-level Kim et al. TCAV with downstream probe ─────
        z_pos = get_sae_features(raw_acts[pos_idx], sae, act_mean, act_std)
        z_neg = get_sae_features(raw_acts[neg_idx], sae, act_mean, act_std)
        probe_w, probe_acc = train_probe(z_pos, z_neg)
        m_score, m_pval = compute_model_tcav_score(
            cav, W_enc, probe_w,
            raw_acts[pos_idx], sae, act_mean, act_std,
            pos_acts_n, neg_acts_n,
            n_random=args.n_random, random_state=42,
        )

        cav_vecs.append(cav)
        cav_accs.append(acc)
        alignments.append(aln)
        firing_rates.append(rate_pos)
        firing_rates_neg.append(rate_neg)
        delta_rates.append(delta)
        p_values.append(p_adj)
        tcav_scores_list.append(tcav_sc)
        tcav_p_values_list.append(tcav_pv)
        model_tcav_scores.append(m_score)
        model_tcav_p_values.append(m_pval)
        probe_accuracies.append(probe_acc)
        concept_sizes.append(n_pos)

        n_sig_dr = int((p_adj < 0.05).sum())
        print(
            f"    Δrate sig={n_sig_dr}/{n_features}  "
            f"model TCAV={m_score:.3f}  p={m_pval:.3f}  "
            f"probe_acc={probe_acc:.3f}"
        )

    # ── 8. Save ──────────────────────────────────────────────────────────────
    tcav_cache = {
        # ── Concept metadata ──────────────────────────────────────────────
        "concept_names":       concept_names,
        "cav_vectors":         torch.tensor(np.stack(cav_vecs)),          # (C, E)
        "cav_accuracies":      cav_accs,                                  # list[float]
        "concept_sizes":       concept_sizes,                             # list[int]
        # ── Per-feature metrics ───────────────────────────────────────────
        "alignments":          torch.tensor(np.stack(alignments)),        # (C, n_features)
        "firing_rates":        torch.tensor(np.stack(firing_rates)),      # (C, n_features) pos
        "firing_rates_neg":    torch.tensor(np.stack(firing_rates_neg)),  # (C, n_features) neg
        "delta_rates":         torch.tensor(np.stack(delta_rates)),       # (C, n_features)
        "p_values":            torch.tensor(np.stack(p_values)),          # (C, n_features) BH-adj z-test
        # ── Variant A: weight-space TCAV scores (per feature) ────────────
        # NOT proper Kim et al. — purely weight-space, no examples used.
        # Discrete {0/K, …, K/K}; masses at 0 and 1 for high-accuracy CAVs.
        "tcav_scores":         torch.tensor(np.stack(tcav_scores_list)),  # (C, n_features)
        "tcav_p_values":       torch.tensor(np.stack(tcav_p_values_list)),# (C, n_features)
        # ── Variant C: model-level Kim et al. TCAV (per concept) ─────────
        # Proper Kim et al.: fraction of concept examples where downstream
        # probe sensitivity S_k(x) = Σ_i[w_i×alignment_i×active_i(x)] > 0.
        # Range [0,1], 0.5=chance; single score per (concept, layer).
        "model_tcav_scores":   model_tcav_scores,                         # list[float] len C
        "model_tcav_p_values": model_tcav_p_values,                       # list[float] len C
        "probe_accuracies":    probe_accuracies,                          # list[float] len C
        # ── Metadata ──────────────────────────────────────────────────────
        "n_features":          n_features,
        "embed_dim":           embed_dim,
        "experiment":          args.experiment,
    }

    out_dir  = ROOT / "results" / "tcav" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tcav_cache.pt"
    torch.save(tcav_cache, str(out_path))
    print(f"\n[TCAV] Saved → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
