"""Build App Cache for the SAE4EEG Interactive Explorer.

Reads a named experiment from ``results/experiments/<name>/metadata.json``,
loads the corresponding SAE, SpectralDecoder and codebook artefacts, computes every
datum the Streamlit app needs, and writes a single self-contained
``results/experiments/<name>/app_cache.pt``.

The app never touches model weights, HDF5 files, or runs any training.
It loads only this cache file.

App cache schema
----------------
Feature Explorer
  feature_stats        list[dict]   128 items from feature_stats.json
                                    keys: feature, fire_rate_pct,
                                    n_patches_active,
                                    mean_activation, max_activation,
                                    label_correlation, label_p_value
  feature_explanations list[dict]   128 items from feature_explanations.json
                                    keys: feature, description,
                                    boosted_band, suppressed_band,
                                    band_effects, cluster, is_representative
  feature_band_deltas  Tensor       (n_features, n_bands)
                                    differential band power per feature
                                    (feature ON − baseline, linear amplitude)
  feature_amp_diff     Tensor       (n_features, n_bins)
                                    full differential amplitude spectrum
  feature_freqs        ndarray      (n_bins,) frequency axis in Hz
  feature_band_names   list[str]    ['delta','theta','alpha',...]
  feature_cooccurrence Tensor       (n_features, n_features) Jaccard similarity
  feature_dashboard_imgs  dict[int, bytes]
                                    PNG bytes of the dashboard images that exist
                                    in results/features/dashboards/
                                    keyed by feature index

Codebook Browser
  codebook_n_clusters  int
  codebook_cluster_sizes  Tensor    (K,)
  codebook_band_labels    list[str] dominant band per cluster
  codebook_amp_log1p   Tensor       (K, n_bins) centroid amplitude
  codebook_freqs       ndarray      (n_bins,)
  codebook_recon_waveforms  Tensor  (K, 19, 128) per-channel waveforms
  codebook_exemplar_recon_r Tensor  (K,) per-channel Pearson r
  codebook_sort_order  Tensor       (K,) rank order (highest power first)
  codebook_band_powers dict[str, ndarray]  (K,) per-band power per cluster
  codebook_centroids_emb  Tensor    (K, 128) centroid embeddings
  codebook_umap_coords    ndarray   (n_tokens, 2)  — computed here
  codebook_umap_labels    ndarray   (n_tokens,)    cluster id per token
  codebook_umap_token_labels  ndarray (n_tokens,) 0/1 label per token (if available)
  codebook_umap_meta      dict[str, ndarray]       per-token metadata strings
                                    keys: age_group, gender, classification,
                                    indication_group, medication_group,
                                    recording_date, clinic

Layer Activation UMAPs  (optional, built with --layer-umap-layers)
  layer_umap_layers          list[int]
  layer_umap_coords          dict[int → ndarray(n_tokens, 2)]
  layer_umap_cluster_labels  ndarray(n_tokens,)  spectral cluster IDs (same for all layers)

Morphology Probe
  morph_names          list[str]    8 morphology names
  morph_waveforms      ndarray      (8, 19, 128) synthetic waveforms
  morph_match_cluster_ids  ndarray  (8, 5)
  morph_match_dists    ndarray      (8, 5)
  morph_match_cosines  ndarray      (8, 5)
  morph_match_waveforms  Tensor     (8, 5, 19, 128)
  morph_match_sizes    ndarray      (8, 5)
  morph_match_dom_bands  list[list[str]]  (8, 5)
  morph_inter_dist     ndarray      (8, 8) Euclidean inter-morph distances

Metadata (echoed from metadata.json for display)
  meta                 dict

Usage::

    uv run tools/build_app_cache.py --experiment exp1_k8_layer2
    uv run tools/build_app_cache.py --experiment exp1_k8_layer2 --max-tokens 10000
"""
import sys
import json
import argparse
import io
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# ── project imports ─────────────────────────────────────────────────────────
from mecheeg.spectral_decoder import SpectralDecoderTrainer, CLINICAL_BANDS
from mecheeg.sae import SparseAutoencoder, ActivationExtractor
from mecheeg.dataset import get_dataloaders, StandardizeLabel, H5PYDatasetLabeled, V4ResampleTransform
from mecheeg.encoders import load_encoder

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FS = 128          # overridden from meta at runtime
PATCH_SIZE = 128  # overridden from meta at runtime
TARGET_LAYER = 2  # overridden from meta at runtime

_ENCODER_DATA = {
    "sleepfm":          "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.0":     "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.1":     "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.3":     "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.4":     "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.5":     "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.6":     "data/D4-v3-preprocessed-v2",
    "sleepfm_v2.7":     "data/D4-v3-preprocessed-v2",
    "sleepfm_granular": "data/D4-v4-preprocessed-10s",
    "reve":             "data/D4-v3-preprocessed-v1",
    "labram":           "data/D4-v3-preprocessed-v1",
}

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7",  "C3",  "Cz", "C4", "T8",
    "T5",  "P3",  "Pz", "P4", "T6", "O1", "O2",
]

# HDF5 metadata fields to collect for UMAP colouring in the app
METADATA_FIELDS = [
    "age_group", "gender", "classification",
    "indication_group", "medication_group", "recording_date", "clinic",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def remove_module_from_state_dict(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()}


def _load_model(encoder_name: str = "sleepfm", weights_path=None):
    if encoder_name == "sleepfm":
        encoder = load_encoder("sleepfm",
                               weights_path=weights_path or ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt")
    else:
        kwargs = {"weights_path": weights_path} if weights_path else {}
        encoder = load_encoder(encoder_name, **kwargs)
    return encoder.to(DEVICE).eval()


def _collect_tokens(model, spectral_decoder_trainer, val_loader, max_tokens):
    """Collect embeddings, spectral targets, raw patches, and labels.

    Handles both SleepFM (cross-channel tokens) and REVE (per-channel tokens).

    For ``EncoderBackend`` instances, ``model.encode()`` only returns the
    *last* layer's activation. To target an earlier layer, hook
    ``get_hookable_layers()[TARGET_LAYER]`` so we capture exactly the
    activation the SAE was trained against. For the SetTransformer legacy
    path (non-backend), the same ``TARGET_LAYER`` is used for the hook key.
    """
    from mecheeg.encoders import EncoderBackend

    is_backend = isinstance(model, EncoderBackend)
    reve_mode  = is_backend and hasattr(model, "channel_names")
    # LaBraM blocks emit (B, 1+C*S, D) with a CLS prefix token. The encode()
    # path strips it; the hook path does not — so we strip here for parity.
    n_prefix_strip = int(getattr(model, "n_prefix_tokens", 0)) if is_backend else 0

    embed_list, spec_list, raw_list, label_list = [], [], [], []
    total = 0
    tokens_per_window = 1  # filled from first batch

    # Pre-build extractor when we need a non-last layer. For the legacy
    # SetTransformer path, always extract via hook (preserves prior behaviour).
    extractor = None
    if is_backend:
        all_layers = model.get_hookable_layers()
        n_layers   = len(all_layers)
        if TARGET_LAYER != n_layers - 1:
            extractor = ActivationExtractor(
                model.model, layers=[all_layers[TARGET_LAYER]],
            )

    from tqdm import tqdm
    pbar = tqdm(val_loader, desc="Collecting tokens", leave=False)
    for batch in pbar:
        x = batch[0].to(DEVICE)
        y = batch[1] if len(batch) > 1 else None
        B, C, T = x.shape

        with torch.no_grad():
            if is_backend and extractor is not None:
                extractor.clear()
                _ = model.encode(x)
                layer_acts = extractor.get_activations()[0]
                # REVE blocks emit (B, C, S, E); flatten to (B, C*S, E).
                if layer_acts.dim() == 4:
                    B2, Cc, Ss, Ee = layer_acts.shape
                    layer_acts = layer_acts.reshape(B2, Cc * Ss, Ee)
                if n_prefix_strip and layer_acts.dim() == 3:
                    layer_acts = layer_acts[:, n_prefix_strip:, :]
            elif is_backend:
                layer_acts = model.encode(x)  # last layer (matches TARGET_LAYER)
            else:
                extractor = ActivationExtractor(model)
                extractor.clear()
                _ = model(x)
                layer_acts = extractor.get_activations()[TARGET_LAYER]

        N_tok = layer_acts.shape[1]

        if reve_mode:
            C_enc  = len(model.channel_names)
            S_reve = N_tok // C_enc
            stride = T // S_reve
            max_st = T - PATCH_SIZE
            patches_list = []
            for s_idx in range(S_reve):
                st = min(s_idx * stride, max_st)
                patches_list.append(x[:, :C_enc, st : st + PATCH_SIZE])
            patches_bcsp = torch.stack(patches_list, dim=2)  # (B, C_enc, S_reve, P)
            patches = patches_bcsp.reshape(B * C_enc * S_reve, 1, PATCH_SIZE)
            n_new = B * C_enc * S_reve
            if y is not None:
                label_list.append(
                    y.unsqueeze(1).unsqueeze(1)
                     .expand(-1, C_enc, S_reve).reshape(B * C_enc * S_reve))
        else:
            S      = T // PATCH_SIZE
            T_used = S * PATCH_SIZE
            patches = x[:, :, :T_used].reshape(B, C, S, PATCH_SIZE)
            patches = patches.permute(0, 2, 1, 3).reshape(B * S, C, PATCH_SIZE)
            n_new  = B * S
            if y is not None:
                label_list.append(y.unsqueeze(1).expand(-1, S).reshape(B * S))

        if total == 0:
            tokens_per_window = n_new // B

        spec_targets = spectral_decoder_trainer.spectral.extract(patches)
        embed_list.append(layer_acts.reshape(n_new, -1).cpu())
        spec_list.append(spec_targets.cpu())
        raw_list.append(patches.cpu())

        total += n_new
        pbar.set_postfix(tokens=f"{total:,}")
        if total >= max_tokens:
            break

    pbar.close()
    if extractor is not None:
        extractor.remove_hooks()

    embeddings = torch.cat(embed_list)[:max_tokens]
    spectral   = torch.cat(spec_list)[:max_tokens]
    raw_full   = torch.cat(raw_list)[:max_tokens]
    token_labels = torch.cat(label_list)[:max_tokens] if label_list else None
    return embeddings, spectral, raw_full, token_labels, tokens_per_window


def _collect_token_metadata(val_dataset, fields, n_tokens, tokens_per_window):
    """Load per-window HDF5 metadata and expand to per-token arrays.

    Reads directly from the dataset's index_map — no extra forward pass needed.
    Windows are iterated in DataLoader order (shuffle=False), so indices align
    with the tokens collected by _collect_tokens.
    """
    import h5py as _h5

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
                except Exception as exc:
                    print(f"  [warn] metadata/{field}: {exc}")

    token_meta = {}
    for field in fields:
        arr      = window_meta[field]                            # (n_windows,)
        expanded = np.repeat(arr, tokens_per_window)[:n_tokens]  # (n_tokens,)
        token_meta[field] = expanded
        print(f"  {field}: {len(np.unique(expanded))} unique values")

    # Subject IDs — already in memory, no HDF5 read needed
    subj = val_dataset.index_map["subjects"][:n_windows]
    if torch.is_tensor(subj):
        subj = subj.numpy()
    token_meta["subject_id"] = np.repeat(subj.astype(str), tokens_per_window)[:n_tokens]
    print(f"  subject_id: {len(np.unique(token_meta['subject_id']))} unique subjects")

    return token_meta


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_section(meta, spectral_decoder_trainer, embeddings=None, token_labels=None,
                           token_meta=None, enr_embeddings=None, enr_token_meta=None):
    """Build the Feature Explorer section of the cache."""
    print("\n[Feature Explorer]")

    # -- Load JSON artefacts (both are optional) ----------------------
    stats_path = ROOT / meta.get("feature_stats", "__missing__")
    expl_path  = ROOT / meta.get("feature_explanations", "__missing__")
    if stats_path.exists():
        feature_stats = json.loads(stats_path.read_text())
    else:
        feature_stats = []
        print("  [info] feature_stats.json not found — will compute from tokens if available")
    if expl_path.exists():
        feature_explanations = json.loads(expl_path.read_text())
    else:
        feature_explanations = []
        print("  [warn] feature_explanations not found — descriptions will be absent")

    # Derive n_features from SAE checkpoint when stats are missing
    sae_ckpt_tmp = torch.load(ROOT / meta["sae_checkpoint"], weights_only=False,
                              map_location="cpu")
    embed_dim_tmp = sae_ckpt_tmp.get("embed_dim", meta.get("embed_dim", 128))
    expansion_tmp = meta["expansion"]
    n_features = (len(feature_stats) if feature_stats
                  else int(embed_dim_tmp * expansion_tmp))
    print(f"  {n_features} features, {len(feature_explanations)} explanations")

    # -- Load SAE checkpoint ------------------------------------------
    sae_ckpt = torch.load(ROOT / meta["sae_checkpoint"], weights_only=False,
                          map_location="cpu")
    embed_dim = sae_ckpt.get("embed_dim", meta.get("embed_dim", 128))
    sae = SparseAutoencoder(embed_dim, expansion=meta["expansion"],
                            mode="topk", k=meta["k"])
    sae.load_state_dict(sae_ckpt["sae_state_dict"])
    act_mean = sae_ckpt["act_mean"]
    act_std  = sae_ckpt["act_std"]
    print(f"  Loaded SAE: expansion={meta['expansion']}, k={meta['k']}")

    # -- Compute feature stats from tokens if JSON not available ------
    if not feature_stats and embeddings is not None:
        from scipy import stats as _scipy_stats
        print(f"  Computing feature stats from {len(embeddings)} tokens ...")
        sae.to(DEVICE).eval()
        x_norm = ((embeddings.to(DEVICE) - act_mean.to(DEVICE))
                  / act_std.to(DEVICE).clamp(min=1e-8))
        _BATCH = 4096
        z_chunks = []
        with torch.no_grad():
            for _s in range(0, len(x_norm), _BATCH):
                z_chunks.append(sae.encode(x_norm[_s:_s + _BATCH]).cpu())
        z_all = torch.cat(z_chunks, dim=0)   # (N, n_features)
        labs  = token_labels.float() if token_labels is not None else None
        for _i in range(n_features):
            zi        = z_all[:, _i]
            fire_mask = zi > 0
            fire_pct  = float(fire_mask.float().mean() * 100)
            n_active  = int(fire_mask.sum())
            mean_act  = float(zi[fire_mask].mean()) if fire_mask.any() else 0.0
            max_act   = float(zi.max())
            if labs is not None:
                r, p = _scipy_stats.pearsonr(zi.numpy(), labs.numpy())
            else:
                r, p = float("nan"), float("nan")
            feature_stats.append({
                "feature":           _i,
                "fire_rate_pct":     fire_pct,
                "n_patches_active":  n_active,
                "mean_activation":   mean_act,
                "max_activation":    max_act,
                "label_correlation": float(r),
                "label_p_value":     float(p),
            })
        # -- Metadata enrichment — subject-level, BH-corrected ---------------
        #
        # Uses the full-dataset tokens (enr_embeddings / enr_token_meta) so all
        # subjects are represented, not just the ~10% in the val set.
        # Token-level tests are anti-conservative because tokens from the same
        # subject are correlated.  Fix: aggregate to subject level first, then
        # compare groups with Mann-Whitney U (non-parametric, robust to small n).
        # Apply Benjamini-Hochberg FDR correction across all (feature, cat) pairs
        # within each field.  Require >= MIN_SUBJ subjects per category.
        #
        MIN_SUBJ = 5

        _enr_meta = enr_token_meta if enr_token_meta is not None else token_meta
        _enr_emb  = enr_embeddings  if enr_embeddings  is not None else embeddings

        feature_meta_enrichment: list[dict] = [{} for _ in range(n_features)]
        if _enr_meta and _enr_meta.get("subject_id") is not None and _enr_emb is not None:
            from scipy.stats import mannwhitneyu as _mwu
            from scipy.stats import false_discovery_control as _bh

            # Compute SAE activations for the full-dataset tokens (batch-wise to
            # avoid materializing the whole 8M-token tensor on GPU at once). Only
            # the boolean firing pattern is needed downstream, so we convert each
            # chunk to bool before accumulating to keep memory at ~1 byte/element
            # rather than 4 bytes/element (matters at E≥4 where n_features × N_full
            # crosses 10s of GB).
            print(f"  Computing SAE activations for {len(_enr_emb)} enrichment tokens ...")
            sae.to(DEVICE).eval()
            _am = act_mean.to(DEVICE)
            _as = act_std.to(DEVICE).clamp(min=1e-8)
            _fire_chunks = []
            with torch.no_grad():
                for _s in range(0, len(_enr_emb), _BATCH):
                    _chunk = _enr_emb[_s:_s + _BATCH].to(DEVICE)
                    _norm  = (_chunk - _am) / _as
                    _fire_chunks.append((sae.encode(_norm) > 0).cpu().numpy())
            sae.cpu()

            fire_np    = np.concatenate(_fire_chunks, axis=0)  # (N_full, n_features) bool
            subj_arr   = _enr_meta["subject_id"]          # (N_full,) str
            uniq_subj  = np.unique(subj_arr)
            print(f"  Enrichment: {len(uniq_subj)} subjects across {len(fire_np)} tokens")

            # Subject-level mean firing rate: (n_subjects, n_features)
            subj_fr = np.stack([
                fire_np[subj_arr == s].mean(axis=0) for s in uniq_subj
            ])                                        # (S, n_features)

            def _age_sort_key_bac(v: str) -> int:
                try:
                    return int(v.split("-")[0].replace("+", ""))
                except ValueError:
                    return 999

            for _field in ("age_group", "gender", "indication_group", "medication_group"):
                _tok_arr = _enr_meta.get(_field)
                if _tok_arr is None:
                    continue

                # Map subject → category (majority vote across its tokens)
                _subj_cat = {}
                for s in uniq_subj:
                    _tok_mask = subj_arr == s
                    _vals, _cnts = np.unique(_tok_arr[_tok_mask], return_counts=True)
                    _subj_cat[s] = str(_vals[np.argmax(_cnts)])

                _subj_cat_arr = np.array([_subj_cat[s] for s in uniq_subj])

                if _field == "age_group":
                    _cats = sorted(
                        set(v for v in _subj_cat_arr if v not in ("", "unknown")),
                        key=_age_sort_key_bac,
                    )
                else:
                    _cats = sorted(
                        set(v for v in _subj_cat_arr if v not in ("", "unknown"))
                    )

                # Collect all (feature, cat) raw p-values for BH correction
                _all_ratios: list[tuple[str, int, float]] = []   # (cat, feat_i, ratio)
                _all_raw_p:  list[float]                  = []

                for _cat in _cats:
                    _in  = _subj_cat_arr == _cat
                    _out = (~_in) & np.isin(_subj_cat_arr, _cats)  # exclude unknowns
                    if _in.sum() < MIN_SUBJ:
                        continue
                    _fr_in  = subj_fr[_in]   # (n_in,  n_features)
                    _fr_out = subj_fr[_out]  # (n_out, n_features)
                    _global_fr = subj_fr[np.isin(_subj_cat_arr, _cats)].mean(axis=0)

                    for _fi in range(n_features):
                        try:
                            _, _p = _mwu(_fr_in[:, _fi], _fr_out[:, _fi],
                                         alternative="two-sided")
                        except ValueError:
                            _p = 1.0
                        _ratio = float(_fr_in[:, _fi].mean()) / max(float(_global_fr[_fi]), 1e-8)
                        _all_ratios.append((_cat, _fi, _ratio))
                        _all_raw_p.append(_p)

                if not _all_raw_p:
                    continue

                # BH correction across all (feature, cat) pairs for this field
                _adj_p = _bh(np.array(_all_raw_p), method="bh")

                for (_cat, _fi, _ratio), _p_adj in zip(_all_ratios, _adj_p):
                    _feat_enr = feature_meta_enrichment[_fi].setdefault(_field, [])
                    _feat_enr.append((_cat, _ratio, float(_p_adj)))

            print(f"  Metadata enrichment computed (subject-level, BH-corrected, "
                  f"n_subjects={len(uniq_subj)})")
        else:
            feature_meta_enrichment = []

        sae.cpu()
        print(f"  Computed stats for {len(feature_stats)} features")
    else:
        feature_meta_enrichment = []
    if not feature_stats:
        print("  [warn] No tokens available — fire rates / label correlations "
              "will be absent from the cache")

    # -- Decode feature directions through SpectralDecoder ------------------------
    # Decoder weight columns are the feature directions
    directions = sae.decoder.weight.T.detach().clone()  # (n_features, 128)
    baseline_raw = act_mean.unsqueeze(0)                # (1, 128)
    alpha = 2.0
    raw_dirs = directions * act_std
    activated_raw = baseline_raw + alpha * raw_dirs     # (n_features, 128)

    amp_base, _, _ = spectral_decoder_trainer.decode_direction(
        baseline_raw.to(DEVICE), denormalise=True)
    amp_act, _, _  = spectral_decoder_trainer.decode_direction(
        activated_raw.to(DEVICE), denormalise=True)
    amp_base = amp_base.squeeze(0).cpu()                # (n_bins,)
    amp_act  = amp_act.cpu()                             # (n_features, n_bins)
    amp_diff = amp_act - amp_base.unsqueeze(0)           # (n_features, n_bins)

    # -- Per-feature per-band differential power ----------------------
    band_names = list(CLINICAL_BANDS.keys())
    freqs = spectral_decoder_trainer.spectral.freqs                   # (n_bins,) ndarray
    feature_band_deltas = np.zeros((n_features, len(band_names)),
                                   dtype=np.float32)
    for j, (bname, (f_lo, f_hi)) in enumerate(CLINICAL_BANDS.items()):
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if mask.sum() > 0:
            feature_band_deltas[:, j] = (
                amp_diff[:, mask].mean(dim=1).numpy()
            )
    print(f"  Band delta matrix: {feature_band_deltas.shape}")

    # -- Co-occurrence (Jaccard similarity) ---------------------------
    # We don't re-run the encoder here (expensive); instead we derive
    # a co-occurrence proxy from the band_effects in feature_explanations
    # (features that boost the same band tend to co-activate).
    # If a richer co-occurrence is needed, run explore_features first.
    band_effect_rows = [
        [e["band_effects"].get(b, 0.0) for b in band_names]
        for e in feature_explanations
    ]
    if band_effect_rows:
        band_effect_mat = np.array(band_effect_rows, dtype=np.float32)
    else:
        # No explanations available; fall back to band deltas as proxy
        band_effect_mat = feature_band_deltas  # (n_features, n_bands)

    # Cosine similarity as co-occurrence proxy
    norms = np.linalg.norm(band_effect_mat, axis=1, keepdims=True).clip(min=1e-8)
    normed = band_effect_mat / norms
    cooccurrence = (normed @ normed.T).astype(np.float32)   # (n_features, n_features)

    # -- Dashboard images: PNG bytes keyed by feature index -----------
    encoder_name = meta.get("encoder", "sleepfm")
    dash_dir = ROOT / "results" / "features" / encoder_name / "dashboards"
    dashboard_imgs = {}
    for png_path in sorted(dash_dir.glob("*.png")):
        # Filenames: 01a_fire_F103.png, 01b_label0_F21.png, etc.
        stem = png_path.stem
        parts = stem.split("_")
        # Last part is F<idx>
        try:
            feat_idx = int(parts[-1].lstrip("F"))
            dashboard_imgs[feat_idx] = png_path.read_bytes()
        except (ValueError, IndexError):
            pass
    print(f"  Dashboard images loaded: {len(dashboard_imgs)} PNGs")

    return {
        "feature_stats":            feature_stats,
        "feature_explanations":     feature_explanations,
        "feature_band_deltas":      torch.tensor(feature_band_deltas),
        "feature_amp_diff":         amp_diff,
        "feature_amp_baseline":     amp_base,
        "feature_freqs":            freqs,
        "feature_band_names":       band_names,
        "feature_cooccurrence":     torch.tensor(cooccurrence),
        "feature_dashboard_imgs":   dashboard_imgs,
        "feature_meta_enrichment":  feature_meta_enrichment,
    }


def _codebook_subsample_idx(N: int, max_umap_tokens: int = 10_000):
    """Return (sub_idx, sub_n) — deterministic subsample used across UMAP sections."""
    sub_n = min(max_umap_tokens, N)
    rng = np.random.default_rng(42)
    return rng.choice(N, sub_n, replace=False), sub_n


def _build_codebook_section(codebook, embeddings, token_labels,
                             token_meta=None, max_umap_tokens=10_000):
    """Build the Codebook Browser section of the cache."""
    print("\n[Codebook Browser]")
    K = codebook["n_clusters"]
    print(f"  {K} clusters")

    # -- UMAP of token embeddings, coloured by cluster label -----------
    emb_np = embeddings.numpy()
    labels_np = codebook["labels"].numpy() if isinstance(
        codebook["labels"], torch.Tensor) else codebook["labels"]

    N = len(emb_np)
    sub_idx, sub_n = _codebook_subsample_idx(N, max_umap_tokens)
    emb_sub = emb_np[sub_idx]
    labels_sub = labels_np[sub_idx]
    tok_label_sub = (token_labels[sub_idx].numpy()
                     if token_labels is not None else None)

    meta_sub = {}
    if token_meta:
        for field, vals in token_meta.items():
            if len(vals) >= N:
                meta_sub[field] = vals[sub_idx]

    print(f"  Computing UMAP ({sub_n} tokens) ...", flush=True)
    import umap as umap_module
    reducer = umap_module.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.3,
        random_state=42, metric="euclidean",
    )
    umap_coords = reducer.fit_transform(emb_sub).astype(np.float32)
    print(f"  UMAP done: {umap_coords.shape}")

    return {
        "codebook_n_clusters":       K,
        "codebook_cluster_sizes":    codebook["cluster_sizes"],
        "codebook_band_labels":      codebook["cluster_band_label"],
        "codebook_amp_log1p":        codebook["centroid_amp_log1p"],
        "codebook_freqs":            codebook["freqs"],
        "codebook_recon_waveforms":  codebook["recon_waveforms"],
        "codebook_exemplar_raw_patches": codebook.get("exemplar_raw_patches"),  # (K, C, P) or None
        "codebook_exemplar_recon_r": codebook["exemplar_recon_r"],
        "codebook_sort_order":       codebook["sort_order"],
        "codebook_band_powers":      codebook["band_powers"],
        "codebook_centroids_emb":    codebook["centroids_emb"],
        "codebook_umap_coords":      umap_coords,
        "codebook_umap_labels":      labels_sub,
        "codebook_umap_token_labels": tok_label_sub,
        "codebook_umap_meta":        meta_sub,
        # Raw token embeddings for the UMAP subsample — used by the concept-steering
        # linear probe so it operates on actual token embeddings rather than centroids.
        "codebook_umap_embeddings":  emb_sub.astype(np.float32),
    }


def _build_layer_umap_section(layer_acts: dict, layer_indices: list,
                               codebook, max_umap_tokens: int = 10_000):
    """Compute UMAP of encoder activations at each specified layer.

    Uses the same deterministic subsample index as the codebook UMAP so that
    tokens are comparable across layers.  Colors come from the spectral
    codebook cluster labels (encoder-layer-independent).

    New cache keys
    --------------
    layer_umap_layers          list[int]
    layer_umap_coords          dict[int → ndarray(sub_n, 2)]
    layer_umap_cluster_labels  ndarray(sub_n,)  — spectral cluster IDs
    """
    print(f"\n[Layer UMAPs] layers={layer_indices}")
    import umap as umap_module

    labels_all = codebook["labels"]
    if torch.is_tensor(labels_all):
        labels_all = labels_all.numpy()

    # Use same subsample size as codebook UMAP
    N_ref = len(labels_all)
    sub_idx, sub_n = _codebook_subsample_idx(N_ref, max_umap_tokens)
    labels_sub = labels_all[sub_idx]

    coords_per_layer = {}
    for L in sorted(layer_indices):
        if L not in layer_acts:
            print(f"  Layer {L}: activations not found, skipping")
            continue
        acts = layer_acts[L]            # (N_collected, E)
        if len(acts) < sub_n:
            print(f"  Layer {L}: only {len(acts)} tokens < {sub_n}, skipping")
            continue
        acts_sub = acts[sub_idx].numpy()
        print(f"  Layer {L}: UMAP({sub_n} × {acts_sub.shape[1]}) ...", flush=True)
        reducer = umap_module.UMAP(
            n_components=2, n_neighbors=30, min_dist=0.3,
            random_state=42, metric="euclidean",
        )
        coords_per_layer[L] = reducer.fit_transform(acts_sub).astype(np.float32)
        print(f"  Layer {L}: done {coords_per_layer[L].shape}")

    if not coords_per_layer:
        print("  No layer UMAPs produced.")
        return {}

    return {
        "layer_umap_layers":         sorted(coords_per_layer.keys()),
        "layer_umap_coords":         coords_per_layer,
        "layer_umap_cluster_labels": labels_sub,
    }


def _build_morphology_section(model, codebook, spectral_decoder_trainer,
                              val_loader, n_batches_scale=5,
                              patch_size=128, target_layer=None):
    """Build the Morphology Probe section of the cache."""
    print("\n[Morphology Probe]")

    # Import morphology generators and helpers from probe_morphology
    sys.path.insert(0, str(ROOT / "tools"))
    from probe_morphology import (
        MORPHOLOGIES, embed_synthetic, find_nearest_centroids,
        get_dominant_band, estimate_data_scale,
    )

    # Estimate real data scale for normalisation
    data_mean, data_std = estimate_data_scale(
        spectral_decoder_trainer, model, val_loader, n_batches=n_batches_scale)
    print(f"  Data scale: mean={data_mean:.4f}, std={data_std:.4f}")

    sort_order = codebook["sort_order"].numpy()
    rank_map = {int(sort_order[r]): r for r in range(len(sort_order))}

    names, waveforms_list = [], []
    match_ids_list, dists_list, cosines_list = [], [], []
    match_waves_list, match_raw_list, sizes_list, dom_bands_list = [], [], [], []

    for name, gen_fn in MORPHOLOGIES.items():
        waveform = gen_fn(fs=patch_size, n=patch_size)
        # Normalise to match real data distribution
        w_std = waveform.std()
        if w_std > 1e-8:
            waveform_norm = (waveform - waveform.mean()) / w_std * data_std + data_mean
        else:
            waveform_norm = waveform

        emb = embed_synthetic(model, waveform_norm, target_layer=target_layer)
        top_idx, dists, cosines = find_nearest_centroids(
            emb, codebook, top_k=5)

        mw = torch.stack([
            codebook["recon_waveforms"][c_id] for c_id in top_idx
        ])  # (5, 19, 128)
        raw_ex = codebook.get("exemplar_raw_patches")
        if raw_ex is not None:
            mr = torch.stack([raw_ex[c_id] for c_id in top_idx])  # (5, C, 128)
        else:
            mr = None
        sizes = [int(codebook["cluster_sizes"][c_id]) for c_id in top_idx]
        dom_bands = [get_dominant_band(codebook, c_id, spectral_decoder_trainer)
                     for c_id in top_idx]

        names.append(name)
        waveforms_list.append(waveform)
        match_ids_list.append(top_idx)
        dists_list.append(dists)
        cosines_list.append(cosines)
        match_waves_list.append(mw)
        match_raw_list.append(mr)
        sizes_list.append(sizes)
        dom_bands_list.append(dom_bands)
        print(f"  {name}: nearest cluster dist={dists[0]:.2f}, cos={cosines[0]:.3f}")

    # Inter-morphology distances
    def _norm_waveform(g):
        w = g(fs=patch_size, n=patch_size)
        return (w - w.mean()) / max(w.std(), 1e-8) * data_std + data_mean

    embs_all = np.array([
        embed_synthetic(model, _norm_waveform(g), target_layer=target_layer)
        for g in MORPHOLOGIES.values()
    ])
    n = len(names)
    inter_dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            inter_dist[i, j] = np.linalg.norm(embs_all[i] - embs_all[j])
    print(f"  Inter-morphology distance matrix: {inter_dist.shape}")

    return {
        "morph_names":             names,
        "morph_waveforms":         np.array(waveforms_list, dtype=np.float32),
        "morph_match_cluster_ids": np.array(match_ids_list),
        "morph_match_dists":       np.array(dists_list),
        "morph_match_cosines":     np.array(cosines_list),
        "morph_match_waveforms":   torch.stack(match_waves_list),
        "morph_match_raw_patches": torch.stack(match_raw_list) if match_raw_list[0] is not None else None,  # (8, 5, C, P)
        "morph_match_sizes":       np.array(sizes_list),
        "morph_match_dom_bands":   dom_bands_list,
        "morph_inter_dist":        inter_dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build Streamlit app cache for a named experiment")
    parser.add_argument("--experiment", "-e", required=True,
                        help="Experiment name (matches results/experiments/<name>/)")
    parser.add_argument("--max-tokens", type=int, default=20_000,
                        help="Max validation tokens to collect (default: 20000)")
    parser.add_argument("--enr-max-tokens", type=int, default=None,
                        help="Cap on enrichment tokens from full dataset (default: all). "
                             "Set for large encoders (e.g. REVE E>=8) to avoid OOM.")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP computation (faster, no codebook browser map)")
    parser.add_argument("--skip-morphology", action="store_true",
                        help="Skip morphology probe (faster rebuild)")
    parser.add_argument("--layer-umap-layers",
                        help="Comma-separated layer indices for per-layer activation UMAPs "
                             "(e.g. '0,4,8,12,16,21').  Requires an extra forward pass.  "
                             "Omit to skip.")
    args = parser.parse_args()

    exp_dir = ROOT / "results" / "experiments" / args.experiment
    meta_path = exp_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata.json found at {meta_path}\n"
            f"Create one first (see results/experiments/exp1_k8_layer2/metadata.json "
            f"as a template)."
        )

    meta = json.loads(meta_path.read_text())
    cache_path = exp_dir / "app_cache.pt"

    print("=" * 72)
    print(f"  Building app cache: {args.experiment}")
    print(f"  Output: {cache_path}")
    print("=" * 72)

    # ── Load shared artefacts ─────────────────────────────────────────
    print("\n[Loading shared artefacts]")
    encoder_name   = meta.get("encoder", "sleepfm")
    encoder_embed  = meta.get("embed_dim", 128)
    encoder_patch  = meta.get("patch_size", 128)
    encoder_layer  = meta.get("target_layer", None)
    data_path     = meta.get("data_path",
                             _ENCODER_DATA.get(encoder_name, "data/D4-v3-preprocessed-v2"))
    _transform    = V4ResampleTransform() if "D4-v4" in str(data_path) else StandardizeLabel()
    weights_path  = meta.get("weights_path", None)
    if weights_path is not None:
        weights_path = ROOT / weights_path

    # Propagate PATCH_SIZE and TARGET_LAYER into module scope for _collect_tokens.
    # TARGET_LAYER must reflect the experiment's metadata; the previous
    # default-of-2 silently extracted the wrong layer for layers != 2.
    global PATCH_SIZE, TARGET_LAYER
    PATCH_SIZE = encoder_patch
    if encoder_layer is None:
        raise ValueError(
            f"metadata.json for {args.experiment} is missing 'target_layer'"
        )
    TARGET_LAYER = int(encoder_layer)
    print(f"  TARGET_LAYER set to {TARGET_LAYER}")

    model = _load_model(encoder_name, weights_path=weights_path)
    print(f"  {encoder_name} encoder loaded (embed_dim={encoder_embed})")

    spectral_decoder_trainer = SpectralDecoderTrainer(embed_dim=encoder_embed, device=DEVICE)
    spectral_decoder_trainer.load(str(ROOT / meta["spectral_decoder_checkpoint"]))
    spectral_decoder_trainer.spectral_decoder.to(DEVICE).eval()
    # Move normalisation stats to the same device as the model
    for attr in ("embed_mean", "embed_std", "target_mean", "target_std"):
        val = getattr(spectral_decoder_trainer, attr, None)
        if val is not None:
            setattr(spectral_decoder_trainer, attr, val.to(DEVICE))
    print("  SpectralDecoder loaded")

    codebook = torch.load(ROOT / meta["codebook_path"],
                          weights_only=False, map_location="cpu")
    print(f"  Codebook loaded ({codebook['n_clusters']} clusters)")

    from torch.utils.data import DataLoader as _DataLoader
    gen = get_dataloaders(
        train_path=str(ROOT / data_path),
        transformer=_transform,
    )
    _, _, val_loader, _ = next(gen)
    print("  Validation data loader ready")

    # Full dataset loader — used for demographic enrichment so all subjects
    # are represented (val set is only ~10% of subjects).
    full_ds = H5PYDatasetLabeled(str(ROOT / data_path), transform=_transform)
    full_loader = _DataLoader(full_ds, batch_size=val_loader.batch_size,
                              shuffle=False, num_workers=0)
    print(f"  Full dataset: {len(full_ds)} windows, "
          f"{len(np.unique(full_ds.subjects))} subjects")

    # ── Collect tokens ────────────────────────────────────────────────
    print(f"\n[Collecting up to {args.max_tokens} validation tokens]")
    embeddings, _, _, token_labels, tokens_per_window = _collect_tokens(
        model, spectral_decoder_trainer, val_loader, args.max_tokens)
    print(f"  Collected {len(embeddings)} tokens  "
          f"({tokens_per_window} tokens/window)")

    print("\n[Collecting token metadata (val set — for UMAP/codebook)]")
    token_meta = _collect_token_metadata(
        val_loader.dataset, METADATA_FIELDS, len(embeddings), tokens_per_window)

    # Collect enrichment tokens from full dataset — all subjects (or capped).
    # Use a subject-interleaved window order so a token cap covers subjects
    # uniformly rather than exhausting long recordings first.
    enr_max_tokens = len(full_ds) * tokens_per_window
    if args.enr_max_tokens is not None:
        enr_max_tokens = min(enr_max_tokens, args.enr_max_tokens)
    print(f"\n[Collecting enrichment tokens from full dataset "
          f"(up to {enr_max_tokens} tokens)]")

    _subj_arr = full_ds.index_map["subjects"]
    if torch.is_tensor(_subj_arr):
        _subj_arr = _subj_arr.numpy()
    _unique_subj = np.unique(_subj_arr)
    from collections import defaultdict as _dd
    _subj_wins: dict = _dd(list)
    for _wi, _s in enumerate(_subj_arr):
        _subj_wins[_s].append(_wi)
    _max_wins = max(len(v) for v in _subj_wins.values())
    _perm: list = []
    for _round in range(_max_wins):
        for _s in _unique_subj:
            if _round < len(_subj_wins[_s]):
                _perm.append(_subj_wins[_s][_round])
    _perm_t = torch.tensor(_perm, dtype=torch.long)

    # Temporarily reorder index_map so both _collect_tokens and
    # _collect_token_metadata walk windows in the same interleaved order.
    _orig = {k: v.clone() if torch.is_tensor(v) else v.copy()
             for k, v in full_ds.index_map.items()}
    for _k, _v in full_ds.index_map.items():
        if torch.is_tensor(_v):
            full_ds.index_map[_k] = _v[_perm_t]
        elif isinstance(_v, np.ndarray):
            full_ds.index_map[_k] = _v[_perm]
    enr_loader = _DataLoader(full_ds, batch_size=val_loader.batch_size,
                             shuffle=False, num_workers=0)

    enr_embeddings, _, _, enr_token_labels, _ = _collect_tokens(
        model, spectral_decoder_trainer, enr_loader, enr_max_tokens)
    print(f"  Collected {len(enr_embeddings)} enrichment tokens")
    enr_token_meta = _collect_token_metadata(
        full_ds, METADATA_FIELDS, len(enr_embeddings), tokens_per_window)

    # Restore original index_map order (full_loader cluster assignment uses it)
    for _k, _v in _orig.items():
        full_ds.index_map[_k] = _v

    # ── Assign full-dataset tokens to codebook clusters ───────────────
    # Stores cluster labels for ALL subjects so the Concept Steering page
    # shows population-level concept-projection distributions (not val-only).
    print("\n[Full-dataset cluster assignment]")
    _centroids = codebook["centroids_emb"].float()   # (K, embed_dim)
    _N_full    = len(enr_embeddings)
    _bs        = 4096
    _full_cl   = np.empty(_N_full, dtype=np.int32)
    for _s in range(0, _N_full, _bs):
        _chunk = enr_embeddings[_s:_s + _bs].float()
        _dists = torch.cdist(_chunk, _centroids)
        _full_cl[_s:_s + _bs] = _dists.argmin(dim=1).numpy()
    print(f"  Assigned {_N_full:,} tokens → {_centroids.shape[0]} clusters")

    # ── Build each section ────────────────────────────────────────────
    cache = {}
    cache["meta"] = meta
    cache["built_at"] = datetime.now().isoformat(timespec="seconds")

    # Full-dataset demographic labels + cluster membership (used by Concept Steering)
    cache["full_token_cluster_labels"] = _full_cl
    cache["full_token_meta"] = {f: arr for f, arr in enr_token_meta.items()}
    if token_labels is not None:
        cache["token_label_classes"] = token_labels.numpy()   # (N,) int, val-set per-token label

    cache.update(_build_feature_section(meta, spectral_decoder_trainer, embeddings, token_labels,
                                         token_meta, enr_embeddings, enr_token_meta))

    if not args.skip_umap:
        cache.update(_build_codebook_section(
            codebook, embeddings, token_labels, token_meta=token_meta))
    else:
        print("\n[Codebook Browser] Skipped (--skip-umap)")

    if args.layer_umap_layers:
        layer_indices = sorted(int(x) for x in args.layer_umap_layers.split(","))
        from mecheeg.sae import SAETrainer
        trainer_tmp = SAETrainer(
            embed_dim=encoder_embed, target_layers=layer_indices, device=DEVICE)
        print(f"\n[Layer UMAPs] Collecting activations for layers {layer_indices} "
              f"(up to {args.max_tokens} tokens) ...")
        gen_lu = get_dataloaders(
            train_path=str(ROOT / data_path), transformer=_transform)
        _, _, val_loader_lu, _ = next(gen_lu)
        multi_layer_acts = trainer_tmp.collect_activations(
            model, val_loader_lu, max_tokens=args.max_tokens)
        cache.update(_build_layer_umap_section(
            multi_layer_acts, layer_indices, codebook))
    else:
        print("\n[Layer UMAPs] Skipped (use --layer-umap-layers to enable)")

    if not args.skip_morphology:
        # Re-load val_loader for data scale estimation (stateless generator)
        gen2 = get_dataloaders(
            train_path=str(ROOT / data_path),
            transformer=_transform,
        )
        _, _, val_loader2, _ = next(gen2)
        cache.update(_build_morphology_section(
            model, codebook, spectral_decoder_trainer, val_loader2,
            patch_size=encoder_patch, target_layer=encoder_layer))
    else:
        print("\n[Morphology Probe] Skipped (--skip-morphology)")

    # ── Save ──────────────────────────────────────────────────────────
    torch.save(cache, cache_path)
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"\n✅  Saved {cache_path}  ({size_mb:.1f} MB)")
    print(f"    Keys: {len(cache)}")
    print(f"    Built at: {cache['built_at']}")


if __name__ == "__main__":
    main()
