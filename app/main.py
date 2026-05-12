
"""SAE4EEG Interactive Explorer — Streamlit app entry point.

Run with:
    uv run --group app streamlit run app/main.py

Selection hierarchy
-------------------
  Model   → pill buttons  (SleepFM | REVE)
  Variant → pill buttons  (Pretrained | Finetuned)
  Layer   → dropdown      (discovered from available checkpoints)

The Feature Explorer works directly from SAE + spectral decoder checkpoints (no pre-built
cache required). Other pages fall back to any matching pre-built app_cache.pt in
results/experiments/.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from scipy.signal import butter, sosfiltfilt

from mecheeg.encoders import MODEL_CARDS

ROOT = Path(__file__).resolve().parent.parent


def _bandpass(x: np.ndarray, fs: float, lo: float = 1.0, hi: float = 40.0) -> np.ndarray:
    """Bandpass filter (C, T) patch along the time axis.
    Uses reflect-padding to reduce edge artefacts on short (128-sample) clips.
    Low cutoff raised to 1 Hz — 0.5 Hz needs >2 s to resolve cleanly."""
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    padlen = min(3 * 12, x.shape[-1] - 1)
    return sosfiltfilt(sos, x, axis=-1, padlen=padlen)


def _eeg_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize (C, T) patch for display: global 95th-percentile scale so
    relative channel amplitudes are preserved (weak channels stay small)."""
    scale = np.percentile(np.abs(x), 95).clip(min=1e-9)
    return x / scale

st.set_page_config(
    page_title="Mechanistic Interpretability for a Large EEG Encoder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
BAND_COLORS = {
    "delta":     "#1f77b4",
    "theta":     "#2ca02c",
    "alpha":     "#ff7f0e",
    "low-beta":  "#d62728",
    "high-beta": "#9467bd",
    "gamma":     "#8c564b",
}
CLINICAL_BANDS = {
    "delta":     (1,  4),
    "theta":     (4,  8),
    "alpha":     (8,  13),
    "low-beta":  (13, 20),
    "high-beta": (20, 30),
    "gamma":     (30, 45),
}
_ENCODER_LABELS = {k: v["display_name"] for k, v in MODEL_CARDS.items()}
_ENCODER_SPECTRAL_DECODER_CFG = {
    "sleepfm":      {"fs": 128, "n_fft": 128},
    "sleepfm_v2.1": {"fs": 128, "n_fft": 128},
    "sleepfm_v2.3": {"fs": 128, "n_fft": 128},
    "sleepfm_v2.4": {"fs": 128, "n_fft": 128},
    "sleepfm_v2.5": {"fs": 128, "n_fft": 128},
    "reve":         {"fs": 200, "n_fft": 200},
}
# HDF5 metadata fields → display names for explorer color options
META_LABELS: Dict[str, str] = {
    "subject_id":       "Subject",
    "age_group":        "Age group",
    "gender":           "Sex / gender",
    "classification":   "Classification",
    "indication_group": "Indication",
    "medication_group": "Medication group",
    "recording_date":   "Recording year",
    "clinic":           "Clinic",
}
# Fields where low cluster diversity flags a confound (everything except Subject itself)
_META_CONFOUND_CHECK_FIELDS = {k for k in META_LABELS if k != "subject_id"}
_META_QUAL_PALETTE = px.colors.qualitative.Plotly

_FEAT_VIZ_DIR = ROOT / "results" / "feature_viz"
_PATCH_SIZE   = 128      # samples per token (1 s @ 128 Hz for SleepFM)

# Channels to show in the tokenizer cluster EEG waveform viewer (name → 0-based index)
_EEG_DISPLAY_CHANNELS: Dict[str, int] = {"F3": 4, "Cz": 11, "O1": 25}
_CHANNEL_NAMES = [
    "Fp1", "Fp2",
    "F9", "F7", "F3", "Fz", "F4", "F8", "F10",
    "T9", "T7", "C3", "Cz", "C4", "T8", "T10",
    "TP7", "TP8",
    "P9", "P7", "P3", "Pz", "P4", "P8", "P10",
    "O1", "O2",
]
_EEG_BANDS = {
    "δ": (0.5,  4,  "rgba(68,119,170,0.15)"),
    "θ": (4,    8,  "rgba(102,204,238,0.15)"),
    "α": (8,   13,  "rgba(34,136,51,0.15)"),
    "β": (13,  30,  "rgba(204,187,68,0.15)"),
    "γ": (30,  60,  "rgba(238,102,119,0.15)"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

# Map folder name -> base encoder. Only finetuned variants are listed.
# `*_local/` folders are local re-trainings that ship the E=1 SAEs missing
# from their canonical siblings (`sleepfm_finetuned/` carries only the E>1
# sweep, `reve_qjbe08/` carries E>=2 only). Pretrained-dashboard folders
# (`sleepfm/`, `reve/`), `*_granular`, and `sleepfm_v2.*` variants are
# intentionally absent.
_ALLOWED_FOLDERS: Dict[str, str] = {
    "sleepfm_finetuned":       "sleepfm",
    "sleepfm_finetuned_local": "sleepfm",
    "labram":                  "labram",
    "reve_qjbe08":             "reve",
    "reve_local":              "reve",
}


def _parse_sae_filename(stem: str) -> Optional[Tuple[float, int, int]]:
    """Parse `sae_<encoder>_exp<E>_k<K>_layer<L>` → (expansion, layer, k).

    Returns None if any field is missing or malformed.
    """
    try:
        expansion = float(stem.split("_exp")[1].split("_")[0])
        layer = int(stem.split("_layer")[1])
        k_val = int(stem.split("_k")[1].split("_")[0])
    except (IndexError, ValueError):
        return None
    return expansion, layer, k_val


def _scan_experiment_metadata() -> Dict[Tuple[str, float, int, int], str]:
    """Map (encoder, expansion, layer, k) → experiment name.

    Walks `results/experiments/*/metadata.json` once and records each
    one's (encoder × `sae_checkpoint`-parsed filename) tuple. Keyed by
    the selection tuple rather than the absolute SAE path so the lookup
    works whether the SAE lives in the canonical folder, in a re-training
    folder like `sleepfm_finetuned_local/`, or has been moved.
    """
    out: Dict[Tuple[str, float, int, int], str] = {}
    exp_root = ROOT / "results" / "experiments"
    if not exp_root.exists():
        return out
    for exp_dir in sorted(exp_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        meta_path = exp_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            rel = meta.get("sae_checkpoint")
            encoder = meta.get("encoder")
            if not rel or not encoder:
                continue
            stem = Path(rel).stem
            parsed = _parse_sae_filename(stem)
            if parsed is None:
                continue
            E, L, K = parsed
            out.setdefault((encoder, E, L, K), exp_dir.name)
        except Exception:
            continue
    return out


def discover_runs() -> Tuple[
    Dict[str, Dict[float, Dict[int, Dict[int, Path]]]],
    Dict[Tuple[str, float, int, int], Optional[str]],
    Dict[Tuple[str, float, int, int], str],
]:
    """Scan `results/features/` and return all available SAE checkpoints.

    Returns
    -------
    runs        : {encoder: {expansion: {layer: {k: sae_path}}}}
    exp_for     : {(encoder, expansion, layer, k): experiment_name or None}
    folder_for  : {(encoder, expansion, layer, k): folder_name}
    """
    runs: Dict[str, Dict[float, Dict[int, Dict[int, Path]]]] = {}
    exp_for: Dict[Tuple[str, float, int, int], Optional[str]] = {}
    folder_for: Dict[Tuple[str, float, int, int], str] = {}

    features_root = ROOT / "results" / "features"
    if not features_root.exists():
        return runs, exp_for, folder_for

    selection_to_exp = _scan_experiment_metadata()

    for folder in sorted(features_root.iterdir()):
        if not folder.is_dir() or folder.name not in _ALLOWED_FOLDERS:
            continue
        encoder = _ALLOWED_FOLDERS[folder.name]
        for ckpt in sorted(folder.glob("sae_*.pt")):
            parsed = _parse_sae_filename(ckpt.stem)
            if parsed is None:
                continue
            expansion, layer, k_val = parsed
            runs.setdefault(encoder, {}).setdefault(expansion, {}).setdefault(layer, {})
            key = (encoder, expansion, layer, k_val)
            if k_val not in runs[encoder][expansion][layer]:
                runs[encoder][expansion][layer][k_val] = ckpt
                exp_for[key] = selection_to_exp.get(key)
                folder_for[key] = folder.name

    # Sort each level
    for enc in runs:
        runs[enc] = {
            E: {layer: dict(sorted(ks.items())) for layer, ks in sorted(layers.items())}
            for E, layers in sorted(runs[enc].items())
        }
    return runs, exp_for, folder_for


def _spectral_decoder_path(folder_name: str) -> Optional[Path]:
    # The XAE module was renamed to SpectralDecoder; the disk layout used by
    # train_spectral_decoder.py writes to results/spectral_decoder/, while the
    # currently deployed checkpoints (pulled from gs://sae4eeg-app-assets) still
    # live under the original results/xae/ tree. Check both.
    candidates = [
        ROOT / "results" / "spectral_decoder" / folder_name / "spectral_decoder_checkpoint.pt",
        ROOT / "results" / "spectral_decoder" / "spectral_decoder_checkpoint.pt",  # legacy
        ROOT / "results" / "xae" / folder_name / "xae_checkpoint.pt",
        ROOT / "results" / "xae" / "xae_checkpoint.pt",
    ]
    return next((p for p in candidates if p.exists()), None)


def _app_cache_path(exp_name: Optional[str]) -> Optional[Path]:
    """Return the app_cache.pt for the named experiment, or None."""
    if not exp_name:
        return None
    p = ROOT / "results" / "experiments" / exp_name / "app_cache.pt"
    return p if p.exists() else None


def _tcav_cache_path(exp_name: Optional[str]) -> Optional[Path]:
    """Return the tcav_cache.pt for the named experiment, or None."""
    if not exp_name:
        return None
    p = ROOT / "results" / "tcav" / exp_name / "tcav_cache.pt"
    return p if p.exists() else None


def _attention_cache_path(exp_name: Optional[str]) -> Optional[Path]:
    """Return the attention_cache.pt for the named experiment, or None."""
    if not exp_name:
        return None
    p = ROOT / "results" / "experiments" / exp_name / "attention_cache.pt"
    return p if p.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (Streamlit resource-cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Computing spectral features …")
def load_run_data(sae_path: str, spectral_decoder_path: Optional[str]) -> dict:
    """
    Load SAE + spectral decoder and compute per-feature spectral profiles.
    All heavy maths happen here; the UI only calls this once per selection.
    """
    from mecheeg.sae import SparseAutoencoder
    from mecheeg.spectral_decoder import SpectralDecoderTrainer

    sae_ckpt   = torch.load(sae_path, map_location="cpu", weights_only=False)
    embed_dim  = sae_ckpt.get("embed_dim", 128)
    expansion  = sae_ckpt.get("expansion", 1.0)
    k          = sae_ckpt.get("k", 8)
    layer      = sae_ckpt.get("target_layer", 2)
    encoder    = sae_ckpt.get("encoder", "sleepfm")

    sae = SparseAutoencoder(embed_dim, expansion=expansion, mode="topk", k=k)
    sae.load_state_dict(sae_ckpt["sae_state_dict"])
    sae.eval()

    act_mean  = sae_ckpt["act_mean"]
    act_std   = sae_ckpt["act_std"]
    W_dec     = sae.decoder.weight.T.detach()    # (n_features, embed_dim)
    n_features = W_dec.shape[0]

    data = {
        "n_features": n_features,
        "embed_dim":  embed_dim,
        "expansion":  expansion,
        "k":          k,
        "layer":      layer,
        "encoder":    encoder,
        "has_spectral_decoder":    False,
    }

    if spectral_decoder_path and Path(spectral_decoder_path).exists():
        cfg         = _ENCODER_SPECTRAL_DECODER_CFG.get(encoder, {"fs": 128, "n_fft": 128})
        spectral_decoder_trainer = SpectralDecoderTrainer(embed_dim=embed_dim, **cfg)
        spectral_decoder_trainer.load(spectral_decoder_path)
        spectral_decoder_trainer.spectral_decoder.eval()

        # Baseline: average token (no feature active)
        baseline   = act_mean.unsqueeze(0)                         # (1, E)
        # Feature ON: baseline shifted by each decoder direction
        activated  = baseline + 2.0 * (W_dec * act_std)           # (n_feat, E)

        amp_base, _, _ = spectral_decoder_trainer.decode_direction(baseline,  denormalise=True)
        amp_act,  _, _ = spectral_decoder_trainer.decode_direction(activated, denormalise=True)
        amp_base = amp_base.squeeze(0).cpu()
        amp_act  = amp_act.cpu()
        amp_diff = amp_act - amp_base.unsqueeze(0)                 # (n_feat, n_bins)

        freqs      = np.array(spectral_decoder_trainer.spectral.freqs)
        band_names = list(CLINICAL_BANDS.keys())
        band_deltas = np.zeros((n_features, len(band_names)), dtype=np.float32)
        for j, (_, (f_lo, f_hi)) in enumerate(CLINICAL_BANDS.items()):
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            if mask.sum() > 0:
                band_deltas[:, j] = amp_diff[:, mask].mean(dim=1).numpy()

        data.update({
            "has_spectral_decoder":              True,
            "feature_amp_diff":     amp_diff,             # (n_feat, n_bins)
            "feature_amp_baseline": amp_base,             # (n_bins,)
            "feature_freqs":        freqs,
            "feature_band_deltas":  torch.tensor(band_deltas),
            "feature_band_names":   band_names,
        })

    return data


@st.cache_resource(show_spinner="Loading app cache …")
def load_app_cache(path: str) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


@st.cache_resource(show_spinner="Loading TCAV cache …")
def load_tcav_cache(path: str) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


@st.cache_resource(show_spinner="Loading attention cache …")
def load_attention_cache(path: str) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


@st.cache_data(show_spinner="Loading layer UMAP cache …", ttl=3600)
def load_layer_umap_cache(path: str) -> dict:
    cache = torch.load(path, weights_only=False, map_location="cpu")
    # Expose token_positions as a token_meta field so it can be used as a colour axis
    if "token_positions" in cache and "token_meta" in cache:
        if "token_position" not in cache["token_meta"]:
            cache["token_meta"]["token_position"] = np.array(cache["token_positions"])
    return cache


@st.cache_data(show_spinner=False)
def _load_eeg_patch(hdf5_path: str, local_idx: int, token_pos: int) -> np.ndarray:
    """Load a 1-s EEG patch (C, 128) from an HDF5 file.

    token_pos is the token offset within the window (in token units, not samples).
    The corresponding sample start = token_pos * _PATCH_SIZE.
    """
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        window = f["data"][local_idx]          # (C, T)
    C, T = window.shape
    start = min(int(token_pos) * _PATCH_SIZE, T - _PATCH_SIZE)
    start = max(start, 0)
    return window[:, start : start + _PATCH_SIZE].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.title("Mechanistic Interpretability for EEG Foundation Models")
    st.sidebar.markdown("---")

    runs_all, exp_for, folder_for = discover_runs()
    if not runs_all:
        st.sidebar.error("No SAE checkpoints found under results/features/")
        st.stop()

    # ── Data-availability filter ──────────────────────────────────────
    # By default, hide configurations that have no app_cache.pt — most
    # tabs are uninteresting without it. The actual toggle widget is
    # rendered at the bottom of the sidebar under "Advanced" so reviewers
    # see the model selectors first; we read its persisted state here.
    only_with_cache = st.session_state.get("filter_with_cache", True)

    def _filter_runs(src: dict) -> dict:
        if not only_with_cache:
            return src
        out: dict = {}
        for enc, by_E in src.items():
            for E, by_L in by_E.items():
                for L, by_K in by_L.items():
                    for K in by_K:
                        exp_name = exp_for.get((enc, E, L, K))
                        if exp_name and (
                            ROOT / "results" / "experiments" / exp_name / "app_cache.pt"
                        ).exists():
                            out.setdefault(enc, {}).setdefault(E, {}).setdefault(L, {})[K] = by_K[K]
        return out

    runs = _filter_runs(runs_all)
    if not runs:
        st.sidebar.warning(
            "No configs have a built `app_cache.pt` yet. Toggle off "
            "**Only configs with data** to browse SAE checkpoints without "
            "downstream caches."
        )
        runs = runs_all

    # ── Encoder ───────────────────────────────────────────────────────
    _ENC_LABELS = {"sleepfm": "SleepFM", "reve": "REVE", "labram": "LaBraM"}
    _ENC_ORDER  = ["sleepfm", "reve", "labram"]
    enc_keys    = [e for e in _ENC_ORDER if e in runs]
    enc_display = [_ENC_LABELS[e] for e in enc_keys]

    st.sidebar.markdown("**Model**")
    enc_choice = st.sidebar.pills(
        "Model", enc_display,
        default=enc_display[0],
        label_visibility="collapsed",
        key="enc_pills",
    )
    encoder = enc_keys[enc_display.index(enc_choice)]

    # ── Helpers ───────────────────────────────────────────────────────
    # Subtle row of snap-point labels under a select_slider. The slider's
    # own value indicator handles "which one is selected", so this row is
    # purely informational — kept light/muted to avoid visual clutter.
    def _tick_row(labels: list[str]) -> None:
        html = (
            '<div style="display:flex;justify-content:space-between;'
            'font-size:0.68rem;color:rgba(150,150,150,0.75);'
            'padding:0 8px;margin:-4px 0 10px;letter-spacing:0.02em;">'
            + "".join(f"<span>{lbl}</span>" for lbl in labels)
            + "</div>"
        )
        st.sidebar.markdown(html, unsafe_allow_html=True)

    # ── Expansion ─────────────────────────────────────────────────────
    expansions = sorted(runs[encoder].keys())
    def _fmt_E(E: float) -> str:
        return f"{int(E)}×" if float(E).is_integer() else f"{E}×"

    st.sidebar.markdown("**Expansion**")
    if len(expansions) >= 2:
        _E_default = 1.0 if 1.0 in expansions else expansions[0]
        expansion = st.sidebar.select_slider(
            "Expansion", expansions,
            value=_E_default,
            format_func=_fmt_E,
            label_visibility="collapsed",
            key="E_slider",
        )
        _tick_row([_fmt_E(E) for E in expansions])
    else:
        expansion = expansions[0]
        st.sidebar.caption(f"Only {_fmt_E(expansion)} available for {enc_choice}")

    # ── Layer ─────────────────────────────────────────────────────────
    # Some (encoder, E) combinations only have a subset of layers — that's
    # surfaced naturally by the dropdown options.
    layers = sorted(runs[encoder][expansion].keys())

    st.sidebar.markdown("**Layer**")
    layer = st.sidebar.selectbox(
        "Layer", layers,
        format_func=lambda L: f"Layer {L}",
        label_visibility="collapsed",
        key="layer_sel",
    )

    # ── k (sparsity) ──────────────────────────────────────────────────
    # k is exposed as a sparsity percentage (k / n_features) so the user
    # picks the desired sparsity rather than a raw integer. Most encoders
    # ship one k for every E (a fixed low k=8); SleepFM at layer 2 also
    # has higher-k "k-scaled" variants.
    _EMBED_DIM: Dict[str, int] = {"sleepfm": 128, "labram": 200, "reve": 512}
    embed_dim = _EMBED_DIM.get(encoder, 128)
    n_features = int(embed_dim * float(expansion))

    k_options = sorted(runs[encoder][expansion][layer].keys())
    def _fmt_k_pct(k: int) -> str:
        return f"{100.0 * k / n_features:.2f}%"

    st.sidebar.markdown("**Sparsity**")
    if len(k_options) >= 2:
        k_sel = st.sidebar.select_slider(
            "Sparsity", k_options,
            value=k_options[0],
            format_func=_fmt_k_pct,
            label_visibility="collapsed",
            key="k_slider",
        )
        _tick_row([_fmt_k_pct(k) for k in k_options])
    else:
        k_sel = k_options[0]
        st.sidebar.caption(f"{_fmt_k_pct(k_sel)} (k = {k_sel})")

    selection_key = (encoder, expansion, layer, k_sel)
    sae_path    = str(runs[encoder][expansion][layer][k_sel])
    folder_name = folder_for[selection_key]
    exp_name    = exp_for[selection_key]
    spectral_decoder_p = _spectral_decoder_path(folder_name)

    # ── Model overview ────────────────────────────────────────────────
    st.sidebar.markdown("---")
    try:
        meta      = torch.load(sae_path, map_location="cpu", weights_only=False)
        meta_E    = meta.get("expansion", expansion)
        meta_k    = meta.get("k", k_sel)
        edim      = meta.get("embed_dim", "?")
        n_feat    = (
            int(float(edim) * float(meta_E))
            if isinstance(edim, (int, float)) and isinstance(meta_E, (int, float))
            else "?"
        )
    except Exception:
        meta_E = expansion
        meta_k = k_sel
        edim = n_feat = "?"

    if exp_name:
        st.sidebar.caption(f"`{exp_name}`")
    else:
        st.sidebar.caption(f"`{folder_name}` (no experiment metadata)")

    if isinstance(meta_k, (int, float)) and isinstance(n_feat, int) and n_feat > 0:
        sparsity_pct = 100.0 * float(meta_k) / n_feat
        sparsity_lbl = f"{sparsity_pct:.2f}%"
    else:
        sparsity_lbl = "?"

    ca, cb = st.sidebar.columns(2)
    ca.metric("Features",  f"{n_feat:,}" if isinstance(n_feat, int) else n_feat)
    cb.metric("Embed dim", str(edim))
    cc, cd = st.sidebar.columns(2)
    cc.metric("Expansion", f"{meta_E}×")
    cd.metric("Sparsity",  sparsity_lbl, help=f"Top-k = {meta_k}")

    def _dot(ok: bool) -> str:
        return "🟢" if ok else "🟠"

    has_cache = _app_cache_path(exp_name) is not None
    has_tcav  = _tcav_cache_path(exp_name) is not None
    has_attn  = _attention_cache_path(exp_name) is not None
    st.sidebar.markdown(
        f"{_dot(spectral_decoder_p is not None)} spectral decoder &nbsp;&nbsp; "
        f"{_dot(has_cache)} Cache &nbsp;&nbsp; "
        f"{_dot(has_tcav)} TCAV &nbsp;&nbsp; "
        f"{_dot(has_attn)} Attn",
        unsafe_allow_html=True,
    )

    # ── Page navigation ───────────────────────────────────────────────
    st.sidebar.markdown("---")
    _PAGES = [
        "Home",
        "Feature Explorer",
        "Layer Explorer",
        "TCAV Explorer",
        "Concept Steering",
        "Taxonomy & Steering",
        "Attention Explorer",
    ]
    page = st.sidebar.radio(
        "Page",
        _PAGES,
        label_visibility="collapsed",
        key="page_nav",
    )

    # ── Advanced (data-availability filter) ────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar.expander("Advanced"):
        st.toggle(
            "Only configs with data",
            value=True,
            help=(
                "When on, the selectors above only show (encoder, expansion, "
                "layer, sparsity) combinations that have a precomputed "
                "`app_cache.pt`. Turn off to see every trained SAE on disk."
            ),
            key="filter_with_cache",
        )

    # ── Version ───────────────────────────────────────────────────────
    try:
        import tomllib
        from datetime import date
        with open(ROOT / "pyproject.toml", "rb") as _f:
            _ver = tomllib.load(_f)["project"]["version"]
        _date = date.today().strftime("%Y-%m-%d")
    except Exception:
        _ver = "?"
        _date = ""
    st.sidebar.markdown("---")
    st.sidebar.caption(f"v{_ver} · {_date}")

    return (
        sae_path,
        str(spectral_decoder_p) if spectral_decoder_p else None,
        folder_name,
        exp_name,
        layer,
        encoder,
        page,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 0 — Home
# ─────────────────────────────────────────────────────────────────────────────

def page_home() -> None:
    st.title("Mechanistic Interpretability for EEG Foundation Models")
    st.caption("Anonymous Authors · Affiliations withheld for anonymous review")
    st.markdown(
        "EEG foundation models achieve state-of-the-art clinical performance, yet their "
        "internal computations remain opaque. We apply TopK Sparse Autoencoders across "
        "SleepFM, REVE, and LaBraM to extract interpretable feature dictionaries grounded "
        "in a clinical taxonomy, and use **concept steering** with a target-vs-off-target "
        "metric to expose three regimes — selectively steerable, encoded-but-entangled, "
        "and non-encoded. A **spectral decoder** maps the interventions back to amplitude "
        "spectra, turning latent manipulations into physiologically interpretable "
        "frequency signatures."
    )

    st.divider()

    # ── Reading the paper alongside the app ──────────────────────────────────
    st.subheader("Reading the paper alongside the app")
    st.markdown(
        "Each tab below corresponds to a section or figure of the paper. The sidebar "
        "drives all tabs: pick a model + expansion + layer, and every tab updates."
    )

    tabs_map = [
        ("Feature Explorer",      "§3.1",
         "Per-feature spectral signatures via the spectral decoder",
         "Figure 6"),
        ("Layer Explorer",        "§3",
         "Animated joint UMAP showing token trajectories across encoder layers",
         "Figure 2"),
        ("TCAV Explorer",         "§4 · Fig 4",
         "Concept attribution via Testing with Concept Activation Vectors",
         "Figure 4"),
        ("Concept Steering",      "§5–6 · Figs 4 / 5 / 6",
         "Clamping concept-aligned features to the target centroid (single config, hands-on)",
         "Figure 5"),
        ("Taxonomy & Steering",   "§3.2 · Fig 2 + §3.3 · Fig 3 + §5–6 · Figs 4–6",
         "Paper figures + interactive Fig 5 (any encoder × layer × concept)",
         "Figure 3"),
        ("Attention Explorer",    "supplementary",
         "Encoder self-attention alongside SAE features (not in paper)",
         None),
    ]

    st.markdown(
        """
<style>
.home-tab-ref {
    display: inline-block;
    background: rgba(120, 120, 140, 0.10);
    color: #666;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.78rem;
    letter-spacing: 0.02em;
    margin-bottom: 0.6rem;
}
.home-tab-title {
    font-size: 1.35rem;
    font-weight: 600;
    margin: 0.1rem 0 0.35rem 0;
    line-height: 1.25;
}
.home-tab-blurb {
    color: #555;
    font-size: 0.95rem;
    line-height: 1.45;
    margin-bottom: 0.9rem;
}
.home-tab-placeholder {
    aspect-ratio: 16 / 9;
    border-radius: 8px;
    background: linear-gradient(135deg, #f3f3f6 0%, #e7e7ee 100%);
    display: flex; align-items: center; justify-content: center;
    color: #aaa; font-size: 0.85rem; font-style: italic;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stImage"] img {
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
</style>
""",
        unsafe_allow_html=True,
    )

    _home_thumb_dir = ROOT / "tools" / "paper_figures"
    for title, paper_ref, blurb, thumb_fig in tabs_map:
        with st.container(border=True):
            cols = st.columns([0.42, 0.58], gap="large", vertical_alignment="center")
            with cols[0]:
                thumb_path = (_home_thumb_dir / thumb_fig / "figure.png") if thumb_fig else None
                if thumb_path and thumb_path.exists():
                    st.image(str(thumb_path), use_container_width=True)
                else:
                    st.markdown(
                        "<div class='home-tab-placeholder'>No paper figure</div>",
                        unsafe_allow_html=True,
                    )
            with cols[1]:
                st.markdown(
                    f"<span class='home-tab-ref'>{paper_ref}</span>"
                    f"<div class='home-tab-title'>{title}</div>"
                    f"<div class='home-tab-blurb'>{blurb}</div>",
                    unsafe_allow_html=True,
                )
                if st.button(f"Open {title} →",
                             key=f"home_nav_{title}",
                             use_container_width=False):
                    st.session_state["_page_nav_pending"] = title
                    st.rerun()
        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

    st.caption(
        "Default selection is the paper's primary experiment: **SleepFM, E=1, layer 2**. "
        "Caches (TCAV, attention, layer-UMAP) are pre-built for the canonical configurations; "
        "tabs that require a missing cache show a clear build hint."
    )

    # ── Glossary ───────────────────────────────────────────────────────────────
    with st.expander("Glossary"):
        st.markdown(
            """
| Term | Meaning |
|------|---------|
| **SAE** | Sparse Autoencoder — a bottleneck network trained on encoder representations to find a sparse, over-complete dictionary of interpretable features. |
| **Spectral decoder** | A small spectral decoder that maps encoder token embeddings to amplitude + phase per EEG frequency band. |
| **App cache** | Pre-computed `.pt` file containing UMAP coordinates, cluster assignments, morphology embeddings, and other heavy computations. Built by `build_app_cache.py`. |
| **TCAV cache** | Pre-computed concept activation vectors and per-feature TCAV scores. Built by `run_tcav.py`. |
| **Top-k SAE** | SAE variant where exactly *k* features are active per token (hard sparsity). |
| **Fire rate** | Fraction of tokens for which a given SAE feature is non-zero. |
| **Dominant band** | EEG frequency band with the largest positive amplitude change when the feature fires. |
| **CAV** | Concept Activation Vector — a linear classifier trained to separate concept tokens from random tokens in encoder activation space. |
| **TCAV score (Variant A)** | Weight-space alignment: fraction of k-fold CAVs where the SAE encoder direction and the CAV point in the same direction. Binary per feature (0 or 1). |
| **TCAV score (Variant C)** | Kim et al. formulation: fraction of concept examples where the model-level directional derivative w.r.t. the concept is positive. Continuous and example-dependent. |
| **Δ fire rate** | Difference in firing rate between concept tokens and the baseline. Positive = feature fires more on concept EEG. |
| **BH-adjusted p** | Benjamini–Hochberg FDR-corrected p-value. |
| **Label correlation** | Pearson *r* between a feature's per-token activation and the binary normal/abnormal label. |
| **Token** | One patch of EEG processed by the encoder. SleepFM: 5 s per channel-averaged patch. REVE: 1 s per single-channel patch (19 channels). LaBraM: 1 s patches across 19 channels. |
| **UMAP** | Uniform Manifold Approximation and Projection — non-linear dimensionality reduction used to visualise high-dimensional token embeddings in 2D. |
| **Attention** | Transformer self-attention weights, inspected per layer. High *attention received* (column sum) indicates tokens the model treats as contextually important. |
| **Concept steering** | Intervention that zeros out SAE features aligned with a concept to test whether the concept is causally encoded by those features. |
| **Feature taxonomy** | Three-way classification of SAE features: *separable* (one clear spectral pattern), *entangled* (multiple mixed patterns), or *spurious* (noise / dead). |
"""
        )

    # ── Pipeline overview ──────────────────────────────────────────────────────
    with st.expander("Pipeline overview"):
        st.markdown(
            """
```
EEG window  →  Encoder (frozen)  →  layer activations  →  TopK SAE  →  sparse features
                                                                          │
                                                ┌─────────────────────────┼───────────────────────┐
                                                ▼                         ▼                       ▼
                                       Spectral Decoder            TCAV (CAV probe)        Concept Steering
                                       (per-feature                (concept attribution    (clamp top features
                                        spectral signature)         per concept)            to donor centroid)
```

| Stage | Method | Paper § | App tab |
|-------|--------|---------|---------|
| **Encoder** | SleepFM / LaBraM / REVE, all binary-finetuned on BrainCapture normal/abnormal | §2 | (sidebar selector) |
| **SAE** | TopK Sparse Autoencoder over layer activations, sweep over expansion E ∈ {1, 2, 4, 8, 16, 32, 64} and sparsity k | §2.1 | (sidebar selector) |
| **Spectral signatures** | Linear decoder mapping each encoder token to amplitude per frequency band; per-feature signatures from gradient with respect to active features | §3.1 | Feature Explorer |
| **Taxonomy** | Three-way split (separable / entangled / spurious) based on spectral coherence + co-firing structure, sweep across encoder × layer × expansion | §3.2–3.3 · Figs 2, 3 | Taxonomy & Steering |
| **TCAV** | Linear concept activation vectors trained on encoder activations; Variant C (Kim et al.) for headline scores | §4 · Fig 4 | TCAV Explorer |
| **Concept Steering** | Clamp top-N TCAV-aligned features to the donor centroid; target / off-target AUROC drop with random-direction baselines | §5–6 · Figs 5, 6 | Concept Steering · Taxonomy & Steering |

**Status indicators in the sidebar:** green dot = artifact exists on disk; orange = missing (build buttons appear in the relevant tab).
"""
        )

    # ── Model Cards ────────────────────────────────────────────────────────────
    with st.expander("Model Cards"):
        _model_cards_content()


def _model_cards_content() -> None:
    st.title("Model Cards")
    st.caption(
        "The paper studies three EEG foundation encoders. All are binary-finetuned "
        "on the BrainCapture normal/abnormal classification task before SAE training."
    )

    # Paper encoders: SleepFM (v1.1, 3 layers, the original SetTransformer),
    # LaBraM-Base (12 layers), REVE (22 layers).
    paper_models = [
        ("sleepfm", "SleepFM"),
        ("labram",  "LaBraM"),
        ("reve",    "REVE"),
    ]

    for key, header in paper_models:
        card  = MODEL_CARDS[key]
        specs = card["specs"]

        st.markdown("---")
        st.markdown(f"### {header}")

        has_patch = "patch_size" in specs
        n_cols = 7 if has_patch else 6
        cols = st.columns(n_cols)
        cols[0].metric("Embed dim",   specs["embed_dim"])
        cols[1].metric("Layers",      specs["layers"])
        cols[2].metric("Tokenizer",   specs["tokenizer"])
        idx = 3
        if has_patch:
            patch_label = f"{specs['patch_size'] // specs['sample_rate_hz']} s"
            cols[idx].metric("Patch size", patch_label)
            idx += 1
        cols[idx].metric("Sample rate", f"{specs['sample_rate_hz']} Hz"); idx += 1
        cols[idx].metric("Pos. emb.",   "Yes" if specs["position_emb"] else "No"); idx += 1
        cols[idx].metric("Optimizer",   specs["optimizer"])

        st.markdown(f"**Reconstruction:** {specs['reconstruction']}")
        st.markdown(f"**Pretraining:** {card['pretraining']}")
        st.markdown(f"**{card['notes_label']}:** {card['notes']}")


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Feature Explorer
# ─────────────────────────────────────────────────────────────────────────────

def page_features(data: dict, app_cache: Optional[dict], folder_name: str, layer: int,
                  exp_name: Optional[str] = None,
                  attn_cache: Optional[dict] = None) -> None:
    enc_label = _ENCODER_LABELS.get(data["encoder"], data["encoder"].upper())
    st.title("Feature Explorer")
    st.caption("**Paper §3.1** — per-feature spectral signatures via the spectral decoder.")
    st.markdown(
        f"SAE features learned on **{enc_label} layer {data['layer']}** activations. "
        "Each feature is a direction in embedding space; its **spectral signature** "
        "shows which EEG frequency bands it boosts or suppresses."
    )

    if not data["has_spectral_decoder"]:
        st.warning(
            "No spectral decoder checkpoint found for this run — spectral visualisations unavailable. "
            "Train a spectral decoder first:  `uv run tools/train_spectral_decoder.py --encoder "
            f"{data['encoder']} --tag ...`"
        )
        return

    n_features   = data["n_features"]
    band_names   = data["feature_band_names"]
    band_deltas  = data["feature_band_deltas"].numpy()   # (n_feat, n_bands)
    amp_diff     = data["feature_amp_diff"].numpy()      # (n_feat, n_bins)
    amp_base     = data["feature_amp_baseline"].numpy()  # (n_bins,)
    freqs        = data["feature_freqs"]                 # (n_bins,)

    # Optional rich stats from pre-built cache
    has_cache    = app_cache is not None
    feature_stats  = app_cache.get("feature_stats",       []) if has_cache else []
    feature_expls  = app_cache.get("feature_explanations", []) if has_cache else []
    cooccurrence   = (
        app_cache["feature_cooccurrence"].numpy()
        if has_cache and "feature_cooccurrence" in app_cache
        else None
    )
    dash_imgs      = app_cache.get("feature_dashboard_imgs", {}) if has_cache else {}

    # Load TCAV cache early so abnormality score can be used as a sort key
    tcav_scores_per_feature: dict[int, float] = {}
    _tcav_p = _tcav_cache_path(exp_name) if folder_name else None
    if _tcav_p:
        try:
            _tc = load_tcav_cache(str(_tcav_p))
            if _tc and "tcav_scores" in _tc and "concept_names" in _tc:
                _cnames = _tc["concept_names"]
                if "abnormal" in _cnames:
                    _ab_idx = _cnames.index("abnormal")
                    _ab_scores = _tc["tcav_scores"][_ab_idx]  # (n_features,)
                    for _fi, _sc in enumerate(_ab_scores.tolist()):
                        tcav_scores_per_feature[_fi] = float(_sc)
        except Exception:
            pass
    has_tcav_scores = bool(tcav_scores_per_feature)

    # Load prototype data early so selectivity can be used as a sort key
    _proto_run_name = _latest_proto_run(folder_name or "", layer) if folder_name else None
    _proto_data     = _load_prototypes(_proto_run_name) if _proto_run_name else None
    has_proto       = _proto_data is not None
    sel_scores:    dict[int, float] = {}
    joint_scores:  dict[int, float] = {}
    abnorm_ratios: dict[int, float] = {}
    real_example_idxs: set[int] = set()

    # Abnormal ratio from attention cache (independent of proto)
    if attn_cache is not None and "top_window_labels" in attn_cache:
        _twl = attn_cache["top_window_labels"]  # (n_features, K)
        for _fi in range(int(attn_cache["n_features"])):
            _lbls = _twl[_fi].float().numpy()
            abnorm_ratios[_fi] = float(np.mean(_lbls > 0.5))
    if has_proto:
        _fi_list  = _proto_data["feature_indices"].tolist()
        _sel_hist = _proto_data["selectivity_hists"]
        for _ii, _fi in enumerate(_fi_list):
            sel_scores[_fi] = float(_sel_hist[_ii, -1])
        _tj = _proto_data.get("top_joint_scores", None)
        if _tj is not None:
            for _ii, _fi in enumerate(_fi_list):
                joint_scores[_fi] = float(_tj[_ii])
        _ex = _proto_data.get("example_patches", None)
        if _ex is not None:
            # shape (N, n_examples, C, P) — has real examples if any patch is non-zero
            for _ii, _fi in enumerate(_fi_list):
                if _ex[_ii].any():
                    real_example_idxs.add(int(_fi))

    # Load metadata enrichment early for sorting
    _all_meta_enr = (app_cache or {}).get("feature_meta_enrichment", [])
    _meta_enr_max: dict[str, dict[int, float]] = {"age_group": {}, "gender": {}}
    for _fi, _fenr in enumerate(_all_meta_enr):
        for _field in ("age_group", "gender"):
            _cats = _fenr.get(_field, [])
            _meta_enr_max[_field][_fi] = max((r for _, r, _ in _cats), default=float("nan"))

    def _stat(i: int, key: str, default=float("nan")):
        return feature_stats[i].get(key, default) if i < len(feature_stats) else default

    def _expl(i: int, key: str, default=""):
        return feature_expls[i].get(key, default) if i < len(feature_expls) else default

    # Derive dominant band from spectral decoder band deltas (always available)
    dominant_band = [
        band_names[int(np.argmax(np.abs(band_deltas[i])))]
        for i in range(n_features)
    ]

    # ── Controls ──────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        sort_options = ["Feature index", "Dominant band"]
        if has_cache:
            sort_options += ["Fire rate ↓", "Mean activation ↓", "Max activation ↓",
                             "Label correlation ↓", "Label correlation ↑"]
        if has_tcav_scores:
            sort_options.append("TCAV abnormal ↓")
        if has_proto:
            sort_options.append("Selectivity ↓")
            if joint_scores:
                sort_options.append("Attention score ↓")
        if abnorm_ratios:
            sort_options.append("Abnormal ratio ↓")
        if _all_meta_enr:
            sort_options += ["Age enrichment ↓", "Sex enrichment ↓"]
        _default_sort = (
            "Selectivity ↓" if has_proto
            else "TCAV abnormal ↓" if has_tcav_scores
            else "Fire rate ↓" if has_cache
            else "Feature index"
        )
        sort_by = st.selectbox("Sort features by", sort_options,
                               index=sort_options.index(_default_sort))
    with col_ctrl2:
        filter_band = st.selectbox("Filter by dominant band", ["All"] + band_names)
    with col_ctrl3:
        sig_only = st.checkbox(
            "Significant only (p < 0.05)",
            value=False,
            disabled=not has_cache,
            help="Requires pre-built cache with label correlation statistics.",
        )

    # ── Build rows ────────────────────────────────────────────────────
    rows = []
    for i in range(n_features):
        desc = _expl(i, "description") or f"Feature {i}"
        rows.append({
            "idx":       i,
            "fire_rate": _stat(i, "fire_rate_pct"),
            "corr":      _stat(i, "label_correlation"),
            "p":         _stat(i, "label_p_value", default=1.0),
            "band":        dominant_band[i],
            "description": desc,
            "cluster":     _expl(i, "cluster", default=-1),
            "mean_act":      _stat(i, "mean_activation"),
            "max_act":       _stat(i, "max_activation"),
            "selectivity":   sel_scores.get(i, float("nan")),
            "joint_score":   joint_scores.get(i, float("nan")),
            "abnorm_ratio":  abnorm_ratios.get(i, float("nan")),
            "tcav_abnormal": tcav_scores_per_feature.get(i, float("nan")),
            "age_enr":       _meta_enr_max["age_group"].get(i, float("nan")),
            "sex_enr":       _meta_enr_max["gender"].get(i, float("nan")),
        })

    if filter_band != "All":
        rows = [r for r in rows if r["band"] == filter_band]
    if sig_only and has_cache:
        rows = [r for r in rows if r["p"] < 0.05]

    sort_map = {
        "Feature index":       lambda r: r["idx"],
        "Dominant band":       lambda r: (r["band"], r["idx"]),
        "Fire rate ↓":         lambda r: -r["fire_rate"] if not np.isnan(r["fire_rate"]) else 0,
        "Mean activation ↓":   lambda r: -r["mean_act"] if not np.isnan(r["mean_act"]) else 0,
        "Max activation ↓":    lambda r: -r["max_act"]  if not np.isnan(r["max_act"])  else 0,
        "Label correlation ↓": lambda r: r["corr"] if not np.isnan(r["corr"]) else 0,
        "Label correlation ↑": lambda r: -r["corr"] if not np.isnan(r["corr"]) else 0,
        "TCAV abnormal ↓":     lambda r: -r["tcav_abnormal"] if not np.isnan(r["tcav_abnormal"]) else 0,
        "Selectivity ↓":       lambda r: -r["selectivity"] if not np.isnan(r["selectivity"]) else 0,
        "Attention score ↓":   lambda r: -r["joint_score"] if not np.isnan(r["joint_score"]) else 0,
        "Abnormal ratio ↓":    lambda r: -r["abnorm_ratio"] if not np.isnan(r["abnorm_ratio"]) else 0,
        "Age enrichment ↓":    lambda r: -r["age_enr"] if not np.isnan(r["age_enr"]) else 0,
        "Sex enrichment ↓":    lambda r: -r["sex_enr"] if not np.isnan(r["sex_enr"]) else 0,
    }
    rows.sort(key=sort_map[sort_by])

    st.markdown(f"**{len(rows)} features** matching filters")

    # ── Feature selector ──────────────────────────────────────────────
    def _feat_option(r: dict) -> str:
        if sort_by == "Attention score ↓" and not np.isnan(r["joint_score"]):
            metric = f"attn={r['joint_score']:.4f}"
        elif sort_by == "Selectivity ↓" and not np.isnan(r["selectivity"]):
            metric = f"sel={r['selectivity']:.0%}"
        elif sort_by == "TCAV abnormal ↓" and not np.isnan(r["tcav_abnormal"]):
            metric = f"tcav={r['tcav_abnormal']:.0%}"
        elif sort_by in ("Fire rate ↓",) and not np.isnan(r["fire_rate"]):
            metric = f"fr={r['fire_rate']:.1f}%"
        elif sort_by == "Mean activation ↓" and not np.isnan(r["mean_act"]):
            metric = f"mean={r['mean_act']:.3f}"
        elif sort_by == "Max activation ↓" and not np.isnan(r["max_act"]):
            metric = f"max={r['max_act']:.3f}"
        elif sort_by in ("Label correlation ↓", "Label correlation ↑") and not np.isnan(r["corr"]):
            metric = f"r={r['corr']:+.3f}"
        elif sort_by == "Age enrichment ↓" and not np.isnan(r["age_enr"]):
            metric = f"age×{r['age_enr']:.2f}"
        elif sort_by == "Sex enrichment ↓" and not np.isnan(r["sex_enr"]):
            metric = f"sex×{r['sex_enr']:.2f}"
        else:
            metric = r["band"]
        star = " ★" if r["idx"] in real_example_idxs else ""
        abn   = r["abnorm_ratio"]
        abn_badge = f"  abn={abn:.0%}" if not np.isnan(abn) else ""
        return f"F{r['idx']}{star}  [{metric}]{abn_badge}"

    feat_options = [_feat_option(r) for r in rows]
    if not feat_options:
        st.info("No features match the current filters.")
        return
    chosen_label = st.selectbox("Select feature", feat_options)
    chosen_idx   = rows[feat_options.index(chosen_label)]["idx"]

    # ── Feature landscape overview ────────────────────────────────────
    with st.expander("Feature landscape — all features overview"):
        _has_xy = has_cache and any(not np.isnan(r["fire_rate"]) for r in rows)
        if not _has_xy and not has_proto:
            st.info("Overview requires app cache or prototype data.")
        else:
            _ov_x_opts = []
            if has_cache:
                _ov_x_opts += ["Fire rate (%)", "Label correlation (r)"]
            if has_proto:
                _ov_x_opts.append("Selectivity")
            _ov_y_opts = list(_ov_x_opts)  # same choices for both axes

            _ov_col1, _ov_col2 = st.columns(2)
            with _ov_col1:
                _x_choice = st.selectbox("X axis", _ov_x_opts, key="ov_x")
            with _ov_col2:
                _y_default = 1 if len(_ov_y_opts) > 1 else 0
                _y_choice = st.selectbox("Y axis", _ov_y_opts, index=_y_default, key="ov_y")

            def _ov_val(r: dict, metric: str) -> float:
                if metric == "Fire rate (%)":      return r["fire_rate"]
                if metric == "Label correlation (r)": return r["corr"]
                if metric == "Selectivity":        return r["selectivity"]
                return float("nan")

            _xs = [_ov_val(r, _x_choice) for r in rows]
            _ys = [_ov_val(r, _y_choice) for r in rows]
            _bands = [r["band"] for r in rows]
            _idxs  = [r["idx"] for r in rows]

            _band_color_map = {b: BAND_COLORS.get(b, "#888") for b in set(_bands)}

            _fig_ov = go.Figure()
            for _band in sorted(set(_bands)):
                _mask = [b == _band for b in _bands]
                _bxs  = [x for x, m in zip(_xs, _mask) if m]
                _bys  = [y for y, m in zip(_ys, _mask) if m]
                _bids = [i for i, m in zip(_idxs, _mask) if m]
                _fig_ov.add_trace(go.Scattergl(
                    x=_bxs, y=_bys,
                    mode="markers",
                    name=_band,
                    marker=dict(color=_band_color_map[_band], size=7, opacity=0.7),
                    hovertext=[f"F{i}" for i in _bids],
                    hoverinfo="text+x+y",
                ))

            # Highlight selected feature
            _sel_row = next((r for r in rows if r["idx"] == chosen_idx), None)
            if _sel_row is not None:
                _sx = _ov_val(_sel_row, _x_choice)
                _sy = _ov_val(_sel_row, _y_choice)
                if not np.isnan(_sx) and not np.isnan(_sy):
                    _fig_ov.add_trace(go.Scatter(
                        x=[_sx], y=[_sy],
                        mode="markers",
                        name=f"Selected (F{chosen_idx})",
                        marker=dict(color="crimson", size=13, symbol="star",
                                    line=dict(color="white", width=1)),
                        showlegend=True,
                    ))

            _fig_ov.update_layout(
                xaxis_title=_x_choice,
                yaxis_title=_y_choice,
                height=340,
                margin=dict(t=20, b=40, l=50, r=10),
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(_fig_ov, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1 — Real Patches
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Real Patches")

    real_patch_fs   = int(attn_cache["fs"])        if attn_cache else 128
    real_patch_size = int(attn_cache["patch_size"]) if attn_cache else _PATCH_SIZE
    real_ch_labels  = list(attn_cache["channel_names"]) if attn_cache else _CHANNEL_NAMES[:19]

    # Label filter control
    _label_filter = st.radio(
        "Show patches from",
        ["All", "Abnormal (label=1)", "Normal (label=0)"],
        horizontal=True,
        key="patch_label_filter",
    ) if attn_cache is not None else "All"
    _label_target: float | None = (
        1.0 if _label_filter == "Abnormal (label=1)" else
        0.0 if _label_filter == "Normal (label=0)" else
        None
    )

    def _render_patch_row(patches: list[np.ndarray], title: str) -> None:
        from plotly.subplots import make_subplots as _msp
        _n   = len(patches)
        _sp  = 2.2
        _t   = np.arange(real_patch_size) / real_patch_fs
        _C   = len(real_ch_labels)
        fig  = _msp(rows=1, cols=_n, subplot_titles=[f"#{i+1}" for i in range(_n)],
                    shared_yaxes=True, horizontal_spacing=0.02)
        for _i, _p in enumerate(patches):
            _p = _eeg_normalize(_bandpass(_p, fs=real_patch_fs))
            for _c in range(_C):
                fig.add_trace(go.Scatter(
                    x=_t.tolist(), y=(_p[_c] + (_C-1-_c)*_sp).tolist(),
                    mode="lines", line=dict(width=0.9, color="steelblue"), showlegend=False,
                    hovertemplate=f"<b>{real_ch_labels[_c]}</b><br>t=%{{x:.3f}} s<extra></extra>",
                ), row=1, col=_i+1)
        fig.update_yaxes(tickmode="array",
                         tickvals=[(_C-1-_c)*_sp for _c in range(_C)],
                         ticktext=real_ch_labels, tickfont=dict(size=10), row=1, col=1)
        for _i in range(2, _n+1):
            fig.update_yaxes(showticklabels=False, row=1, col=_i)
        for _i in range(1, _n+1):
            fig.update_xaxes(title_text="Time (s)", title_font=dict(size=10), row=1, col=_i)
        fig.update_layout(title=title, height=max(600, _C*22), margin=dict(l=70, r=10, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

    if attn_cache is not None and chosen_idx < int(attn_cache["n_features"]):
        K_c          = int(attn_cache["K"])
        ps_c         = int(attn_cache["patch_size"])
        local_idxs   = attn_cache["top_window_idx"][chosen_idx]     # (K,)
        win_labels   = attn_cache["top_window_labels"][chosen_idx]  # (K,)
        has_attn_wts = attn_cache.get("windows_attn") is not None

        act_patches:   list[np.ndarray] = []
        joint_patches: list[np.ndarray] = []

        # Re-rank cached windows by peak token activation (not mean) so that
        # sharp transients like spikes surface first.
        _eligible_k = [
            _k for _k in range(K_c)
            if _label_target is None or round(float(win_labels[_k])) == int(_label_target)
        ]
        _eligible_k.sort(
            key=lambda _k: float(
                attn_cache["windows_feat_acts"][int(local_idxs[_k]), :, chosen_idx]
                .float().max()
            ),
            reverse=True,
        )

        for _k in _eligible_k:
            _li = int(local_idxs[_k])
            _fa = attn_cache["windows_feat_acts"][_li, :, chosen_idx].float().numpy()  # (S,)
            _eeg = attn_cache["windows_eeg"][_li].float().numpy()                       # (27, T)

            # Row 1: raw activation peak
            _act_tok = int(np.argmax(_fa))
            act_patches.append(_eeg[:, _act_tok*ps_c:(_act_tok+1)*ps_c][:len(real_ch_labels)])

            # Row 2: joint activation × attention peak
            if has_attn_wts:
                _aw    = attn_cache["windows_attn"][_li].float().numpy()
                _ar    = _aw.mean(axis=0).mean(axis=0)
                _joint = (_fa / (_fa.max() + 1e-8)) * (_ar / (_ar.max() + 1e-8))
            else:
                _joint = _fa
            _jt = int(np.argmax(_joint))
            joint_patches.append(_eeg[:, _jt*ps_c:(_jt+1)*ps_c][:len(real_ch_labels)])

            if len(act_patches) >= 5:
                break

        _fl = f" — {_label_filter.lower()}" if _label_target is not None else ""
        if act_patches:
            _render_patch_row(act_patches,  f"Top activating tokens{_fl} — ranked by activation (1 s each)")
            _render_patch_row(joint_patches, f"Top activating tokens{_fl} — ranked by activation × attention (1 s each)")
        else:
            _filter_msg = f" with label '{_label_filter}'" if _label_target is not None else ""
            st.caption(f"No activating windows found{_filter_msg} for this feature.")
    elif attn_cache is None:
        st.info("No attention cache found — build one with `tools/build_attention_cache.py`.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2 — Statistics
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Statistics")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        fire_rate = _stat(chosen_idx, "fire_rate_pct")
        mean_act  = _stat(chosen_idx, "mean_activation")
        corr      = _stat(chosen_idx, "label_correlation")
        pval      = _stat(chosen_idx, "label_p_value", default=1.0)

        st.metric("Fire rate",
                  f"{fire_rate:.1f}%" if not np.isnan(fire_rate) else "—")
        st.metric("Mean activation (when active)",
                  f"{mean_act:.3f}" if not np.isnan(mean_act) else "—")
        if not np.isnan(corr):
            _corr_interp = (
                "fires more on **abnormal** EEG" if corr > 0.05
                else "fires more on **normal** EEG" if corr < -0.05
                else "no clear label preference"
            )
            _sig = "★ significant" if pval < 0.05 else "not significant"
            st.metric(
                "Label correlation (r)",
                f"{corr:+.4f}",
                help=(
                    "Pearson r between this feature's activation and the binary "
                    "abnormal/normal label across all tokens.  "
                    "r > 0 → fires more on abnormal EEG.  "
                    "r < 0 → fires more on normal EEG.  "
                    "r ≈ 0 → no label preference."
                ),
            )
            st.caption(f"→ {_corr_interp}  ({_sig}, p={pval:.2e})")
        else:
            st.metric("Label correlation (r)", "—",
                      help="Requires pre-built app cache.")
        desc = _expl(chosen_idx, "description") or f"Feature {chosen_idx}"
        st.markdown(f"**Description:** {desc}")
        st.markdown(f"**Dominant band:** {dominant_band[chosen_idx]}")


    with col_b:
        st.markdown("**Differential amplitude spectrum**")
        diff = amp_diff[chosen_idx]
        base = amp_base
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(
            x=freqs, y=np.expm1(base), mode="lines", name="Baseline",
            line=dict(color="#aaaaaa", width=1.5, dash="dot"),
        ))
        fig_spec.add_trace(go.Scatter(
            x=freqs, y=np.expm1(base + diff), mode="lines", name="Feature active",
            line=dict(color="#E53935", width=2),
        ))
        for bname, (f_lo, f_hi) in CLINICAL_BANDS.items():
            fig_spec.add_vrect(
                x0=f_lo, x1=f_hi, fillcolor=BAND_COLORS[bname], opacity=0.06,
                layer="below", line_width=0,
                annotation_text=bname, annotation_position="top left", annotation_font_size=9,
            )
        fig_spec.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="Amplitude",
            height=320, margin=dict(t=10, b=40, l=50, r=10),
            legend=dict(x=0.7, y=0.95),
        )
        st.plotly_chart(fig_spec, use_container_width=True)

    # ── Band powers + metadata enrichment (side by side) ─────────────────────
    _feat_meta_enr = _all_meta_enr
    _enr_data = _feat_meta_enr[chosen_idx] if _feat_meta_enr and chosen_idx < len(_feat_meta_enr) else {}

    _has_age = "age_group" in _enr_data
    _has_sex = "gender" in _enr_data
    _col_widths = [1] + ([2] if _has_age else []) + ([1] if _has_sex else [])
    _chart_cols = st.columns(_col_widths)
    _chart_col_iter = iter(_chart_cols)

    # Band power delta
    with next(_chart_col_iter):
        bd = band_deltas[chosen_idx]
        fig_bar = go.Figure(go.Bar(
            x=band_names, y=bd.tolist(),
            marker_color=[BAND_COLORS[b] if bd[i] >= 0 else "#aaaaaa" for i, b in enumerate(band_names)],
            text=[f"{v:+.3f}" for v in bd], textposition="outside",
            cliponaxis=False,
        ))
        fig_bar.update_layout(
            title="Band Δ",
            xaxis_title=None, yaxis_title="Δ amplitude",
            height=300, margin=dict(t=30, b=10, l=40, r=10),
            yaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="grey"),
            font=dict(size=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    def _vertical_enr_bar(cats_sorted: list, colors: list, title: str):
        _names = [c for c, _, _ in cats_sorted]
        _vals  = [r for _, r, _ in cats_sorted]
        _tips  = [f"{c}: ×{r:.2f} (p={p:.2e})" for c, r, p in cats_sorted]
        _sig   = ["★" if p < 0.05 else "" for _, _, p in cats_sorted]
        _y_max = max(_vals) if _vals else 2.0
        fig = go.Figure(go.Bar(
            x=_names, y=_vals,
            marker_color=colors,
            marker_opacity=0.85,
            text=[f"×{v:.1f}{s}" for v, s in zip(_vals, _sig)],
            textposition="outside",
            textfont=dict(size=9),
            cliponaxis=False,
            showlegend=False,
            customdata=_tips,
            hovertemplate="%{customdata}<extra></extra>",
        ))
        fig.add_hline(y=1.0, line_width=1, line_dash="dot", line_color="#666666")
        fig.update_layout(
            title=title,
            xaxis=dict(tickfont=dict(size=9), tickangle=-35, automargin=True),
            yaxis=dict(range=[0, max(_y_max * 1.3, 2.1)], zeroline=False,
                       showticklabels=False, showgrid=False),
            height=300, margin=dict(t=30, b=10, l=10, r=10),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#fafafa", size=10),
            showlegend=False,
        )
        return fig

    if _has_age:
        with next(_chart_col_iter):
            _cats_age = _enr_data["age_group"]
            def _ak_fe(v: str) -> int:
                try: return int(v.split("-")[0].replace("+", ""))
                except ValueError: return 999
            _cats_age_sorted = sorted(_cats_age, key=lambda x: _ak_fe(x[0]))
            _known_age = [c for c, _, _ in _cats_age_sorted if c != "unknown"]
            _grad_age = px.colors.sample_colorscale(
                "Plasma", [i / max(len(_known_age) - 1, 1) for i in range(len(_known_age))]
            )
            _age_cmap = {v: _grad_age[i] for i, v in enumerate(_known_age)}
            _age_cmap["unknown"] = "#666666"
            _clrs_age = [_age_cmap.get(c, "#aaaaaa") for c, _, _ in _cats_age_sorted]
            st.plotly_chart(
                _vertical_enr_bar(_cats_age_sorted, _clrs_age, "Age group enrichment"),
                use_container_width=True, config={"displayModeBar": False},
            )

    if _has_sex:
        with next(_chart_col_iter):
            _cats_sex = sorted(_enr_data["gender"], key=lambda x: -x[1])
            _sex_pal = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8"]
            _sex_cmap = {c: _sex_pal[i % len(_sex_pal)] for i, (c, _, _) in enumerate(
                sorted(_enr_data["gender"], key=lambda x: x[0])
            )}
            _clrs_sex = [_sex_cmap.get(c, "#aaaaaa") for c, _, _ in _cats_sex]
            st.plotly_chart(
                _vertical_enr_bar(_cats_sex, _clrs_sex, "Sex enrichment"),
                use_container_width=True, config={"displayModeBar": False},
            )
    st.caption("Enrichment = firing rate in group ÷ global firing rate  (×1 = baseline, ★ p < 0.05)")

    with st.expander("Co-occurrence with other features"):
        if cooccurrence is not None and chosen_idx < cooccurrence.shape[0]:
            co_row = cooccurrence[chosen_idx]
        else:
            norms  = np.linalg.norm(band_deltas, axis=1, keepdims=True).clip(min=1e-8)
            normed = band_deltas / norms
            co_row = (normed @ normed[chosen_idx])
        top_co = np.argsort(co_row)[::-1][:10]
        top_co = [i for i in top_co if i != chosen_idx][:8]
        fig_co = go.Figure(go.Bar(
            x=[f"F{i}" for i in top_co], y=[float(co_row[i]) for i in top_co],
            marker_color="#5C6BC0",
        ))
        fig_co.update_layout(
            title="Top co-occurring features (band-effect profile similarity)",
            height=260, margin=dict(t=40, b=30, l=40, r=10),
        )
        st.plotly_chart(fig_co, use_container_width=True)

    if chosen_idx in dash_imgs:
        with st.expander("Top-activating EEG windows (dashboard image)"):
            st.image(dash_imgs[chosen_idx],
                     caption=f"Feature F{chosen_idx} — top-activating windows",
                     use_container_width=True)

    with st.expander("All features — sortable table"):
        import pandas as pd
        df = pd.DataFrame(rows).rename(columns={
            "idx": "Feature", "fire_rate": "Fire %", "corr": "r(label)",
            "p": "p-value", "band": "Dominant band", "description": "Description",
        })
        st.dataframe(df.drop(columns=["cluster"]), use_container_width=True, height=400)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3 — Prototype
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Prototype")

    if _proto_run_name is None:
        st.info(
            "No prototype runs found for this model/layer in `results/feature_viz/`. "
            "Run `tools/feature_maximization.py --experiment <name>` first."
        )
    else:
        st.caption(f"Run: `{_proto_run_name}`")
        proto = _proto_data
        feat_indices_list = proto["feature_indices"].tolist()

        if chosen_idx not in feat_indices_list:
            st.info(f"Feature {chosen_idx} has no prototype in run `{_proto_run_name}`.")
        else:
            sel_idx   = feat_indices_list.index(chosen_idx)
            _has_slim = "peak_patches" in proto
            feat_act_h = proto["feat_act_hists"]
            sel_h      = proto["selectivity_hists"]
            final_zs   = proto["final_zs"]
            peak_tokens = proto["peak_tokens"].tolist()
            fs          = int(proto["fs"])
            emp_rms     = proto.get("empirical_spatial_rms", None)

            fi       = feat_indices_list[sel_idx]
            peak_tok = peak_tokens[sel_idx]
            fa_hist  = feat_act_h[sel_idx]
            s_hist   = sel_h[sel_idx]
            fz       = final_zs[sel_idx]

            if _has_slim:
                peak_patch = proto["peak_patches"][sel_idx].astype(np.float32)
            else:
                sig        = proto["signals"][sel_idx]
                peak_patch = sig[:, peak_tok * _PATCH_SIZE:(peak_tok + 1) * _PATCH_SIZE]

            C_p       = peak_patch.shape[0]
            P_samples = peak_patch.shape[-1]
            ch_labels = _CHANNEL_NAMES[:C_p]
            final_sel = float(s_hist[-1])
            sel_pct   = final_sel * 100
            t_s       = np.arange(P_samples) / fs
            spacing   = 2.2

            proto_crop = _bandpass(peak_patch, fs=fs)
            proto_norm = _eeg_normalize(proto_crop)
            fig_proto  = go.Figure()
            for c in range(C_p):
                y_off = (C_p - 1 - c) * spacing
                fig_proto.add_trace(go.Scatter(
                    x=t_s.tolist(), y=(proto_norm[c] + y_off).tolist(),
                    mode="lines", line=dict(width=1.0, color="salmon"),
                    showlegend=False,
                    hovertemplate=f"<b>{ch_labels[c]}</b><br>t=%{{x:.3f}} s<extra></extra>",
                ))
            fig_proto.update_layout(
                title="Gradient-ascent prototype — peak token (1 s)",
                xaxis_title="Time (s)",
                yaxis=dict(tickmode="array",
                           tickvals=[(C_p - 1 - c) * spacing for c in range(C_p)],
                           ticktext=ch_labels, tickfont=dict(size=11)),
                height=max(600, C_p * 22),
                margin=dict(l=60, r=10, t=40, b=40),
            )
            st.plotly_chart(fig_proto, use_container_width=True)

            with st.expander(f"Feature purity — {sel_pct:.1f}%"):
                k_floor = 1.0 / 8 * 100
                st.info(
                    f"**Feature purity: {sel_pct:.1f}%** — "
                    f"at the peak token, {sel_pct:.1f}% of all SAE activation energy is on this feature "
                    f"(chance = {k_floor:.1f}%, max = 100%)."
                )
                col3, col4 = st.columns(2)
                with col3:
                    if emp_rms is not None and emp_rms.shape[0] > sel_idx:
                        rms_vals  = emp_rms[sel_idx, :C_p]
                        title_sp  = "Spatial RMS — real data (top-100 patches)"
                        bar_color = "steelblue"
                    else:
                        rms_vals  = np.sqrt((peak_patch[:C_p] ** 2).mean(axis=-1))
                        title_sp  = "Spatial RMS — gradient-ascent peak token"
                        bar_color = "salmon"
                    fig_sp = go.Figure(go.Bar(
                        x=rms_vals.tolist(), y=ch_labels,
                        orientation="h", marker_color=bar_color,
                    ))
                    fig_sp.update_layout(
                        title=title_sp, xaxis_title="RMS amplitude",
                        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                        height=max(350, C_p * 16), margin=dict(l=60, r=10, t=40, b=40),
                    )
                    st.plotly_chart(fig_sp, use_container_width=True)
                with col4:
                    top8_idx  = np.argsort(fz)[::-1][:8]
                    top8_vals = fz[top8_idx]
                    total_z   = top8_vals.sum() + 1e-9
                    fractions = top8_vals / total_z
                    colors    = ["crimson" if idx == fi else "#5B9BD5" for idx in top8_idx]
                    fig_sel = go.Figure(go.Bar(
                        x=[f"f{idx}" for idx in top8_idx], y=fractions.tolist(),
                        marker_color=colors,
                        hovertext=[f"Feature {idx}<br>z={v:.3f}<br>{f:.1%} of top-k total"
                                   for idx, v, f in zip(top8_idx, top8_vals, fractions)],
                        hoverinfo="text",
                    ))
                    fig_sel.add_hline(y=1.0 / 8, line_dash="dot", line_color="gray", opacity=0.6,
                                     annotation_text="uniform (1/k)", annotation_position="top right")
                    fig_sel.update_layout(
                        title=f"Feature purity — {final_sel:.1%} of total SAE activation<br>"
                              f"<sup>Red bar = target feature.</sup>",
                        yaxis=dict(title="Fraction of total z", tickformat=".0%", range=[0, 1.05]),
                        height=320, margin=dict(l=50, r=20, t=60, b=40),
                    )
                    st.plotly_chart(fig_sel, use_container_width=True)

            with st.expander("Optimisation dynamics"):
                steps = list(range(1, len(fa_hist) + 1))
                fig_dyn = go.Figure()
                fig_dyn.add_trace(go.Scatter(x=steps, y=fa_hist.tolist(),
                    name="Feature activation (post-TopK)",
                    line=dict(color="#2196F3", width=1.5), yaxis="y1"))
                fig_dyn.add_trace(go.Scatter(x=steps, y=s_hist.tolist(),
                    name="Feature purity (z_i / Σz)",
                    line=dict(color="#FF9800", width=1.5, dash="dash"), yaxis="y2"))
                fig_dyn.add_hline(y=1.0 / 8, line_dash="dot", line_color="#FF9800", opacity=0.4,
                                  annotation_text="1/k floor", annotation_position="bottom right",
                                  yref="y2")
                fig_dyn.update_layout(
                    title="Gradient-ascent optimisation dynamics", xaxis_title="Step",
                    yaxis=dict(title=dict(text="Feature activation", font=dict(color="#2196F3")),
                               tickfont=dict(color="#2196F3")),
                    yaxis2=dict(title=dict(text="Feature purity", font=dict(color="#FF9800")),
                                tickfont=dict(color="#FF9800"),
                                overlaying="y", side="right", range=[0, 1.05]),
                    legend=dict(x=0.01, y=0.99),
                    height=300, margin=dict(l=60, r=60, t=40, b=40),
                )
                st.plotly_chart(fig_dyn, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Build-cache helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_app_cache_button(exp_name: str, key_suffix: str = "") -> None:
    """Show the build command and a button that runs it."""
    cmd = f"uv run tools/build_app_cache.py --experiment {exp_name}"
    st.code(cmd, language="bash")
    if st.button("Build app cache", key=f"build_cache_{exp_name}{key_suffix}"):
        with st.spinner("Building app cache — this may take a few minutes…"):
            result = subprocess.run(
                ["uv", "run", "tools/build_app_cache.py", "--experiment", exp_name],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
        if result.returncode == 0:
            st.success("Cache built successfully — reloading…")
            st.rerun()
        else:
            st.error("Build failed")
            st.code(result.stdout + result.stderr)


# ─────────────────────────────────────────────────────────────────────────────

def page_tcav_explorer(
    data: dict,
    folder_name: str,
    layer: int,
    tcav_cache: Optional[dict],
    app_cache: Optional[dict],
    exp_name: Optional[str] = None,
):
    import pandas as pd

    st.title("TCAV Explorer")
    st.caption("**Paper §4** — concept attribution via Testing with Concept Activation Vectors.")
    enc_label = _ENCODER_LABELS.get(data["encoder"], data["encoder"].upper())
    st.markdown(
        f"**Testing with Concept Activation Vectors** — {enc_label} layer {data['layer']}. "
        "Linear CAVs are trained in encoder activation space; each CAV direction is projected "
        "onto SAE decoder directions to measure feature–concept alignment."
    )

    exp_name = f"{folder_name}_layer{layer}"

    if tcav_cache is None:
        if app_cache is None:
            st.info("TCAV requires an app cache first.")
            _build_app_cache_button(exp_name, "_tcav")
        else:
            tcav_cmd = f"uv run tools/run_tcav.py --experiment {exp_name}"
            st.info("No TCAV cache found for this experiment.")
            st.code(tcav_cmd, language="bash")
            if st.button("Build TCAV cache", key=f"build_tcav_{exp_name}"):
                with st.spinner("Building TCAV cache — this may take a few minutes…"):
                    result = subprocess.run(
                        ["uv", "run", "tools/run_tcav.py", "--experiment", exp_name],
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                    )
                if result.returncode == 0:
                    st.success("TCAV cache built — reloading…")
                    st.rerun()
                else:
                    st.error("Build failed")
                    st.code(result.stdout + result.stderr)
        return

    concept_names = tcav_cache["concept_names"]
    cav_accs      = tcav_cache["cav_accuracies"]   # list[float]
    concept_sizes = tcav_cache["concept_sizes"]    # list[int]
    n_features    = int(tcav_cache["n_features"])
    C             = len(concept_names)

    alignments   = tcav_cache["alignments"]
    firing_rates = tcav_cache["firing_rates"]
    # delta_rate and p_values added in v2 of run_tcav.py; graceful fallback for old caches
    _has_stats = "delta_rates" in tcav_cache
    delta_rates  = tcav_cache["delta_rates"]  if _has_stats else alignments * firing_rates
    p_values_mat = tcav_cache["p_values"]     if _has_stats else None

    for t in [alignments, firing_rates, delta_rates]:
        if torch.is_tensor(t):
            pass  # convert below
    alignments   = alignments.numpy()   if torch.is_tensor(alignments)   else np.array(alignments)
    firing_rates = firing_rates.numpy() if torch.is_tensor(firing_rates) else np.array(firing_rates)
    delta_rates  = delta_rates.numpy()  if torch.is_tensor(delta_rates)  else np.array(delta_rates)
    if p_values_mat is not None:
        p_values_mat = p_values_mat.numpy() if torch.is_tensor(p_values_mat) else np.array(p_values_mat)

    # ── Variant A: weight-space TCAV per feature ─────────────────────────────
    _has_tcav = "tcav_scores" in tcav_cache
    tcav_scores_mat_raw = tcav_cache["tcav_scores"]   if _has_tcav else None
    tcav_p_mat_raw      = tcav_cache["tcav_p_values"] if _has_tcav else None
    tcav_scores_mat_np = (tcav_scores_mat_raw.numpy() if torch.is_tensor(tcav_scores_mat_raw)
                          else np.array(tcav_scores_mat_raw)) if _has_tcav else None
    tcav_p_mat_np      = (tcav_p_mat_raw.numpy() if torch.is_tensor(tcav_p_mat_raw)
                          else np.array(tcav_p_mat_raw)) if _has_tcav else None

    # ── Variant C: model-level Kim et al. TCAV per concept ───────────────────
    _has_model_tcav   = "model_tcav_scores" in tcav_cache
    model_tcav_scores = tcav_cache.get("model_tcav_scores")   # list[float] len C
    model_tcav_pvals  = tcav_cache.get("model_tcav_p_values") # list[float] len C
    probe_accuracies  = tcav_cache.get("probe_accuracies")    # list[float] len C

    # Primary ranking metric for heatmap / band section: delta_rate
    tcav_scores_mat = delta_rates   # (C, n_features)

    band_concept_names = [n for n in concept_names if n != "abnormal"]
    has_abnormal = "abnormal" in concept_names
    ab_idx       = concept_names.index("abnormal") if has_abnormal else None
    ab_valid     = (
        has_abnormal
        and not np.isnan(cav_accs[ab_idx])
        and cav_accs[ab_idx] > 0.52
    )

    # Per-feature dominant band: highest-scoring band concept
    band_indices = [concept_names.index(b) for b in band_concept_names
                    if not np.isnan(cav_accs[concept_names.index(b)])]
    if band_indices:
        band_scores_sub = tcav_scores_mat[band_indices, :]   # (n_bands, n_features)
        dominant_band_idx = np.argmax(band_scores_sub, axis=0)
        valid_bands = [band_concept_names[i]
                       for i in [concept_names.index(b) for b in band_concept_names
                                 if not np.isnan(cav_accs[concept_names.index(b)])]]
        feature_dominant_band = [valid_bands[dominant_band_idx[i]] for i in range(n_features)]
    else:
        feature_dominant_band = ["unknown"] * n_features

    # Pull label correlations from app_cache if available
    feature_stats = (app_cache or {}).get("feature_stats", [])
    label_r   = np.array([
        s.get("label_correlation", float("nan")) for s in feature_stats
    ] if feature_stats else [float("nan")] * n_features, dtype=np.float32)
    label_p   = np.array([
        s.get("label_p_value", float("nan")) for s in feature_stats
    ] if feature_stats else [float("nan")] * n_features, dtype=np.float32)
    has_label_corr = not np.all(np.isnan(label_r))

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1 — Abnormality Analysis (primary)
    # ═══════════════════════════════════════════════════════════════════
    st.subheader("Abnormality analysis")

    if not ab_valid:
        acc_str = f"{cav_accs[ab_idx]:.3f}" if has_abnormal and not np.isnan(cav_accs[ab_idx]) else "n/a"
        st.warning(
            f"Abnormality CAV accuracy = {acc_str} (chance level). "
            "The normal/abnormal distinction may not be linearly separable at this layer, "
            "or token labels were not available when the TCAV cache was built."
        )
    else:
        ab_score    = tcav_scores_mat[ab_idx]    # delta_rate: (n_features,)
        ab_aln      = alignments[ab_idx]
        ab_rate     = firing_rates[ab_idx]
        ab_pval     = p_values_mat[ab_idx] if p_values_mat is not None else np.ones(n_features)
        ab_sig      = ab_pval < 0.05
        ab_tcav     = tcav_scores_mat_np[ab_idx] if _has_tcav else None  # Variant A per-feature
        ab_tcav_p   = tcav_p_mat_np[ab_idx]      if _has_tcav else None

        # ── Model-level TCAV headline (Variant C) ─────────────────────
        if _has_model_tcav and model_tcav_scores is not None:
            m_score = model_tcav_scores[ab_idx]
            m_pval  = model_tcav_pvals[ab_idx]
            p_acc   = probe_accuracies[ab_idx]
            _is_nan = m_score != m_score  # nan check
            if not _is_nan:
                _sig_badge = "significant" if m_pval < 0.05 else "not significant"
                _sig_color = "green" if m_pval < 0.05 else "grey"
                st.info(
                    f"**Model-level TCAV score (Kim et al.): {m_score:.3f}** "
                    f"(0.5 = chance)  ·  p = {m_pval:.3f} — "
                    f":{_sig_color}[{_sig_badge}]  ·  "
                    f"Probe accuracy: {p_acc:.3f}  ·  "
                    f"CAV accuracy: {cav_accs[ab_idx]:.3f}"
                )
                st.caption(
                    "Model TCAV = fraction of abnormal-class examples where the probe's "
                    "prediction increases when activations are nudged in the concept direction: "
                    "S(x) = Σᵢ [w_probe_i × (W_enc_i · v_C) × I(zᵢ(x)>0)] > 0. "
                    "This is Variant C — the proper Kim et al. formulation with a downstream "
                    "predictor. Per-feature metrics below use Variant A (weight-space) and "
                    "Δ firing rate."
                )

        # Rank: significant features first, then by delta_rate descending
        ranked = np.lexsort((-ab_score, ab_pval))   # primary: p ascending, secondary: delta desc

        # ── Left: ranked feature table  ──────────────────────────────
        col_tbl, col_scatter = st.columns([1, 1])

        with col_tbl:
            n_sig_ab = int(ab_sig.sum())
            st.markdown(
                f"**{n_sig_ab}/{n_features} features enriched** on abnormal tokens "
                f"(BH-adj p < 0.05 · {concept_sizes[ab_idx]:,} abnormal tokens)"
            )
            top_n_tbl = 20
            tbl_rows = []
            for rank, fi in enumerate(ranked[:top_n_tbl]):
                r_val = float(label_r[fi]) if has_label_corr and not np.isnan(label_r[fi]) else None
                # B1 = fraction of folds where W_enc[i]·v_C > 0 (alignment sign stability)
                b1 = float(ab_tcav[fi]) if ab_tcav is not None else None
                # B2 = B1 × firing_rate (alignment AND activation frequency)
                b2 = round(b1 * float(ab_rate[fi]), 4) if b1 is not None else None
                row = {
                    "Rank":        rank + 1,
                    "Feature":     f"F{fi}",
                    "Δ fire rate": round(float(ab_score[fi]), 4),
                    "p (BH-adj)":  f"{ab_pval[fi]:.2e}",
                    "Sig.":        "✓" if ab_sig[fi] else "",
                    "B1 (align.)": round(b1, 3) if b1 is not None else "—",
                    "B2 (×rate)":  b2 if b2 is not None else "—",
                    "Alignment":   round(float(ab_aln[fi]), 4),
                    "Fire rate":   round(float(ab_rate[fi]), 3),
                }
                if ab_tcav_p is not None:
                    row["B1 p"] = f"{ab_tcav_p[fi]:.2e}"
                row["Label r"]   = round(r_val, 4) if r_val is not None else "—"
                row["Dom. band"] = feature_dominant_band[fi]
                tbl_rows.append(row)
            st.dataframe(
                pd.DataFrame(tbl_rows),
                use_container_width=True,
                height=420,
                hide_index=True,
            )
            st.caption(
                "**Δ fire rate** = firing_rate_abnormal − firing_rate_baseline (controls for background activity).  "
                "**B1** = fraction of K fold-CAVs where W_enc[i]·v_C > 0 "
                "(standard per-feature Kim et al.; {0,0.1,…,1}, 0.5=chance).  "
                "**B2** = B1 × firing_rate (B1 weighted by how often the feature fires on concept examples).  "
                "Ranked by BH-corrected p-value (Δ fire rate z-test), then Δ fire rate."
            )

        # ── Right: scatter ────────────────────────────────────────────
        with col_scatter:
            marker_colors = [BAND_COLORS.get(feature_dominant_band[i], "#888888")
                             for i in range(n_features)]
            marker_sizes   = [9 if ab_sig[i] else 5 for i in range(n_features)]
            marker_opacity = [1.0 if ab_sig[i] else 0.35 for i in range(n_features)]
            marker_symbols = ["circle" if ab_sig[i] else "circle-open"
                              for i in range(n_features)]
            top8 = set(ranked[:8].tolist())

            if ab_tcav is not None:
                x_vals  = ab_tcav.tolist()
                x_title = "TCAV score  (fraction of folds with positive directional derivative; 0.5 = chance)"
                title   = "TCAV score vs Δ firing rate (abnormality)"
                caption = (
                    "x = traditional TCAV score: fraction of k-fold CAV runs where W_enc[i]·v_C > 0.  "
                    "Range [0,1]; 0.5 = chance.  "
                    "y = firing rate enrichment on abnormal tokens (vs baseline).  "
                    "Filled circles = BH-adj p < 0.05.  Coloured by dominant EEG band."
                )
            elif has_label_corr:
                x_vals  = label_r.tolist()
                x_title = "Label correlation r  (Pearson, + = fires more on abnormal)"
                title   = "Label correlation vs Δ firing rate (abnormality)"
                caption = (
                    "x = Pearson r between feature firing and abnormality label.  "
                    "y = firing rate enrichment on abnormal tokens (vs baseline).  "
                    "Filled circles = BH-adj p < 0.05.  "
                    "Top-right = significant by both measures."
                )
            else:
                x_vals  = ab_aln.tolist()
                x_title = "CAV alignment (cosine similarity with abnormality direction)"
                title   = "CAV alignment vs Δ firing rate (abnormality)"
                caption = (
                    "x = cosine similarity of SAE decoder direction with abnormality CAV.  "
                    "y = firing rate enrichment on abnormal tokens.  "
                    "Filled circles = BH-adj p < 0.05.  Coloured by dominant EEG band."
                )

            if ab_tcav is not None:
                _custom = np.stack([
                    np.arange(n_features),
                    ab_aln,
                    ab_rate,
                    ab_pval,
                    ab_score,
                    ab_tcav,
                    ab_tcav_p,
                ], axis=1)
                _hover = (
                    "<b>F%{customdata[0]:.0f}</b><br>"
                    "TCAV score = %{customdata[5]:.3f}<br>"
                    "TCAV p = %{customdata[6]:.2e}<br>"
                    "Δ fire rate = %{customdata[4]:.4f}<br>"
                    "p (BH-adj) = %{customdata[3]:.2e}<br>"
                    "Alignment = %{customdata[1]:.3f}<br>"
                    "Fire rate (concept) = %{customdata[2]:.3f}"
                    "<extra></extra>"
                )
            else:
                _custom = np.stack([
                    np.arange(n_features),
                    ab_aln,
                    ab_rate,
                    ab_pval,
                    ab_score,
                ], axis=1)
                _hover = (
                    "<b>F%{customdata[0]:.0f}</b><br>"
                    "Δ fire rate = %{customdata[4]:.4f}<br>"
                    "p (BH-adj) = %{customdata[3]:.2e}<br>"
                    "Alignment = %{customdata[1]:.3f}<br>"
                    "Fire rate (concept) = %{customdata[2]:.3f}"
                    "<extra></extra>"
                )
            fig_ab = go.Figure(go.Scattergl(
                x=x_vals,
                y=ab_score.tolist(),
                mode="markers+text",
                marker=dict(
                    color=marker_colors,
                    size=marker_sizes,
                    opacity=marker_opacity,
                    symbol=marker_symbols,
                    line=dict(width=1, color="rgba(255,255,255,0.6)"),
                ),
                text=[f"F{i}" if i in top8 else "" for i in range(n_features)],
                textposition="top center",
                textfont=dict(size=9, color="white"),
                customdata=_custom,
                hovertemplate=_hover,
            ))
            fig_ab.add_hrect(y0=0, y1=float(ab_score.max()) * 1.1,
                             fillcolor="rgba(229,57,53,0.04)", line_width=0)
            if ab_tcav is not None:
                # Vertical reference line at 0.5 (chance TCAV score)
                fig_ab.add_vline(x=0.5, line_width=1, line_dash="dash",
                                 line_color="rgba(255,255,255,0.3)",
                                 annotation_text="chance", annotation_font_size=10,
                                 annotation_font_color="rgba(255,255,255,0.5)")
            fig_ab.update_layout(
                title=title,
                xaxis_title=x_title,
                yaxis_title="Δ firing rate  (abnormal − baseline)",
                height=420,
                margin=dict(t=50, b=55, l=60, r=10),
                xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="grey"),
                yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="grey"),
            )
            st.plotly_chart(fig_ab, use_container_width=True)
            st.caption(caption)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2 — CAV quality summary (compact)
    # ═══════════════════════════════════════════════════════════════════
    with st.expander("CAV quality summary", expanded=False):
        summary_rows = []
        for i, name in enumerate(concept_names):
            acc = cav_accs[i]
            row: dict = {
                "Concept":      name,
                "CAV acc.":     f"{acc:.3f}" if not np.isnan(acc) else "—",
                "CAV quality":  ("✓ good"   if not np.isnan(acc) and acc >= 0.65 else
                                 "⚠ weak"   if not np.isnan(acc) and acc >= 0.55 else
                                 "✗ chance" if not np.isnan(acc) else "—"),
                "# tokens":     f"{concept_sizes[i]:,}",
                "Top feature":  f"F{int(np.argmax(alignments[i]))}" if not np.isnan(acc) else "—",
            }
            if _has_model_tcav and model_tcav_scores is not None:
                ms = model_tcav_scores[i]
                mp = model_tcav_pvals[i]
                pa = probe_accuracies[i]
                _nan = ms != ms
                row["Model TCAV"] = f"{ms:.3f}" if not _nan else "—"
                row["TCAV p"]     = f"{mp:.3f}" if not _nan else "—"
                row["Probe acc."] = f"{pa:.3f}" if not _nan else "—"
            summary_rows.append(row)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        st.caption(
            "**CAV acc.** ≥ 0.65 = reliable; 0.55–0.65 = weak; ≤ 0.55 = chance.  "
            "**Model TCAV** = Kim et al. model-level score ([0,1], 0.5=chance); "
            "**Probe acc.** = 5-fold CV accuracy of the downstream z-feature probe."
        )

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3 — Band concepts (secondary / contextual)
    # ═══════════════════════════════════════════════════════════════════
    st.subheader("Band concepts")
    st.markdown(
        "Which features encode specific EEG frequency bands?  "
        "Cross-referencing band alignment with abnormality alignment reveals the "
        "spectral substrate of pathology (e.g. a feature that is both delta-selective "
        "and abnormality-selective encodes **pathological delta**)."
    )

    valid_band_concepts = [
        name for name in band_concept_names
        if not np.isnan(cav_accs[concept_names.index(name)])
        and cav_accs[concept_names.index(name)] > 0.52
    ]
    if not valid_band_concepts:
        st.info("No band CAVs exceeded chance level for this layer.")
    else:
        col_sel, col_n = st.columns([3, 1])
        with col_sel:
            chosen_band = st.selectbox("Select band concept", valid_band_concepts,
                                       key="band_concept_sel")
        with col_n:
            top_n = st.number_input("Top N features", min_value=5, max_value=50,
                                    value=15, step=5, key="band_top_n")

        bi      = concept_names.index(chosen_band)
        b_aln   = alignments[bi]
        b_rate  = firing_rates[bi]
        b_score = tcav_scores_mat[bi]   # delta_rate for this band
        b_pval  = p_values_mat[bi] if p_values_mat is not None else np.ones(n_features)
        top_idx = np.argsort(b_score)[::-1][:top_n]

        col_a, col_b = st.columns([1, 1])

        with col_a:
            band_color = BAND_COLORS.get(chosen_band, "#5C6BC0")
            # Colour bars by abnormality score: saturate toward red for high abnormality
            bar_colors = [
                "#E53935" if (ab_valid and ab_score[i] > np.percentile(ab_score, 80))
                else band_color
                for i in top_idx
            ]
            fig_band_bar = go.Figure(go.Bar(
                x=[f"F{i}" for i in top_idx],
                y=[float(b_score[i]) for i in top_idx],
                marker_color=bar_colors,
                text=[
                    (f"★ p={b_pval[i]:.1e}" if b_pval[i] < 0.05
                     else f"p={b_pval[i]:.1e}")
                    for i in top_idx
                ],
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate="<b>F%{x}</b><br>Δ fire rate: %{y:.4f}<extra></extra>",
            ))
            fig_band_bar.update_layout(
                title=f"Top features for {chosen_band} "
                      + ("(★ = also top-20% abnormality)" if ab_valid else ""),
                xaxis_title="SAE feature",
                yaxis_title="Δ firing rate  (concept − baseline)",
                height=360,
                margin=dict(t=55, b=30, l=50, r=10),
                yaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="grey"),
            )
            st.plotly_chart(fig_band_bar, use_container_width=True)

        with col_b:
            # If abnormality available: scatter band-score vs abnormality-score
            if ab_valid:
                point_colors = [BAND_COLORS.get(chosen_band, "#5C6BC0")] * n_features
                in_top = np.zeros(n_features, dtype=bool)
                in_top[top_idx] = True
                fig_cross = go.Figure(go.Scattergl(
                    x=b_score.tolist(),
                    y=ab_score.tolist(),
                    mode="markers",
                    marker=dict(
                        color=[band_color if in_top[i] else "#333333"
                               for i in range(n_features)],
                        size=[8 if in_top[i] else 4 for i in range(n_features)],
                        opacity=[1.0 if in_top[i] else 0.3 for i in range(n_features)],
                        line=dict(width=0),
                    ),
                    text=[f"F{i}" for i in range(n_features)],
                    hoverinfo="text",
                ))
                fig_cross.add_hrect(
                    y0=float(np.percentile(ab_score, 80)), y1=float(ab_score.max()),
                    fillcolor="rgba(229,57,53,0.06)", line_width=0,
                )
                fig_cross.update_layout(
                    title=f"{chosen_band} TCAV vs abnormality TCAV",
                    xaxis_title=f"{chosen_band} TCAV score",
                    yaxis_title="Abnormality TCAV score",
                    height=360,
                    margin=dict(t=50, b=50, l=60, r=10),
                    xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="grey"),
                    yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="grey"),
                )
                st.plotly_chart(fig_cross, use_container_width=True)
                st.caption(
                    f"Top-right = features selective for both **{chosen_band}** "
                    "and **abnormality** — candidate pathological signatures."
                )
            else:
                # Fallback: alignment vs firing rate
                fig_bscat = go.Figure(go.Scattergl(
                    x=b_aln.tolist(), y=b_rate.tolist(), mode="markers",
                    marker=dict(color=band_color, size=6, opacity=0.7,
                                line=dict(width=0)),
                    text=[f"F{i}" for i in range(n_features)],
                    hoverinfo="text",
                ))
                fig_bscat.update_layout(
                    title="Alignment vs firing rate",
                    xaxis_title="CAV alignment", yaxis_title="Firing rate",
                    height=360, margin=dict(t=50, b=50, l=60, r=10),
                    xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="grey"),
                )
                st.plotly_chart(fig_bscat, use_container_width=True)

        # ── Feature × concept heatmap (all concepts, top features) ───
        with st.expander("Feature × concept heatmap"):
            st.markdown(
                "TCAV scores across all concepts for the most prominent features. "
                "Reveals which features are concept-specific vs. broadly active."
            )
            union_top: set[int] = set()
            for i in range(C):
                if not np.isnan(cav_accs[i]):
                    union_top.update(np.argsort(tcav_scores_mat[i])[::-1][:10].tolist())
            union_top_list = sorted(union_top)

            if union_top_list:
                heatmap_data = tcav_scores_mat[:, union_top_list]
                fig_heat = go.Figure(go.Heatmap(
                    z=heatmap_data,
                    x=[f"F{i}" for i in union_top_list],
                    y=concept_names,
                    colorscale="RdBu",
                    zmid=0,
                    colorbar=dict(title="TCAV score"),
                    hoverongaps=False,
                    hovertemplate="Concept: %{y}<br>Feature: %{x}<br>Score: %{z:.3f}<extra></extra>",
                ))
                fig_heat.update_layout(
                    height=max(300, 50 * C + 80),
                    margin=dict(t=10, b=60, l=90, r=10),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

        # ── Concept comparison ────────────────────────────────────────
        with st.expander("Compare two concepts"):
            cc1, cc2 = st.columns(2)
            with cc1:
                concept_a = st.selectbox("Concept A", concept_names, key="cmp_a")
            with cc2:
                concept_b = st.selectbox(
                    "Concept B", concept_names,
                    index=min(1, len(concept_names) - 1),
                    key="cmp_b",
                )
            ia = concept_names.index(concept_a)
            ib = concept_names.index(concept_b)
            diff = tcav_scores_mat[ia] - tcav_scores_mat[ib]
            top_diff = np.argsort(np.abs(diff))[::-1][:20]

            fig_cmp = go.Figure(go.Bar(
                x=[f"F{i}" for i in top_diff],
                y=[float(diff[i]) for i in top_diff],
                marker_color=[
                    BAND_COLORS.get(concept_a, "#5C6BC0") if diff[i] > 0
                    else BAND_COLORS.get(concept_b, "#E53935")
                    for i in top_diff
                ],
                hovertemplate="<b>F%{x}</b><br>Δ TCAV: %{y:.3f}<extra></extra>",
            ))
            fig_cmp.update_layout(
                title=f"TCAV difference: {concept_a} − {concept_b}",
                xaxis_title="SAE feature",
                yaxis_title=f"Δ TCAV  ({concept_a} − {concept_b})",
                height=320, margin=dict(t=50, b=30, l=60, r=10),
                yaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="grey"),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.caption(
                f"Positive = more aligned with **{concept_a}**.  "
                f"Negative = more aligned with **{concept_b}**."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Page 6 — Layer Explorer
# ─────────────────────────────────────────────────────────────────────────────

_LABEL_COLORS = {0: "#4CAF50", 1: "#F44336", -1: "#aaaaaa"}
_LABEL_NAMES  = {0: "Normal",  1: "Abnormal", -1: "Unknown"}

_GRAN_LABEL_COLORS = {
    0: "#4CAF50",       # Normal
    1: "#ff7f0e",       # Diffuse slowing
    2: "#1f77b4",       # Focal slowing
    3: "#d62728",       # Focal sharp waves
    4: "#9467bd",       # Focal spike-wave
    5: "#8c564b",       # Gen. spike-wave
    6: "#e377c2",       # Gen. polyspike-wave
    7: "#7f7f7f",       # Gen. sharp waves
    8: "#bcbd22",       # Burst suppression
    9: "#17becf",       # Epileptic seizure
    -1: "#aaaaaa"       # Unknown
}
_GRAN_LABEL_NAMES = {
    0: "Normal",
    1: "Diffuse slowing",
    2: "Focal slowing",
    3: "Focal sharp waves",
    4: "Focal spike-wave",
    5: "Gen. spike-wave",
    6: "Gen. polyspike-wave",
    7: "Gen. sharp waves",
    8: "Burst suppression",
    9: "Epileptic seizure",
    -1: "Unknown"
}


@st.cache_data(show_spinner=False, hash_funcs={dict: id})
def _make_animation_figure(
    cache: dict, color_by: str, token_meta: Optional[Dict[str, Any]] = None,
    highlight: Optional[str] = None,
) -> go.Figure:
    """Build a Plotly animated scatter — one frame per encoder layer."""
    layers  = cache["layers"]
    band    = cache["band"]     # (N,) dtype=object
    label   = cache["label"]   # (N,) int32
    subject = cache["subject"] # (N,) int32
    if token_meta is None:
        token_meta = {}

    # ── Colour assignment ─────────────────────────────────────────────────────
    legend_entries: Optional[Dict[str, str]] = None

    point_labels: List[str] = []  # category label per point, used for highlight opacity

    if color_by == "Band":
        point_labels = [str(b) for b in band]
        colors = [BAND_COLORS.get(lb, "#aaaaaa") for lb in point_labels]
        legend_entries = {
            b: BAND_COLORS.get(b, "#aaaaaa")
            for b in sorted(set(point_labels))
        }

    elif color_by == "Label":
        max_label = np.max(label)
        names_dict = _GRAN_LABEL_NAMES if max_label > 1 else _LABEL_NAMES
        colors_dict = _GRAN_LABEL_COLORS if max_label > 1 else _LABEL_COLORS
        
        point_labels = [names_dict.get(int(l), str(l)) for l in label]
        colors = [colors_dict.get(int(l), "#aaaaaa") for l in label]
        legend_entries = {
            names_dict[v]: colors_dict[v]
            for v in sorted(set(int(l) for l in label))
            if v in names_dict
        }

    elif color_by == "Subject":
        unique_subjects = sorted(set(int(s) for s in subject if int(s) >= 0))
        n_subj = len(unique_subjects)
        if n_subj <= 26:
            palette = px.colors.qualitative.Alphabet[:n_subj]
        else:
            palette = [f"hsl({360 * i // n_subj},70%,50%)" for i in range(n_subj)]
        subj_color_map = {s: palette[i] for i, s in enumerate(unique_subjects)}
        subj_color_map[-1] = "#aaaaaa"
        point_labels = [str(int(s)) for s in subject]
        colors = [subj_color_map.get(int(s), "#aaaaaa") for s in subject]
        legend_entries = None  # Too many to list in a legend

    elif color_by == "Cluster":
        cluster_labels = token_meta.get("_cluster_labels", np.zeros(len(band), dtype=np.int32))
        n_cl = int(cluster_labels.max()) + 1
        pal = _META_QUAL_PALETTE
        point_labels = [str(int(c)) for c in cluster_labels]
        colors = [pal[int(c) % len(pal)] for c in cluster_labels]
        legend_entries = {f"Cluster {i}": pal[i % len(pal)] for i in range(n_cl)}

    elif color_by in token_meta:
        raw_vals = token_meta[color_by]
        if color_by == "recording_date":
            raw_vals = np.array([v[:4] if len(v) >= 4 else v for v in raw_vals], dtype=object)

        if color_by == "token_position":
            # Sequential colorscale — position is ordinal (sort numerically)
            unique_vals = sorted(set(str(v) for v in raw_vals), key=lambda x: int(x))
            gradient = px.colors.sample_colorscale("Viridis", [i / max(len(unique_vals) - 1, 1) for i in range(len(unique_vals))])
            val_to_color = {v: gradient[i] for i, v in enumerate(unique_vals)}
        elif color_by == "age_group":
            # Sort by leading number so '4-9' < '10-19'; put 'unknown' last in grey
            def _age_sort_key(v: str) -> int:
                try:
                    return int(v.split("-")[0].replace("+", ""))
                except ValueError:
                    return 999
            known_vals = sorted(
                set(str(v) for v in raw_vals if str(v) not in ("", "unknown")),
                key=_age_sort_key,
            )
            gradient = px.colors.sample_colorscale("Plasma", [i / max(len(known_vals) - 1, 1) for i in range(len(known_vals))])
            val_to_color = {v: gradient[i] for i, v in enumerate(known_vals)}
            val_to_color["unknown"] = "#666666"
            unique_vals = known_vals + (["unknown"] if "unknown" in set(str(v) for v in raw_vals) else [])
        else:
            unique_vals = sorted(set(str(v) for v in raw_vals if str(v) != ""))
            pal = _META_QUAL_PALETTE
            val_to_color = {v: pal[i % len(pal)] for i, v in enumerate(unique_vals)}

        point_labels = [str(v) for v in raw_vals]
        colors = [val_to_color.get(lb, "#888888") for lb in point_labels]
        legend_entries = {v: val_to_color[v] for v in unique_vals if v in val_to_color}

    else:  # fallback
        unique_subjects = sorted(set(int(s) for s in subject if int(s) >= 0))
        n_subj = len(unique_subjects)
        if n_subj <= 26:
            palette = px.colors.qualitative.Alphabet[:n_subj]
        else:
            palette = [f"hsl({360 * i // n_subj},70%,50%)" for i in range(n_subj)]
        subj_color_map = {s: palette[i] for i, s in enumerate(unique_subjects)}
        subj_color_map[-1] = "#aaaaaa"
        point_labels = [str(int(s)) for s in subject]
        colors = [subj_color_map.get(int(s), "#aaaaaa") for s in subject]
        legend_entries = None

    # ── Per-point opacity (highlight one category, dim others) ────────────────
    if highlight and highlight != "All":
        opacities = [0.9 if lb == highlight else 0.08 for lb in point_labels]
        sizes = [5 if lb == highlight else 3 for lb in point_labels]
    else:
        opacities = 0.65
        sizes = 4

    # ── Hover text (shared across all frames) ─────────────────────────────────
    _tok_pos = cache.get("token_positions")
    
    max_label = np.max(label)
    names_dict = _GRAN_LABEL_NAMES if max_label > 1 else _LABEL_NAMES
    
    hover = [
        f"band: {b}<br>label: {names_dict.get(int(l), str(l))}<br>subj: {int(s)}"
        + (f"<br>tok_pos: {int(_tok_pos[i])}" if _tok_pos is not None else "")
        for i, (b, l, s) in enumerate(zip(band, label, subject))
    ]

    marker_cfg = dict(color=colors, size=sizes, opacity=opacities, line=dict(width=0))

    # ── Global viewport — covers all layers so points can move within it ─────
    all_x = np.concatenate([cache["xy"][L][:, 0] for L in layers])
    all_y = np.concatenate([cache["xy"][L][:, 1] for L in layers])
    _xpad = (all_x.max() - all_x.min()) * 0.05
    _ypad = (all_y.max() - all_y.min()) * 0.05
    x_range = [float(all_x.min() - _xpad), float(all_x.max() + _xpad)]
    y_range = [float(all_y.min() - _ypad), float(all_y.max() + _ypad)]

    # ── Animation frames — data only, no layout overrides ────────────────────
    frames = [
        go.Frame(
            data=[go.Scatter(
                x=cache["xy"][L][:, 0].tolist(),
                y=cache["xy"][L][:, 1].tolist(),
                mode="markers",
                marker=marker_cfg,
                hovertext=hover,
                hoverinfo="text",
            )],
            name=str(L),
        )
        for L in layers
    ]

    # ── Legend dummy traces ───────────────────────────────────────────────────
    legend_traces = []
    if legend_entries:
        for name, color in legend_entries.items():
            legend_traces.append(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=color, size=8),
                name=name, showlegend=True,
            ))

    first_xy = cache["xy"][layers[0]]

    fig = go.Figure(
        data=[go.Scatter(
            x=first_xy[:, 0].tolist(),
            y=first_xy[:, 1].tolist(),
            mode="markers",
            marker=marker_cfg,
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )] + legend_traces,
        frames=frames,
    )

    fig.update_layout(
        title=f"{cache['display_name']} — Joint UMAP across layers",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="UMAP-1",
                   range=x_range),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="UMAP-2",
                   range=y_range),
        height=600,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(t=50, b=60, l=40, r=10),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0, x=0.5, xanchor="center", yanchor="top",
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=800, redraw=False),
                        transition=dict(duration=600, easing="cubic-in-out"),
                        fromcurrent=True,
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=False),
                    )],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            y=0.02, x=0.08, xanchor="left", len=0.84,
            currentvalue=dict(prefix="", font=dict(size=14, color="#fafafa")),
            transition=dict(duration=400, easing="cubic-in-out"),
            steps=[
                dict(
                    args=[[str(L)], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=400, easing="cubic-in-out"),
                    )],
                    label="Tokenizer" if L == -1 else str(L),
                    method="animate",
                )
                for L in layers
            ],
        )],
    )
    return fig


@st.cache_data(show_spinner=False)
def _load_spectral_decoder_comparison_metrics() -> Optional[dict]:
    """Load the JSON produced by compare_spectral_decoder.py, or None if not yet built."""
    p = ROOT / "results" / "spectral_decoder" / "spectral_decoder_comparison_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _load_all_tcav_caches_for_comparison() -> dict:
    """
    Scan results/tcav/ and load model_tcav_scores + cav_accuracies from every
    available cache.  Returns {exp_name: {...}} keyed by experiment directory name.
    """
    tcav_root = ROOT / "results" / "tcav"
    data: dict = {}
    if not tcav_root.exists():
        return data
    for exp_dir in sorted(tcav_root.iterdir()):
        cache_p = exp_dir / "tcav_cache.pt"
        if not cache_p.exists():
            continue
        try:
            cache = torch.load(str(cache_p), map_location="cpu", weights_only=False)
            model_tcav = cache.get("model_tcav_scores")
            if model_tcav is None:
                continue
            concept_names = list(cache.get("concept_names", []))
            cav_acc       = cache.get("cav_accuracies")
            probe_acc     = cache.get("probe_accuracies")
            data[exp_dir.name] = {
                "concept_names":     concept_names,
                "model_tcav_scores": (model_tcav.tolist() if torch.is_tensor(model_tcav)
                                      else list(model_tcav)),
                "cav_accuracies":    (cav_acc.tolist() if torch.is_tensor(cav_acc)
                                      else list(cav_acc)) if cav_acc is not None else None,
                "probe_accuracies":  (probe_acc.tolist() if torch.is_tensor(probe_acc)
                                      else list(probe_acc)) if probe_acc is not None else None,
            }
        except Exception:
            continue
    return data


def _exp_name_to_encoder_layer(exp_name: str):
    """Parse 'sleepfm_v2.4_layer5' → ('sleepfm_v2.4', 5)."""
    try:
        parts = exp_name.rsplit("_layer", 1)
        enc   = parts[0]
        layer = int(parts[1])
        return enc, layer
    except (IndexError, ValueError):
        return exp_name, -1


# ─────────────────────────────────────────────────────────────────────────────
def _render_tokenizer_clusters(cache: dict, k: int) -> None:
    """Show per-cluster band/label breakdown and representative EEG waveforms."""
    km_data       = cache["kmeans"][k]
    cluster_lbls  = km_data["labels"]           # (N,)
    band_arr      = cache["band"]
    label_arr     = cache["label"]
    file_indices  = cache["file_indices"]
    local_indices = cache["local_indices"]
    token_positions = cache["token_positions"]
    token_meta    = cache.get("token_meta", {})
    N             = len(cluster_lbls)

    # Resolve HDF5 file list from the data_path stored in the cache
    data_path = Path(cache.get("data_path", ROOT / "data" / "D4-v3-preprocessed-v2"))
    h5_paths  = sorted(data_path.glob("*.hdf5"))

    pal = _META_QUAL_PALETTE

    # ── Pre-compute consistent color maps for metadata fields ─────────────────
    def _age_sort_key_tc(v: str) -> int:
        try:
            return int(v.split("-")[0].replace("+", ""))
        except ValueError:
            return 999

    _age_arr = np.array(token_meta["age_group"]) if "age_group" in token_meta else None
    if _age_arr is not None:
        _age_known = sorted(
            set(str(v) for v in _age_arr if str(v) not in ("", "unknown")),
            key=_age_sort_key_tc,
        )
        _age_gradient = px.colors.sample_colorscale(
            "Plasma", [i / max(len(_age_known) - 1, 1) for i in range(len(_age_known))]
        )
        _age_color_map: dict[str, str] = {v: _age_gradient[i] for i, v in enumerate(_age_known)}
        _age_color_map["unknown"] = "#666666"
        _age_all_vals = _age_known + (["unknown"] if "unknown" in set(str(v) for v in _age_arr) else [])
    else:
        _age_color_map = {}
        _age_all_vals: list[str] = []

    # Global band counts for enrichment
    _uv_band_gl, _uc_band_gl = np.unique(band_arr, return_counts=True)
    _gl_band = dict(zip(_uv_band_gl.tolist(), _uc_band_gl.tolist()))

    _gender_arr = np.array(token_meta["gender"]) if "gender" in token_meta else None
    if _gender_arr is not None:
        _gender_vals = sorted(set(str(v) for v in _gender_arr if str(v) not in ("", "unknown")))
        _gender_pal = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8"]
        _gender_color_map: dict[str, str] = {v: _gender_pal[i % len(_gender_pal)] for i, v in enumerate(_gender_vals)}
        _gender_color_map["unknown"] = "#666666"
    else:
        _gender_color_map = {}

    for ci in range(k):
        mask = cluster_lbls == ci
        n_cl = int(mask.sum())
        if n_cl == 0:
            continue

        color = pal[ci % len(pal)]
        st.markdown(
            f"<span style='color:{color}; font-size:1.1em; font-weight:bold'>"
            f"Cluster {ci}</span> &nbsp;— {n_cl:,} tokens ({100 * n_cl / N:.1f}%)",
            unsafe_allow_html=True,
        )
        col_stats, col_wave = st.columns([1, 2])

        with col_stats:
            # ── All bar plots (enrichment vs. population baseline) ────────────
            from plotly.subplots import make_subplots as _msp_tc

            # Collect (title, [(label, enrichment, hover_text), ...], [colors])
            _bar_data: list[tuple[str, list[tuple[str, float, str]], list[str] | None]] = []

            def _enrichment(cluster_counts: dict[str, int], global_counts: dict[str, int], n_cl: int, N: int) -> list[tuple[str, float, str]]:
                """cluster% / global% — 1.0 = baseline."""
                out = []
                for v, cc in sorted(cluster_counts.items(), key=lambda x: -x[1]):
                    gc = global_counts.get(v, 0)
                    g_pct = 100 * gc / N if N > 0 else 0
                    c_pct = 100 * cc / n_cl if n_cl > 0 else 0
                    ratio = c_pct / g_pct if g_pct > 0 else 0.0
                    out.append((str(v), ratio, f"{c_pct:.0f}% vs {g_pct:.0f}% global (×{ratio:.2f})"))
                return out

            # Band
            _ub, _uc = np.unique(band_arr[mask], return_counts=True)
            _cl_band = dict(zip(_ub.tolist(), _uc.tolist()))
            _enr_band = _enrichment(_cl_band, _gl_band, n_cl, N)
            if _enr_band:
                _band_clrs = [BAND_COLORS.get(v, "#aaaaaa") for v, _, _ in _enr_band]
                _bar_data.append(("Band", _enr_band, _band_clrs))

            # Label
            _lv_map = {0: ("normal", "#4CAF50"), 1: ("abnormal", "#F44336")}
            _lbl_cl  = {nm: int((label_arr[mask] == lv).sum()) for lv, (nm, _) in _lv_map.items()}
            _lbl_gl  = {nm: int((label_arr == lv).sum())       for lv, (nm, _) in _lv_map.items()}
            _lbl_enr = _enrichment(_lbl_cl, _lbl_gl, n_cl, N)
            if _lbl_enr:
                _lbl_clrs = [_lv_map[[lv for lv, (nm2, _) in _lv_map.items() if nm2 == nm][0]][1]
                             for nm, _, _ in _lbl_enr]
                _bar_data.append(("Label", _lbl_enr, _lbl_clrs))

            # Age group — all values in sorted order
            if _age_arr is not None:
                _sub_age = _age_arr[mask]
                _cl_age = {v: int((_sub_age == v).sum()) for v in _age_all_vals}
                _gl_age = {v: int((_age_arr == v).sum()) for v in _age_all_vals}
                _enr_age = [
                    (v, (_cl_age[v] / n_cl) / (_gl_age[v] / N) if _gl_age[v] > 0 else 0.0,
                     f"{100*_cl_age[v]/n_cl:.0f}% vs {100*_gl_age[v]/N:.0f}% global (×{(_cl_age[v]/n_cl)/(_gl_age[v]/N):.2f})"
                     if _gl_age[v] > 0 else f"{100*_cl_age[v]/n_cl:.0f}% (no global data)")
                    for v in _age_all_vals
                ]
                _bar_data.append(("Age group", _enr_age, [_age_color_map.get(v, "#aaaaaa") for v, _, _ in _enr_age]))

            # Gender
            if _gender_arr is not None:
                _sub_gen = _gender_arr[mask]
                _uv_gl, _uc_gl = np.unique(_gender_arr, return_counts=True)
                _gl_gen = dict(zip(_uv_gl.tolist(), _uc_gl.tolist()))
                _uv_cl, _uc_cl = np.unique(_sub_gen, return_counts=True)
                _cl_gen = dict(zip(_uv_cl.tolist(), _uc_cl.tolist()))
                _enr_gen = _enrichment(_cl_gen, _gl_gen, n_cl, N)
                if _enr_gen:
                    _bar_data.append(("Sex", _enr_gen, [_gender_color_map.get(v, "#aaaaaa") for v, _, _ in _enr_gen]))

            if _bar_data:
                _nrows = len(_bar_data)
                _fig_meta = _msp_tc(
                    rows=_nrows, cols=1,
                    subplot_titles=[d[0] for d in _bar_data],
                    vertical_spacing=0.06,
                )
                _x_max = max(v for _, cats, _ in _bar_data for _, v, _ in cats) if _bar_data else 2.0
                _x_range = [0, max(_x_max * 1.25, 2.1)]
                for _ri, (_title, _cats, _colors) in enumerate(_bar_data, start=1):
                    _names = [c[0] for c in _cats]
                    _vals  = [c[1] for c in _cats]
                    _tips  = [c[2] for c in _cats]
                    _clrs  = _colors if _colors else [color] * len(_cats)
                    _fig_meta.add_trace(
                        go.Bar(
                            x=_vals, y=_names,
                            orientation="h",
                            marker_color=_clrs,
                            marker_opacity=0.85,
                            text=[f"×{v:.1f}" for v in _vals],
                            textposition="outside",
                            textfont=dict(size=9),
                            cliponaxis=False,
                            showlegend=False,
                            customdata=_tips,
                            hovertemplate="%{customdata}<extra></extra>",
                        ),
                        row=_ri, col=1,
                    )
                    # Baseline reference line at 1×
                    _fig_meta.add_vline(
                        x=1.0, line_width=1, line_dash="dot", line_color="#666666",
                        row=_ri, col=1,
                    )
                    _fig_meta.update_xaxes(
                        range=_x_range, showticklabels=False,
                        showgrid=False, zeroline=False,
                        row=_ri, col=1,
                    )
                    _fig_meta.update_yaxes(
                        autorange="reversed",
                        tickmode="array",
                        tickvals=_names,
                        ticktext=_names,
                        tickfont=dict(size=9),
                        showgrid=False,
                        automargin=True,
                        row=_ri, col=1,
                    )
                _fig_meta.update_layout(
                    height=sum(max(70, 20 * len(cats)) for _, cats, _ in _bar_data),
                    margin=dict(l=0, r=30, t=20, b=5),
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="#fafafa", size=9),
                    showlegend=False,
                )
                for _ann in _fig_meta.layout.annotations:
                    _ann.font = dict(size=10, color="#aaaaaa")
                    _ann.x = 0
                    _ann.xanchor = "left"
                st.plotly_chart(_fig_meta, use_container_width=True, config={"displayModeBar": False})

        with col_wave:
            idxs = np.where(mask)[0]
            rng  = np.random.default_rng(seed=42 + ci)
            chosen = rng.choice(idxs, size=min(4, len(idxs)), replace=False)

            fig_w = go.Figure()
            for ex_i, tok_idx in enumerate(chosen):
                fi  = int(file_indices[tok_idx])
                li  = int(local_indices[tok_idx])
                tp  = int(token_positions[tok_idx])
                if fi >= len(h5_paths):
                    continue
                patch = _load_eeg_patch(str(h5_paths[fi]), li, tp)  # (C, 128)
                t_ax  = np.arange(_PATCH_SIZE) / _PATCH_SIZE        # 0..1 s
                x_off = ex_i * 1.15
                _ch_spacing = 2.5
                for ch_i, ch_name in enumerate(_CHANNEL_NAMES):
                    if ch_i >= patch.shape[0]:
                        continue
                    sig = patch[ch_i]
                    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
                    fig_w.add_trace(go.Scatter(
                        x=(t_ax + x_off).tolist(),
                        y=(sig + ch_i * _ch_spacing).tolist(),
                        mode="lines",
                        line=dict(width=1, color=color),
                        showlegend=False,
                        hovertemplate=f"{ch_name} ex{ex_i+1}<extra></extra>",
                    ))
            _ch_spacing = 2.5
            fig_w.update_layout(
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(
                    tickmode="array",
                    tickvals=[i * _ch_spacing for i in range(len(_CHANNEL_NAMES))],
                    ticktext=_CHANNEL_NAMES,
                    tickfont=dict(size=9), showgrid=False,
                ),
                height=520, margin=dict(l=45, r=5, t=5, b=5),
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#fafafa"),
                showlegend=False,
            )
            st.plotly_chart(fig_w, use_container_width=True)
        st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy & Steering — paper figures (with one interactive panel)
# ─────────────────────────────────────────────────────────────────────────────

_PAPER_FIG_DIR = ROOT / "tools" / "paper_figures"


def _show_paper_image(rel_dir: str, caption: str) -> None:
    p = _PAPER_FIG_DIR / rel_dir / "figure.png"
    if p.exists():
        st.image(str(p), use_container_width=True, caption=caption)
    else:
        st.info(f"`{p.relative_to(ROOT)}` not found.")


@st.cache_data(show_spinner="Loading steering-curve data …")
def _load_steering_curves_data() -> Dict[Tuple[str, str], dict]:
    """Load Figure 5 data — returns {(concept, experiment) → entry-dict}."""
    p = _PAPER_FIG_DIR / "Figure 5" / "data.npz"
    if not p.exists():
        return {}
    raw = np.load(p, allow_pickle=True)
    out: Dict[Tuple[str, str], dict] = {}
    for k in raw.files:
        e = json.loads(str(raw[k]))
        out[(e["concept"], e["experiment"])] = e
    return out


_EXP_RE_SLEEPFM = re.compile(r"^sleepfm_finetuned(?:_exp(\d+))?_layer(\d+)$")
_EXP_RE_LABRAM  = re.compile(r"^labram_layer(\d+)(?:_exp(\d+))?$")
_EXP_RE_REVE    = re.compile(r"^reve_qjbe08(?:_exp(\d+))?_layer(\d+)$")


def _parse_experiment(exp: str) -> Optional[Tuple[str, int, int]]:
    """Parse experiment name into (family, expansion, layer_0idx)."""
    if exp.endswith("_multi"):
        return None
    m = _EXP_RE_SLEEPFM.match(exp)
    if m:
        E = int(m.group(1)) if m.group(1) else 1
        return ("SleepFM", E, int(m.group(2)))
    m = _EXP_RE_LABRAM.match(exp)
    if m:
        E = int(m.group(2)) if m.group(2) else 1
        return ("LaBraM", E, int(m.group(1)))
    m = _EXP_RE_REVE.match(exp)
    if m:
        E = int(m.group(1)) if m.group(1) else 1
        return ("REVE", E, int(m.group(2)))
    return None


@st.cache_data(show_spinner="Loading taxonomy data …")
def _load_taxonomy_data() -> Dict[str, dict]:
    """Load Figure 3 data — {exp_name: {separable, entangled, dead}}."""
    p = _PAPER_FIG_DIR / "Figure 3" / "data.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


_TAXONOMY_ENCODERS = [
    {"name": "SleepFM", "layers": [0, 1, 2],         "exp_for":
        lambda L, E: f"sleepfm_finetuned_layer{L}" if E == 1 else f"sleepfm_finetuned_exp{E}_layer{L}"},
    {"name": "LaBraM",  "layers": list(range(12)),   "exp_for":
        lambda L, E: f"labram_layer{L}" if E == 1 else f"labram_layer{L}_exp{E}"},
    {"name": "REVE",    "layers": list(range(22)),   "exp_for":
        lambda L, E: f"reve_qjbe08_layer{L}" if E == 1 else f"reve_qjbe08_exp{E}_layer{L}"},
]
_TAXONOMY_EXPANSIONS = [1, 2, 4, 8, 16, 32, 64]
_TAXONOMY_METRICS = {
    "Separable": ("separable", "Greens", "Fraction of concept-enriched SAE features that respond to one and only one concept."),
    "Entangled": ("entangled", "Oranges", "Fraction sharing one neural primitive across multiple labels (e.g. age + pathology via δ/θ)."),
    "Dead":      ("dead",      "Reds",    "Fraction of SAE features that never fire — capacity-limited."),
}


def _render_taxonomy_subtab() -> None:
    st.markdown(
        "**Figure 3 (§3.3)** — fraction of concept-enriched SAE features in each of three taxonomy classes "
        "(Separable / Entangled / Dead), across layers × expansion factors, per encoder. "
        "Hover any cell to see exact values. Hatched cells = no SAE trained at that (ℓ, E)."
    )

    taxonomy = _load_taxonomy_data()
    if not taxonomy:
        st.info("`tools/paper_figures/Figure 3/data.json` not found.")
        return

    metric_name = st.radio(
        "Metric",
        list(_TAXONOMY_METRICS.keys()),
        horizontal=True,
        key="taxonomy_metric",
        help=" · ".join(f"**{k}** — {v[2]}" for k, v in _TAXONOMY_METRICS.items()),
    )
    field, cmap, blurb = _TAXONOMY_METRICS[metric_name]
    st.caption(blurb)

    n_enc = len(_TAXONOMY_ENCODERS)
    cols = st.columns(n_enc, gap="medium")
    for col, cfg in zip(cols, _TAXONOMY_ENCODERS):
        with col:
            layers = cfg["layers"]
            Es     = _TAXONOMY_EXPANSIONS
            mat    = np.full((len(layers), len(Es)), np.nan)
            for i, L in enumerate(layers):
                for j, E in enumerate(Es):
                    entry = taxonomy.get(cfg["exp_for"](L, E))
                    if entry is not None:
                        mat[i, j] = entry.get(field, np.nan) * 100.0
            hover_text = [[f"L{L} · E={E}<br>{metric_name}: {mat[i,j]:.1f}%"
                           if not np.isnan(mat[i,j]) else f"L{L} · E={E}<br>no SAE trained"
                           for j, E in enumerate(Es)] for i, L in enumerate(layers)]
            fig = go.Figure(data=go.Heatmap(
                z=mat,
                x=[f"{E}×" for E in Es],
                y=[f"L{L}" for L in layers],
                colorscale=cmap,
                zmin=0, zmax=100,
                hoverinfo="text",
                text=hover_text,
                colorbar=dict(title=dict(text="%", side="right"), tickformat=".0f", thickness=10),
            ))
            fig.update_layout(
                title=f"<b>{cfg['name']}</b>",
                height=max(220, 32 * len(layers) + 110),
                margin=dict(l=40, r=10, t=40, b=40),
                xaxis_title="Expansion",
                yaxis_title="Layer",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show paper Figure 3 (static composite)"):
        _show_paper_image("Figure 3", "Figure 3 — Monosemanticity taxonomy (paper composite)")

    st.markdown("---")
    st.markdown(
        "**Figure 2 (§3.2)** — SAE-faithfulness layer sweep. Test AUROC of a linear probe trained on "
        "mean-pooled embeddings as layer-ℓ activations are replaced by their TopK-SAE reconstructions."
    )
    _show_paper_image("Figure 2", "Figure 2 — SAE-faithfulness layer sweep")


def _render_steering_static_subtab() -> None:
    st.markdown(
        "**Figure 4 (§5)** — per-encoder, per-layer encoding strength (top row) and steering selectivity (bottom row) "
        "for six clinical concepts (Age, Pathology, Sex, Epileptic Activity, ASM medication, Psychiatric medication)."
    )
    _show_paper_image("Figure 4", "Figure 4 — Cross-model concept encoding & steering selectivity")

    st.markdown("---")
    st.markdown(
        "**Figure 6 (§6)** — spectrum-level concept steering on SleepFM L2 (E=8). Decoded spectrum after "
        "clamping the top *n* TCAV-aligned features to the normal-class centroid."
    )
    _show_paper_image("Figure 6", "Figure 6 — Spectrum-level concept steering")


def _render_steering_curves_subtab() -> None:
    st.markdown(
        "**Figure 5 (§5)** — interactive variant. Pick encoder family, expansion ratio, layer, and target concept. "
        "**Red** curve = target-concept AUROC, **blue** = Pathology (off-target), grey dotted = random-direction "
        "baseline. AUROC at *f=0* tells you how well the concept is *encoded*; the gap between target and "
        "off-target curves (vs the random baseline) tells you how *selectively* it can be steered."
    )
    data = _load_steering_curves_data()
    if not data:
        st.info("Figure 5 data not found at `tools/paper_figures/Figure 5/data.npz`.")
        return

    parsed: Dict[str, Tuple[str, int, int]] = {}
    for _, exp in data.keys():
        if exp not in parsed:
            p = _parse_experiment(exp)
            if p is not None:
                parsed[exp] = p

    concepts = sorted({c for c, _ in data})
    families = [f for f in ("SleepFM", "LaBraM", "REVE")
                if any(v[0] == f for v in parsed.values())]

    col_f, col_e, col_l, col_c = st.columns([1.2, 1.2, 1.2, 1.6])
    with col_f:
        family = st.selectbox("Encoder", families, key="steering_curves_family")
    exps_in_family = {exp: (E, L) for exp, (fam, E, L) in parsed.items() if fam == family}
    available_E = sorted({E for E, _ in exps_in_family.values()})
    with col_e:
        E_sel = st.selectbox("Expansion E", available_E, key="steering_curves_E", index=0)
    available_L = sorted({L for exp, (E, L) in exps_in_family.items() if E == E_sel})
    with col_l:
        L_sel = st.selectbox("Layer (0-indexed)", available_L, key="steering_curves_L", index=0)

    experiment = next((exp for exp, (E, L) in exps_in_family.items()
                       if E == E_sel and L == L_sel), None)

    # Concept selector: greyed-out labels for concepts without data at this config.
    ordered = [c for c in ("Age", "Pathology", "Gender",
                           "Medication (ASM)", "Medication (Psychiatric)") if c in concepts]
    controls = [c for c in concepts if c not in ordered]
    all_concepts = ordered + controls
    available_here = {c for c in all_concepts if experiment is not None and (c, experiment) in data}

    def _label(c: str) -> str:
        return c if c in available_here else f"{c} — not run for this config"

    with col_c:
        concept = st.selectbox(
            "Concept (target)",
            all_concepts,
            format_func=_label,
            key="steering_curves_concept",
        )

    if experiment is None:
        st.info("No data for this configuration.")
        return

    entry = data.get((concept, experiment))
    if entry is None:
        # Coverage hint: which configs DO have this concept?
        with_concept = sorted({exp for c, exp in data.keys() if c == concept})
        in_family = [exp for exp in with_concept if parsed.get(exp, (None,))[0] == family]
        st.warning(
            f"**`{concept}`** has not been run for **{family} · L{L_sel} · E={E_sel}**. "
            f"Across all encoders, {len(with_concept)}/{len(parsed)} configs have it; "
            f"within {family}, {len(in_family)}/{sum(1 for v in parsed.values() if v[0]==family)} do."
        )
        st.caption(
            "**Coverage note:** *Age* was run on every (encoder × layer × expansion). "
            "*Pathology*, *Gender*, *Medication (ASM)* and *Medication (Psychiatric)* were only run on the "
            "primary-expansion configs plus a handful of higher-E variants — the rest would need a fresh "
            "steering pass against the per-token labels."
        )
        return

    fracs    = np.asarray(entry["fracs"], dtype=float)
    tgt_auc  = np.asarray(entry["tgt_auc"], dtype=float)
    off_auc  = np.asarray(entry["off_auc"], dtype=float)
    rand_tgt = np.asarray(entry.get("rand_tgt_auc_mean", []), dtype=float)
    rand_off = np.asarray(entry.get("rand_off_auc_mean", []), dtype=float)

    # ── Headline metrics (AUROC₀ + excess steerability) ────────────────────────
    boot_tgt0 = np.asarray(entry.get("bootstrap_tgt_auc0", []), dtype=float)
    auroc0      = float(boot_tgt0.mean()) if boot_tgt0.size else float(tgt_auc[0])
    auroc0_std  = float(boot_tgt0.std(ddof=1)) if boot_tgt0.size > 1 else 0.0

    ba = np.asarray(entry.get("bootstrap_areas", []), dtype=float)
    br = np.asarray(entry.get("bootstrap_rand_areas", []), dtype=float)
    if ba.size and ba.size == br.size:
        excess_mean = float((ba - br).mean())
        excess_std  = float((ba - br).std(ddof=1)) if ba.size > 1 else 0.0
    else:
        area      = float(entry.get("area", 0.0))
        rand_area = float(entry.get("rand_area_mean", 0.0))
        excess_mean = area - rand_area
        excess_std  = float(entry.get("rand_area_std", 0.0))

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "AUROC₀ — concept encoding strength",
            f"{auroc0:.3f}",
            delta=f"± {auroc0_std:.3f}" if auroc0_std > 0 else None,
            delta_color="off",
            help="Target-concept linear-probe AUROC at clamping fraction f=0 (no intervention). "
                 "Bootstrap mean ± std across resamples of the probed token set.",
        )
    with m2:
        sign = "+" if excess_mean >= 0 else ""
        st.metric(
            "Excess steerability",
            f"{sign}{excess_mean:.3f}",
            delta=f"± {excess_std:.3f}" if excess_std > 0 else None,
            delta_color="off",
            help="(target − off-target) AUROC area swept across f, minus the random-direction baseline. "
                 "Positive = concept can be steered more than a random direction; near-zero = entangled.",
        )
    with m3:
        st.metric(
            "Sample size",
            f"{entry.get('n_target_per_group', '?')} / group",
            delta=f"{entry.get('n_features', '?')} SAE features",
            delta_color="off",
            help="Tokens per group used for the AUROC computation, and SAE dictionary size.",
        )

    fig = go.Figure()
    if rand_tgt.size == fracs.size:
        fig.add_scatter(x=fracs, y=rand_tgt, mode="lines", name="random — target",
                        line=dict(color="#bbbbbb", dash="dot", width=1))
    if rand_off.size == fracs.size:
        fig.add_scatter(x=fracs, y=rand_off, mode="lines", name="random — Pathology",
                        line=dict(color="#888888", dash="dot", width=1))
    fig.add_scatter(x=fracs, y=tgt_auc, mode="lines+markers",
                    name=f"{concept} (target)",
                    line=dict(color="#c0392b", width=2.5))
    fig.add_scatter(x=fracs, y=off_auc, mode="lines+markers",
                    name="Pathology (off-target)",
                    line=dict(color="#1f6fb4", width=2.5))
    fig.update_layout(
        height=460,
        xaxis_title="Clamping fraction f",
        yaxis_title="AUROC",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=70, b=50),
    )
    fig.update_yaxes(range=[0.0, 1.0])
    fig.add_annotation(
        text=f"<b>{family} · layer {L_sel} · E={E_sel}</b> &nbsp;·&nbsp; target = {concept}",
        xref="paper", yref="paper", x=0.5, y=1.18,
        showarrow=False, font=dict(size=14),
        xanchor="center",
    )
    st.plotly_chart(fig, use_container_width=True)


def page_taxonomy_steering() -> None:
    """Paper-aligned Taxonomy & Steering figures with one interactive panel."""
    st.title("Taxonomy & Steering")
    st.caption(
        "**Paper §3.2 · Fig 2 + §3.3 · Fig 3 + §5–6 · Figs 4 / 5 / 6** — the paper's quantitative figures, "
        "with an interactive variant of Figure 5."
    )
    tab_tax, tab_steer, tab_curves = st.tabs([
        "Taxonomy",
        "Steering (paper figures)",
        "Steering curves (interactive)",
    ])
    with tab_tax:
        _render_taxonomy_subtab()
    with tab_steer:
        _render_steering_static_subtab()
    with tab_curves:
        _render_steering_curves_subtab()



def page_layer_explorer(folder_name: str) -> None:
    """Layer Explorer — animated joint UMAP showing token trajectories across layers."""
    st.header("Layer Explorer")
    st.caption("**Paper §3** — animated joint UMAP showing token trajectories across encoder layers.")
    st.caption(
        "Animated joint UMAP: the same tokens appear at every layer in a shared "
        "coordinate space, so you can see how representations evolve with depth.  "
        "Use the Play button or drag the layer slider."
    )

    umap_dir  = ROOT / "results" / "layer_umap"
    cache_path = umap_dir / folder_name / "umap_cache.pt"

    if not cache_path.exists():
        st.info(
            f"No layer UMAP cache found for **{folder_name}**.  Build it with:\n\n"
            f"```\nuv run tools/build_layer_umap_cache.py --encoder {folder_name}\n```\n\n"
            f"Cache will be saved to `results/layer_umap/{folder_name}/umap_cache.pt`."
        )
        return

    cache = load_layer_umap_cache(str(cache_path))

    # ── Controls ──────────────────────────────────────────────────────────────
    token_meta = cache.get("token_meta", {})
    meta_fields = [k for k, v in token_meta.items()
                   if v is not None and len(np.unique(v)) > 1]
    has_kmeans = bool(cache.get("kmeans")) and -1 in cache.get("layers", [])

    color_options = (["Cluster"] if has_kmeans else []) + ["Band", "Label", "Subject"] + meta_fields

    col_cb, col_hl, col_info = st.columns([2, 2, 2])
    with col_cb:
        color_by = st.selectbox(
            "Color by",
            options=color_options,
            index=color_options.index("Label") if "Label" in color_options else 0,
            key="layer_explorer_color_by",
        )
    with col_hl:
        if color_by == "Cluster" and has_kmeans:
            available_ks = sorted(cache["kmeans"].keys())
            k_clusters = st.slider(
                "K clusters",
                min_value=min(available_ks), max_value=max(available_ks),
                value=6, step=1, key="layer_explorer_k",
            )
            _cat_vals = [f"Cluster {i}" for i in range(k_clusters)]
        elif color_by == "Band":
            k_clusters = 6
            _cat_vals = sorted(set(str(b) for b in cache["band"]))
        elif color_by == "Label":
            max_label = max(int(l) for l in cache["label"])
            names_dict = _GRAN_LABEL_NAMES if max_label > 1 else _LABEL_NAMES
            k_clusters = len([v for v in sorted(set(int(l) for l in cache["label"])) if v in names_dict])
            _cat_vals = [names_dict[v] for v in sorted(set(int(l) for l in cache["label"])) if v in names_dict]
        elif color_by in token_meta:
            k_clusters = 6
            _raw = token_meta[color_by]
            if color_by == "recording_date":
                _raw = [v[:4] if len(v) >= 4 else v for v in _raw]
            if color_by == "token_position":
                _cat_vals = sorted(set(str(v) for v in _raw), key=lambda x: int(x))
            elif color_by == "age_group":
                def _age_sort_key(v: str) -> int:
                    try:
                        return int(v.split("-")[0].replace("+", ""))
                    except ValueError:
                        return 999
                _known = sorted(set(str(v) for v in _raw if str(v) not in ("", "unknown")), key=_age_sort_key)
                _cat_vals = _known + (["unknown"] if "unknown" in set(str(v) for v in _raw) else [])
            else:
                _cat_vals = sorted(set(str(v) for v in _raw if str(v) != ""))
        else:
            k_clusters = 6
            _cat_vals = []
        highlight = st.selectbox(
            "Highlight",
            options=["All"] + _cat_vals,
            key="layer_explorer_highlight",
        ) if _cat_vals else "All"
    with col_info:
        n_tok  = cache["n_tokens"]
        layers = cache["layers"]
        layer_range = f"Tokenizer – Layer {layers[-1]}" if -1 in layers else f"{layers[0]}–{layers[-1]}"
        st.markdown(
            f"**{n_tok:,}** tokens &nbsp;·&nbsp; "
            f"**{len(layers)}** steps ({layer_range}) &nbsp;·&nbsp; "
            f"Joint UMAP (consistent coordinates)"
        )

    # Inject cluster labels into token_meta for the figure builder
    if color_by == "Cluster" and has_kmeans:
        token_meta_fig = dict(token_meta)
        token_meta_fig["_cluster_labels"] = cache["kmeans"][k_clusters]["labels"]
        _hl_val = highlight.replace("Cluster ", "") if highlight != "All" else "All"
    else:
        token_meta_fig = token_meta
        _hl_val = highlight

    # ── Animated figure ───────────────────────────────────────────────────────
    fig = _make_animation_figure(cache, color_by, token_meta=token_meta_fig, highlight=_hl_val)
    st.plotly_chart(fig, use_container_width=True)

    # ── Band distribution summary ─────────────────────────────────────────────
    with st.expander("Band distribution"):
        band_arr = cache["band"]
        unique_bands, counts = np.unique(band_arr, return_counts=True)
        rows = sorted(zip(counts.tolist(), unique_bands.tolist()), reverse=True)
        cols = st.columns(min(len(rows), 4))
        for i, (cnt, bname) in enumerate(rows):
            with cols[i % len(cols)]:
                color = BAND_COLORS.get(str(bname), "#aaaaaa")
                pct   = 100 * cnt / n_tok
                st.markdown(
                    f"<span style='color:{color}; font-weight:bold'>{bname}</span><br>"
                    f"{cnt:,} &nbsp;({pct:.1f}%)",
                    unsafe_allow_html=True,
                )

    # ── Tokenizer cluster analysis ────────────────────────────────────────────
    if has_kmeans and "file_indices" in cache:
        with st.expander("Tokenizer cluster analysis", expanded=(color_by == "Cluster")):
            _render_tokenizer_clusters(cache, k_clusters)



# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Feature Prototypes page
# ─────────────────────────────────────────────────────────────────────────────

def _latest_proto_run(folder_name: str, layer: int) -> str | None:
    """Return the latest prototype run for this model+layer, or None."""
    if not _FEAT_VIZ_DIR.exists():
        return None
    prefix = f"{folder_name}_layer{layer}"
    candidates = sorted(
        d.name for d in _FEAT_VIZ_DIR.iterdir()
        if d.is_dir() and (d / "prototypes.npz").exists()
        and d.name.startswith(prefix)
    )
    return candidates[-1] if candidates else None


@st.cache_data
def _load_prototypes(run_name: str) -> dict:
    path = _FEAT_VIZ_DIR / run_name / "prototypes.npz"
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ─────────────────────────────────────────────────────────────────────────────
# Page — Attention Explorer
# ─────────────────────────────────────────────────────────────────────────────

# Channel colour map matching explore_features.py dashboards (index = channel position)
_ATTN_CH_COLORS = [
    "#1f77b4", "#aec7e8",                               # Fp1, Fp2
    "#17becf", "#9edae5", "#2ca02c", "#98df8a", "#17becf",  # F7–F8
    "#2ca02c", "#98df8a", "#bcbd22", "#98df8a", "#2ca02c",  # T7–T8
    "#d62728", "#e6550d", "#fd8d3c", "#e6550d", "#d62728",  # T5–T6
    "#9467bd", "#8c6bb1",                               # O1, O2
]


def page_attention_explorer(
    data: dict,
    attn_cache: Optional[dict],
    folder_name: str,
    layer: int,
) -> None:
    st.title("Attention Explorer")
    st.caption("Supplementary — encoder self-attention alongside SAE features (not a paper figure).")
    st.markdown(
        "Inspect temporal self-attention from **transformer layer 0** alongside "
        "SAE feature activations for individual 60-second windows. "
        "High *attention received* (column sum of the attention matrix) "
        "indicates tokens that the model treats as contextually important."
    )

    exp_name = f"{folder_name}_layer{layer}"

    if attn_cache is None:
        st.info(
            f"No attention cache found for **{exp_name}**. Build it with:\n\n"
            f"```\nuv run tools/build_attention_cache.py --experiment {exp_name}\n```"
        )
        return

    n_features  = int(attn_cache["n_features"])
    n_heads     = int(attn_cache["n_heads"])
    S           = int(attn_cache["S"])
    fs          = int(attn_cache["fs"])
    K           = int(attn_cache["K"])
    ch_names    = list(attn_cache["channel_names"])
    has_attn    = attn_cache["windows_attn"] is not None and n_heads > 0

    top_idx      = attn_cache["top_window_idx"]       # (n_features, K) long
    top_acts     = attn_cache["top_window_mean_acts"]  # (n_features, K)
    top_labels   = attn_cache["top_window_labels"]     # (n_features, K)
    wins_eeg     = attn_cache["windows_eeg"]           # (n_unique, C, T) float16
    wins_facts   = attn_cache["windows_feat_acts"]     # (n_unique, S, n_features) float16
    wins_attn    = attn_cache["windows_attn"]          # (n_unique, n_heads, S, S) float16 or None

    C = wins_eeg.shape[1]
    n_ch_display = min(C, len(ch_names))
    t_token = np.linspace(0.5, S - 0.5, S)            # token centres in seconds

    # ── Controls ──────────────────────────────────────────────────────
    cc1, cc2, cc3 = st.columns([2, 2, 1])

    # Optionally load TCAV for sorted feature list
    tcav_path = _tcav_cache_path(exp_name)
    tcav_cache = load_tcav_cache(str(tcav_path)) if tcav_path else None

    feat_labels: list[str] = []
    for fi in range(n_features):
        best_act = float(top_acts[fi, 0])
        feat_labels.append(f"Feature {fi}  (max mean act: {best_act:.3f})")

    with cc1:
        feat_idx = st.selectbox(
            "Feature",
            options=list(range(n_features)),
            format_func=lambda i: feat_labels[i],
            key="attn_feat_idx",
        )

    with cc2:
        rank = st.radio(
            "Window rank",
            options=list(range(K)),
            format_func=lambda r: f"Top {r + 1}  (act={float(top_acts[feat_idx, r]):.3f},"
                                  f" label={int(top_labels[feat_idx, r])})",
            horizontal=True,
            key="attn_win_rank",
        )

    with cc3:
        show_heads = st.checkbox("Per-head heatmaps", value=True, key="attn_show_heads")

    # ── Load window data ──────────────────────────────────────────────
    from plotly.subplots import make_subplots

    local_win = int(top_idx[feat_idx, rank])
    eeg_raw   = wins_eeg[local_win].float().numpy()                   # (C, T)
    feat_ts   = wins_facts[local_win, :, feat_idx].float().numpy()    # (S,)
    attn_mat: Optional[np.ndarray] = None
    if has_attn:
        attn_mat = wins_attn[local_win].float().numpy()               # (n_heads, S, S)

    # Bandpass-filter EEG for display
    eeg_bp = _bandpass(eeg_raw[:n_ch_display], fs=fs)                 # (n_ch, T)
    eeg_norm = _eeg_normalize(eeg_bp)

    t_eeg = np.arange(eeg_norm.shape[1]) / fs                        # (T,) in seconds

    # ── Figure 1: EEG traces + feature activation + attention ─────────
    n_rows = 3 if has_attn else 2
    row_h  = [0.70, 0.15, 0.15] if has_attn else [0.75, 0.25]
    fig_eeg = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_h,
        vertical_spacing=0.04,
    )

    # Row 1: EEG traces (stacked with channel offsets)
    spacing = 4.0
    for ci in range(n_ch_display):
        trace = eeg_norm[ci]
        offset = (n_ch_display - 1 - ci) * spacing
        color = _ATTN_CH_COLORS[ci] if ci < len(_ATTN_CH_COLORS) else "#888"
        fig_eeg.add_trace(
            go.Scattergl(
                x=t_eeg, y=trace + offset,
                mode="lines",
                line=dict(color=color, width=0.6),
                name=ch_names[ci],
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Channel label tick marks
    tick_vals = [(n_ch_display - 1 - ci) * spacing for ci in range(n_ch_display)]
    tick_text = ch_names[:n_ch_display]
    fig_eeg.update_yaxes(
        tickvals=tick_vals, ticktext=tick_text, tickfont=dict(size=9),
        row=1, col=1,
    )
    fig_eeg.update_xaxes(showticklabels=False, row=1, col=1)

    # Row 2: feature activation per token
    feat_norm = feat_ts / (feat_ts.max() + 1e-8)
    fig_eeg.add_trace(
        go.Scatter(
            x=t_token, y=feat_norm,
            mode="lines+markers",
            line=dict(color="#e6550d", width=1.8),
            marker=dict(size=4),
            name=f"Feature {feat_idx} activation",
            fill="tozeroy",
            fillcolor="rgba(230,85,13,0.15)",
        ),
        row=2, col=1,
    )
    fig_eeg.update_yaxes(title_text="Act (norm)", title_font=dict(size=10), row=2, col=1)
    fig_eeg.update_xaxes(showticklabels=False, row=2, col=1)

    # Row 3 (optional): attention received per token
    if has_attn and attn_mat is not None:
        attn_avg     = attn_mat.mean(axis=0)                          # (S, S) mean over heads
        attn_recv    = attn_avg.mean(axis=0)                          # (S,) column means
        attn_recv_n  = attn_recv / (attn_recv.max() + 1e-8)
        fig_eeg.add_trace(
            go.Scatter(
                x=t_token, y=attn_recv_n,
                mode="lines+markers",
                line=dict(color="#1f77b4", width=1.8),
                marker=dict(size=4),
                name="Attention received (mean heads)",
                fill="tozeroy",
                fillcolor="rgba(31,119,180,0.12)",
            ),
            row=3, col=1,
        )
        fig_eeg.update_yaxes(title_text="Attn (norm)", title_font=dict(size=10), row=3, col=1)
        fig_eeg.update_xaxes(title_text="Time (s)", row=3, col=1)
    else:
        fig_eeg.update_xaxes(title_text="Time (s)", row=2, col=1)

    fig_eeg.update_layout(
        height=520 if has_attn else 450,
        margin=dict(l=60, r=20, t=40, b=40),
        title_text=f"Feature {feat_idx} — Top {rank + 1} window "
                   f"(mean act={float(top_acts[feat_idx, rank]):.3f}, "
                   f"label={int(top_labels[feat_idx, rank])})",
        title_font=dict(size=13),
    )
    st.plotly_chart(fig_eeg, use_container_width=True)

    # ── Top patches by activation × attention ─────────────────────────
    if has_attn and attn_mat is not None:
        attn_avg_pre   = attn_mat.mean(axis=0)                # (S, S)
        attn_recv_pre  = attn_avg_pre.mean(axis=0)            # (S,) attention received
        feat_norm_pre  = feat_ts / (feat_ts.max() + 1e-8)
        attn_norm_pre  = attn_recv_pre / (attn_recv_pre.max() + 1e-8)
        joint_score    = feat_norm_pre * attn_norm_pre        # (S,)

        n_top_patches      = min(5, S)
        top_tok_idx        = np.argsort(joint_score)[::-1][:n_top_patches]
        patch_size_samples = int(attn_cache["patch_size"])
        spacing_ex         = 2.2
        t_patch            = np.arange(patch_size_samples) / fs

        subplot_titles = [f"#{i + 1}" for i in range(n_top_patches)]
        fig_top = make_subplots(
            rows=1, cols=n_top_patches,
            subplot_titles=subplot_titles,
            shared_yaxes=True,
            horizontal_spacing=0.02,
        )
        for col_i, tok_i in enumerate(top_tok_idx):
            start      = int(tok_i) * patch_size_samples
            patch      = eeg_raw[:n_ch_display, start : start + patch_size_samples]
            patch_bp   = _bandpass(patch, fs=fs)
            patch_norm = _eeg_normalize(patch_bp)
            for ci in range(n_ch_display):
                y_off = (n_ch_display - 1 - ci) * spacing_ex
                fig_top.add_trace(
                    go.Scatter(
                        x=t_patch.tolist(), y=(patch_norm[ci] + y_off).tolist(),
                        mode="lines", line=dict(width=0.9, color="steelblue"),
                        showlegend=False,
                        hovertemplate=f"<b>{ch_names[ci]}</b><br>t=%{{x:.3f}} s<extra></extra>",
                    ),
                    row=1, col=col_i + 1,
                )

        fig_top.update_yaxes(
            tickmode="array",
            tickvals=[(n_ch_display - 1 - c) * spacing_ex for c in range(n_ch_display)],
            ticktext=ch_names[:n_ch_display],
            tickfont=dict(size=10),
            row=1, col=1,
        )
        for col_i in range(2, n_top_patches + 1):
            fig_top.update_yaxes(showticklabels=False, row=1, col=col_i)
        for col_i in range(1, n_top_patches + 1):
            fig_top.update_xaxes(
                range=[0, 1.0],
                showticklabels=False, showgrid=False, zeroline=False,
                row=1, col=col_i,
            )

        # Scale bar: |——— 1 second ———|
        y_bar  = -2.2
        cap_h  = 0.3
        x_end  = float(t_patch[-1])   # 127/128 ≈ 0.992
        for col_i in range(n_top_patches):
            xref = "x" if col_i == 0 else f"x{col_i + 1}"
            for shape in [
                dict(type="line", x0=0,     x1=x_end, y0=y_bar,         y1=y_bar),
                dict(type="line", x0=0,     x1=0,     y0=y_bar - cap_h, y1=y_bar + cap_h),
                dict(type="line", x0=x_end, x1=x_end, y0=y_bar - cap_h, y1=y_bar + cap_h),
            ]:
                fig_top.add_shape(**shape, xref=xref, yref="y",
                                  line=dict(color="#555", width=1.5))
            fig_top.add_annotation(
                x=x_end / 2, y=y_bar - 0.7,
                text="1 second", xref=xref, yref="y",
                showarrow=False, font=dict(size=10, color="#555"),
            )

        fig_top.update_yaxes(
            range=[y_bar - 1.5, (n_ch_display - 1) * spacing_ex + 1.5],
            row=1, col=1,
        )
        fig_top.update_layout(
            title="Top 1-second patches — activation × attention received",
            height=max(600, n_ch_display * 22) + 60,
            margin=dict(l=70, r=10, t=60, b=20),
        )
        st.plotly_chart(fig_top, use_container_width=True)

        score_cols = st.columns(n_top_patches)
        for col_i, tok_i in enumerate(top_tok_idx):
            with score_cols[col_i]:
                st.caption(f"Second {int(tok_i)}")
                st.caption(f"Activation {feat_ts[tok_i]:.3f}")
                st.caption(f"Attention {attn_recv_pre[tok_i]:.4f}")

    # ── Figure 2: attention matrices ──────────────────────────────────
    if not has_attn or attn_mat is None:
        st.info("Attention weights not available for this encoder.")
        return

    st.subheader("Layer-0 Temporal Attention")
    st.caption(
        "Each cell (query → key) shows how much attention token *query* pays to "
        "token *key*. Columns with high sum = tokens heavily attended to by others."
    )

    # Mean attention heatmap
    attn_mean_np = attn_mat.mean(axis=0)                              # (S, S)
    fig_mean = go.Figure(go.Heatmap(
        z=attn_mean_np,
        x=list(range(S)),
        y=list(range(S)),
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="Weight", thickness=12, len=0.7),
    ))
    fig_mean.update_layout(
        height=380,
        title="Mean attention (averaged over all heads)",
        title_font=dict(size=12),
        xaxis_title="Key token (time →)",
        yaxis_title="Query token (time →)",
        margin=dict(l=60, r=40, t=50, b=50),
    )
    st.plotly_chart(fig_mean, use_container_width=True)

    # Per-head attention heatmaps
    if show_heads and n_heads > 0:
        with st.expander(f"Per-head attention matrices ({n_heads} heads)", expanded=False):
            n_cols = 4
            n_rows_h = (n_heads + n_cols - 1) // n_cols
            fig_heads = make_subplots(
                rows=n_rows_h, cols=n_cols,
                subplot_titles=[f"Head {h}" for h in range(n_heads)],
                horizontal_spacing=0.06,
                vertical_spacing=0.12,
            )
            for h in range(n_heads):
                r = h // n_cols + 1
                c = h % n_cols + 1
                fig_heads.add_trace(
                    go.Heatmap(
                        z=attn_mat[h],
                        colorscale="Blues",
                        showscale=(h == n_heads - 1),
                        colorbar=dict(thickness=10, len=0.4),
                    ),
                    row=r, col=c,
                )
                fig_heads.update_xaxes(showticklabels=False, row=r, col=c)
                fig_heads.update_yaxes(showticklabels=False, row=r, col=c)

            fig_heads.update_layout(
                height=220 * n_rows_h,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_heads, use_container_width=True)



# ─────────────────────────────────────────────────────────────────────────────
# Concept Steering
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading SAE weights …")
def _load_sae_for_steering(sae_path: str):
    """Load only the SAE weights needed for concept steering (no spectral decoder)."""
    from mecheeg.sae import SparseAutoencoder

    ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
    embed_dim = ckpt.get("embed_dim", 128)
    expansion = ckpt.get("expansion", 8)
    k = ckpt.get("k", 32)
    sae = SparseAutoencoder(embed_dim, expansion=expansion, mode="topk", k=k)
    sae.load_state_dict(ckpt["sae_state_dict"])
    sae.eval()
    # W_dec rows = feature directions in normalised embedding space
    W_dec = sae.decoder.weight.T.detach()  # (n_features, embed_dim)
    return sae, W_dec, ckpt["act_mean"].float(), ckpt["act_std"].float()


def _build_concept_weights(
    enr_list: list,
    n_features: int,
    field: str,
    category: str,
    p_thresh: float = 0.05,
) -> torch.Tensor:
    """Weight vector over SAE features for a demographic category.

    w_i = (ratio - 1) * -log10(p_adj)  if BH q < p_thresh else 0.

    Combines effect size (ratio - 1) with statistical confidence (-log10 p),
    so features that are both strongly enriched and highly significant get
    proportionally more weight in the concept direction.
    """
    weights = torch.zeros(n_features)
    for i, feat_enr in enumerate(enr_list):
        for cat, ratio, p_adj in feat_enr.get(field, []):
            if cat == category and p_adj < p_thresh:
                effect = float(ratio) - 1.0
                confidence = -np.log10(max(float(p_adj), 1e-300))
                weights[i] = effect * confidence
    return weights


def _concept_direction(
    W_dec: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project concept weight vector through decoder to get embedding-space direction.

    Returns (direction: (embed_dim,), norm: scalar tensor).
    W_dec is (n_features, embed_dim), so W_dec.T @ weights → (embed_dim,).
    """
    d = W_dec.T @ weights
    return d, d.norm()


_STEERING_FIG_DIR = Path("paper/concept_steering_figures")

_TAXONOMY_COLORS = {
    "separable": "#74c476",
    "entangled": "#fd8d3c",
    "null":      "#9e9e9e",
}

_CONCEPT_GROUPS: List[Dict[str, Any]] = [
    {
        "title": "Age: 0–3 → 50+",
        "feats": "75",
        "taxonomy": "entangled",
        "result": "Clean in both normal and abnormal EEG — δ drops fully, abnormal lands in adult-abnormal band",
        "note": "Age features encode maturation δ/θ shared with pathological slowing. "
                "Both conditions steer into their respective adult reference bands.",
        "figures": [
            ("Normal EEG", "concept_steering_spectral_decoder_age_normal.png"),
            ("Abnormal EEG", "concept_steering_spectral_decoder_age_abnormal.png"),
        ],
    },
    {
        "title": "Classification: All Abnormal → Normal",
        "feats": "128 (TCAV)",
        "taxonomy": "entangled",
        "result": "Strong — δ drops, α emerges",
        "note": "TCAV captures the shared δ/θ subspace. "
                "Both age and pathology features load on the same primitive.",
        "figures": [
            ("All ages", "concept_steering_spectral_decoder_classification_all.png"),
        ],
    },
    {
        "title": "Classification: Adult Abnormal → Adult Normal",
        "feats": "128 (TCAV)",
        "taxonomy": "separable",
        "result": "Clean removal",
        "note": "Adult pathological δ is not entangled with developmental δ — "
                "steering is clean with little residual.",
        "figures": [
            ("Adult (20+)", "concept_steering_spectral_decoder_classification_adult_all.png"),
        ],
    },
    {
        "title": "Classification: Child Abnormal → Child Normal",
        "feats": "128 (TCAV)",
        "taxonomy": "entangled",
        "result": "Partial — residual δ above child-normal ref",
        "note": "Developmental δ cannot be fully removed via pathology features — "
                "the same neural primitive underlies both maturation and pathological slowing.",
        "figures": [
            ("Child (0–19)", "concept_steering_spectral_decoder_classification_child_all.png"),
        ],
    },
    {
        "title": "Epileptiform → Normal  /  Other Abnormal → Normal",
        "feats": "128 (TCAV)",
        "taxonomy": "entangled",
        "result": "Strong — same signature as all-abnormal for both subtypes",
        "note": "Epileptiform and other-abnormal share the same δ/θ SAE representation at layer 2.",
        "figures": [
            ("Epileptiform → Normal", "concept_steering_spectral_decoder_classification_epileptiform_all.png"),
            ("Other Abnormal → Normal", "concept_steering_spectral_decoder_classification_other_abnormal_all.png"),
        ],
    },
    {
        "title": "Epileptiform → Other Abnormal",
        "feats": "0",
        "taxonomy": "null",
        "result": "Null — zero enriched features",
        "note": "No SAE features significantly discriminate the two pathology subtypes. "
                "Both share the same δ/θ representation — the encoder has not learned "
                "epileptiform-specific morphology (spikes, ictal dynamics) as a separable dimension.",
        "figures": [
            ("Epileptiform vs Other", "concept_steering_spectral_decoder_classification_epileptiform_vs_other_all.png"),
        ],
    },
    {
        "title": "Pediatric Classification (age-matched)",
        "feats": "128 (TCAV)",
        "taxonomy": "entangled",
        "result": "Partial — more residual δ than adult equivalent",
        "note": "Age-matched reference (child abnormal → child normal) isolates pathological excess "
                "above the developmental baseline. Greater residual than adult confirms deeper "
                "age/pathology entanglement in the pediatric population.",
        "figures": [
            ("Child abn → child norm", "concept_steering_spectral_decoder_classification_pediatric_all.png"),
        ],
    },
    {
        "title": "Gender: F → M",
        "feats": "32",
        "taxonomy": "separable",
        "result": "Null — near-zero effect in both normal and abnormal EEG",
        "note": "Negative control. EEG spectral content does not meaningfully differ by sex. "
                "Validates method specificity — the approach only produces a signal when a "
                "genuine biological concept is encoded.",
        "figures": [
            ("Normal EEG", "concept_steering_spectral_decoder_gender_normal.png"),
            ("Abnormal EEG", "concept_steering_spectral_decoder_gender_abnormal.png"),
        ],
    },
    {
        "title": "Medication: ASM → None",
        "feats": "30",
        "taxonomy": "entangled",
        "result": "Weak — confounded by underlying diagnosis",
        "note": "The ASM spectral signature (θ/α slowing) disappears when controlling for "
                "pathology status, confirming the enrichment captures diagnosis rather than "
                "medication per se.",
        "figures": [
            ("Normal EEG", "concept_steering_spectral_decoder_medication_asm_normal.png"),
            ("Abnormal EEG", "concept_steering_spectral_decoder_medication_asm_abnormal.png"),
        ],
    },
    {
        "title": "Medication: Psychiatric → None",
        "feats": "17",
        "taxonomy": "separable",
        "result": "Moderate — β/γ suppression (opposite direction to ASM)",
        "note": "The SAE separates two pharmacologically distinct classes with opposite "
                "spectral signatures: ASM → θ/α slowing; psychiatric → β/γ increase. "
                "Psychiatric effect persists in normal EEGs (not confounded by pathology).",
        "figures": [
            ("Normal EEG", "concept_steering_spectral_decoder_medication_psych_normal.png"),
        ],
    },
]


def _taxonomy_badge(taxonomy: str) -> str:
    color = _TAXONOMY_COLORS.get(taxonomy, "#9e9e9e")
    labels = {
        "separable": "Separable-monosemantic",
        "entangled":  "Entangled-monosemantic",
        "null":       "Null result",
    }
    label = labels.get(taxonomy, taxonomy)
    return (
        f'<span style="background:{color};color:#111;padding:2px 8px;'
        f'border-radius:4px;font-size:0.8em;font-weight:600">{label}</span>'
    )


def _render_steering_gallery() -> None:
    st.markdown(
        "SAE concept steering replaces source-group feature activations with the target-group "
        "centroid in SAE space, then decodes via the spectral decoder to obtain an amplitude spectrum. "
        "Results classify into a three-way taxonomy:"
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'{_taxonomy_badge("separable")}  \nOne concept, clean spectral shift, no residual.',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'{_taxonomy_badge("entangled")}  \nMultiple clinical labels sharing one neural primitive.',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'{_taxonomy_badge("null")}  \nZero enriched features — concepts indistinguishable at this layer.',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    for concept in _CONCEPT_GROUPS:
        tax = concept["taxonomy"]
        color = _TAXONOMY_COLORS.get(tax, "#9e9e9e")
        st.markdown(
            f"#### {concept['title']} &nbsp; {_taxonomy_badge(tax)}",
            unsafe_allow_html=True,
        )
        st.caption(f"**{concept['feats']}** enriched features · {concept['result']}")

        figs = concept["figures"]
        cols = st.columns(len(figs))
        for col, (label, fname) in zip(cols, figs):
            path = _STEERING_FIG_DIR / fname
            with col:
                if path.exists():
                    st.image(str(path), caption=label, use_container_width=True)
                else:
                    st.caption(f"*(figure not found: {fname})*")

        st.markdown(
            f'<div style="border-left:3px solid {color};padding:6px 12px;'
            f'margin:4px 0 16px 0;color:#ccc;font-size:0.9em">{concept["note"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")


def page_concept_steering(
    app_cache: Optional[dict], sae_path: str, folder_name: str, layer: int
) -> None:
    st.title("Concept Steering")
    st.caption("**Paper §5, §6 · Figures 4 / 5 / 6** — clamping concept-aligned features to the target centroid.")

    tab_gallery, tab_interactive = st.tabs(["Results Gallery", "Interactive Clamping"])

    with tab_gallery:
        _render_steering_gallery()

    with tab_interactive:
        st.caption(
            "Zero out the SAE features that encode a demographic concept — "
            "does the concept become undecodable?"
        )
        _page_concept_steering_interactive(app_cache, sae_path)


def _page_concept_steering_interactive(
    app_cache: Optional[dict], sae_path: str
) -> None:
    if not app_cache:
        st.info("No app cache found. Run `tools/build_app_cache.py` for this experiment.")
        return

    enr_list = app_cache.get("feature_meta_enrichment", [])
    if not enr_list:
        st.info("No demographic enrichment data in cache. Rebuild app cache.")
        return

    if not sae_path or not Path(sae_path).exists():
        st.info("SAE checkpoint not found for this experiment.")
        return

    sae, W_dec, act_mean, act_std = _load_sae_for_steering(sae_path)
    n_features = W_dec.shape[0]

    # ── Controls ──────────────────────────────────────────────────────────────
    _FIELD_LABELS = {
        "age_group": "Age group",
        "gender": "Sex",
        "indication_group": "Indication",
        "medication_group": "Medication",
    }
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        field = st.selectbox(
            "Demographic field",
            list(_FIELD_LABELS),
            format_func=lambda x: _FIELD_LABELS[x],
        )

    cats: set[str] = set()
    for fe in enr_list:
        for cat, _r, _p in fe.get(field, []):
            cats.add(cat)
    def _cat_sort_key(v: str) -> tuple:
        try:
            return (0, int(v.split("-")[0].replace("+", "")))
        except ValueError:
            return (1, v)
    cats_sorted = sorted(cats, key=_cat_sort_key)

    with col2:
        source_cat = st.selectbox("Concept to remove", cats_sorted)

    # Build weights early so n_sig_source is available for sliders
    w_source = _build_concept_weights(enr_list, n_features, field, source_cat)
    n_sig_source = int((w_source != 0).sum().item())

    with col3:
        n_clamp = st.slider(
            "Top N features to zero out", 1, max(1, n_sig_source), min(20, max(1, n_sig_source)),
            key="n_clamp_slider",
        )
    with col4:
        clamp_scale = st.slider(
            "Scale  (0 = zero out · 1 = unchanged · >1 = amplify)",
            0.0, 2.0, 0.0, 0.05, key="clamp_scale_slider",
        )

    if n_sig_source == 0:
        st.warning(f"No significant features for {source_cat} (BH q < 0.05). Try a different field or category.")
        return

    # ── Clamped feature table ─────────────────────────────────────────────────
    _enr_idx = w_source.nonzero(as_tuple=True)[0]
    _enr_sorted = _enr_idx[w_source[_enr_idx].argsort(descending=True)]
    clamp_features = _enr_sorted[:n_clamp]

    feat_band_names: list = app_cache.get("feature_band_names", [])
    feat_band_deltas = app_cache.get("feature_band_deltas")

    st.markdown(
        f"**{n_sig_source}** features significantly enriched for `{source_cat}` "
        f"(BH q < 0.05) · zeroing out top **{n_clamp}**"
    )
    tbl_rows = []
    for rank, fi in enumerate(clamp_features.tolist()):
        fe = enr_list[fi]
        ratio, p_adj = None, None
        for cat, r, p in fe.get(field, []):
            if cat == source_cat:
                ratio, p_adj = r, p
                break
        dom_band = ""
        if feat_band_deltas is not None and len(feat_band_names) > 0:
            dom_band = feat_band_names[int(np.argmax(np.abs(feat_band_deltas[fi])))]
        tbl_rows.append({
            "Rank": rank + 1,
            "Feature": f"F{fi}",
            "Enrichment (×)": f"{ratio:.2f}" if ratio is not None else "—",
            "BH q": f"{p_adj:.2e}" if p_adj is not None else "—",
            "Dom. band": dom_band,
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

    # ── Token selection ───────────────────────────────────────────────────────
    centroids_emb: Optional[torch.Tensor] = app_cache.get("codebook_centroids_emb")
    if centroids_emb is None:
        st.info("No codebook centroids in cache.")
        return

    centroids_norm = (centroids_emb.float() - act_mean) / (act_std + 1e-8)
    with torch.no_grad():
        z_all = sae.encode(centroids_norm)                              # (K, n_features)
        concept_scores = z_all[:, clamp_features].mean(dim=1).numpy()  # (K,)
    cluster_idx = int(np.argmax(concept_scores))
    x_norm = centroids_norm[cluster_idx]

    # ── Feature clamping ──────────────────────────────────────────────────────
    with torch.no_grad():
        z_orig     = sae.encode(x_norm.unsqueeze(0)).squeeze(0)
        x_hat_norm = sae.decode(z_orig.unsqueeze(0)).squeeze(0)
        z_steered  = z_orig.clone()
        z_steered[clamp_features] = z_steered[clamp_features] * clamp_scale
        x_steered_norm = sae.decode(z_steered.unsqueeze(0)).squeeze(0)  # noqa: F841

    recon_err = float((x_hat_norm - x_norm).norm() / (x_norm.norm() + 1e-8))
    st.caption(
        f"Source token: cluster {cluster_idx} "
        f"(mean enriched-feature activation = {concept_scores[cluster_idx]:.3f}) "
        f"· recon error {recon_err:.3f}"
    )

    # ── Linear probe: is the concept still decodable after clamping? ──────────
    try:
        import warnings as _warnings
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.model_selection import StratifiedKFold as _SKF
        from sklearn.model_selection import cross_val_score as _cvs
        from sklearn.exceptions import ConvergenceWarning as _CW

        _probe_emb = app_cache.get("codebook_umap_embeddings")
        _probe_meta = (
            app_cache.get("codebook_umap_embeddings_meta")
            or app_cache.get("codebook_umap_meta")
            or {}
        )
        _probe_field = _probe_meta.get(field) if _probe_meta else None

        if _probe_emb is not None and _probe_field is not None:
            _probe_raw  = np.array([str(v) for v in _probe_field])
            _known_mask = _probe_raw != ""
            X_raw   = _probe_emb[_known_mask].astype(np.float32)
            y_probe = (_probe_raw[_known_mask] == source_cat).astype(int)
            _data_source_probe = f"{int(_known_mask.sum()):,} tokens"
        else:
            K_cb    = centroids_emb.shape[0]
            y_probe = np.zeros(K_cb, dtype=int)
            X_raw   = centroids_emb.float().numpy()
            _data_source_probe = f"{K_cb} centroids (fallback)"

        _rng = np.random.default_rng(42)
        _pi, _ni = np.where(y_probe == 1)[0], np.where(y_probe == 0)[0]
        _sub = np.concatenate([
            _rng.choice(_pi, min(200, len(_pi)), replace=False),
            _rng.choice(_ni, min(200, len(_ni)), replace=False),
        ])
        X_raw, y_probe = X_raw[_sub], y_probe[_sub]
        n_pos, n_neg = int(y_probe.sum()), int((y_probe == 0).sum())

        X_probe = ((torch.tensor(X_raw) - act_mean) / (act_std + 1e-8)).numpy()

        def _steer(X: np.ndarray, scale: float) -> np.ndarray:
            """Encode ALL tokens, clamp enriched features in ALL tokens, decode ALL.

            Clamping both source and negative ensures the probe cannot use
            'enriched features are zero → must be source class' as a signal.
            The retrained probe must rely on other features to distinguish groups.
            AUROC drop = those features were the discriminative dimensions.
            """
            with torch.no_grad():
                zi = sae.encode(torch.tensor(X, dtype=torch.float32)).clone()
                zi[:, clamp_features] *= scale
                return sae.decode(zi).numpy()

        n_splits = min(3, n_pos, n_neg)
        if n_splits >= 2:
            cv  = _SKF(n_splits=n_splits, shuffle=True, random_state=42)
            clf = _LR(max_iter=500, C=1.0, class_weight="balanced")
            probe_scales = [1.0, 0.5, 0.0]
            bar_labels   = ["scale=1\n(unchanged)", "scale=0.5\n(half)", "scale=0\n(zero out)"]
            probe_means, probe_stds = [], []
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", _CW)
                for s in probe_scales:
                    scores = _cvs(clf, _steer(X_probe, s), y_probe, cv=cv, scoring="roc_auc")
                    probe_means.append(float(scores.mean()))
                    probe_stds.append(float(scores.std()))

            auc_before = probe_means[0]
            _idx_min   = int(np.argmin(probe_means))
            min_auc    = probe_means[_idx_min]
            delta      = auc_before - min_auc
            verdict    = (
                "✓ concept substantially removed" if delta > 0.15 else
                "~ partial removal"               if delta > 0.05 else
                "✗ concept still decodable"
            )
            ci = [1.96 * s / np.sqrt(n_splits) for s in probe_stds]
            bar_colors = ["#00cc88" if i == _idx_min else "#636efa" for i in range(len(probe_scales))]

            fig_probe = go.Figure()
            fig_probe.add_trace(go.Bar(
                x=bar_labels, y=probe_means,
                error_y=dict(type="data", array=ci, visible=True,
                             color="rgba(255,255,255,0.6)", thickness=2, width=6),
                marker_color=bar_colors,
                text=[f"{v:.3f}" for v in probe_means],
                textposition="outside", textfont=dict(size=11),
                hovertemplate="%{x}<br>AUROC=%{y:.3f}<extra></extra>",
                showlegend=False,
            ))
            fig_probe.add_hline(
                y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                annotation_text="chance (0.5)", annotation_position="right",
                annotation_font_color="rgba(255,255,255,0.4)",
            )
            _y_max = max(m + e for m, e in zip(probe_means, ci))
            fig_probe.update_layout(
                title=f"Linear probe AUROC — is <b>{source_cat}</b> still decodable after clamping?",
                height=320, xaxis=dict(showgrid=False),
                yaxis=dict(title="AUROC (3-fold CV)", range=[0.4, min(_y_max + 0.08, 1.05)],
                           showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#fafafa", size=11),
                margin=dict(t=50, b=40, l=55, r=20), bargap=0.4,
            )
            st.plotly_chart(fig_probe, use_container_width=True)
            st.caption(
                f"Enriched features clamped in **all** tokens (scale 1→0.5→0), probe retrained. "
                f"AUROC drop = those features were the discriminative dimensions; "
                f"residual AUROC = concept still encoded elsewhere. "
                f"AUROC: **{auc_before:.3f}** → **{min_auc:.3f}** "
                f"(Δ = {delta:+.3f}) · **{verdict}** · "
                f"balanced LR, {n_splits}-fold CV, "
                f"{n_pos} {source_cat!r} / {n_neg} other ({_data_source_probe})."
            )
        else:
            st.info(f"Too few {source_cat!r} tokens ({n_pos}) for cross-validation.")
    except ImportError:
        st.info("scikit-learn not available.")

    # ── Feature snapshot ──────────────────────────────────────────────────────
    delta_z = (z_steered - z_orig).numpy()
    k_val = sae.k
    active_orig  = set(z_orig.topk(k_val).indices.numpy().tolist())
    active_steer = set(z_steered.topk(k_val).indices.numpy().tolist())
    active = np.array(sorted(active_orig | active_steer))
    active = active[np.argsort(delta_z[active])[::-1]]
    fig_diff = go.Figure()
    fig_diff.add_bar(
        name="Original",
        x=[f"F{i}" for i in active],
        y=z_orig.numpy()[active].tolist(),
        marker_color="rgba(150,150,150,0.7)",
    )
    fig_diff.add_bar(
        name=f"Clamped (scale={clamp_scale})",
        x=[f"F{i}" for i in active],
        y=z_steered.numpy()[active].tolist(),
        marker_color="rgba(214,39,40,0.85)",
    )
    fig_diff.update_layout(
        title=f"Feature snapshot — cluster {cluster_idx} (highest concept activation)",
        barmode="group", height=280,
        margin=dict(l=50, r=20, t=45, b=60),
        xaxis_tickangle=-45, yaxis_title="SAE activation",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=11),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_diff, use_container_width=True)


def main():
    if "_page_nav_pending" in st.session_state:
        st.session_state["page_nav"] = st.session_state.pop("_page_nav_pending")
    sae_path, spectral_decoder_path, folder_name, exp_name, layer, encoder, page = sidebar()

    if page == "Home":
        page_home()
        return

    if page == "Feature Explorer":
        run_data   = load_run_data(sae_path, spectral_decoder_path)
        app_cache  = load_app_cache(str(p)) if (p := _app_cache_path(exp_name)) else None
        attn_cache = load_attention_cache(str(p)) if (p := _attention_cache_path(exp_name)) else None
        page_features(run_data, app_cache, folder_name, layer, exp_name=exp_name, attn_cache=attn_cache)
        return


    if page == "Layer Explorer":
        page_layer_explorer(folder_name)
        return


    if page == "TCAV Explorer":
        run_data  = load_run_data(sae_path, spectral_decoder_path)
        app_cache = load_app_cache(str(p)) if (p := _app_cache_path(exp_name)) else None
        tcav_cache = load_tcav_cache(str(p)) if (p := _tcav_cache_path(exp_name)) else None
        page_tcav_explorer(run_data, folder_name, layer, tcav_cache, app_cache, exp_name=exp_name)
        return

    if page == "Concept Steering":
        app_cache = load_app_cache(str(p)) if (p := _app_cache_path(exp_name)) else None
        page_concept_steering(app_cache, sae_path, folder_name, layer)
        return

    if page == "Attention Explorer":
        run_data   = load_run_data(sae_path, spectral_decoder_path)
        attn_cache = load_attention_cache(str(p)) if (p := _attention_cache_path(exp_name)) else None
        page_attention_explorer(run_data, attn_cache, folder_name, layer)
        return

    if page == "Taxonomy & Steering":
        page_taxonomy_steering()
        return



if __name__ == "__main__":
    main()
