
"""Interactive companion to "Mechanistic Interpretability of EEG Foundation
Models via Sparse Autoencoders" (NeurIPS 2026).

Run with:
    uv run --group app streamlit run app/main.py

The sidebar exposes only the paper's operating points: encoder × layer at
E=1, k=8, plus the E=8, k=64 SleepFM L3 dictionary behind Figure 6. Every
page reads pre-built caches under results/ (SAE + spectral decoder
checkpoints, app_cache.pt, tcav_cache.pt, layer UMAP, attention) and the
camera-ready figures and seeded steering data bundled in app/static/paper/.
"""
from __future__ import annotations

import json
import os
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
    page_title="EEG Foundation Model Interpretability",
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


def _canon_folder(folder: str) -> str:
    """`*_local/` re-training folders share experiments with their canonical sibling."""
    return _SPECTRAL_DECODER_CANONICAL.get(folder, folder)


def _scan_experiment_metadata() -> Dict[Tuple[str, float, int, int], str]:
    """Map (canonical SAE folder, expansion, layer, k) → experiment name.

    Keyed by the folder of the experiment's `sae_checkpoint`, so pretrained
    experiments (`results/features/reve/…`) never shadow finetuned ones
    (`results/features/reve_qjbe08/…`). When several experiments share one SAE,
    the canonical `{folder}_layer{L}` wins (e.g. `sleepfm_finetuned_layer2` over
    the sweep variant `sleepfm_finetuned_exp1_k8_layer2`), else the shortest name.
    """
    out: Dict[Tuple[str, float, int, int], str] = {}
    exp_root = ROOT / "results" / "experiments"
    if not exp_root.exists():
        return out
    for exp_dir in sorted(exp_root.iterdir(), key=lambda p: (len(p.name), p.name)):
        meta_path = exp_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            rel = json.loads(meta_path.read_text()).get("sae_checkpoint")
        except (OSError, json.JSONDecodeError):
            continue
        if not rel:
            continue
        parsed = _parse_sae_filename(Path(rel).stem)
        if parsed is None:
            continue
        E, L, K = parsed
        folder = _canon_folder(Path(rel).parent.name)
        key = (folder, E, L, K)
        if exp_dir.name == f"{folder}_layer{L}":
            out[key] = exp_dir.name          # canonical name always wins
        else:
            out.setdefault(key, exp_dir.name)
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
                exp_for[key] = selection_to_exp.get((_canon_folder(folder.name), expansion, layer, k_val))
                folder_for[key] = folder.name

    # Sort each level
    for enc in runs:
        runs[enc] = {
            E: {layer: dict(sorted(ks.items())) for layer, ks in sorted(layers.items())}
            for E, layers in sorted(runs[enc].items())
        }
    return runs, exp_for, folder_for


# Folders that share a spectral decoder with another folder. `*_local/`
# retrains only ship the E=1 SAEs that are missing from their canonical
# sibling; the canonical folder owns the xae checkpoint and we reuse it.
_SPECTRAL_DECODER_CANONICAL: Dict[str, str] = {
    "sleepfm_finetuned_local": "sleepfm_finetuned",
    "reve_local":              "reve_qjbe08",
}


def _spectral_decoder_path(folder_name: str) -> Optional[Path]:
    # The XAE module was renamed to SpectralDecoder; the disk layout used by
    # train_spectral_decoder.py writes to results/spectral_decoder/, while the
    # currently deployed checkpoints (pulled from gs://sae4eeg-app-assets) still
    # live under the original results/xae/ tree. Check both.
    search_folders = [folder_name]
    canonical = _SPECTRAL_DECODER_CANONICAL.get(folder_name)
    if canonical and canonical != folder_name:
        search_folders.append(canonical)

    candidates: list[Path] = []
    for f in search_folders:
        candidates.extend([
            ROOT / "results" / "spectral_decoder" / f / "spectral_decoder_checkpoint.pt",
            ROOT / "results" / "xae" / f / "xae_checkpoint.pt",
        ])
    candidates.extend([
        ROOT / "results" / "spectral_decoder" / "spectral_decoder_checkpoint.pt",  # legacy
        ROOT / "results" / "xae" / "xae_checkpoint.pt",
    ])
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

@st.cache_resource(show_spinner="Computing spectral features …", max_entries=8)
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


@st.cache_resource(show_spinner="Loading app cache …", max_entries=4)
def load_app_cache(path: str) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


@st.cache_resource(show_spinner="Loading TCAV cache …", max_entries=8)
def load_tcav_cache(path: str) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


@st.cache_resource(show_spinner="Loading attention cache …", max_entries=3)
def load_attention_cache(path: str) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


@st.cache_resource(show_spinner="Loading layer UMAP cache …", max_entries=3)
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

# Configurations exposed in the hosted app: the paper's operating points
# (Appendix B, Table 4). Every encoder uses E=1, k=8 at every layer; a few
# layers additionally offer the sparsity-matched E=8, k=64 dictionary behind
# Figure 6 (SleepFM L3) and Appendix G (LaBraM L12, REVE L9). Everything else
# on disk stays reachable from the analysis scripts but is not a knob here.
_PAPER_DICTS: List[Tuple[float, int, str]] = [
    (1.0, 8,  "Standard · E=1, k=8"),
    (8.0, 64, "Sparsity-matched · E=8, k=64"),
]
_ENC_LABELS = {"sleepfm": "SleepFM", "labram": "LaBraM", "reve": "REVE"}
_ENC_ORDER  = ["sleepfm", "labram", "reve"]
_PAGES = [
    "Home",
    "Feature Explorer",
    "TCAV Explorer",
    "Concept Steering",
    "Taxonomy & Steering",
    "Layer Explorer",
    "Attention Explorer",
]


@st.cache_resource(show_spinner=False)
def _paper_runs() -> Tuple[Dict[str, Dict[int, Dict[Tuple[float, int], Tuple[str, str, Optional[str]]]]], str]:
    """{encoder: {layer: {(E, k): (sae_path, folder, exp_name)}}} for the paper configs.

    Only configs with a built app cache are kept. Cached for the process
    lifetime: the deployed results/ tree is read-only.
    """
    runs_all, exp_for, folder_for = discover_runs()
    wanted = {(E, k) for E, k, _ in _PAPER_DICTS}
    out: Dict[str, Dict[int, Dict[Tuple[float, int], Tuple[str, str, Optional[str]]]]] = {}
    for enc, by_E in runs_all.items():
        for E, by_L in by_E.items():
            for L, by_K in by_L.items():
                for K, path in by_K.items():
                    if (E, K) not in wanted:
                        continue
                    exp = exp_for.get((enc, E, L, K))
                    if _app_cache_path(exp) is None:
                        continue
                    out.setdefault(enc, {}).setdefault(L, {})[(E, K)] = (
                        str(path), folder_for[(enc, E, L, K)], exp,
                    )
    try:
        import tomllib
        with open(ROOT / "pyproject.toml", "rb") as f:
            version = tomllib.load(f)["project"]["version"]
    except Exception:
        version = "?"
    return out, version


def sidebar():
    st.sidebar.markdown(
        "### Mechanistic Interpretability of EEG Foundation Models\n"
        "<span style='font-size:0.8rem;opacity:0.7'>NeurIPS 2026 · interactive companion</span>",
        unsafe_allow_html=True,
    )

    runs, version = _paper_runs()
    if not runs:
        st.sidebar.error("No SAE checkpoints with an app cache found under results/.")
        st.stop()

    enc_keys = [e for e in _ENC_ORDER if e in runs]
    enc_display = [_ENC_LABELS[e] for e in enc_keys]
    enc_choice = st.sidebar.segmented_control(
        "Model", enc_display, default=enc_display[0], key="enc_pills",
    ) or enc_display[0]
    encoder = enc_keys[enc_display.index(enc_choice)]

    # Layers are 1-based in the paper; code and caches are 0-based.
    layers = sorted(runs[encoder])
    _default_L = 2 if encoder == "sleepfm" and 2 in layers else layers[-1]
    layer = st.sidebar.selectbox(
        "Layer", layers,
        index=layers.index(_default_L),
        format_func=lambda L: f"L{L + 1}",
        key=f"layer_sel_{encoder}",
    )

    dicts = runs[encoder][layer]
    if len(dicts) > 1:
        labels = {(E, k): lbl for E, k, lbl in _PAPER_DICTS if (E, k) in dicts}
        E_k = st.sidebar.radio(
            "Dictionary", list(labels), format_func=labels.get, key="dict_sel",
            help="E = expansion (dictionary size / embedding dim), k = active features per token. "
                 "E=8, k=64 is the dictionary behind Fig. 6 (SleepFM L3) and App. G.",
        )
    else:
        E_k = next(iter(dicts))
    sae_path, folder_name, exp_name = dicts[E_k]
    spectral_decoder_p = _spectral_decoder_path(folder_name)

    st.sidebar.divider()
    page = st.sidebar.radio("Page", _PAGES, label_visibility="collapsed", key="page_nav")

    st.sidebar.divider()
    st.sidebar.caption(
        f"E={E_k[0]:g} · k={E_k[1]} · `{exp_name}`  \n"
        "[Paper](https://github.com/BrainCapture/mechanistic-interpretability-for-eeg-foundation-models/blob/main/paper/Mechanistic_Interpretability_of_EEG_Foundation_Models_via_Sparse_Autoencoders.pdf)"
        " · [Code](https://github.com/BrainCapture/mechanistic-interpretability-for-eeg-foundation-models)"
        f" · v{version}"
    )

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

_STATIC = ROOT / "app" / "static"
_PAPER_STATIC = _STATIC / "paper"
_PAPER_URL = (
    "https://github.com/BrainCapture/mechanistic-interpretability-for-eeg-foundation-models/"
    "blob/main/paper/Mechanistic_Interpretability_of_EEG_Foundation_Models_via_Sparse_Autoencoders.pdf"
)
_CODE_URL = "https://github.com/BrainCapture/mechanistic-interpretability-for-eeg-foundation-models"

_HOME_PAGES = [
    ("Feature Explorer",    "§3.2–3.3 · Stages I–II",
     "Browse every SAE feature: its spectral signature, firing statistics and top-activating EEG.",
     "feature_explorer"),
    ("TCAV Explorer",       "§3.4 · Stage III",
     "Which features encode abnormality, age, sex and medication, and how strongly.",
     "tcav_explorer"),
    ("Concept Steering",    "§3.5 · Fig. 6",
     "Clamp concept-aligned features to a target group and watch the decoded spectrum move.",
     "concept_steering"),
    ("Taxonomy & Steering", "§4.3–4.4 · Figs. 3–5",
     "The paper's taxonomy and selectivity results, with every steering curve explorable.",
     "taxonomy_steering"),
    ("Layer Explorer",      "supplementary",
     "Animated UMAP of the same tokens as they move through the encoder's layers.",
     "layer_explorer"),
    ("Attention Explorer",  "supplementary",
     "Encoder self-attention shown alongside the SAE features.",
     "attention_explorer"),
]


def _go_to(title: str) -> None:
    st.session_state["_page_nav_pending"] = title


def page_home() -> None:
    st.title("Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders")
    st.markdown(
        "William Lehn-Schiøler · Magnus Ruud Kjær · Rahul Thapa · Magnus Guldberg Pedersen · "
        "Anton Mosquera Storgaard · Nick Williams · Andreas Brink-Kjær · Tue Lehn-Schiøler · "
        "Sadasivan Puthusserypady · James Zou · Lars Kai Hansen  \n"
        f"**NeurIPS 2026** · [Paper (PDF)]({_PAPER_URL}) · [Code]({_CODE_URL})"
    )
    st.markdown(
        "We train TopK sparse autoencoders on three EEG transformers (**SleepFM, REVE, LaBraM**), "
        "map the learned features to clinical concepts with TCAV, and test them causally with "
        "**concept steering**. A **spectral decoder** turns each intervention back into an "
        "amplitude spectrum a clinician can read."
    )

    fig1 = _PAPER_STATIC / "fig1_pipeline.png"
    if fig1.exists():
        st.image(str(fig1), width="stretch",
                 caption="Figure 1. The four-stage pipeline, run on a frozen, finetuned encoder.")

    st.subheader("Explore")
    st.caption(
        "Pick a **model** and **layer** in the sidebar; every page follows that choice. "
        "The default is the paper's primary configuration: SleepFM, layer L3."
    )
    thumb_dir = _STATIC / "home_thumbs"
    for row in range(0, len(_HOME_PAGES), 3):
        cols = st.columns(3, gap="medium")
        for col, (title, ref, blurb, slug) in zip(cols, _HOME_PAGES[row:row + 3]):
            with col, st.container(border=True):
                thumb = thumb_dir / f"{slug}.png"
                if thumb.exists():
                    st.image(str(thumb), width="stretch")
                st.markdown(f"**{title}**  \n<span style='font-size:0.8rem;opacity:0.65'>{ref}</span>",
                            unsafe_allow_html=True)
                st.caption(blurb)
                st.button(f"Open {title} →", key=f"home_nav_{title}",
                          on_click=_go_to, args=(title,))

    with st.expander("Glossary"):
        st.markdown(
            """
| Term | Meaning |
|------|---------|
| **SAE** | Sparse autoencoder: learns a sparse, over-complete dictionary of features from an encoder layer's activations. |
| **TopK SAE** | SAE in which exactly *k* features are active per token. |
| **E (expansion)** | Dictionary size divided by the embedding dimension. |
| **Spectral decoder** | Small MLP mapping a token embedding to its amplitude (and phase) spectrum. |
| **Token** | One EEG patch seen by the encoder. SleepFM: channel-averaged; REVE and LaBraM: one channel, one second. |
| **Fire rate** | Fraction of tokens on which a feature is non-zero. |
| **CAV** | Concept activation vector: a linear direction separating concept tokens from the rest. |
| **TCAV score** | Fraction of concept examples whose prediction is pushed up by the concept direction (Kim et al.). |
| **Concept steering** | Clamping the most concept-aligned SAE features to the target group's centroid, then decoding. |
| **AUROC₀** | Probe AUROC before any clamping: how strongly the concept is encoded. |
| **Excess selectivity Δ̃** | Area between off-target and target AUROC curves under clamping, minus the mean of 200 random rankings. |
| **Taxonomy** | Separable (one concept), entangled (several concepts share a feature), dead (never informative). |
"""
        )

    with st.expander("Model cards"):
        _model_cards_content()


def _model_cards_content() -> None:
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
    st.caption("**§3.2–3.3.** Each SAE feature, decoded to an amplitude spectrum by the spectral decoder.")
    st.markdown(
        f"SAE features learned on **{enc_label} layer {data['layer']}** activations. "
        "Each feature is a direction in embedding space; its **spectral signature** "
        "shows which EEG frequency bands it boosts or suppresses."
    )

    if not data["has_spectral_decoder"]:
        st.info(
            "Spectral-decoder data isn't available for this encoder yet. "
            "Pick a different model in the sidebar — the home-page **data-availability** "
            "heatmap shows which (encoder, layer, expansion) cells are fully explorable."
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

    # Abnormal ratio from attention cache (independent of feature visualisations)
    abnorm_ratios: dict[int, float] = {}
    if attn_cache is not None and "top_window_labels" in attn_cache:
        _twl = attn_cache["top_window_labels"]  # (n_features, K)
        for _fi in range(int(attn_cache["n_features"])):
            _lbls = _twl[_fi].float().numpy()
            abnorm_ratios[_fi] = float(np.mean(_lbls > 0.5))

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

    # Derive dominant band from SpectralDecoder band deltas (always available)
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
        if abnorm_ratios:
            sort_options.append("Abnormal ratio ↓")
        if _all_meta_enr:
            sort_options += ["Age enrichment ↓", "Sex enrichment ↓"]
        _default_sort = (
            "TCAV abnormal ↓" if has_tcav_scores
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
        "Abnormal ratio ↓":    lambda r: -r["abnorm_ratio"] if not np.isnan(r["abnorm_ratio"]) else 0,
        "Age enrichment ↓":    lambda r: -r["age_enr"] if not np.isnan(r["age_enr"]) else 0,
        "Sex enrichment ↓":    lambda r: -r["sex_enr"] if not np.isnan(r["sex_enr"]) else 0,
    }
    rows.sort(key=sort_map[sort_by])

    st.markdown(f"**{len(rows)} features** matching filters")

    # ── Feature selector ──────────────────────────────────────────────
    def _feat_option(r: dict) -> str:
        if sort_by == "TCAV abnormal ↓" and not np.isnan(r["tcav_abnormal"]):
            metric = f"tcav={r['tcav_abnormal']:.0%}"
        elif sort_by == "Fire rate ↓" and not np.isnan(r["fire_rate"]):
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
        abn = r["abnorm_ratio"]
        abn_badge = f"  abn={abn:.0%}" if not np.isnan(abn) else ""
        return f"F{r['idx']}  [{metric}]{abn_badge}"

    feat_options = [_feat_option(r) for r in rows]
    if not feat_options:
        st.info("No features match the current filters.")
        return
    chosen_label = st.selectbox("Select feature", feat_options)
    chosen_idx   = rows[feat_options.index(chosen_label)]["idx"]

    # ── Feature landscape overview ────────────────────────────────────
    with st.expander("Feature landscape — all features overview"):
        _has_xy = has_cache and any(not np.isnan(r["fire_rate"]) for r in rows)
        if not _has_xy:
            st.info("Overview requires an app cache for this experiment.")
        else:
            _ov_x_opts = []
            if has_cache:
                _ov_x_opts += ["Fire rate (%)", "Label correlation (r)"]
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
            st.plotly_chart(_fig_ov, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
        if _show_build_hints():
            st.info("No attention cache found — build one with `tools/build_attention_cache.py`.")
        else:
            st.info(
                "Attention cache isn't built for this config. Attention Explorer is "
                "currently available for `sleepfm_finetuned_layer{0,1,2}`."
            )

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
        st.plotly_chart(fig_spec, width="stretch")

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
        st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})

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
                width="stretch", config={"displayModeBar": False},
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
                width="stretch", config={"displayModeBar": False},
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
        st.plotly_chart(fig_co, width="stretch")

    if chosen_idx in dash_imgs:
        with st.expander("Top-activating EEG windows (dashboard image)"):
            st.image(dash_imgs[chosen_idx],
                     caption=f"Feature F{chosen_idx} — top-activating windows",
                     width="stretch")

    with st.expander("All features — sortable table"):
        import pandas as pd
        df = pd.DataFrame(rows).rename(columns={
            "idx": "Feature", "fire_rate": "Fire %", "corr": "r(label)",
            "p": "p-value", "band": "Dominant band", "description": "Description",
        })
        st.dataframe(df.drop(columns=["cluster"]), width="stretch", height=400)

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Build-cache helper
# ─────────────────────────────────────────────────────────────────────────────

def _show_build_hints() -> bool:
    """True when running locally with a writable results/ tree.

    The hosted Cloud Run app is stateless and can't actually build caches;
    showing build buttons there would just confuse reviewers. We only show
    the dev workflow when the results directory is writable.
    """
    try:
        return os.access(str(ROOT / "results"), os.W_OK) \
            and os.environ.get("MECHEEG_REVIEWER_MODE", "0") != "1"
    except OSError:
        return False


def _build_app_cache_button(exp_name: str, key_suffix: str = "") -> None:
    """Show the build command and a button that runs it (dev only)."""
    if not _show_build_hints():
        st.info(
            "App-cache data for this config isn't built yet. Pick a config marked "
            "green in the **data-availability heatmap** on the home page to "
            "explore everything end-to-end."
        )
        return
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
    st.caption("**§3.4.** Concept attribution with Testing with Concept Activation Vectors (TCAV).")
    enc_label = _ENCODER_LABELS.get(data["encoder"], data["encoder"].upper())
    st.markdown(
        f"**Testing with Concept Activation Vectors** — {enc_label} layer {data['layer']}. "
        "Linear CAVs are trained in encoder activation space; each CAV direction is projected "
        "onto SAE decoder directions to measure feature–concept alignment."
    )

    exp_name = f"{folder_name}_layer{layer}"

    if tcav_cache is None:
        if app_cache is None:
            st.info(
                "TCAV data isn't built for this config. Pick a config marked green "
                "in the home-page **data-availability heatmap** for a ready-to-explore TCAV view."
            )
            _build_app_cache_button(exp_name, "_tcav")
        elif not _show_build_hints():
            st.info(
                "TCAV cache not yet built for **" + exp_name + "**. "
                "The SAE and app cache are available; concept attribution will appear once the "
                "TCAV build queue reaches this config (see the **data-availability heatmap** "
                "on the home page)."
            )
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
                width="stretch",
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
            st.plotly_chart(fig_ab, width="stretch")
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
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
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
            st.plotly_chart(fig_band_bar, width="stretch")

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
                st.plotly_chart(fig_cross, width="stretch")
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
                st.plotly_chart(fig_bscat, width="stretch")

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
                st.plotly_chart(fig_heat, width="stretch")

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
            st.plotly_chart(fig_cmp, width="stretch")
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

# Sleep stage colours (Wong palette subset, colourblind-safe)
_SLEEP_STAGE_COLORS: Dict[str, str] = {
    "Wake":      "#E69F00",   # amber
    "N1":        "#56B4E9",   # sky blue
    "N2":        "#0072B2",   # deep blue
    "N3":        "#009E73",   # teal
    "REM":       "#CC79A7",   # pink/purple
    "Move":      "#95a5a6",   # grey
    "Unscored":  "#7f8c8d",   # dark grey
    "Unknown":   "#aaaaaa",
}
_SLEEP_STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM", "Move", "Unscored", "Unknown"]

# AHI (apnea-hypopnea index) is a continuous events/hour value. Bin it into the
# standard AASM clinical OSA-severity categories so the Layer Explorer colours by
# meaningful groups instead of ~1,300 distinct float values.
#   Normal <5 · Mild 5–15 · Moderate 15–30 · Severe ≥30
_AHI_ORDER = ["Normal (<5)", "Mild (5–15)", "Moderate (15–30)", "Severe (≥30)", "unknown"]
_AHI_COLORS = {
    "Normal (<5)":      "#1a9850",  # green
    "Mild (5–15)":      "#fee08b",  # yellow
    "Moderate (15–30)": "#fc8d59",  # orange
    "Severe (≥30)":     "#d73027",  # red
    "unknown":          "#999999",  # grey
}


def _ahi_category(value: Any) -> str:
    """Map a raw AHI value to its AASM clinical severity bin."""
    try:
        ahi = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if ahi < 5:
        return "Normal (<5)"
    if ahi < 15:
        return "Mild (5–15)"
    if ahi < 30:
        return "Moderate (15–30)"
    return "Severe (≥30)"


@st.cache_data(show_spinner=False, hash_funcs={dict: id})
def _make_animation_figure(
    cache: dict, color_by: str, token_meta: Optional[Dict[str, Any]] = None,
    highlight: Optional[str] = None,
    umap_label: str = "Joint UMAP",
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

    elif color_by == "sleep_stage_name" and "sleep_stage_name" in token_meta:
        raw_vals = token_meta["sleep_stage_name"]
        point_labels = [str(v) for v in raw_vals]
        colors = [_SLEEP_STAGE_COLORS.get(lb, "#aaaaaa") for lb in point_labels]
        present = set(point_labels)
        legend_entries = {s: _SLEEP_STAGE_COLORS[s] for s in _SLEEP_STAGE_ORDER if s in present and s in _SLEEP_STAGE_COLORS}

    elif color_by == "micro_events" and "micro_events" in token_meta:
        _EV_COLORS = {
            "spindle":        "#009E73",  # teal
            "slow_wave":      "#0072B2",  # deep blue
            "eye_movement":   "#CC79A7",  # pink/purple
            "arousal":        "#D55E00",  # vermilion
            "apnea_oa":       "#E69F00",  # amber
            "apnea_ca":       "#F0E442",  # yellow
            "apnea_hypopnea": "#56B4E9",  # sky blue
            "desaturation":   "#999999",  # grey
            "multiple":       "#ffffff",  # white (multiple events)
            "none":           "#333333",  # dark grey
        }
        _EV_ORDER = ["spindle", "slow_wave", "eye_movement", "arousal",
                     "apnea_oa", "apnea_ca", "apnea_hypopnea", "desaturation",
                     "multiple", "none"]
        raw_vals = token_meta["micro_events"]
        point_labels = [str(v) for v in raw_vals]
        colors = [_EV_COLORS.get(lb, "#aaaaaa") for lb in point_labels]
        present = set(point_labels)
        legend_entries = {e: _EV_COLORS[e] for e in _EV_ORDER if e in present}

    elif color_by == "ahi_a0h3a" and "ahi_a0h3a" in token_meta:
        # Bin continuous AHI into ordered AASM severity categories
        point_labels = [_ahi_category(v) for v in token_meta["ahi_a0h3a"]]
        colors = [_AHI_COLORS.get(lb, "#888888") for lb in point_labels]
        present = set(point_labels)
        legend_entries = {c: _AHI_COLORS[c] for c in _AHI_ORDER if c in present}

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
        title=f"{cache['display_name']} — {umap_label} across layers",
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
                    label=cache.get("layer_labels", {}).get(L, "Tokenizer" if L == -1 else str(L)),
                    method="animate",
                )
                for L in layers
            ],
        )],
    )
    return fig


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
                st.plotly_chart(_fig_meta, width="stretch", config={"displayModeBar": False})

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
            st.plotly_chart(fig_w, width="stretch")
        st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy & Steering — paper figures (with one interactive panel)
# ─────────────────────────────────────────────────────────────────────────────

def _show_paper_image(fname: str, caption: str) -> None:
    p = _PAPER_STATIC / fname
    if p.exists():
        st.image(str(p), width="stretch", caption=caption)
    else:
        st.info(f"`{p.relative_to(ROOT)}` not found.")


@st.cache_resource(show_spinner=False)
def _load_seeded_steering() -> Dict[str, dict]:
    """Seeded steering sweeps behind Figures 4–5: {experiment: {layer, cells[]}}.

    Each cell is one (concept, SAE seed) sweep. Built from
    tools/paper_figures/Figure 5/data.json in the release repo.
    """
    p = _PAPER_STATIC / "steering_seeded.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource(show_spinner=False)
def _load_taxonomy_data() -> Dict[str, dict]:
    """Figure 3 data: {exp_name: {separable, entangled, dead}}."""
    p = _PAPER_STATIC / "taxonomy.json"
    return json.loads(p.read_text()) if p.exists() else {}


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
    "Separable": ("separable", "Greens", "Share of concept-enriched features that respond to exactly one concept."),
    "Entangled": ("entangled", "Oranges", "Share of features that respond to several concepts (e.g. age and pathology via δ/θ)."),
    "Dead":      ("dead",      "Reds",    "Share of features that never carry concept information."),
}


def _render_taxonomy_subtab() -> None:
    st.markdown(
        "**Figure 3 (§4.3).** Fraction of concept-enriched SAE features in each taxonomy class, "
        "by layer and expansion, for each encoder. Hover a cell for its value; blank cells had no SAE trained."
    )
    taxonomy = _load_taxonomy_data()
    if not taxonomy:
        st.info("Taxonomy data not found.")
        return

    metric_name = st.segmented_control(
        "Metric", list(_TAXONOMY_METRICS), default="Separable", key="taxonomy_metric",
    ) or "Separable"
    field, cmap, blurb = _TAXONOMY_METRICS[metric_name]
    st.caption(blurb)

    cols = st.columns(len(_TAXONOMY_ENCODERS), gap="medium")
    for col, cfg in zip(cols, _TAXONOMY_ENCODERS):
        layers, Es = cfg["layers"], _TAXONOMY_EXPANSIONS
        mat = np.full((len(layers), len(Es)), np.nan)
        for i, L in enumerate(layers):
            for j, E in enumerate(Es):
                entry = taxonomy.get(cfg["exp_for"](L, E))
                if entry is not None:
                    mat[i, j] = entry.get(field, np.nan) * 100.0
        hover = [[f"L{L + 1} · E={E}<br>{metric_name}: {mat[i, j]:.1f}%"
                  if not np.isnan(mat[i, j]) else f"L{L + 1} · E={E}<br>no SAE trained"
                  for j, E in enumerate(Es)] for i, L in enumerate(layers)]
        fig = go.Figure(go.Heatmap(
            z=mat, x=[f"{E}×" for E in Es], y=[f"L{L + 1}" for L in layers],
            colorscale=cmap, zmin=0, zmax=100, hoverinfo="text", text=hover,
            colorbar=dict(title=dict(text="%", side="right"), tickformat=".0f", thickness=10),
        ))
        fig.update_layout(
            title=f"<b>{cfg['name']}</b>", height=max(220, 32 * len(layers) + 110),
            margin=dict(l=40, r=10, t=40, b=40), xaxis_title="Expansion", yaxis_title="Layer",
            yaxis=dict(autorange="reversed"),
        )
        col.plotly_chart(fig, width="stretch")

    with st.expander("Paper Figure 3 (static)"):
        _show_paper_image("fig3_taxonomy.png", "Figure 3. Monosemanticity taxonomy across expansion and depth.")
    with st.expander("Paper Figure 2: SAE faithfulness at every layer"):
        _show_paper_image(
            "fig2_faithfulness.png",
            "Figure 2. Probe AUROC with layer-ℓ activations replaced by SAE reconstructions (5-fold CV).",
        )


def _render_steering_static_subtab() -> None:
    _show_paper_image(
        "fig4_steering.png",
        "Figure 4 (§4.4). Encoding strength AUROC₀ (top) and excess selectivity Δ̃ (bottom), "
        "E=1, k=8, mean ± sd over 5 SAE seeds; dots mark cells selective in ≥3/5 seeds.",
    )
    st.divider()
    _show_paper_image(
        "fig5_curves.png",
        "Figure 5. Nine representative steering sweeps across the encoding × selectivity landscape.",
    )
    st.divider()
    _show_paper_image(
        "fig6_spectral.png",
        "Figure 6 (§4.5). Spectrum-level steering, abnormal → normal, SleepFM L3 (E=8, k=64): "
        "clamping the top n = 104 and 164 TCAV-aligned features.",
    )


_STEER_FAMILIES = [
    ("SleepFM", "sleepfm_finetuned_layer"),
    ("LaBraM",  "labram_layer"),
    ("REVE",    "reve_qjbe08_layer"),
]
_STEER_CONCEPTS = [
    ("Pathology", "Abnormality"), ("Age", "Age"), ("Gender", "Sex"),
    ("Medication (Psychiatric)", "Medication (psychiatric)"), ("Medication (ASM)", "Medication (ASM)"),
]
_SEL_P, _SEL_MIN = 0.05, 0.05


def _render_steering_curves_subtab() -> None:
    st.markdown(
        "Pick any cell of Figure 4 to see its steering sweep. **Red**: target-concept AUROC; "
        "**blue**: off-target AUROC (age when the target is abnormality, otherwise abnormality); "
        "dotted: mean of 200 random feature rankings. Shading is ± 1 sd over 5 SAE seeds."
    )
    data = _load_seeded_steering()
    if not data:
        st.info("Seeded steering data not found.")
        return

    c1, c2, c3 = st.columns([1, 1, 1.6])
    fam_names = [f for f, _ in _STEER_FAMILIES]
    family = c1.selectbox("Encoder", fam_names, key="steer_family")
    prefix = dict(_STEER_FAMILIES)[family]
    exps = sorted((v["layer"], e) for e, v in data.items() if e.startswith(prefix))
    layer_idx = c2.selectbox("Layer", [L for L, _ in exps], format_func=lambda L: f"L{L + 1}",
                             index=len(exps) - 1, key=f"steer_layer_{family}")
    exp = dict(exps)[layer_idx]
    concept = c3.selectbox("Target concept", [c for c, _ in _STEER_CONCEPTS],
                           format_func=dict(_STEER_CONCEPTS).get, key="steer_concept")
    off_name = "Age" if concept == "Pathology" else "Abnormality"

    cells = [c for c in data[exp]["cells"] if c["concept"] == concept]
    if not cells:
        st.info("This cell was not run.")
        return

    fr = np.asarray(cells[0]["fracs"], dtype=float)
    stack = lambda key: np.asarray([c[key] for c in cells], dtype=float)  # noqa: E731
    tgt, off = stack("tgt_auc"), stack("off_auc")
    ntgt, noff = stack("null_tgt_auc_mean"), stack("null_off_auc_mean")
    ex = np.asarray([c["excess_delta"] for c in cells])
    a0 = np.asarray([c["auroc0"] for c in cells])
    n_sel = sum(1 for c in cells if c["p_value"] < _SEL_P and c["excess_delta"] >= _SEL_MIN)
    sd = lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0  # noqa: E731

    if a0.mean() < 0.70:
        regime = "Weakly encoded"
    elif n_sel >= 0.6 * len(cells):
        regime = "Selectively steerable"
    else:
        regime = "Encoded but entangled"

    m1, m2, m3 = st.columns(3)
    m1.metric("AUROC₀ (encoding)", f"{a0.mean():.3f} ± {sd(a0):.3f}")
    m2.metric("Excess selectivity Δ̃", f"{ex.mean():+.3f} ± {sd(ex):.3f}")
    m3.metric("Regime", regime, help=f"Selective in {n_sel}/{len(cells)} seeds "
              f"(p < {_SEL_P} against random rankings and Δ̃ ≥ {_SEL_MIN}).")

    fig = go.Figure()
    for arr, color, name in [(tgt, "192,57,43", f"{dict(_STEER_CONCEPTS)[concept]} (target)"),
                             (off, "31,111,180", f"{off_name} (off-target)")]:
        mu, s = arr.mean(0), arr.std(0, ddof=1) if len(arr) > 1 else np.zeros_like(fr)
        fig.add_scatter(x=np.r_[fr, fr[::-1]], y=np.r_[mu + s, (mu - s)[::-1]], fill="toself",
                        fillcolor=f"rgba({color},0.15)", line=dict(width=0),
                        hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=fr, y=mu, mode="lines", name=name, line=dict(color=f"rgb({color})", width=2.5))
    fig.add_scatter(x=fr, y=ntgt.mean(0), mode="lines", name="random: target",
                    line=dict(color="rgb(192,57,43)", dash="dot", width=1))
    fig.add_scatter(x=fr, y=noff.mean(0), mode="lines", name="random: off-target",
                    line=dict(color="rgb(31,111,180)", dash="dot", width=1))
    fig.update_layout(
        height=440, xaxis_title="Clamping fraction f", yaxis_title="Probe AUROC",
        yaxis=dict(range=[min(0.4, float(np.min([tgt.min(), off.min()])) - 0.05), 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    st.plotly_chart(fig, width="stretch")


def page_taxonomy_steering() -> None:
    st.title("Taxonomy & Steering")
    st.caption("The paper's quantitative results (§4.3–4.5, Figs. 2–6) with every steering sweep explorable.")
    tab_curves, tab_steer, tab_tax = st.tabs(["Steering curves", "Paper figures", "Taxonomy"])
    with tab_curves:
        _render_steering_curves_subtab()
    with tab_steer:
        _render_steering_static_subtab()
    with tab_tax:
        _render_taxonomy_subtab()


def page_layer_explorer(folder_name: str) -> None:
    """Layer Explorer — animated joint UMAP showing token trajectories across layers."""
    st.header("Layer Explorer")
    st.caption("**Supplementary.** Animated joint UMAP of token trajectories across encoder layers.")
    st.caption(
        "Animated joint UMAP: the same tokens appear at every layer in a shared "
        "coordinate space, so you can see how representations evolve with depth.  "
        "Use the Play button or drag the layer slider."
    )

    umap_dir  = ROOT / "results" / "layer_umap"

    # `*_local/` folders hold E=1 re-trainings of the same encoder; the layer
    # UMAP depends only on the encoder, so fall back to the canonical folder.
    actual_folder = folder_name
    if not (umap_dir / actual_folder / "umap_cache.pt").exists():
        actual_folder = _SPECTRAL_DECODER_CANONICAL.get(folder_name, folder_name)

    cache_path = umap_dir / actual_folder / "umap_cache.pt"

    if not cache_path.exists():
        if _show_build_hints():
            st.info(
                f"No layer UMAP cache found for **{actual_folder}**.  Build it with:\n\n"
                f"```\nuv run tools/build_layer_umap_cache.py --encoder {actual_folder}"
                f" --max-tokens 5000 --max-per-subject 3 --equalize-stages --add-perlayer\n```\n\n"
                f"Cache will be saved to `results/layer_umap/{actual_folder}/umap_cache.pt`."
            )
        else:
            st.info(
                "Layer UMAP isn't built for this encoder. Pick **SleepFM** in the sidebar "
                "for the animated layer trajectory."
            )
        return

    cache = load_layer_umap_cache(str(cache_path))

    # ── Controls ──────────────────────────────────────────────────────────────
    token_meta = dict(cache.get("token_meta", {}))  # mutable copy

    # Build a combined "micro_events" field from individual per-token event fields
    # so users can see all sleep micro-events on a single colour axis.
    _MICRO_EVENT_FIELDS = ["spindle", "slow_wave", "eye_movement", "arousal",
                           "apnea_oa", "apnea_ca", "apnea_hypopnea", "desaturation"]
    _avail_ev = [f for f in _MICRO_EVENT_FIELDS
                 if f in token_meta and token_meta[f] is not None]
    if _avail_ev:
        n_tok_total = len(next(iter(token_meta.values())))
        micro = []
        for i in range(n_tok_total):
            active = [f for f in _avail_ev if str(token_meta[f][i]) == "1"]
            micro.append(active[0] if len(active) == 1 else
                         "multiple" if len(active) > 1 else "none")
        token_meta["micro_events"] = np.array(micro, dtype=object)

    # Fields covered by micro_events composite + raw integer duplicate
    _HIDDEN_META_FIELDS = {
        "sleep_stage",  # raw int — sleep_stage_name is cleaner
        "spindle", "slow_wave", "eye_movement", "arousal",
        "apnea_oa", "apnea_ca", "apnea_hypopnea", "desaturation", "respiratory",
    }
    _META_DISPLAY_NAMES: dict[str, str] = {
        "sleep_stage_name": "Sleep stage",
        "micro_events": "Micro events",
        "age_group": "Age group",
        "gender": "Gender",
        "race": "Race",
        "osa_severity": "OSA severity",
        "ahi_a0h3a": "AHI",
        "mesa_site": "MESA site",
        "sleep_medication": "Sleep medication",
        "arousal_index_group": "Arousal index",
        "spindle_density_group": "Spindle density",
        "slow_wave_density_group": "Slow-wave density",
        "subject_id": "Subject ID",
        "Label": "Label",
        "Band": "Band",
        "Subject": "Subject",
        "Cluster": "Cluster",
    }

    meta_fields = [k for k, v in token_meta.items()
                   if v is not None and len(np.unique(v)) > 1
                   and k not in _HIDDEN_META_FIELDS]
    has_kmeans = bool(cache.get("kmeans")) and -1 in cache.get("layers", [])
    has_perlayer = "xy_perlayer" in cache
    has_lda = "xy_lda" in cache

    # Projection mode toggle (only shown when extra projections are available)
    if has_perlayer or has_lda:
        mode_options = ["Joint (trajectory)"]
        if has_perlayer:
            mode_options.append("Per-layer (structure)")
        if has_lda:
            mode_options.append("LDA (sleep stage)")
        umap_mode = st.radio(
            "Projection mode",
            options=mode_options,
            index=0,
            horizontal=True,
            key="layer_explorer_umap_mode",
            help=(
                "**Joint**: one UMAP fit across all layers — consistent coordinates, "
                "smooth animation, but within-layer structure averaged out.  "
                "**Per-layer**: independent UMAP per layer — each frame shows that "
                "layer's own geometry.  "
                "**LDA**: projects onto the top-2 Linear Discriminant directions for "
                "sleep stage — directly shows what a linear probe sees; if stages are "
                "linearly separable they will cluster visibly here."
            ),
        )
        use_perlayer = umap_mode == "Per-layer (structure)"
        use_lda = umap_mode == "LDA (sleep stage)"
        cache = dict(cache)  # shallow copy so we can swap xy safely
        if use_perlayer:
            cache["xy"] = cache["xy_perlayer"]
        elif use_lda:
            cache["xy"] = cache["xy_lda"]
    else:
        use_perlayer = False
        use_lda = False

    color_options = (["Cluster"] if has_kmeans else []) + ["Band", "Label", "Subject"] + meta_fields

    # For MESA caches that carry sleep_stage_name, default to that field so the
    # plot is immediately meaningful without manual dropdown selection.
    _default_color = (
        "sleep_stage_name" if "sleep_stage_name" in color_options
        else ("Label" if "Label" in color_options else color_options[0])
    )

    col_cb, col_hl, col_info = st.columns([2, 2, 2])
    with col_cb:
        color_by = st.selectbox(
            "Color by",
            options=color_options,
            index=color_options.index(_default_color),
            key="layer_explorer_color_by",
            format_func=lambda x: _META_DISPLAY_NAMES.get(
                x, x.replace("_", " ").title()
            ),
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
        elif color_by == "sleep_stage_name" and "sleep_stage_name" in token_meta:
            k_clusters = 5
            _raw = token_meta["sleep_stage_name"]
            _present = set(str(v) for v in _raw if str(v) != "")
            _cat_vals = [s for s in _SLEEP_STAGE_ORDER if s in _present]
        elif color_by == "micro_events" and "micro_events" in token_meta:
            _EV_ORDER_HL = ["spindle", "slow_wave", "eye_movement", "arousal",
                            "apnea_oa", "apnea_ca", "apnea_hypopnea", "desaturation",
                            "multiple", "none"]
            _present_ev = set(str(v) for v in token_meta["micro_events"] if str(v) != "")
            k_clusters = len(_present_ev)
            _cat_vals = [e for e in _EV_ORDER_HL if e in _present_ev]
        elif color_by == "ahi_a0h3a" and "ahi_a0h3a" in token_meta:
            _present_ahi = {_ahi_category(v) for v in token_meta["ahi_a0h3a"]}
            _cat_vals = [c for c in _AHI_ORDER if c in _present_ahi]
            k_clusters = len(_cat_vals)
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
        _ll = cache.get("layer_labels", {})
        if _ll:
            layer_range = f"{_ll.get(layers[0], str(layers[0]))} – {_ll.get(layers[-1], str(layers[-1]))}"
        elif -1 in layers:
            layer_range = f"Tokenizer – Layer {layers[-1]}"
        else:
            layer_range = f"{layers[0]}–{layers[-1]}"
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
    if use_lda:
        _umap_label = "LDA (sleep stage)"
    elif use_perlayer:
        _umap_label = "Per-layer UMAP"
    else:
        _umap_label = "Joint UMAP"
    fig = _make_animation_figure(cache, color_by, token_meta=token_meta_fig, highlight=_hl_val, umap_label=_umap_label)
    st.plotly_chart(fig, width="stretch")

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
        if _show_build_hints():
            st.info(
                f"No attention cache found for **{exp_name}**. Build it with:\n\n"
                f"```\nuv run tools/build_attention_cache.py --experiment {exp_name}\n```"
            )
        else:
            st.info(
                "Attention cache isn't built for this config. Attention Explorer is a "
                "supplementary tab; the canonical view is on `sleepfm_finetuned_layer{0,1,2}`."
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
    st.plotly_chart(fig_eeg, width="stretch")

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
        st.plotly_chart(fig_top, width="stretch")

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
    st.plotly_chart(fig_mean, width="stretch")

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
            st.plotly_chart(fig_heads, width="stretch")



# ─────────────────────────────────────────────────────────────────────────────
# Concept Steering
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading SAE weights …", max_entries=4)
def _load_sae_for_steering(sae_path: str):
    """Load only the SAE weights needed for concept steering (no SpectralDecoder)."""
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


def page_concept_steering(
    app_cache: Optional[dict], sae_path: str, folder_name: str, layer: int
) -> None:
    st.title("Concept Steering")
    st.caption(
        "**§3.5, Fig. 6.** Clamp the SAE features most aligned with a concept to the "
        "target group's values, then check whether a probe can still read the concept."
    )
    with st.expander("Paper result: Figure 6, spectrum-level steering"):
        _show_paper_image(
            "fig6_spectral.png",
            "Abnormal → normal on SleepFM L3 (E=8, k=64). "
            "Clamping the top n = 104 and 164 features suppresses δ/θ and restores the α peak.",
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
    st.dataframe(pd.DataFrame(tbl_rows), width="stretch", hide_index=True)

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
            st.plotly_chart(fig_probe, width="stretch")
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
    st.plotly_chart(fig_diff, width="stretch")


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
