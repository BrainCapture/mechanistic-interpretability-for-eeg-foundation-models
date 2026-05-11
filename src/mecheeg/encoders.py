"""
encoders.py — Swappable encoder backends for SAE4EEG
=====================================================
Defines the ``EncoderBackend`` protocol and two concrete implementations:

  SleepFMBackend  — wraps the local ``SetTransformer`` model (sleepfm.py).
                    Native sample rate: 128 Hz.  embed_dim = 128.
                    Hooks into transformer_encoder.layers for activation
                    extraction.

  REVEBackend     — wraps ``brain-bzh/reve-base`` from HuggingFace.
                    Native sample rate: 200 Hz.  embed_dim = 768 (Base).
                    Requires HF login + EDPB agreement (gated model).
                    Expects 200 Hz input (use the v1 dataset, not v2).
                    Uses hard-coded standard 10-20 electrode positions for
                    the 19-channel montage used in the D4 dataset.

Usage
-----
    from mecheeg.encoders import load_encoder

    # SleepFM (default, weights from sleepfm_weights.pt)
    backend = load_encoder("sleepfm", weights_path="sleepfm_weights.pt")

    # REVE (requires `pip install transformers` and HF credentials)
    backend = load_encoder("reve")

    # Collect per-token activations:  (B, S, embed_dim)
    tokens = backend.encode(batch)   # batch at encoder's native sample_rate_in

API contract
------------
Every ``EncoderBackend`` guarantees:

  .encode(x)               → Tensor (B, S, embed_dim)
                             x must be at ``sample_rate_in`` Hz — callers
                             should use the dataset matching this rate.

  .encode_flat(x)          → Tensor (B*S, embed_dim)   (flattened tokens)

  .get_hookable_layers()   → list[nn.Module]
                             Ordered list of transformer layers that
                             ActivationExtractor can hook into.

  .embed_dim               → int
  .sample_rate_in          → int  (native input rate expected by this backend)
  .model                   → nn.Module  (underlying model)

Dataset selection
-----------------
Each backend declares its required input sample rate via ``sample_rate_in``.
Tools should select the matching dataset version:

  SleepFMBackend  sample_rate_in=128  →  D4-v3-preprocessed-v2
  REVEBackend     sample_rate_in=200  →  D4-v3-preprocessed-v1
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

# ── 10-20 system electrode positions (x, y, z in metres, MNI space) ─────────
# Standard 19-channel clinical montage used in the D4 dataset.
# Coordinates taken from MNE-Python's standard_1020 montage, same values
# as in brain-bzh/reve/index.html.
_POSITIONS_10_20_19CH = {
    "Fp1": (-0.02813, 0.08611, 0.01542),
    "Fp2": ( 0.02845, 0.08558, 0.01530),
    "F7":  (-0.07026, 0.04247, -0.01142),
    "F3":  (-0.05024, 0.05311,  0.04219),
    "Fz":  ( 0.00031, 0.05851,  0.06646),
    "F4":  ( 0.05184, 0.05430,  0.04081),
    "F8":  ( 0.07254, 0.04490, -0.01142),
    "T7":  (-0.07127,-0.02349, -0.04605),   # T3 in older 10-20 notation
    "C3":  (-0.06536,-0.01163,  0.06436),
    "Cz":  ( 0.00040,-0.00917,  0.10024),
    "C4":  ( 0.06712,-0.01090,  0.06358),
    "T8":  ( 0.07127,-0.02349, -0.04605),   # T4
    "T5":  (-0.07127,-0.07255, -0.00250),   # P7 / T5
    "P3":  (-0.05301,-0.07879,  0.05594),
    "Pz":  ( 0.00033,-0.08111,  0.08262),
    "P4":  ( 0.05567,-0.07856,  0.05656),
    "T6":  ( 0.07127,-0.07255, -0.00250),   # P8 / T6
    "O1":  (-0.02941,-0.11245,  0.00884),
    "O2":  ( 0.02984,-0.11216,  0.00880),
}

# Ordered to match the D4 dataset channel order
_D4_CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7",  "C3",  "Cz", "C4", "T8",
    "T5",  "P3",  "Pz", "P4", "T6",
    "O1",  "O2",
]


def _get_positions_tensor(channel_names: List[str]) -> torch.Tensor:
    """Return (C, 3) float32 tensor of electrode 3D positions."""
    coords = []
    for name in channel_names:
        if name not in _POSITIONS_10_20_19CH:
            raise KeyError(
                f"Unknown channel '{name}'. Add it to _POSITIONS_10_20_19CH."
            )
        coords.append(_POSITIONS_10_20_19CH[name])
    return torch.tensor(coords, dtype=torch.float32)


def _remove_module_prefix(state_dict: dict) -> dict:
    """Strip 'module.' prefix added by DataParallel wrappers."""
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBackend(ABC):
    """
    Common interface for all encoder backends.

    Subclasses must set ``self.embed_dim`` and implement
    ``encode()``, ``get_hookable_layers()``, and ``to()``.
    """

    embed_dim: int       # dimensionality of token embeddings
    sample_rate_in: int  # always 128 (callers always pass 128 Hz data)
    model: nn.Module     # underlying PyTorch model

    # ------------------------------------------------------------------
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : (B, C, T) float32 tensor, **128 Hz** sample rate.

        Returns
        -------
        Tensor of shape (B, S, embed_dim) — per-token embeddings.
        """
        ...

    def encode_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: returns (B*S, embed_dim)."""
        out = self.encode(x)
        B, S, E = out.shape
        return out.reshape(B * S, E)

    # ------------------------------------------------------------------
    @abstractmethod
    def get_hookable_layers(self) -> List[nn.Module]:
        """
        Return the ordered list of transformer layers that
        ActivationExtractor should hook into.  Index 0 = first layer, etc.
        """
        ...

    # ------------------------------------------------------------------
    @abstractmethod
    def to(self, device: str | torch.device) -> "EncoderBackend":
        """Move underlying model to device; return self."""
        ...

    def eval(self) -> "EncoderBackend":
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "EncoderBackend":
        self.model.train(mode)
        return self

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"embed_dim={self.embed_dim}, "
            f"layers={len(self.get_hookable_layers())})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SleepFM backend
# ─────────────────────────────────────────────────────────────────────────────

class SleepFMBackend(EncoderBackend):
    """
    Wraps the local SetTransformer model (SleepFM architecture).

    Parameters
    ----------
    weights_path : str or Path, optional
        Path to ``sleepfm_weights.pt``.  If *None* the model is returned
        with random weights (useful for tests).
    patch_size : int
        Tokenisation patch size in samples.  128 samples = 1 second @ 128 Hz.
    embed_dim : int
        Transformer embedding dimension.  Must match the checkpoint.
    target_layer : int
        Index of the transformer layer whose activations will be used as
        SAE input.  0-indexed.  Default: 2 (last layer of the 3-layer model).
    """

    sample_rate_in = 128  # SleepFM was trained at 128 Hz

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        patch_size: int = 128,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        pooling_head: int = 8,
        dropout: float = 0.3,
        target_layer: int = 2,
    ):
        from mecheeg.sleepfm import SetTransformer  # local import to avoid cycles

        self.embed_dim    = embed_dim
        self.patch_size   = patch_size
        self.target_layer = target_layer

        self.model = SetTransformer(
            in_channels=1,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            pooling_head=pooling_head,
            dropout=dropout,
        )

        if weights_path is not None:
            weights_path = Path(weights_path)
            ckpt = torch.load(weights_path, weights_only=False, map_location="cpu")
            raw_sd = ckpt.get("state_dict", ckpt)
            # Strip DataParallel 'module.' prefix
            raw_sd = _remove_module_prefix(raw_sd)
            # Strip PyTorch Lightning 'encoder.' prefix if present
            if any(k.startswith("encoder.") for k in raw_sd):
                raw_sd = {
                    (k[len("encoder."):] if k.startswith("encoder.") else k): v
                    for k, v in raw_sd.items()
                }
            missing, unexpected = self.model.load_state_dict(raw_sd, strict=False)
            if missing:
                print(f"  [warn] {len(missing)} keys not in checkpoint: {missing[:3]}…")
            if unexpected:
                print(f"  [warn] {len(unexpected)} unexpected keys ignored: {unexpected[:3]}…")
            print(f"[SleepFMBackend] Loaded weights from {weights_path}")

        self.model.eval()

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T) at 128 Hz.
        Returns (B, S, embed_dim) where S = T // patch_size.
        """
        # SetTransformer forward returns (pooled: (B, E), embedding: (B, S, E))
        with torch.no_grad():
            _, embedding = self.model(x)
        return embedding  # (B, S, E)

    # ------------------------------------------------------------------
    def get_hookable_layers(self) -> List[nn.Module]:
        """Returns the transformer encoder layers (nn.TransformerEncoderLayer)."""
        return list(self.model.transformer_encoder.layers)

    # ------------------------------------------------------------------
    def to(self, device: str | torch.device) -> "SleepFMBackend":
        self.model.to(device)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# REVE backend
# ─────────────────────────────────────────────────────────────────────────────

class REVEBackend(EncoderBackend):
    """
    Wraps the ``brain-bzh/reve-base`` HuggingFace model.

    REVE requires input at **200 Hz**.  Pass data from the D4-v3-preprocessed-v1
    dataset (native 200 Hz); no resampling is performed.

    Prerequisites
    -------------
    1. Install transformers::

         uv add transformers

    2. Log into HuggingFace and accept the EDPB usage agreement::

         huggingface-cli login

    Parameters
    ----------
    model_name : str
        HuggingFace repo ID for the REVE model.
    positions_name : str
        HuggingFace repo ID for the REVE position bank.
    channel_names : list[str], optional
        Ordered channel names matching the data.  Defaults to the standard
        19-channel 10-20 montage used in the D4 dataset.
    target_layer : int, optional
        Index into the transformer layers for SAE activation extraction.
        Default: ``-1`` (last layer).
    embed_dim : int, optional
        Override embed_dim discovery.  Set automatically from the model config
        if not provided.
    """

    sample_rate_in = 200  # REVE requires 200 Hz input; use D4-v3-preprocessed-v1

    def __init__(
        self,
        model_name: str = "brain-bzh/reve-base",
        positions_name: str = "brain-bzh/reve-positions",
        channel_names: Optional[List[str]] = None,
        target_layer: int = -1,
        embed_dim: Optional[int] = None,
        weights_path: Optional[str | Path] = None,
    ):
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for REVEBackend. "
                "Install with: uv add transformers"
            ) from e

        self.channel_names = channel_names or _D4_CHANNEL_NAMES
        self.target_layer  = target_layer

        print(f"[REVEBackend] Loading position bank from '{positions_name}' …")
        self._pos_bank = AutoModel.from_pretrained(
            positions_name, trust_remote_code=True
        )
        self._pos_bank.eval()

        print(f"[REVEBackend] Loading model from '{model_name}' …")
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()

        # Overlay finetuned weights if a local checkpoint is provided.
        # Expected format: PyTorch Lightning .ckpt with state_dict keys like
        # "encoder.<original_key>" (the Lightning module wraps REVE as self.encoder).
        if weights_path is not None:
            weights_path = Path(weights_path)
            print(f"[REVEBackend] Overlaying finetuned weights from '{weights_path}' …")
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            raw_sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            # Strip "encoder." prefix; discard head.* and other non-backbone keys
            backbone_sd = {
                k[len("encoder."):]: v
                for k, v in raw_sd.items()
                if k.startswith("encoder.")
            }
            missing, unexpected = self.model.load_state_dict(backbone_sd, strict=False)
            if missing:
                print(f"  [warn] {len(missing)} keys not in checkpoint: {missing[:3]}…")
            if unexpected:
                print(f"  [warn] {len(unexpected)} unexpected keys ignored: {unexpected[:3]}…")
            print(f"  [ok] Loaded {len(backbone_sd)} backbone parameters")

        # Discover embed_dim from model config
        if embed_dim is not None:
            self.embed_dim = embed_dim
        elif hasattr(self.model, "config"):
            cfg = self.model.config
            self.embed_dim = getattr(
                cfg, "hidden_size",
                getattr(cfg, "embed_dim",
                getattr(cfg, "d_model", 768))
            )
        else:
            self.embed_dim = 768  # REVE-Base default

        print(f"[REVEBackend] embed_dim={self.embed_dim}, "
              f"input_sr={self.sample_rate_in} Hz")

        # Pre-compute electrode positions (C, 3) from the position bank
        self._positions_1ch: torch.Tensor = self._load_positions()

    # ------------------------------------------------------------------
    def _load_positions(self) -> torch.Tensor:
        """
        Query the REVE position bank once and cache the (C, 3) tensor.
        Falls back to local hard-coded coordinates if the bank doesn't
        recognise a channel name.
        """
        try:
            with torch.no_grad():
                pos = self._pos_bank(self.channel_names)  # (C, 3)
            return pos.cpu()
        except Exception:
            # Fallback: use hard-coded 10-20 positions
            return _get_positions_tensor(self.channel_names).cpu()

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T) at 200 Hz (use D4-v3-preprocessed-v1 dataset).
        C may be larger than len(channel_names); the first len(channel_names)
        channels are selected (standard 10-20 montage occupies indices 0–18).
        Returns (B, S, embed_dim).
        """
        B = x.shape[0]
        device = next(self.model.parameters()).device

        # Select channels matching the position bank (first N channels)
        n_ch = len(self.channel_names)
        if x.shape[1] != n_ch:
            x = x[:, :n_ch, :]

        # 1. Expand electrode positions for batch
        positions = self._positions_1ch.to(device)           # (C, 3)
        positions = positions.unsqueeze(0).expand(B, -1, -1) # (B, C, 3)

        # 2. Forward pass
        with torch.no_grad():
            output = self.model(x.float().to(device), positions)

        # 3. Extract per-token embeddings
        #    HuggingFace models return BaseModelOutput with .last_hidden_state
        #    REVE returns a plain tensor of shape (B, C, S, E).
        if hasattr(output, "last_hidden_state"):
            tokens = output.last_hidden_state  # (B, S, E) or (B, C, S, E)
        elif isinstance(output, torch.Tensor):
            tokens = output
        else:
            tokens = output[0]

        # REVE outputs (B, C, S, E) — flatten channel × time → (B, C*S, E)
        if tokens.dim() == 4:
            B2, C, S, E = tokens.shape
            tokens = tokens.reshape(B2, C * S, E)

        return tokens.cpu()

    # ------------------------------------------------------------------
    def get_hookable_layers(self) -> List[nn.Module]:
        """
        Returns the list of transformer block layers inside REVE.

        REVE's TransformerBackbone exposes its blocks under
        ``model.transformer.layers`` — a ModuleList of 22 blocks, each
        a ModuleList([Attention, FeedForward]).

        A ``ModuleList`` has no ``forward()`` method and therefore cannot
        receive PyTorch forward hooks.  We hook the ``FeedForward`` sub-module
        (index 1) of each block, which outputs ``(B, C*S, E)`` after the
        full attention + MLP pass — equivalent to the standard "block output".

        Falls back to a broader attribute search if the architecture changes.
        """
        # Primary path for brain-bzh/reve-base:
        # transformer.layers is a ModuleList of 22 blocks, each ModuleList([Attn, FF])
        try:
            blocks = list(self.model.transformer.layers)
            if blocks:
                hookable = []
                for block in blocks:
                    children = list(block.children())
                    # Hook FeedForward (last child) as the "block output"
                    hookable.append(children[-1])
                return hookable
        except AttributeError:
            pass

        candidates = [
            # Common HuggingFace transformer attribute paths
            lambda m: list(m.encoder.layer),
            lambda m: list(m.transformer.layer),
            lambda m: list(m.encoder.layers),
            lambda m: list(m.blocks),
            lambda m: list(m.layers),
        ]
        for fn in candidates:
            try:
                layers = fn(self.model)
                if layers:
                    return layers
            except AttributeError:
                continue

        # Last resort: collect all nn.Module children that look like blocks
        # (have a self-attention sub-module)
        blocks = []
        for mod in self.model.modules():
            children = list(mod.children())
            if any(hasattr(c, "self_attn") or hasattr(c, "attention") for c in children):
                blocks.append(mod)
        if blocks:
            return blocks

        raise RuntimeError(
            "Could not discover transformer layers in the REVE model. "
            "Please inspect the model architecture and subclass REVEBackend "
            "to override get_hookable_layers()."
        )

    # ------------------------------------------------------------------
    def to(self, device: str | torch.device) -> "REVEBackend":
        self.model.to(device)
        self._pos_bank.to(device)
        self._positions_1ch = self._positions_1ch.to(device)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# LaBraM backend
# ─────────────────────────────────────────────────────────────────────────────

class LaBraMBackend(EncoderBackend):
    """
    Wraps the LaBraM-Base ``NeuralTransformer`` (Jiang et al., ICLR 2024).

    Native sample rate: **200 Hz**.  Use the D4-v3-preprocessed-v1 dataset
    (200 Hz, 60-second windows = 12000 samples).  Patch size is 200 samples
    (1 second per token).  The 27-channel D4 input is sliced down to the
    first 19 channels (standard 10-20 montage) before the encoder.

    Parameters
    ----------
    weights_path : str or Path
        Path to a finetuned LaBraM checkpoint produced by
        ``tools/finetune_labram_binary.py``.  Expected layout::

            ckpt = {
                "state_dict": {"encoder.<...>": ..., "head.<...>": ...},
                "config": {...},
            }

        The ``encoder.`` prefix is stripped on load.  If ``None``, the
        encoder is initialised from braindecode's pretrained weights at
        ``checkpoints/pretrained/labram/labram-base.pth``.
    target_layer : int
        Index into ``self.model.blocks`` for SAE activation extraction.
        Default: 11 (last of 12 transformer blocks).
    max_time_patches : int
        Length of the temporal embedding.  Defaults to 60 to match the
        60-second context used by the binary finetune.
    """

    sample_rate_in = 200
    embed_dim = 200

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        target_layer: int = 11,
        max_time_patches: int = 60,
        n_channels: int = 19,
        init_values: float = 0.1,
    ):
        from mecheeg.labram import labram_base_patch200_200, remap_braindecode_state_dict

        self.target_layer = target_layer
        self.n_channels = n_channels
        self.max_time_patches = max_time_patches
        # Per-channel-per-second token layout. The 19 channel names mirror REVE's
        # 10-20 montage so that SpectralDecoder/SAE token-collection helpers take the same
        # per-channel branch (rather than the SleepFM channel-pooled branch).
        self.channel_names = _D4_CHANNEL_NAMES[:n_channels]
        # forward_features(return_all_tokens=True) emits 1 CLS prefix token before
        # the C*S patch tokens — must be stripped before reshape.
        self.n_prefix_tokens = 1

        self.model = labram_base_patch200_200(
            max_time_patches=max_time_patches,
            init_values=init_values,
        )

        if weights_path is None:
            # Fall back to upstream pretrained weights
            default_pre = Path("checkpoints/pretrained/labram/labram-base.pth")
            if default_pre.exists():
                weights_path = default_pre
            else:
                print("[LaBraMBackend] No weights_path provided and pretrained file "
                      f"not found at {default_pre}; encoder is randomly initialised.")

        if weights_path is not None:
            weights_path = Path(weights_path)
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            raw_sd = ckpt.get("state_dict", ckpt)
            # Two cases:
            #  (a) braindecode pretrained — keys like "position_embedding", "blocks.0.mlp.0..."
            #  (b) local finetune — keys like "encoder.pos_embed", "head.weight"
            if any(k.startswith("encoder.") for k in raw_sd):
                backbone_sd = {
                    k[len("encoder."):]: v
                    for k, v in raw_sd.items()
                    if k.startswith("encoder.")
                }
            else:
                backbone_sd = remap_braindecode_state_dict(
                    raw_sd, target_time_patches=max_time_patches
                )

            missing, unexpected = self.model.load_state_dict(backbone_sd, strict=False)
            if missing:
                print(f"  [warn] {len(missing)} missing keys: {missing[:3]}…")
            if unexpected:
                print(f"  [warn] {len(unexpected)} unexpected keys: {unexpected[:3]}…")
            print(f"[LaBraMBackend] Loaded weights from {weights_path}")

        self.model.eval()

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T) at 200 Hz.  T must be divisible by 200.  C may exceed 19;
            the first 19 channels are selected.

        Returns
        -------
        Tensor of shape (B, C*S, embed_dim) where C=n_channels and S=T//200.
        The CLS prefix token from forward_features is stripped here so callers
        receive a clean per-(channel, time) token grid.
        """
        if x.shape[1] != self.n_channels:
            x = x[:, :self.n_channels, :]
        device = next(self.model.parameters()).device
        with torch.no_grad():
            tokens = self.model.forward_features(
                x.to(device), return_all_tokens=True,
            )
        if self.n_prefix_tokens:
            tokens = tokens[:, self.n_prefix_tokens:, :]
        return tokens.cpu()

    # ------------------------------------------------------------------
    def get_hookable_layers(self) -> List[nn.Module]:
        """Return the 12 transformer blocks for forward-hook activation extraction."""
        return list(self.model.blocks)

    # ------------------------------------------------------------------
    def to(self, device: str | torch.device) -> "LaBraMBackend":
        self.model.to(device)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Model cards
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CARDS: dict[str, dict] = {
    # ── SleepFM family ────────────────────────────────────────────────────────
    "sleepfm_v2.0": {
        "display_name": "SleepFM v2.0",
        "family": "SleepFM",
        "specs": {
            "embed_dim":      128,
            "layers":         6,
            "tokenizer":      "CNN",
            "position_emb":   False,
            "reconstruction": "None",
            "optimizer":      "SGD",
            "patch_size":     640,
            "sample_rate_hz": 128,
        },
        "pretraining": (
            "Contrastive learning (InfoNCE with Kendall multi-task weighting) on PSG "
            "recordings. SGD with momentum 0.9, step LR schedule (no weight decay), "
            "and fp32 precision — the original v1.1 training recipe applied to the "
            "v2 CNN-tokenizer architecture."
        ),
        "notes": (
            "SGD/fp32 baseline for the v2 series. Shares the CNN tokenizer, 6-layer "
            "transformer, and patch_size=640 with all other v2 models, but retains "
            "the old optimizer recipe from v1.1 (SGD, step LR, no gradient clipping). "
            "Paired with v2.1, it isolates the contribution of the optimizer upgrade "
            "alone, independent of any changes to the learning objective or spatial structure."
        ),
        "notes_label": "Role in ablation",
    },
    "sleepfm": {
        "display_name": "SleepFM v1.1",
        "family": "SleepFM",
        "specs": {
            "embed_dim":      128,
            "layers":         3,
            "tokenizer":      "Linear",
            "position_emb":   False,
            "reconstruction": "None",
            "optimizer":      "SGD",
            "patch_size":     128,
            "sample_rate_hz": 128,
        },
        "pretraining": (
            "Contrastive learning on large-scale polysomnography (PSG) recordings "
            "spanning EEG, respiratory, cardiac, and EMG modalities. Optimised with SGD."
        ),
        "notes": (
            "We make only minimal changes to the original SleepFM encoder — reducing "
            "temporal compression from 5 seconds to 1 second — to ensure compatibility "
            "with short-duration, non-sleep biosignal events. This isolates the effect "
            "of the pretraining distribution from architectural specialisation, attributing "
            "downstream performance to sleep-based representation learning rather than "
            "design choices."
        ),
        "notes_label": "Our adaptations",
    },
    "sleepfm_v2.1": {
        "display_name": "SleepFM v2.1",
        "family": "SleepFM",
        "specs": {
            "embed_dim":      128,
            "layers":         6,
            "tokenizer":      "CNN",
            "position_emb":   False,
            "reconstruction": "None",
            "optimizer":      "AdamW",
            "patch_size":     640,
            "sample_rate_hz": 128,
        },
        "pretraining": (
            "Contrastive learning (InfoNCE with Kendall multi-task weighting) on PSG "
            "recordings. AdamW with weight decay 0.05, cosine LR schedule, and gradient "
            "clipping — the improved training recipe shared by all v2 models."
        ),
        "notes": (
            "Contrastive-only, position-free baseline for the v2 series. Compared to v1.1 "
            "it doubles the number of transformer layers, replaces the linear tokenizer "
            "with a CNN, and upgrades the optimiser, while keeping the learning objective "
            "as simple as possible. The closest v2 counterpart to v1.1 and the natural "
            "starting point for isolating the effect of each subsequent addition."
        ),
        "notes_label": "Role in ablation",
    },
    "sleepfm_v2.3": {
        "display_name": "SleepFM v2.3",
        "family": "SleepFM",
        "specs": {
            "embed_dim":      128,
            "layers":         6,
            "tokenizer":      "CNN",
            "position_emb":   False,
            "reconstruction": "After CNN tokenizer",
            "optimizer":      "AdamW",
            "patch_size":     640,
            "sample_rate_hz": 128,
        },
        "pretraining": (
            "Joint contrastive learning and masked autoencoding (MAE) with fixed loss "
            "weights (contrastive 1.0, MAE 1.0). AdamW with weight decay 0.05 and "
            "cosine LR schedule."
        ),
        "notes": (
            "Adds a MAE reconstruction head directly after the CNN tokenizer, before "
            "the signal enters the transformer. Because reconstruction happens at the "
            "single-channel token level — prior to any cross-channel mixing — the model "
            "must encode channel-local spectral structure alongside the cross-channel "
            "contrastive signal. Compared to v2.1, this isolates the effect of a "
            "token-level auxiliary objective with position embedding still absent."
        ),
        "notes_label": "Role in ablation",
    },
    "sleepfm_v2.4": {
        "display_name": "SleepFM v2.4",
        "family": "SleepFM",
        "specs": {
            "embed_dim":      128,
            "layers":         6,
            "tokenizer":      "CNN",
            "position_emb":   True,
            "reconstruction": "None",
            "optimizer":      "AdamW",
            "patch_size":     640,
            "sample_rate_hz": 128,
        },
        "pretraining": (
            "Contrastive learning (InfoNCE with Kendall multi-task weighting) on PSG "
            "recordings. AdamW with weight decay 0.05 and cosine LR schedule."
        ),
        "notes": (
            "Contrastive-only counterpart to v2.1 with channel position embeddings added. "
            "The model is explicitly informed of electrode identity, introducing spatial "
            "structure into the transformer without changing the learning objective. "
            "The closest 'full upgrade' of v1.1 — same objective, better training recipe, "
            "and spatial awareness — without any reconstruction auxiliary task."
        ),
        "notes_label": "Role in ablation",
    },
    "sleepfm_v2.5": {
        "display_name": "SleepFM v2.5",
        "family": "SleepFM",
        "specs": {
            "embed_dim":      128,
            "layers":         6,
            "tokenizer":      "CNN",
            "position_emb":   True,
            "reconstruction": "After spatial pooling",
            "optimizer":      "AdamW",
            "patch_size":     640,
            "sample_rate_hz": 128,
        },
        "pretraining": (
            "Joint contrastive learning and spatial MAE with fixed loss weights. "
            "AdamW with weight decay 0.05 and cosine LR schedule."
        ),
        "notes": (
            "Adds a spatial MAE reconstruction head after spatial pooling but before "
            "temporal pooling in the transformer. The model must reconstruct raw channel "
            "signals from representations that have already aggregated information across "
            "the spatial dimension, encouraging coarser-grained spatially-aware encodings "
            "than the token-level MAE in v2.3. Combined with position embeddings, this "
            "completes the 2×2 ablation grid of {position embedding} × {reconstruction stage}."
        ),
        "notes_label": "Role in ablation",
    },
    "sleepfm_v2.6": {
        "display_name": "SleepFM v2.6",
        "family": "SleepFM",
        "specs": {
            "embed_dim":      128,
            "layers":         6,
            "tokenizer":      "CNN",
            "position_emb":   True,
            "reconstruction": "None",
            "optimizer":      "AdamW",
            "patch_size":     128,
            "sample_rate_hz": 128,
        },
        "pretraining": (
            "Contrastive learning (InfoNCE with Kendall multi-task weighting) on PSG "
            "recordings. AdamW with weight decay 0.05 and cosine LR schedule. "
            "1-second patch tokenisation (patch_size=128 at 128 Hz)."
        ),
        "notes": (
            "Contrastive-only model with channel position embeddings and 1-second patches. "
            "Equivalent to v2.4 but with patch_size reduced from 640 (5 s) to 128 (1 s), "
            "giving finer temporal resolution at the cost of shorter context per token. "
            "Isolates the effect of patch granularity in the presence of spatial structure."
        ),
        "notes_label": "Role in ablation",
    },
    "sleepfm_v2.7": {
        "display_name": "SleepFM v2.7",
        "family": "SleepFM",
        "specs": {
            "embed_dim":      128,
            "layers":         6,
            "tokenizer":      "CNN",
            "position_emb":   False,
            "reconstruction": "None",
            "optimizer":      "AdamW",
            "patch_size":     128,
            "sample_rate_hz": 128,
        },
        "pretraining": (
            "Contrastive learning (InfoNCE with Kendall multi-task weighting) on PSG "
            "recordings. AdamW with weight decay 0.05 and cosine LR schedule. "
            "1-second patch tokenisation (patch_size=128 at 128 Hz)."
        ),
        "notes": (
            "Contrastive-only, position-free model with 1-second patches. "
            "Equivalent to v2.1 but with patch_size reduced from 640 (5 s) to 128 (1 s). "
            "Paired with v2.6, it isolates the contribution of channel position embeddings "
            "at the finer 1-second patch granularity."
        ),
        "notes_label": "Role in ablation",
    },
    # ── SleepFM granular (multi-class fine-tune on V4 10-class labels) ─────────
    "sleepfm_granular": {
        "display_name": "SleepFM granular",
        "family": "SleepFM",
        "specs": {
            "embed_dim":      128,
            "layers":         3,
            "tokenizer":      "Linear",
            "position_emb":   False,
            "reconstruction": "None",
            "optimizer":      "AdamW",
            "patch_size":     128,
            "sample_rate_hz": 128,
        },
        "pretraining": (
            "SleepFM v1.1 pretrained on large-scale PSG recordings, then fine-tuned "
            "with 10-class cross-entropy on the D4-v4 dataset (256 Hz, 35,920 windows). "
            "Class-weighted loss to handle 83.7% normal-class imbalance. "
            "Two-phase training: head-only (5 epochs) then full fine-tuning (30 epochs). "
            "Labels: normal (0), diffuse slowing (1), focal slowing (2), focal sharp waves (3), "
            "focal spike-wave (4), gen. spike-wave (5), gen. polyspike-wave (6), "
            "gen. sharp waves (7), burst suppression (8), seizure (9)."
        ),
        "notes": (
            "Trained to separate EEG abnormality subtypes at representation level. "
            "Primary motivation: the binary-trained encoder encodes epileptiform concepts "
            "but is blind to slowing subtypes (diffuse/focal slowing, TCAV p>0.05 at layer 2). "
            "Multi-class supervision forces each subtype to occupy a distinct representation "
            "direction, making slowing concepts TCAV-detectable."
        ),
        "notes_label": "Motivation",
    },
    # ── LaBraM ────────────────────────────────────────────────────────────────
    "labram": {
        "display_name": "LaBraM-Base",
        "family": "LaBraM",
        "specs": {
            "embed_dim":      200,
            "layers":         12,
            "tokenizer":      "Temporal Conv",
            "position_emb":   True,
            "reconstruction": "VQ-NSP (pretraining)",
            "optimizer":      "AdamW",
            "patch_size":     200,
            "sample_rate_hz": 200,
        },
        "pretraining": (
            "Vector-quantized neural spectrum prediction (VQ-NSP) on ~2,500 hours "
            "of EEG from ~20 datasets (Jiang et al., ICLR 2024). The neural tokenizer "
            "is trained first to encode 1-second EEG patches into discrete codes; the "
            "transformer is then pre-trained to reconstruct masked codes."
        ),
        "notes": (
            "Pretrained weights from braindecode/labram-pretrained on HuggingFace "
            "(LaBraM-Base, 5.8M params). Finetuned in-house for binary normal/abnormal "
            "classification on the D4-v3 dataset (200 Hz, 60-second windows). The "
            "temporal embedding is interpolated from 16 to 60 positions to extend "
            "the context from the original 16-second pretraining window."
        ),
        "notes_label": "Access & adaptation",
    },
    # ── REVE ──────────────────────────────────────────────────────────────────
    "reve": {
        "display_name": "REVE",
        "family": "REVE",
        "specs": {
            "embed_dim":      512,
            "layers":         22,
            "tokenizer":      "Patch",
            "position_emb":   True,
            "reconstruction": "None",
            "optimizer":      "—",
            "sample_rate_hz": 200,
        },
        "pretraining": (
            "Self-supervised pretraining on large-scale brain recordings spanning diverse "
            "EEG datasets. Spatial structure is encoded via continuous 3D electrode "
            "coordinates (MNI space) queried from the REVE position bank at inference time."
        ),
        "notes": (
            "Gated on HuggingFace (brain-bzh/reve-base). Requires HF login and acceptance "
            "of the EDPB usage agreement. Produces per-channel, per-second embeddings of "
            "shape (C×S, 512) — 19 channels × S time patches per window."
        ),
        "notes_label": "Access & output shape",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SleepFM v2 backend
# ─────────────────────────────────────────────────────────────────────────────

class SleepFMv2Backend(SleepFMBackend):
    """
    Backend for SleepFM v2.x models (v2.1, v2.3, v2.4, v2.5).

    All v2 models share the SetTransformer architecture with 6 transformer
    layers but use a different CNN tokenizer (``TokenizerV2``) trained with
    patch_size=640 (5-second patches at 128 Hz).  TokenizerV2 uses
    Conv→LayerNorm→ELU blocks only (no BatchNorm), which is why we cannot
    reuse the parent class directly.

    v2.4 and v2.5 store a ``channel_emb`` table in their checkpoints (used
    during pretraining to inject channel-group identity into patch tokens).
    SetTransformer.forward() does not apply this embedding, so it appears as
    an unexpected key and is silently ignored via strict=False.  Transformer
    layer weights load correctly; only the explicit channel-identity signal is
    absent at inference time.
    """

    sample_rate_in = 128

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        patch_size: int = 640,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        pooling_head: int = 8,
        dropout: float = 0.3,
        target_layer: int = 5,
        max_seq_length: int = 128,
    ):
        from mecheeg.sleepfm import SetTransformer, TokenizerV2  # avoid circular

        self.embed_dim    = embed_dim
        self.patch_size   = patch_size
        self.target_layer = target_layer

        # Build SetTransformer skeleton, then replace the tokenizer with the
        # v2 variant before loading weights so state-dict shapes match.
        self.model = SetTransformer(
            in_channels=1,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            pooling_head=pooling_head,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )
        self.model.patch_embedding = TokenizerV2(
            input_size=patch_size, output_size=embed_dim
        )

        if weights_path is not None:
            weights_path = Path(weights_path)
            ckpt   = torch.load(weights_path, weights_only=False, map_location="cpu")
            raw_sd = ckpt.get("state_dict", ckpt)
            raw_sd = _remove_module_prefix(raw_sd)
            missing, unexpected = self.model.load_state_dict(raw_sd, strict=False)
            if missing:
                print(f"  [warn] {len(missing)} keys not in checkpoint: {missing[:3]}…")
            if unexpected:
                print(f"  [warn] {len(unexpected)} unexpected keys ignored: {unexpected[:3]}…")
            print(f"[SleepFMv2Backend] Loaded weights from {weights_path}")

        self.model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

# Standard v2 models: patch_size=640 (5-second patches)
_SLEEPFM_V2_NAMES = {"sleepfm_v2.0", "sleepfm_v2.1", "sleepfm_v2.3", "sleepfm_v2.4", "sleepfm_v2.5"}
# 1-second patch variants: patch_size=128
_SLEEPFM_V2_128P_NAMES = {"sleepfm_v2.6", "sleepfm_v2.7"}


def load_encoder(
    name: str,
    weights_path: Optional[str | Path] = None,
    **kwargs,
) -> EncoderBackend:
    """
    Factory function for encoder backends.

    Parameters
    ----------
    name : str
        ``"sleepfm"``, ``"sleepfm_granular"``, ``"sleepfm_finetuned"``,
        ``"sleepfm_pretrained"``, ``"sleepfm_v2.0"``, ``"sleepfm_v2.1"``,
        ``"sleepfm_v2.3"``, ``"sleepfm_v2.4"``, ``"sleepfm_v2.5"``,
        ``"sleepfm_v2.6"``, ``"sleepfm_v2.7"``,
        or ``"reve"``.
    weights_path : str or Path, optional
        For SleepFM variants: path to the ``.pt`` checkpoint.
        For ``"reve"``: optional path to a finetuned ``.ckpt`` file.
    **kwargs
        Additional keyword arguments forwarded to the backend constructor.

    Returns
    -------
    EncoderBackend

    Examples
    --------
    >>> backend = load_encoder("sleepfm", weights_path="sleepfm_weights.pt")
    >>> backend = load_encoder("sleepfm_v2.1", weights_path="best.pt")
    >>> backend = load_encoder("sleepfm_v2.6", weights_path="best.pt")
    >>> backend = load_encoder("reve")
    """
    name = name.lower().strip()

    if name in ("sleepfm", "sleep_fm", "set_transformer", "sleepfm_granular",
                "sleepfm_finetuned", "sleepfm_pretrained"):
        return SleepFMBackend(weights_path=weights_path, **kwargs)

    if name in _SLEEPFM_V2_NAMES:
        return SleepFMv2Backend(weights_path=weights_path, **kwargs)

    if name in _SLEEPFM_V2_128P_NAMES:
        # v2.6 and v2.7 use 1-second patches and max_seq_length=300
        kwargs.setdefault("patch_size", 128)
        kwargs.setdefault("max_seq_length", 300)
        return SleepFMv2Backend(weights_path=weights_path, **kwargs)

    if name == "reve":
        return REVEBackend(weights_path=weights_path, **kwargs)

    if name in ("labram", "labram_base", "labram_finetuned"):
        return LaBraMBackend(weights_path=weights_path, **kwargs)

    _all = sorted(_SLEEPFM_V2_NAMES | _SLEEPFM_V2_128P_NAMES)
    raise ValueError(
        f"Unknown encoder '{name}'. "
        f"Supported values: 'sleepfm', "
        + ", ".join(f"'{n}'" for n in _all)
        + ", 'reve', 'labram'."
    )
