"""
Explain AE  (SpectralDecoder) — Spectral decoder for EEG token interpretability
=====================================================================

The SetTransformer tokeniser compresses 1-second EEG patches into 128-dim
embedding vectors.  These tokens are opaque — unlike text, we cannot simply
"read" them.  The SpectralDecoder learns to reconstruct interpretable Fourier components
(amplitude + phase of narrow frequency bands) from those token embeddings,
giving us a generative spectral decoder we can point at any direction in
embedding space (e.g. SAE feature directions) and ask: "what does this
sound like in frequency-domain?"

For time-domain visualisation, the SpectralDecoder uses **phase-agnostic waveform
reconstruction**: per-band waveforms are synthesised from the predicted
spectral amplitudes via zero-phase iFFT.  This produces symmetric
"characteristic waveforms" that correctly show frequency content and
relative band power without predicting phase.

Diagnostic analysis showed that phase is NOT encoded in the SetTransformer
embeddings (nearest-neighbour phase consistency ≈ 0.05, circular variance
≈ 0.90).  Amplitude prediction is excellent (R² ≈ 0.80), but phase
prediction is theoretically impossible from these embeddings.  Zero-phase
reconstruction is therefore the optimal approach for visualisation.

Key design choices
------------------
* **Phase representation**: we use (cos φ, sin φ) per band to avoid the
  2π wrap-around discontinuity.  This doubles the phase dimensions but
  makes the regression well-behaved.

* **Loss function**: amplitude MSE + amplitude-weighted phase cosine loss
  + optional embedding round-trip loss.
  Phase is meaningless when amplitude ≈ 0, so we down-weight it.

* **Phase-agnostic waveforms**: ``spectral_to_waveforms()`` reconstructs
  per-band time-domain waveforms from predicted amplitudes via zero-phase
  iFFT — deterministic, symmetric "characteristic waveforms" that show
  what each feature boosts without depending on unpredictable phase.

* **No codebook**: this is a plain deterministic autoencoder — proximity in
  embedding space maps smoothly to proximity in spectral/temporal space.

* **Channel-averaged**: because the SetTransformer spatially pools across
  channels before the temporal transformer, each token already represents a
  channel-averaged view.  The SpectralDecoder reconstructs the *spatially-pooled*
  spectral signature and waveforms.

Pipeline
--------
  raw EEG patch  (19ch × 128 samples)
       ↓  FFT (per channel, then average across channels)
  spectral target  (n_bins amplitudes + n_bins×2 phase components)
       ↑
  SpectralDecoder decoder  ←  128-dim token embedding  (from SetTransformer)
       ↓  zero-phase iFFT (at display-time, not trained)
  band waveforms  (6 bands × 128 samples, phase-agnostic)

Usage
-----
  from spectral_decoder import SpectralTargetExtractor, SpectralDecoder, SpectralDecoderTrainer
  
  # 1. Build spectral targets from raw EEG
  extractor = SpectralTargetExtractor(fs=128, n_fft=128)
  targets = extractor(raw_eeg_patches)   # (N, n_bins * 3)
  
  # 2. Train SpectralDecoder on (token_embedding, spectral_target) pairs
  trainer = SpectralDecoderTrainer(embed_dim=128, n_bins=extractor.n_bins)
  trainer.train(model, dataloader)
  
  # 3. Decode any direction in embedding space
  spectral_decoder = trainer.spectral_decoder
  spectrum = spectral_decoder.decode(sae_feature_direction)    # interpretable!
  waveforms = trainer.spectral_to_waveforms(amp)  # zero-phase per-band waveforms
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mecheeg.sae import ActivationExtractor


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Spectral Target Extractor
# ─────────────────────────────────────────────────────────────────────────────

# Standard clinical EEG frequency bands for labelling / grouping
CLINICAL_BANDS = {
    "delta":       (0.5,  4.0),
    "theta":       (4.0,  8.0),
    "alpha":       (8.0, 13.0),
    "low-beta":   (13.0, 20.0),
    "high-beta":  (20.0, 30.0),
    "gamma":      (30.0, 45.0),
}


class SpectralTargetExtractor:
    """
    Convert raw EEG patches into spectral targets for SpectralDecoder training.

    For each 1-second EEG patch (C channels × T samples):
      1. Compute FFT per channel
      2. Average amplitude and phase across channels (matching the
         spatial pooling that the SetTransformer already did)
      3. Return amplitude + (cos φ, sin φ) per frequency bin

    Parameters
    ----------
    fs      : sampling rate in Hz
    n_fft   : FFT length (defaults to patch length, i.e. fs for 1 s patches)
    f_min   : minimum frequency to keep (Hz), default 0.5
    f_max   : maximum frequency to keep (Hz), default None → Nyquist
    """

    def __init__(
        self,
        fs: int = 128,
        n_fft: Optional[int] = None,
        f_min: float = 0.5,
        f_max: Optional[float] = None,
    ):
        self.fs = fs
        self.n_fft = n_fft or fs  # 1-second window → n_fft = fs
        self.f_min = f_min
        self.f_max = f_max or (fs / 2.0)

        # Compute the frequency axis once
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / fs)
        self._freq_mask = (freqs >= f_min) & (freqs <= self.f_max)
        self.freqs = freqs[self._freq_mask]
        self.n_bins = len(self.freqs)

        # Map each bin to a clinical band name
        self.bin_labels = []
        for f in self.freqs:
            label = "other"
            for name, (lo, hi) in CLINICAL_BANDS.items():
                if lo <= f < hi:
                    label = name
                    break
            self.bin_labels.append(label)

    @property
    def target_dim(self) -> int:
        """Total output dimensionality: amplitude + cos(phase) + sin(phase)."""
        return self.n_bins * 3

    def __call__(self, eeg_patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_patches: (N, C, T) raw EEG patches — one patch per token

        Returns:
            targets: (N, n_bins * 3) — [amplitude | cos(phase) | sin(phase)]
        """
        return self.extract(eeg_patches)

    def extract(self, eeg_patches: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral targets for a batch of EEG patches.

        Args:
            eeg_patches: (N, C, T) raw EEG — N patches, C channels, T samples

        Returns:
            targets: (N, n_bins * 3) float32 tensor on same device as input
        """
        device = eeg_patches.device
        N, C, T = eeg_patches.shape

        # Pad or truncate to n_fft
        if T < self.n_fft:
            eeg_patches = F.pad(eeg_patches, (0, self.n_fft - T))
        elif T > self.n_fft:
            eeg_patches = eeg_patches[..., :self.n_fft]

        # FFT per channel: (N, C, n_fft//2 + 1) complex
        spectrum = torch.fft.rfft(eeg_patches, n=self.n_fft, dim=-1)

        # Keep only the frequency bins we care about
        freq_mask = torch.tensor(self._freq_mask, device=device)
        spectrum = spectrum[:, :, freq_mask]   # (N, C, n_bins)

        # Amplitude and phase (per channel)
        amplitude = spectrum.abs()              # (N, C, n_bins)
        phase = spectrum.angle()                # (N, C, n_bins)

        # Average across channels (matches SetTransformer spatial pooling)
        amp_mean = amplitude.mean(dim=1)        # (N, n_bins)
        # For phase: average the unit-vector representation to handle wrapping
        cos_phase = phase.cos().mean(dim=1)     # (N, n_bins)
        sin_phase = phase.sin().mean(dim=1)     # (N, n_bins)

        # Re-normalise (cos, sin) to unit circle
        norm = (cos_phase ** 2 + sin_phase ** 2).sqrt().clamp(min=1e-8)
        cos_phase = cos_phase / norm
        sin_phase = sin_phase / norm

        # Log-scale amplitude for better dynamic range
        amp_log = torch.log1p(amp_mean)

        # Concatenate: [amplitude | cos(phase) | sin(phase)]
        targets = torch.cat([amp_log, cos_phase, sin_phase], dim=-1)  # (N, 3*n_bins)
        return targets.float()

    def unpack_targets(
        self, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split a target / prediction tensor back into components.

        Args:
            targets: (..., n_bins * 3)

        Returns:
            amplitude:  (..., n_bins)  — log1p-scaled
            cos_phase:  (..., n_bins)
            sin_phase:  (..., n_bins)
        """
        nb = self.n_bins
        amp = targets[..., :nb]
        cos_p = targets[..., nb : 2 * nb]
        sin_p = targets[..., 2 * nb : 3 * nb]
        return amp, cos_p, sin_p

    def amplitude_linear(self, log_amp: torch.Tensor) -> torch.Tensor:
        """Convert log1p amplitude back to linear scale."""
        return torch.expm1(log_amp)

    def phase_angle(
        self, cos_p: torch.Tensor, sin_p: torch.Tensor
    ) -> torch.Tensor:
        """Recover phase angle in radians from (cos, sin) representation."""
        return torch.atan2(sin_p, cos_p)

    def get_band_mask(self, band_name: str) -> np.ndarray:
        """Boolean mask over n_bins for a clinical band (e.g. 'alpha')."""
        return np.array([lbl == band_name for lbl in self.bin_labels])

    def get_band_power(
        self, amplitude: torch.Tensor, band_name: str
    ) -> torch.Tensor:
        """
        Average amplitude within a clinical band.

        Args:
            amplitude: (..., n_bins) — log1p scaled
            band_name: one of CLINICAL_BANDS keys

        Returns:
            (...,) mean log-amplitude in that band
        """
        mask = torch.tensor(
            self.get_band_mask(band_name),
            dtype=torch.bool,
            device=amplitude.device,
        )
        if mask.sum() == 0:
            return torch.zeros(amplitude.shape[:-1], device=amplitude.device)
        return amplitude[..., mask].mean(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 1b.  Temporal Target Extractor (FFT band-pass filtered waveforms)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalTargetExtractor:
    """
    Extract band-pass filtered, channel-averaged time-domain targets.

    For each of the 6 clinical EEG bands the extractor isolates the
    frequency content via FFT masking (brick-wall bandpass), inverse-FFTs
    back to the time domain, and averages across channels (to match the
    SetTransformer's spatial pooling).

    **Implementation**: uses ``torch.fft.rfft`` / ``irfft`` with boolean
    frequency masks — runs entirely on GPU, ~200× faster than the
    equivalent scipy ``sosfiltfilt`` approach and produces exact band
    isolation with zero phase distortion.

    The output is a flat vector of length ``n_bands × patch_size``
    (by default 6 × 128 = 768).  This gives full 128 Hz resolution so
    no high-frequency detail is lost.

    Parameters
    ----------
    fs         : sampling rate (Hz)
    patch_size : number of samples in one token (default: 128 = 1 s)
    """

    # Band names in a fixed canonical order (matches CLINICAL_BANDS dict)
    BAND_NAMES: List[str] = list(CLINICAL_BANDS.keys())

    def __init__(
        self,
        fs: int = 128,
        patch_size: int = 128,
    ):
        self.fs = fs
        self.patch_size = patch_size
        self.n_bands = len(self.BAND_NAMES)
        self.target_dim = self.n_bands * patch_size     # 6 × 128 = 768

        # Pre-compute FFT frequency bins and boolean masks
        n_freq = patch_size // 2 + 1                    # rfft output length
        freqs = torch.fft.rfftfreq(patch_size, d=1.0 / fs)  # (n_freq,)

        # Build stacked float mask: (n_bands, n_freq) — kept on CPU,
        # moved to device lazily in extract()
        masks = []
        for band_name in self.BAND_NAMES:
            f_lo, f_hi = CLINICAL_BANDS[band_name]
            masks.append(((freqs >= f_lo) & (freqs <= f_hi)).float())
        self._mask_cpu = torch.stack(masks)             # (n_bands, n_freq)
        self._mask_gpu: Optional[torch.Tensor] = None   # cached on first use

    # ------------------------------------------------------------------
    def _get_mask(self, device: torch.device) -> torch.Tensor:
        """Return the band mask on *device*, caching the transfer."""
        if self._mask_gpu is None or self._mask_gpu.device != device:
            self._mask_gpu = self._mask_cpu.to(device)
        return self._mask_gpu

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract(self, patches: torch.Tensor) -> torch.Tensor:
        """
        FFT band-pass filter raw EEG patches and channel-average.

        Args:
            patches: (N, C, T) — N patches, C channels, T samples

        Returns:
            targets: (N, n_bands × T) — concatenated band waveforms
                     Layout: [delta_t0..delta_tT | theta_t0..theta_tT | ...]
        """
        N, C, T = patches.shape
        mask = self._get_mask(patches.device)           # (n_bands, n_freq)

        # FFT along time axis: (N, C, T) → (N, C, n_freq)
        X = torch.fft.rfft(patches, dim=-1)

        # Apply each band mask via broadcasting:
        #   X:    (N, C, 1,       n_freq)
        #   mask: (1, 1, n_bands, n_freq)
        # → X_bands: (N, C, n_bands, n_freq)
        X_bands = X.unsqueeze(2) * mask.unsqueeze(0).unsqueeze(0)

        # IFFT back to time domain: (N, C, n_bands, T)
        x_bands = torch.fft.irfft(X_bands, n=T, dim=-1)

        # Average over channels → (N, n_bands, T) → flatten → (N, n_bands*T)
        x_avg = x_bands.mean(dim=1)                    # (N, n_bands, T)
        return x_avg.reshape(N, -1)                     # (N, 768)

    # ------------------------------------------------------------------
    def unpack_targets(
        self, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Unpack flat target vector into per-band waveforms.

        Args:
            targets: (..., n_bands * T)

        Returns:
            dict mapping band_name → (..., T) waveform tensor
        """
        T = self.patch_size
        result = {}
        for b, band_name in enumerate(self.BAND_NAMES):
            result[band_name] = targets[..., b * T : (b + 1) * T]
        return result

    # ------------------------------------------------------------------
    def pack_targets(self, band_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Inverse of unpack_targets: dict of band waveforms → flat tensor."""
        parts = [band_dict[name] for name in self.BAND_NAMES]
        return torch.cat(parts, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 1b'.  Broadband Per-Channel Temporal Target Extractor (v4)
# ─────────────────────────────────────────────────────────────────────────────

class BroadbandTemporalTargetExtractor:
    """
    Drop both band-decomposition and channel-averaging from the temporal
    target. The "v4" target is the raw 1-second waveform per channel — a
    flat vector of length ``n_channels × patch_size``.

    Trains the temporal head to predict broadband per-channel EEG directly
    from the embedding. Removes the two information-loss stages of
    ``TemporalTargetExtractor`` (band-pass + channel-mean) so the head can
    in principle synthesise broadband transients (spikes) on individual
    channels.

    Outputs are packed as ``[ch0_t0..ch0_T | ch1_t0..ch1_T | …]`` so the
    layout mirrors the band-decomposed extractor's per-band layout.
    """

    def __init__(self, fs: int = 128, patch_size: int = 128,
                 n_channels: int = 19):
        self.fs = fs
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_bands = n_channels  # alias used by SpectralDecoder config
        self.target_dim = n_channels * patch_size
        self.BAND_NAMES: List[str] = [f"ch{i:02d}" for i in range(n_channels)]

    @torch.no_grad()
    def extract(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (N, C, T) — if C > n_channels (e.g. dataset has 27
                     channels but encoder uses the first 19), slice the
                     first ``self.n_channels`` to match the encoder's input.
        Returns:
            (N, n_channels * T) flat tensor (broadband, per-channel)
        """
        N, C, T = patches.shape
        if C > self.n_channels:
            patches = patches[:, :self.n_channels, :]
        elif C < self.n_channels:
            raise AssertionError(
                f"BroadbandTemporalTargetExtractor n_channels="
                f"{self.n_channels} but input has only C={C} channels")
        return patches.reshape(N, self.n_channels * T)

    def unpack_targets(self, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Unpack flat (..., C*T) → dict ch_name → (..., T)."""
        T = self.patch_size
        out = {}
        for c, name in enumerate(self.BAND_NAMES):
            out[name] = targets[..., c * T:(c + 1) * T]
        return out

    def pack_targets(self, ch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [ch_dict[n] for n in self.BAND_NAMES]
        return torch.cat(parts, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 1c.  Morphology Target Extractor (transient / waveform / spatial features)
# ─────────────────────────────────────────────────────────────────────────────

class MorphologyTargetExtractor:
    """
    Extract per-token morphology features that complement spectral amplitude.

    These targets are designed to capture EEG patterns that pure FFT-amplitude
    misses — sharpness/transients (line length, NEO, kurtosis, ricker
    response), waveform morphology (slow-wave envelope, Hjorth complexity),
    and spatial topology (per-channel max amplitude).

    Per-token feature layout (target_dim = 8 + n_channels):
      [0] line_length        Σ|Δx| (channel-mean)            — sharpness
      [1] neo_max            max of x² − x_-1·x_+1           — sharpness
      [2] peak_amp           channel-mean of max|x|          — amplitude
      [3] delta_env          mean |1–4 Hz bandpass|          — slow wave
      [4] beta_env           mean |13–30 Hz bandpass|        — fast activity
      [5] kurtosis           excess kurtosis                  — leptokurticity
      [6] hjorth_complexity  std(d²x)/std(dx) ÷ Hjorth mob    — waveform shape
      [7] ricker_response    max |Ricker conv| at ~50 ms     — spike detector
      [8…8+n_channels-1]     per-channel max|x|              — spatial map

    Inputs and outputs use the same standardisation as the spectral
    extractor — i.e. the patches are whatever the encoder sees (already
    z-scored for SleepFM).  Targets are downstream-normalised by the
    SpectralDecoderTrainer (mean/std per-feature), so absolute units don't matter.

    Parameters
    ----------
    fs          : sampling rate (Hz)
    patch_size  : samples per token (default 128 = 1 s at 128 Hz)
    n_channels  : number of channels to use for per-channel and channel-pooled
                  features (default 19 — the 10-20 EEG channels, ignoring the
                  trailing eye/EMG/EKG channels in 27-channel V3/V4 data).
    """

    FEATURE_NAMES_SCALAR: List[str] = [
        "line_length",
        "neo_max",
        "peak_amp",
        "delta_env",
        "beta_env",
        "kurtosis",
        "hjorth_complexity",
        "ricker_response",
    ]

    def __init__(
        self,
        fs: int = 128,
        patch_size: int = 128,
        n_channels: int = 19,
    ):
        self.fs = fs
        self.patch_size = patch_size
        self.n_channels = n_channels

        self.feature_names = list(self.FEATURE_NAMES_SCALAR) + [
            f"peak_amp_ch{i}" for i in range(n_channels)
        ]
        self.target_dim = len(self.feature_names)

        # Precompute frequency masks for δ (1–4 Hz) and β (13–30 Hz)
        n_freq = patch_size // 2 + 1
        freqs = torch.fft.rfftfreq(patch_size, d=1.0 / fs)
        self._delta_mask = ((freqs >= 1.0) & (freqs <= 4.0)).float()
        self._beta_mask = ((freqs >= 13.0) & (freqs <= 30.0)).float()

        # Ricker (Mexican-hat) wavelet at spike scale (~50 ms FWHM ≈ σ=6 samples
        # at 128 Hz).  Length ≈ 6σ + 1.
        sigma = 0.025 * fs / 2          # 0.025 s scale → 6.4 samples / 2
        ricker_len = max(int(6 * sigma) | 1, 9)   # odd
        t = torch.arange(ricker_len, dtype=torch.float32) - ricker_len // 2
        a = (1.0 - (t / sigma) ** 2) * torch.exp(-(t / sigma) ** 2 / 2)
        a = a / a.abs().max().clamp(min=1e-8)
        self._ricker = a.view(1, 1, -1)            # (1, 1, K)
        self._ricker_pad = ricker_len // 2

        # Cached GPU copies
        self._delta_mask_dev: Optional[torch.Tensor] = None
        self._beta_mask_dev: Optional[torch.Tensor] = None
        self._ricker_dev: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def _to_device(self, device: torch.device) -> None:
        if self._delta_mask_dev is None or self._delta_mask_dev.device != device:
            self._delta_mask_dev = self._delta_mask.to(device)
            self._beta_mask_dev = self._beta_mask.to(device)
            self._ricker_dev = self._ricker.to(device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Compute morphology targets for a batch of EEG patches.

        Args:
            patches: (N, C, T) — N patches, C channels, T samples
        Returns:
            targets: (N, target_dim)
        """
        device = patches.device
        self._to_device(device)
        N, C, T = patches.shape
        ch_use = min(self.n_channels, C)
        x = patches[:, :ch_use, :]                 # (N, ch_use, T)

        # 1. Line length — per-channel sum of |Δx|, then mean
        diff = x[:, :, 1:] - x[:, :, :-1]
        line_length = diff.abs().sum(dim=-1).mean(dim=-1)         # (N,)

        # 2. NEO max — Teager-Kaiser energy peak per channel, then mean
        neo = x[:, :, 1:-1] ** 2 - x[:, :, :-2] * x[:, :, 2:]
        neo_max = neo.amax(dim=-1).mean(dim=-1)                   # (N,)

        # 3. Peak amplitude (per channel and pooled)
        peak_per_ch = x.abs().amax(dim=-1)                        # (N, ch_use)
        peak_amp = peak_per_ch.mean(dim=-1)                       # (N,)

        # 4. & 5. Channel-averaged signal for envelope / kurtosis / Hjorth
        x_avg = x.mean(dim=1)                                     # (N, T)
        X = torch.fft.rfft(x_avg, dim=-1)
        x_delta = torch.fft.irfft(X * self._delta_mask_dev, n=T, dim=-1)
        x_beta = torch.fft.irfft(X * self._beta_mask_dev, n=T, dim=-1)
        delta_env = x_delta.abs().mean(dim=-1)                    # (N,)
        beta_env = x_beta.abs().mean(dim=-1)                      # (N,)

        # 6. Kurtosis (excess) of channel-averaged signal
        mu = x_avg.mean(dim=-1, keepdim=True)
        std = x_avg.std(dim=-1, keepdim=True).clamp(min=1e-8)
        z = (x_avg - mu) / std
        kurtosis = (z ** 4).mean(dim=-1) - 3.0                    # (N,)

        # 7. Hjorth complexity = (std(d²x)/std(dx)) / (std(dx)/std(x))
        d1 = x_avg[:, 1:] - x_avg[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        var_x = x_avg.var(dim=-1).clamp(min=1e-8)
        var_d1 = d1.var(dim=-1).clamp(min=1e-8)
        var_d2 = d2.var(dim=-1).clamp(min=1e-8)
        mobility = (var_d1 / var_x).sqrt()
        complexity = ((var_d2 / var_d1).sqrt()
                      / mobility.clamp(min=1e-8))                 # (N,)

        # 8. Ricker (Mexican-hat) response at spike scale: max |conv|
        #    Apply per-channel then take the max across channels — focal spikes
        #    only fire on a few channels and channel-averaging would dilute them.
        ricker = self._ricker_dev                                 # (1, 1, K)
        # x: (N, ch_use, T) — flatten (N*ch_use, 1, T) for conv1d
        flat = x.reshape(N * ch_use, 1, T)
        resp = F.conv1d(flat, ricker, padding=self._ricker_pad).abs()
        resp = resp.view(N, ch_use, -1).amax(dim=-1)              # (N, ch_use)
        ricker_response = resp.amax(dim=-1)                       # (N,) — max across channels

        scalars = torch.stack([
            line_length, neo_max, peak_amp,
            delta_env, beta_env, kurtosis, complexity, ricker_response,
        ], dim=-1)                                                # (N, 8)

        targets = torch.cat([scalars, peak_per_ch], dim=-1)       # (N, 8 + ch_use)
        return targets.float()

    # ------------------------------------------------------------------
    def unpack_targets(self, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split a flat morphology vector into a name→tensor dict."""
        out: Dict[str, torch.Tensor] = {}
        for i, name in enumerate(self.feature_names):
            out[name] = targets[..., i]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SpectralDecoder Model
# ─────────────────────────────────────────────────────────────────────────────

class _ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Linear → add."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class SpectralDecoder(nn.Module):
    """
    Autoencoder that maps token embeddings ↔ spectral representations,
    with an optional temporal decoder head for band-pass filtered waveforms.

    The *decoder* is the star: given any 128-dim direction in embedding space,
    it produces an interpretable spectral signature (amplitude + phase per
    frequency bin).  When the temporal head is enabled, it also produces
    per-band time-domain waveforms (6 bands × 128 samples = 768 dims).

    The encoder exists only for verification — it lets us check that the
    spectral representation retains enough information to reconstruct the
    embedding.  During interpretation we only use the decoder.

    Architecture (v2 — residual, dual-head)
    ----------------------------------------
    Decoder trunk:  embed_dim → project_up → N residual blocks
                         ├─ dec_project_out_spectral → target_dim  (192)
                         └─ dec_project_out_temporal → temporal_dim (768) [optional]
    Encoder:        target_dim → project_up → N residual blocks → embed_dim

    Parameters
    ----------
    embed_dim      : transformer embedding dimensionality (128)
    target_dim     : spectral target dimensionality (n_bins * 3)
    temporal_dim   : temporal target dimensionality (n_bands * patch_size),
                     or None to disable the temporal head
    hidden_dim     : hidden layer width (default: 4× embed_dim)
    n_blocks       : number of residual blocks in each direction (default: 3)
    n_bins         : number of frequency bins (for loss computation)
    n_bands        : number of clinical bands (default: 6)
    patch_size     : samples per token (default: 128)
    dropout        : dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 128,
        target_dim: int = 189,  # 63 bins × 3
        temporal_dim: Optional[int] = None,
        morphology_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_bins: Optional[int] = None,
        n_bands: int = 6,
        patch_size: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.target_dim = target_dim
        self.temporal_dim = temporal_dim       # None → no temporal head
        self.morphology_dim = morphology_dim   # None → no morphology head
        self.hidden_dim = hidden_dim or embed_dim * 4
        self.n_bins = n_bins or (target_dim // 3)
        self.n_bands = n_bands
        self.patch_size = patch_size
        self.n_blocks = n_blocks
        self.dropout = dropout

        # ── Decoder trunk: embedding → shared hidden ─────────────────────
        self.dec_project_in = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.dec_blocks = nn.Sequential(
            *[_ResidualBlock(self.hidden_dim, dropout) for _ in range(n_blocks)]
        )

        # ── Spectral head (always present) ───────────────────────────────
        self.dec_project_out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, target_dim),
        )

        # ── Temporal head (optional) ─────────────────────────────────────
        #   Dedicated residual blocks + projection — the temporal task is
        #   harder than spectral (768 vs 192 dims, waveform shape matters)
        #   so it gets its own capacity beyond the shared trunk.
        if temporal_dim is not None:
            self.dec_temporal_blocks = nn.Sequential(
                *[_ResidualBlock(self.hidden_dim, dropout) for _ in range(2)]
            )
            self.dec_temporal_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, temporal_dim),
            )
        else:
            self.dec_temporal_blocks = None
            self.dec_temporal_head = None

        # ── Morphology head (optional) ───────────────────────────────────
        #   Outputs a small set of waveform / sharpness / spatial scalars
        #   designed to capture spike-and-wave morphology that pure FFT
        #   amplitude misses.  Smaller output dim → only one extra residual
        #   block beyond the shared trunk.
        if morphology_dim is not None:
            self.dec_morph_blocks = _ResidualBlock(self.hidden_dim, dropout)
            self.dec_morph_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim // 2, morphology_dim),
            )
        else:
            self.dec_morph_blocks = None
            self.dec_morph_head = None

        # ── Encoder: spectrum → embedding (for verification) ────────────
        self.enc_project_in = nn.Sequential(
            nn.Linear(target_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.enc_blocks = nn.Sequential(
            *[_ResidualBlock(self.hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.enc_project_out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _decode_trunk(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Shared decoder trunk: embedding → hidden representation."""
        h = self.dec_project_in(embeddings)
        h = self.dec_blocks(h)
        return h

    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Map token embeddings to spectral targets.

        Args:
            embeddings: (..., embed_dim)
        Returns:
            spectral: (..., target_dim)  — [amp | cos_phase | sin_phase]
        """
        h = self._decode_trunk(embeddings)
        return self.dec_project_out(h)

    def decode_temporal(self, embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Map token embeddings to band-pass filtered waveforms.

        Args:
            embeddings: (..., embed_dim)
        Returns:
            temporal: (..., temporal_dim) or None if temporal head is disabled
        """
        if self.dec_temporal_head is None:
            return None
        h = self._decode_trunk(embeddings)
        h = self.dec_temporal_blocks(h)
        return self.dec_temporal_head(h)

    def decode_morphology(self, embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Map token embeddings to morphology scalars / per-channel features.

        Args:
            embeddings: (..., embed_dim)
        Returns:
            morphology: (..., morphology_dim) or None if morphology head is disabled
        """
        if self.dec_morph_head is None:
            return None
        h = self._decode_trunk(embeddings)
        h = self.dec_morph_blocks(h)
        return self.dec_morph_head(h)

    def decode_both(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Decode spectral, temporal, and morphology targets in a single trunk pass.

        Returns:
            spectral_pred:    (..., target_dim)
            temporal_pred:    (..., temporal_dim) or None
            morphology_pred:  (..., morphology_dim) or None
        """
        h = self._decode_trunk(embeddings)
        spectral = self.dec_project_out(h)
        temporal = None
        if self.dec_temporal_head is not None:
            h_t = self.dec_temporal_blocks(h)
            temporal = self.dec_temporal_head(h_t)
        morphology = None
        if self.dec_morph_head is not None:
            h_m = self.dec_morph_blocks(h)
            morphology = self.dec_morph_head(h_m)
        return spectral, temporal, morphology

    def encode(self, spectral: torch.Tensor) -> torch.Tensor:
        """
        Map spectral targets back to embedding space (verification).

        Args:
            spectral: (..., target_dim)
        Returns:
            embeddings: (..., embed_dim)
        """
        h = self.enc_project_in(spectral)
        h = self.enc_blocks(h)
        return self.enc_project_out(h)

    def forward(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Full forward: embedding → spectrum (+ temporal, + morphology) → embedding_hat

        Returns:
            spectral_pred:    (..., target_dim)
            embed_recon:      (..., embed_dim)
            temporal_pred:    (..., temporal_dim) or None
            morphology_pred:  (..., morphology_dim) or None
        """
        spectral_pred, temporal_pred, morphology_pred = self.decode_both(embeddings)
        embed_recon = self.encode(spectral_pred)
        return spectral_pred, embed_recon, temporal_pred, morphology_pred

    def spectral_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        phase_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Combined loss for spectral reconstruction.

        Components:
          1. Amplitude MSE — reconstruction of log-amplitude spectrum
          2. Phase cosine loss — weighted by target amplitude (phase is
             meaningless when amplitude ≈ 0)

        Args:
            pred:   (B, n_bins * 3) — predicted [amp | cos | sin]
            target: (B, n_bins * 3) — ground truth [amp | cos | sin]
            phase_weight: relative weight of phase loss vs amplitude loss

        Returns:
            total_loss, {amp_loss, phase_loss}
        """
        nb = self.n_bins

        # Unpack
        pred_amp = pred[..., :nb]
        pred_cos = pred[..., nb : 2 * nb]
        pred_sin = pred[..., 2 * nb :]

        tgt_amp = target[..., :nb]
        tgt_cos = target[..., nb : 2 * nb]
        tgt_sin = target[..., 2 * nb :]

        # 1. Amplitude loss: plain MSE on log-amplitude
        amp_loss = F.mse_loss(pred_amp, tgt_amp)

        # 2. Phase loss: cosine distance, weighted by target amplitude
        #    cos_distance = 1 - cos(angle_between) = 1 - (cos₁cos₂ + sin₁sin₂)
        cos_sim = pred_cos * tgt_cos + pred_sin * tgt_sin  # (B, n_bins)
        cos_sim = cos_sim.clamp(-1, 1)
        phase_dist = 1.0 - cos_sim                          # (B, n_bins)

        # Weight by target amplitude — high-amplitude bins matter more.
        # Use softmax so weights are always non-negative and sum to 1,
        # even when tgt_amp has been zero-mean normalised (which makes
        # raw values negative and would otherwise flip the loss sign).
        amp_weights = F.softmax(tgt_amp.detach(), dim=-1)
        phase_loss = (phase_dist * amp_weights).sum(dim=-1).mean()

        total = amp_loss + phase_weight * phase_loss

        return total, {"amp_loss": amp_loss, "phase_loss": phase_loss}

    def temporal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        corr_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Loss for time-domain band-pass filtered waveform reconstruction.

        Components:
          1. Per-sample MSE    — point-wise accuracy
          2. 1 − Pearson corr  — waveform shape preservation (per-band)

        The Pearson term is the primary driver of temporal quality — it
        focuses learning on the waveform *shape* (spike morphology, etc.)
        irrespective of amplitude scale.

        Args:
            pred:   (B, n_bands * T) — predicted band waveforms
            target: (B, n_bands * T) — ground truth band waveforms
            corr_weight: relative weight of correlation loss (default 1.0)

        Returns:
            total_loss, {temporal_mse, temporal_corr}
        """
        mse = F.mse_loss(pred, target)

        # Per-band Pearson correlation, averaged over bands and samples
        T = self.patch_size
        corr_sum = 0.0
        n_valid = 0
        for b in range(self.n_bands):
            p = pred[..., b * T : (b + 1) * T]     # (B, T)
            t = target[..., b * T : (b + 1) * T]   # (B, T)
            p_z = p - p.mean(dim=-1, keepdim=True)
            t_z = t - t.mean(dim=-1, keepdim=True)
            num = (p_z * t_z).sum(dim=-1)                  # (B,)
            den = (p_z.norm(dim=-1) * t_z.norm(dim=-1)).clamp(min=1e-8)
            corr = (num / den).mean()                       # scalar
            if not torch.isnan(corr):
                corr_sum += corr
                n_valid += 1

        avg_corr = corr_sum / max(n_valid, 1)
        corr_loss = 1.0 - avg_corr                           # ∈ [0, 2]

        total = mse + corr_weight * corr_loss

        return total, {
            "temporal_mse": mse,
            "temporal_corr": avg_corr,     # for logging (higher=better)
        }

    def morphology_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Loss for morphology head — plain MSE over normalised features.

        Targets are normalised externally (per-feature mean/std) before
        being passed in, so a flat MSE weights all features equally.

        Args:
            pred:    (B, morphology_dim)
            target:  (B, morphology_dim)
        Returns:
            total_loss, {morph_mse}
        """
        mse = F.mse_loss(pred, target)
        return mse, {"morph_mse": mse}

    def full_loss(
        self,
        embeddings: torch.Tensor,
        spectral_targets: torch.Tensor,
        phase_weight: float = 0.5,
        recon_weight: float = 0.1,
        temporal_targets: Optional[torch.Tensor] = None,
        temporal_weight: float = 0.5,
        temporal_corr_weight: float = 0.3,
        morphology_targets: Optional[torch.Tensor] = None,
        morphology_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full training loss: spectral + optional temporal + embedding round-trip.

        Args:
            embeddings:           (B, embed_dim)     — token embeddings
            spectral_targets:     (B, target_dim)    — ground-truth spectra
            phase_weight:         weight for phase loss
            recon_weight:         weight for embedding reconstruction loss
            temporal_targets:     (B, temporal_dim)   — ground-truth band waveforms (optional)
            temporal_weight:      weight for temporal loss relative to spectral
            temporal_corr_weight: weight for correlation term within temporal loss

        Returns:
            total_loss, dict of component losses
        """
        spectral_pred, embed_recon, temporal_pred, morphology_pred = self(embeddings)

        spec_loss, spec_details = self.spectral_loss(
            spectral_pred, spectral_targets, phase_weight
        )

        embed_loss = F.mse_loss(embed_recon, embeddings)

        total = spec_loss + recon_weight * embed_loss

        details = {
            **spec_details,
            "embed_recon_loss": embed_loss,
        }

        # ── Temporal loss (if both targets and head are present) ─────────
        if temporal_targets is not None and temporal_pred is not None:
            temp_loss, temp_details = self.temporal_loss(
                temporal_pred, temporal_targets, temporal_corr_weight
            )
            total = total + temporal_weight * temp_loss
            details.update(temp_details)
            details["temporal_loss"] = temp_loss

        # ── Morphology loss (if both targets and head are present) ───────
        if morphology_targets is not None and morphology_pred is not None:
            morph_loss, morph_details = self.morphology_loss(
                morphology_pred, morphology_targets
            )
            total = total + morphology_weight * morph_loss
            details.update(morph_details)
            details["morphology_loss"] = morph_loss

        details["total"] = total
        return total, details


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SpectralDecoder Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SpectralDecoderTrainer:
    """
    Trains the SpectralDecoder by:
      1. Freezing the SetTransformer
      2. Running EEG data through it to extract (token_embedding, raw_patch) pairs
      3. Computing FFT spectral targets from the raw patches
      4. Optionally computing band-pass filtered temporal targets
      5. Training the SpectralDecoder on (embedding, spectral_target [, temporal_target]) pairs

    Parameters
    ----------
    embed_dim        : transformer embed_dim (128)
    fs               : EEG sampling rate
    n_fft            : FFT length
    f_min, f_max     : frequency range to keep
    hidden_dim       : SpectralDecoder hidden layer width
    lr               : learning rate
    batch_size       : training batch size
    epochs           : number of training epochs
    phase_weight     : weight of phase loss relative to amplitude
    recon_weight     : weight of embedding round-trip loss
    temporal         : if True, enable the temporal decoder head
    temporal_weight  : weight of temporal loss relative to spectral
    temporal_corr_weight : weight of Pearson correlation term in temporal loss
    device           : 'cuda' / 'cpu'
    """

    def __init__(
        self,
        embed_dim: int = 128,
        fs: int = 128,
        n_fft: Optional[int] = None,
        f_min: float = 0.5,
        f_max: Optional[float] = None,
        hidden_dim: Optional[int] = None,
        n_blocks: int = 3,
        lr: float = 1e-3,
        batch_size: int = 2048,
        epochs: int = 30,
        phase_weight: float = 0.5,
        recon_weight: float = 0.1,
        temporal: bool = False,
        temporal_weight: float = 0.5,
        temporal_corr_weight: float = 0.3,
        temporal_mode: str = "bands",   # "bands" or "broadband_perchannel"
        temporal_n_channels: int = 19,
        morphology: bool = False,
        morphology_weight: float = 0.5,
        morphology_n_channels: int = 19,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.embed_dim = embed_dim
        self.fs = fs
        self.phase_weight = phase_weight
        self.recon_weight = recon_weight
        self.temporal_enabled = temporal
        self.temporal_weight = temporal_weight
        self.temporal_corr_weight = temporal_corr_weight
        self.morphology_enabled = morphology
        self.morphology_weight = morphology_weight
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        # Build spectral extractor
        self.spectral = SpectralTargetExtractor(
            fs=fs, n_fft=n_fft, f_min=f_min, f_max=f_max
        )

        # Build temporal extractor (if enabled)
        patch_size = n_fft or fs   # default: 1 s at fs Hz
        self.temporal_extractor = None
        self.temporal_mode = temporal_mode
        temporal_dim: Optional[int] = None
        if temporal:
            if temporal_mode == "bands":
                self.temporal_extractor = TemporalTargetExtractor(
                    fs=fs, patch_size=patch_size
                )
            elif temporal_mode == "broadband_perchannel":
                self.temporal_extractor = BroadbandTemporalTargetExtractor(
                    fs=fs, patch_size=patch_size,
                    n_channels=temporal_n_channels,
                )
            else:
                raise ValueError(f"Unknown temporal_mode={temporal_mode!r}")
            temporal_dim = self.temporal_extractor.target_dim

        # Build morphology extractor (if enabled)
        self.morphology_extractor: Optional[MorphologyTargetExtractor] = None
        morphology_dim: Optional[int] = None
        if morphology:
            self.morphology_extractor = MorphologyTargetExtractor(
                fs=fs, patch_size=patch_size, n_channels=morphology_n_channels,
            )
            morphology_dim = self.morphology_extractor.target_dim

        # Build SpectralDecoder model
        self.spectral_decoder = SpectralDecoder(
            embed_dim=embed_dim,
            target_dim=self.spectral.target_dim,
            temporal_dim=temporal_dim,
            morphology_dim=morphology_dim,
            hidden_dim=hidden_dim,
            n_bins=self.spectral.n_bins,
            n_bands=len(CLINICAL_BANDS),
            patch_size=patch_size,
            n_blocks=n_blocks,
            dropout=dropout,
        )

        # Normalisation stats (set during collect)
        self.embed_mean: Optional[torch.Tensor] = None
        self.embed_std: Optional[torch.Tensor] = None
        self.target_mean: Optional[torch.Tensor] = None
        self.target_std: Optional[torch.Tensor] = None
        self.temporal_mean: Optional[torch.Tensor] = None
        self.temporal_std: Optional[torch.Tensor] = None
        self.morphology_mean: Optional[torch.Tensor] = None
        self.morphology_std: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_pairs(
        self,
        set_transformer: nn.Module,
        dataloader: DataLoader,
        target_layer: int = 2,
        max_tokens: Optional[int] = None,
        pool_channels: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run EEG data through the SetTransformer and collect paired
        (token_embedding, spectral_target [, temporal_target, morphology_target]) data.

        For each EEG window (B, C, T):
          - Extract layer activations → (B, S, E) token embeddings
          - Reshape raw EEG into per-second patches → (B, S, C, patch_size)
          - Compute spectral targets for each patch
          - Optionally compute band-pass filtered temporal targets
          - Optionally compute morphology targets (line length, NEO, …)

        Returns:
            embeddings:         (N_tokens, embed_dim)      — transformer token embeddings
            targets:            (N_tokens, target_dim)      — spectral targets
            temporal_targets:   (N_tokens, temporal_dim)    — temporal targets (or None)
            morphology_targets: (N_tokens, morphology_dim)  — morphology targets (or None)
        """
        set_transformer.eval().to(self.device)
        call_fn: object
        n_enc_channels: Optional[int] = None  # set for per-channel backends (REVE, LaBraM)
        n_prefix_tokens: int = getattr(set_transformer, "n_prefix_tokens", 0)
        try:
            from mecheeg.encoders import EncoderBackend
            if isinstance(set_transformer, EncoderBackend):
                inner_model = set_transformer.model
                hook_layers = set_transformer.get_hookable_layers()
                call_fn = lambda x: set_transformer.encode(x)  # noqa: E731
                # REVE/LaBraM expose channel_names; use len to derive C_enc
                if hasattr(set_transformer, "channel_names"):
                    n_enc_channels = len(set_transformer.channel_names)
            else:
                inner_model = set_transformer
                hook_layers = None
                call_fn = lambda x: inner_model(x)  # noqa: E731
        except ImportError:
            inner_model = set_transformer
            hook_layers = None
            call_fn = lambda x: inner_model(x)  # noqa: E731
        inner_model.eval()
        extractor = ActivationExtractor(inner_model, layers=hook_layers)

        embed_list = []
        target_list = []
        temporal_list = [] if self.temporal_extractor is not None else None
        morphology_list = [] if self.morphology_extractor is not None else None
        total = 0

        patch_size = self.spectral.n_fft  # n_fft = patch length (1 s)

        desc = "Collecting pairs"
        if max_tokens:
            desc += f" (max {max_tokens:,})"
        pbar = tqdm(dataloader, desc=desc, unit="batch", leave=False)

        for batch in pbar:
            x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            B, C, T = x.shape
            S = T // patch_size  # number of 1-second patches

            # Forward pass through encoder to get token embeddings
            extractor.clear()
            _ = call_fn(x)
            acts = extractor.get_activations()
            layer_acts = acts[target_layer]  # (B, N_tok, E) on CPU
            # Strip prefix (CLS) tokens from per-channel backends like LaBraM
            if n_prefix_tokens:
                layer_acts = layer_acts[:, n_prefix_tokens:, :]

            N_tok = layer_acts.shape[1]  # actual tokens per sample

            if n_enc_channels is not None and N_tok != S:
                # REVE-style: per-(channel, time) tokens, shape (B, C_enc*S_reve, E)
                # REVE may use internal stride ≠ patch_size, so S_reve = N_tok // C_enc.
                C_enc  = n_enc_channels
                S_reve = N_tok // C_enc
                # Map S_reve time tokens to patch_size windows via uniform stride.
                stride    = T // S_reve
                max_start = T - patch_size
                patches_list = []
                for s_idx in range(S_reve):
                    st = min(s_idx * stride, max_start)
                    patches_list.append(x[:, :C_enc, st : st + patch_size])
                # (B, C_enc, S_reve, patch_size)
                patches_bcsp = torch.stack(patches_list, dim=2)
                T_used = T

                if pool_channels:
                    # Channel-pool tokens: mean across channels → (B, S_reve, E)
                    # Gives one token per time step with channel-averaged semantics,
                    # matching how SleepFM works (cross-channel pooled tokens).
                    layer_acts = layer_acts.reshape(B, C_enc, S_reve, -1).mean(dim=1)
                    # Channel-averaged spectral targets: (B*S_reve, C_enc, patch_size)
                    patches_flat = patches_bcsp.permute(0, 2, 1, 3).reshape(
                        B * S_reve, C_enc, patch_size
                    )
                else:
                    # Per-channel tokens: (B*C_enc*S_reve, 1, patch_size)
                    patches_flat = patches_bcsp.reshape(B * C_enc * S_reve, 1, patch_size)
            else:
                # SleepFM-style: channel-pooled tokens, shape (B, S, E)
                T_used = S * patch_size
                patches = x[:, :, :T_used].reshape(B, C, S, patch_size)
                patches = patches.permute(0, 2, 1, 3)  # (B, S, C, patch_size)
                patches_flat = patches.reshape(B * S, C, patch_size)

            n_flat = patches_flat.shape[0]  # B*S or B*C*S

            # Compute spectral targets
            spec_targets = self.spectral.extract(patches_flat)  # (n_flat, target_dim)

            # Compute temporal targets (if enabled)
            if self.temporal_extractor is not None:
                temp_targets = self.temporal_extractor.extract(patches_flat)
                temporal_list.append(temp_targets.cpu())

            # Compute morphology targets (if enabled)
            if self.morphology_extractor is not None:
                morph_targets = self.morphology_extractor.extract(patches_flat)
                morphology_list.append(morph_targets.cpu())

            # Flatten embeddings to match patches
            embeds_flat = layer_acts.reshape(n_flat, -1)

            embed_list.append(embeds_flat.cpu())
            target_list.append(spec_targets.cpu())

            total += n_flat
            pbar.set_postfix(tokens=f"{total:,}")
            if max_tokens and total >= max_tokens:
                break

        pbar.close()

        extractor.remove_hooks()

        embeddings = torch.cat(embed_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        temporal_targets = torch.cat(temporal_list, dim=0) if temporal_list is not None else None
        morphology_targets = torch.cat(morphology_list, dim=0) if morphology_list is not None else None

        if max_tokens and len(embeddings) > max_tokens:
            embeddings = embeddings[:max_tokens]
            targets = targets[:max_tokens]
            if temporal_targets is not None:
                temporal_targets = temporal_targets[:max_tokens]
            if morphology_targets is not None:
                morphology_targets = morphology_targets[:max_tokens]

        extras = ""
        if temporal_targets is not None:
            extras += f"  temporal: {temporal_targets.shape}"
        if morphology_targets is not None:
            extras += f"  morphology: {morphology_targets.shape}"
        print(f"[SpectralDecoder] Collected {len(embeddings)} (token, spectrum) pairs  "
              f"| embed: {embeddings.shape}  target: {targets.shape}{extras}")
        return embeddings, targets, temporal_targets, morphology_targets

    # ------------------------------------------------------------------
    def train(
        self,
        set_transformer: nn.Module,
        dataloader: DataLoader,
        target_layer: int = 2,
        val_dataloader: Optional[DataLoader] = None,
        max_tokens: Optional[int] = None,
        patience: int = 15,
        pool_channels: bool = False,
    ) -> "SpectralDecoder":
        """
        Full pipeline: collect pairs → normalise → train SpectralDecoder.

        Returns the trained SpectralDecoder (also stored as self.spectral_decoder).
        """
        # 1. Collect (embedding, spectral_target [, temporal_target, morphology_target]) pairs
        embeddings, targets, temporal_targets, morphology_targets = self.collect_pairs(
            set_transformer, dataloader, target_layer, max_tokens,
            pool_channels=pool_channels,
        )

        # 1b. Filter dead patches (near-zero amplitude → degenerate R²)
        n_bins = self.spectral.n_bins
        amp_std = targets[:, :n_bins].std(dim=1)
        alive_mask = amp_std > 0.01
        n_dead = (~alive_mask).sum().item()
        if n_dead > 0:
            print(f"[SpectralDecoder] Filtered {n_dead} dead patches "
                  f"(near-zero amplitude variance)")
            embeddings = embeddings[alive_mask]
            targets = targets[alive_mask]
            if temporal_targets is not None:
                temporal_targets = temporal_targets[alive_mask]
            if morphology_targets is not None:
                morphology_targets = morphology_targets[alive_mask]

        # 2. Normalise embeddings and spectral targets
        self.embed_mean = embeddings.mean(dim=0)
        self.embed_std = embeddings.std(dim=0).clamp(min=1e-8)
        self.target_mean = targets.mean(dim=0)
        self.target_std = targets.std(dim=0).clamp(min=1e-8)

        embeddings_norm = (embeddings - self.embed_mean) / self.embed_std
        targets_norm = (targets - self.target_mean) / self.target_std

        # 2b. Normalise temporal targets (if enabled)
        temporal_norm = None
        if temporal_targets is not None:
            self.temporal_mean = temporal_targets.mean(dim=0)
            self.temporal_std = temporal_targets.std(dim=0).clamp(min=1e-8)
            temporal_norm = (temporal_targets - self.temporal_mean) / self.temporal_std

        # 2c. Normalise morphology targets (if enabled)
        morphology_norm = None
        if morphology_targets is not None:
            self.morphology_mean = morphology_targets.mean(dim=0)
            self.morphology_std = morphology_targets.std(dim=0).clamp(min=1e-8)
            morphology_norm = (morphology_targets - self.morphology_mean) / self.morphology_std

        msg = (f"[SpectralDecoder] Normalised — embed μ-norm={self.embed_mean.norm():.3f}, "
               f"target μ-norm={self.target_mean.norm():.3f}")
        if self.temporal_mean is not None:
            msg += f", temporal μ-norm={self.temporal_mean.norm():.3f}"
        if self.morphology_mean is not None:
            msg += f", morphology μ-norm={self.morphology_mean.norm():.3f}"
        print(msg)

        # 3. Optional: collect validation pairs
        val_embed_norm = val_target_norm = val_temporal_norm = val_morphology_norm = None
        if val_dataloader is not None:
            val_embeddings, val_targets, val_temporal, val_morphology = self.collect_pairs(
                set_transformer, val_dataloader, target_layer,
                max_tokens=max_tokens // 5 if max_tokens else None,
                pool_channels=pool_channels,
            )
            val_embed_norm = (val_embeddings - self.embed_mean) / self.embed_std
            val_target_norm = (val_targets - self.target_mean) / self.target_std
            if val_temporal is not None:
                val_temporal_norm = (val_temporal - self.temporal_mean) / self.temporal_std
            if val_morphology is not None:
                val_morphology_norm = (val_morphology - self.morphology_mean) / self.morphology_std

        # 4. Train
        self._train_spectral_decoder(
            embeddings_norm, targets_norm,
            val_embed_norm, val_target_norm,
            temporal_norm, val_temporal_norm,
            morphology_norm, val_morphology_norm,
            patience=patience,
        )

        return self.spectral_decoder

    # ------------------------------------------------------------------
    def _train_spectral_decoder(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        val_embeddings: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
        temporal: Optional[torch.Tensor] = None,
        val_temporal: Optional[torch.Tensor] = None,
        morphology: Optional[torch.Tensor] = None,
        val_morphology: Optional[torch.Tensor] = None,
        patience: int = 15,
    ):
        """Core training loop with early stopping."""
        spectral_decoder = self.spectral_decoder.to(self.device)
        opt = torch.optim.AdamW(spectral_decoder.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=self.lr * 0.01
        )

        has_temporal = temporal is not None
        has_morph = morphology is not None

        tensors = [embeddings, targets]
        if has_temporal:
            tensors.append(temporal)
        if has_morph:
            tensors.append(morphology)
        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        wait = 0              # early-stopping counter
        best_epoch = 0

        self.history: Dict[str, list] = {
            "total": [], "amp": [], "phase": [], "recon": [],
            "val_total": [], "val_amp": [], "val_phase": [],
        }
        if has_temporal:
            self.history["temporal_mse"] = []
            self.history["temporal_corr"] = []
            self.history["val_temporal_mse"] = []
            self.history["val_temporal_corr"] = []
        if has_morph:
            self.history["morph_mse"] = []
            self.history["val_morph_mse"] = []

        for epoch in range(self.epochs):
            spectral_decoder.train()
            epoch_loss = 0.0
            epoch_amp = 0.0
            epoch_phase = 0.0
            epoch_recon = 0.0
            epoch_temp_mse = 0.0
            epoch_temp_corr = 0.0
            epoch_morph_mse = 0.0

            batch_pbar = tqdm(
                loader, desc=f"  Epoch {epoch+1:3d}/{self.epochs}",
                unit="batch", leave=False,
            )
            for batch_data in batch_pbar:
                emb_batch = batch_data[0].to(self.device)
                tgt_batch = batch_data[1].to(self.device)
                idx = 2
                temp_batch = None
                if has_temporal:
                    temp_batch = batch_data[idx].to(self.device); idx += 1
                morph_batch = None
                if has_morph:
                    morph_batch = batch_data[idx].to(self.device); idx += 1

                loss, details = spectral_decoder.full_loss(
                    emb_batch, tgt_batch,
                    phase_weight=self.phase_weight,
                    recon_weight=self.recon_weight,
                    temporal_targets=temp_batch,
                    temporal_weight=self.temporal_weight,
                    temporal_corr_weight=self.temporal_corr_weight,
                    morphology_targets=morph_batch,
                    morphology_weight=self.morphology_weight,
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(spectral_decoder.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                epoch_amp += details["amp_loss"].item()
                epoch_phase += details["phase_loss"].item()
                epoch_recon += details["embed_recon_loss"].item()
                if has_temporal:
                    epoch_temp_mse += details["temporal_mse"].item()
                    epoch_temp_corr += details["temporal_corr"].item()
                if has_morph:
                    epoch_morph_mse += details["morph_mse"].item()

            batch_pbar.close()
            scheduler.step()
            n = len(loader)
            lr_now = scheduler.get_last_lr()[0]

            self.history["total"].append(epoch_loss / n)
            self.history["amp"].append(epoch_amp / n)
            self.history["phase"].append(epoch_phase / n)
            self.history["recon"].append(epoch_recon / n)

            log = (f"  Epoch {epoch+1:3d}/{self.epochs} | "
                   f"loss={epoch_loss/n:.4f}  "
                   f"amp={epoch_amp/n:.4f}  "
                   f"phase={epoch_phase/n:.4f}  "
                   f"embed_recon={epoch_recon/n:.4f}")

            if has_temporal:
                self.history["temporal_mse"].append(epoch_temp_mse / n)
                self.history["temporal_corr"].append(epoch_temp_corr / n)
                log += (f"  temp_mse={epoch_temp_mse/n:.4f}  "
                        f"temp_corr={epoch_temp_corr/n:.4f}")
            if has_morph:
                self.history["morph_mse"].append(epoch_morph_mse / n)
                log += f"  morph_mse={epoch_morph_mse/n:.4f}"

            log += f"  lr={lr_now:.2e}"

            # Validation + early stopping
            if val_embeddings is not None:
                val_loss, val_details = self._eval(
                    spectral_decoder, val_embeddings, val_targets, val_temporal, val_morphology,
                )
                self.history["val_total"].append(val_loss)
                self.history["val_amp"].append(val_details["amp_loss"])
                self.history["val_phase"].append(val_details["phase_loss"])
                log += (f"  | val_loss={val_loss:.4f}  "
                        f"val_amp={val_details['amp_loss']:.4f}  "
                        f"val_phase={val_details['phase_loss']:.4f}")
                if has_temporal and "temporal_mse" in val_details:
                    self.history["val_temporal_mse"].append(val_details["temporal_mse"])
                    self.history["val_temporal_corr"].append(val_details["temporal_corr"])
                    log += f"  val_temp_corr={val_details['temporal_corr']:.4f}"
                if has_morph and "morph_mse" in val_details:
                    self.history["val_morph_mse"].append(val_details["morph_mse"])
                    log += f"  val_morph_mse={val_details['morph_mse']:.4f}"
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in spectral_decoder.state_dict().items()}
                    best_epoch = epoch + 1
                    wait = 0
                    log += "  ★"
                else:
                    wait += 1

            print(log)

            # Early stopping check
            if val_embeddings is not None and wait >= patience:
                print(f"\n  ⏹ Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs, "
                      f"best was epoch {best_epoch})")
                break

        # Restore best model if we did validation
        if best_state is not None:
            spectral_decoder.load_state_dict(best_state)
            print(f"  ✓ Restored best validation model "
                  f"(epoch {best_epoch}, val_loss={best_val_loss:.4f})")

        self.spectral_decoder = spectral_decoder.cpu()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval(
        self,
        spectral_decoder: SpectralDecoder,
        val_embeddings: torch.Tensor,
        val_targets: torch.Tensor,
        val_temporal: Optional[torch.Tensor] = None,
        val_morphology: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate on validation set."""
        spectral_decoder.eval()
        total_loss = 0.0
        total_amp = 0.0
        total_phase = 0.0
        total_temp_mse = 0.0
        total_temp_corr = 0.0
        total_morph_mse = 0.0
        n = 0

        for i in range(0, len(val_embeddings), self.batch_size):
            emb = val_embeddings[i : i + self.batch_size].to(self.device)
            tgt = val_targets[i : i + self.batch_size].to(self.device)
            temp = None
            if val_temporal is not None:
                temp = val_temporal[i : i + self.batch_size].to(self.device)
            morph = None
            if val_morphology is not None:
                morph = val_morphology[i : i + self.batch_size].to(self.device)
            loss, details = spectral_decoder.full_loss(
                emb, tgt, self.phase_weight, self.recon_weight,
                temporal_targets=temp,
                temporal_weight=self.temporal_weight,
                temporal_corr_weight=self.temporal_corr_weight,
                morphology_targets=morph,
                morphology_weight=self.morphology_weight,
            )
            total_loss += loss.item()
            total_amp += details["amp_loss"].item()
            total_phase += details["phase_loss"].item()
            if "temporal_mse" in details:
                total_temp_mse += details["temporal_mse"].item()
                total_temp_corr += details["temporal_corr"].item()
            if "morph_mse" in details:
                total_morph_mse += details["morph_mse"].item()
            n += 1

        result = {
            "amp_loss": total_amp / n,
            "phase_loss": total_phase / n,
        }
        if val_temporal is not None:
            result["temporal_mse"] = total_temp_mse / n
            result["temporal_corr"] = total_temp_corr / n
        if val_morphology is not None:
            result["morph_mse"] = total_morph_mse / n

        return total_loss / n, result

    # ------------------------------------------------------------------
    def decode_direction(
        self,
        direction: torch.Tensor,
        denormalise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode any direction in embedding space into spectral components.
        This is the main interpretability entry point.

        Args:
            direction:    (embed_dim,) or (N, embed_dim) vector(s)
            denormalise:  if True, undo target normalisation

        Returns:
            amplitude:  (..., n_bins) — log1p-scaled
            cos_phase:  (..., n_bins)
            sin_phase:  (..., n_bins)
        """
        self.spectral_decoder.eval()
        was_1d = direction.dim() == 1
        if was_1d:
            direction = direction.unsqueeze(0)

        # Normalise input the same way as training data
        if self.embed_mean is not None:
            direction = (direction - self.embed_mean) / self.embed_std

        with torch.no_grad():
            pred = self.spectral_decoder.decode(direction)

        # Denormalise output
        if denormalise and self.target_mean is not None:
            pred = pred * self.target_std + self.target_mean

        amp, cos_p, sin_p = self.spectral.unpack_targets(pred)

        if was_1d:
            amp, cos_p, sin_p = amp.squeeze(0), cos_p.squeeze(0), sin_p.squeeze(0)

        return amp, cos_p, sin_p

    # ------------------------------------------------------------------
    def decode_direction_temporal(
        self,
        direction: torch.Tensor,
        denormalise: bool = True,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Decode any direction in embedding space into per-band time-domain
        waveforms.  Returns None if the temporal head is not enabled.

        Args:
            direction:    (embed_dim,) or (N, embed_dim) vector(s)
            denormalise:  if True, undo temporal normalisation

        Returns:
            dict mapping band_name → (..., patch_size) waveform, or None
        """
        if self.temporal_extractor is None or self.spectral_decoder.dec_temporal_head is None:
            return None

        self.spectral_decoder.eval()
        was_1d = direction.dim() == 1
        if was_1d:
            direction = direction.unsqueeze(0)

        # Normalise input the same way as training data
        if self.embed_mean is not None:
            direction = (direction - self.embed_mean) / self.embed_std

        with torch.no_grad():
            pred = self.spectral_decoder.decode_temporal(direction)

        # Denormalise output
        if denormalise and self.temporal_mean is not None:
            pred = pred * self.temporal_std + self.temporal_mean

        band_waveforms = self.temporal_extractor.unpack_targets(pred)

        if was_1d:
            band_waveforms = {k: v.squeeze(0) for k, v in band_waveforms.items()}

        return band_waveforms

    # ------------------------------------------------------------------
    def decode_direction_morphology(
        self,
        direction: torch.Tensor,
        denormalise: bool = True,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Decode any direction in embedding space into morphology features.

        Args:
            direction:    (embed_dim,) or (N, embed_dim) vector(s)
            denormalise:  if True, undo morphology target normalisation
        Returns:
            dict mapping feature_name → scalar (or (N,)) tensor, or None
        """
        if self.morphology_extractor is None or self.spectral_decoder.dec_morph_head is None:
            return None

        self.spectral_decoder.eval()
        was_1d = direction.dim() == 1
        if was_1d:
            direction = direction.unsqueeze(0)

        if self.embed_mean is not None:
            direction = (direction - self.embed_mean) / self.embed_std

        with torch.no_grad():
            pred = self.spectral_decoder.decode_morphology(direction)

        if denormalise and self.morphology_mean is not None:
            pred = pred * self.morphology_std + self.morphology_mean

        out = self.morphology_extractor.unpack_targets(pred)
        if was_1d:
            out = {k: v.squeeze(0) for k, v in out.items()}
        return out

    # ------------------------------------------------------------------
    def spectral_to_waveforms(
        self,
        amplitude: torch.Tensor,
        patch_size: int = 128,
        fs: int = 128,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase-agnostic waveform reconstruction from spectral amplitudes.

        Constructs a per-band waveform by placing the predicted amplitude
        into the corresponding FFT bins with **zero phase** (all cosines)
        and inverse-FFTing back to the time domain.  The result is a
        symmetric "characteristic waveform" centred at T/2 that faithfully
        shows the frequency content *without* predicting phase.

        This is preferred for visualisation because phase is not encoded
        in the SetTransformer embeddings (NN-consistency ≈ 0.05) and
        cannot be recovered.  The zero-phase shape is deterministic and
        visually interpretable — it shows band power as oscillation
        amplitude and frequency content as oscillation period.

        Args:
            amplitude:   (..., n_bins) log1p-scaled amplitudes
            patch_size:  number of time-domain samples (default 128)
            fs:          sampling rate in Hz (default 128)

        Returns:
            dict mapping band_name → (..., patch_size) waveform tensor
        """
        # Convert log1p → linear amplitude
        amp_lin = torch.expm1(amplitude)               # (..., n_bins)

        n_freq = patch_size // 2 + 1                   # 65
        freq_mask = torch.tensor(self.spectral._freq_mask,
                                 device=amplitude.device)

        # Place amplitudes as real-valued (zero imaginary = zero phase)
        # into the full frequency axis
        shape = amplitude.shape[:-1]                   # batch dims
        full = torch.zeros(*shape, n_freq, device=amplitude.device)
        full[..., freq_mask] = amp_lin                 # all real → phase = 0

        # Build band masks
        freqs_full = torch.fft.rfftfreq(patch_size, d=1.0 / fs)
        result = {}
        for band_name, (f_lo, f_hi) in CLINICAL_BANDS.items():
            band_mask = ((freqs_full >= f_lo) & (freqs_full <= f_hi)).to(
                amplitude.device
            ).float()
            masked = full * band_mask                  # (..., n_freq)
            # iFFT: real input → symmetric time-domain output
            waveform = torch.fft.irfft(
                masked.to(torch.cfloat), n=patch_size, dim=-1
            )
            # Roll so symmetric peak is centred at T/2
            waveform = torch.roll(waveform, shifts=patch_size // 2, dims=-1)
            result[band_name] = waveform.float()

        return result

    # ------------------------------------------------------------------
    def spectral_to_signal(
        self,
        amplitude: torch.Tensor,
        cos_phase: Optional[torch.Tensor] = None,
        sin_phase: Optional[torch.Tensor] = None,
        patch_size: int = 128,
    ) -> torch.Tensor:
        """
        Reconstruct a full broadband time-domain signal from spectral
        components.

        If ``cos_phase``/``sin_phase`` are provided, uses them (predicted
        or ground-truth).  Otherwise falls back to zero-phase (all cosines),
        producing a symmetric waveform centred at T/2.

        Args:
            amplitude:  (..., n_bins) log1p-scaled
            cos_phase:  (..., n_bins) or None → zero phase
            sin_phase:  (..., n_bins) or None → zero phase
            patch_size: number of time-domain samples (default 128)

        Returns:
            signal: (..., patch_size)  reconstructed time-domain waveform
        """
        amp_lin = torch.expm1(amplitude)                   # linear amplitude
        n_freq = patch_size // 2 + 1
        freq_mask = torch.tensor(self.spectral._freq_mask,
                                 device=amplitude.device)

        shape = amplitude.shape[:-1]
        full = torch.zeros(*shape, n_freq, dtype=torch.cfloat,
                           device=amplitude.device)

        if cos_phase is not None and sin_phase is not None:
            phase = torch.atan2(sin_phase, cos_phase)
            full[..., freq_mask] = amp_lin * torch.exp(1j * phase)
        else:
            # zero phase: amplitude is placed as real part only
            full[..., freq_mask] = amp_lin.to(torch.cfloat)

        signal = torch.fft.irfft(full, n=patch_size, dim=-1)

        # For zero-phase, centre the symmetric peak at T/2
        if cos_phase is None:
            signal = torch.roll(signal, shifts=patch_size // 2, dims=-1)

        return signal.float()

    # ------------------------------------------------------------------
    def decode_direction_waveforms(
        self,
        direction: torch.Tensor,
        denormalise: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode embedding direction(s) into phase-agnostic per-band waveforms.

        This is the **recommended** method for visualising what a feature
        "looks like" in the time domain.  It uses the spectral head (which
        has high-quality amplitude prediction, R²≈0.80) and reconstructs
        waveforms via zero-phase iFFT — no temporal head needed.

        Args:
            direction:    (embed_dim,) or (N, embed_dim) vector(s)
            denormalise:  if True, undo target normalisation

        Returns:
            dict mapping band_name → (..., patch_size) waveform tensor
        """
        amp, cos_p, sin_p = self.decode_direction(direction, denormalise=denormalise)
        return self.spectral_to_waveforms(amp)

    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save SpectralDecoder checkpoint."""
        state = {
            "spectral_decoder_state_dict": self.spectral_decoder.state_dict(),
            "embed_mean": self.embed_mean,
            "embed_std": self.embed_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "spectral_config": {
                "fs": self.spectral.fs,
                "n_fft": self.spectral.n_fft,
                "f_min": self.spectral.f_min,
                "f_max": self.spectral.f_max,
                "n_bins": self.spectral.n_bins,
                "freqs": self.spectral.freqs.tolist(),
                "bin_labels": self.spectral.bin_labels,
            },
            "model_config": {
                "embed_dim": self.spectral_decoder.embed_dim,
                "target_dim": self.spectral_decoder.target_dim,
                "temporal_dim": self.spectral_decoder.temporal_dim,
                "morphology_dim": self.spectral_decoder.morphology_dim,
                "hidden_dim": self.spectral_decoder.hidden_dim,
                "n_bins": self.spectral_decoder.n_bins,
                "n_bands": self.spectral_decoder.n_bands,
                "patch_size": self.spectral_decoder.patch_size,
                "n_blocks": self.spectral_decoder.n_blocks,
                "dropout": self.spectral_decoder.dropout,
            },
            # Temporal normalisation stats
            "temporal_mean": self.temporal_mean,
            "temporal_std": self.temporal_std,
            "temporal_enabled": self.temporal_enabled,
            # Morphology normalisation stats
            "morphology_mean": self.morphology_mean,
            "morphology_std": self.morphology_std,
            "morphology_enabled": self.morphology_enabled,
        }
        # Temporal extractor config
        if self.temporal_extractor is not None:
            state["temporal_config"] = {
                "fs": self.temporal_extractor.fs,
                "patch_size": self.temporal_extractor.patch_size,
                "n_bands": self.temporal_extractor.n_bands,
                "mode": self.temporal_mode,
                "n_channels": getattr(self.temporal_extractor,
                                      "n_channels", None),
            }
        # Morphology extractor config
        if self.morphology_extractor is not None:
            state["morphology_config"] = {
                "fs": self.morphology_extractor.fs,
                "patch_size": self.morphology_extractor.patch_size,
                "n_channels": self.morphology_extractor.n_channels,
                "feature_names": self.morphology_extractor.feature_names,
            }
        torch.save(state, path)
        print(f"[SpectralDecoder] Saved checkpoint to {path}")

    def load(self, path: str):
        """Load SpectralDecoder checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=False)

        # Rebuild model with saved config
        cfg = state["model_config"]
        self.spectral_decoder = SpectralDecoder(
            embed_dim=cfg["embed_dim"],
            target_dim=cfg["target_dim"],
            temporal_dim=cfg.get("temporal_dim"),
            morphology_dim=cfg.get("morphology_dim"),
            hidden_dim=cfg["hidden_dim"],
            n_bins=cfg["n_bins"],
            n_bands=cfg.get("n_bands", 6),
            patch_size=cfg.get("patch_size", 128),
            n_blocks=cfg.get("n_blocks", 3),
            dropout=cfg.get("dropout", 0.1),
        )
        # Backward-compat: this module used to be named XAE. Old checkpoints
        # (results/xae/*/xae_checkpoint.pt) store the weights under
        # "xae_state_dict" instead of "spectral_decoder_state_dict".
        sd = state.get("spectral_decoder_state_dict") or state.get("xae_state_dict")
        if sd is None:
            raise KeyError(
                "checkpoint has neither 'spectral_decoder_state_dict' nor "
                "'xae_state_dict'"
            )
        self.spectral_decoder.load_state_dict(sd)

        # Restore normalisation stats
        self.embed_mean = state["embed_mean"]
        self.embed_std = state["embed_std"]
        self.target_mean = state["target_mean"]
        self.target_std = state["target_std"]
        self.temporal_mean = state.get("temporal_mean")
        self.temporal_std = state.get("temporal_std")
        self.temporal_enabled = state.get("temporal_enabled", False)
        self.morphology_mean = state.get("morphology_mean")
        self.morphology_std = state.get("morphology_std")
        self.morphology_enabled = state.get("morphology_enabled", False)

        # Restore spectral config
        scfg = state["spectral_config"]
        self.spectral = SpectralTargetExtractor(
            fs=scfg["fs"], n_fft=scfg["n_fft"],
            f_min=scfg["f_min"], f_max=scfg["f_max"],
        )

        # Restore temporal extractor (if present)
        self.temporal_extractor = None
        self.temporal_mode = "bands"
        if "temporal_config" in state:
            tcfg = state["temporal_config"]
            mode = tcfg.get("mode", "bands")
            self.temporal_mode = mode
            if mode == "broadband_perchannel":
                self.temporal_extractor = BroadbandTemporalTargetExtractor(
                    fs=tcfg["fs"], patch_size=tcfg["patch_size"],
                    n_channels=tcfg.get("n_channels", 19),
                )
            else:
                self.temporal_extractor = TemporalTargetExtractor(
                    fs=tcfg["fs"], patch_size=tcfg["patch_size"],
                )

        # Restore morphology extractor (if present)
        self.morphology_extractor = None
        if "morphology_config" in state:
            mcfg = state["morphology_config"]
            self.morphology_extractor = MorphologyTargetExtractor(
                fs=mcfg["fs"], patch_size=mcfg["patch_size"],
                n_channels=mcfg.get("n_channels", 19),
            )

        extra = ""
        if cfg.get("temporal_dim"):
            extra += f", temporal_dim={cfg['temporal_dim']}"
        if cfg.get("morphology_dim"):
            extra += f", morphology_dim={cfg['morphology_dim']}"
        print(f"[SpectralDecoder] Loaded checkpoint from {path}  "
              f"(n_bins={self.spectral.n_bins}, "
              f"target_dim={self.spectral_decoder.target_dim}{extra})")
