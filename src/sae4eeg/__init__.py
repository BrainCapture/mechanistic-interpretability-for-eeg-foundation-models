# sae4eeg — Mechanistic Interpretability for EEG Foundation Models
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sae4eeg")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = [
    "sleepfm",
    "sae",
    "xae",
    "dataset",
    "encoders",
    # Convenience re-exports
    "EncoderBackend",
    "SleepFMBackend",
    "REVEBackend",
    "load_encoder",
]

from sae4eeg.encoders import EncoderBackend, SleepFMBackend, REVEBackend, load_encoder
