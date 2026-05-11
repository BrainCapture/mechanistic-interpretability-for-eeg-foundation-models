# mecheeg — Mechanistic Interpretability for EEG Foundation Models
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mecheeg")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = [
    "sleepfm",
    "sae",
    "spectral_decoder",
    "dataset",
    "encoders",
    # Convenience re-exports
    "EncoderBackend",
    "SleepFMBackend",
    "REVEBackend",
    "load_encoder",
]

from mecheeg.encoders import EncoderBackend, SleepFMBackend, REVEBackend, load_encoder
