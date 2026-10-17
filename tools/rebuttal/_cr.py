"""Camera-ready rerun switches, shared by the rebuttal scripts.

Defaults reproduce the rebuttal record exactly. Two optional environment variables
redirect a run without editing any script:

  CR_OUT_DIR  write artifacts here instead of results/rebuttal/
  CR_SEED     SAE training seed (overrides the script's hard-coded SEED)
"""
import os
from pathlib import Path


def out_dir(default: Path) -> Path:
    p = Path(os.environ["CR_OUT_DIR"]) if os.environ.get("CR_OUT_DIR") else default
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed(default: int) -> int:
    return int(os.environ.get("CR_SEED", default))
