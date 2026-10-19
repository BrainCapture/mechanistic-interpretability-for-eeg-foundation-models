"""Regenerate the camera-ready appendix figures from the bundled data.json (seeded SAE results).

  uv run python "tools/paper_figures/Appendix/plot.py"
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
with tempfile.TemporaryDirectory() as tmp:
    for name, data in json.loads((HERE / "data.json").read_text()).items():
        (Path(tmp) / name).parent.mkdir(parents=True, exist_ok=True)
        (Path(tmp) / name).write_text(json.dumps(data))
    env = dict(os.environ, FIG_IN=tmp, FIG_OUT_DIR=str(HERE))
    subprocess.run([sys.executable, str(ROOT / "tools/paper_figures/plot_camera_ready_appendix.py")], env=env, check=True)
