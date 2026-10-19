"""Regenerate Figure 4 (concept encoding and steering selectivity) from the bundled data.json (seeded SAE results).

  uv run python "tools/paper_figures/Figure 4/plot.py"
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
with tempfile.TemporaryDirectory() as tmp:
    for name, payload in json.loads((HERE / "data.json").read_text()).items():
        (Path(tmp) / f"{name}.json").write_text(json.dumps(payload))
    env = dict(os.environ, FIG_IN=tmp, FIG_OUT_DIR=str(HERE), FIG_STEM="figure")
    subprocess.run([sys.executable, str(ROOT / "tools/paper_figures/plot_fig4_camera_ready.py")], env=env, check=True)
