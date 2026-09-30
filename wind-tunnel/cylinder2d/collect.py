"""Collect the cylinder runs into results/validation.json.

Usage: python collect.py <runs_dir> <out.json>
"""
import json
import re
import subprocess
import sys
from pathlib import Path

runs, out = Path(sys.argv[1]), Path(sys.argv[2])
here = Path(__file__).parent
cases = [("coarse, walls ±10D", "cyl_coarse", False), ("fine, walls ±10D", "cyl_fine", True),
         ("coarse, walls ±30D", "cyl_coarse_w30", False), ("fine, walls ±30D", "cyl_fine_w30", False)]
meshes = []
for label, d, frames in cases:
    case = runs / d
    log2 = case / "log.pimpleFoam.2"
    if not (log2.exists() and "\nEnd" in log2.read_text()):
        continue  # missing or still running
    r = json.loads(subprocess.check_output([sys.executable, str(here / "analyze.py"), str(case), "100"]))
    cells = int(re.search(r"cells:\s+(\d+)", (case / "log.checkMesh").read_text()).group(1))
    meshes.append(dict(level=label, case=d, cells=cells, frames=frames, **r))
res = dict(
    case="2D laminar cylinder, Re = 100 (D = 1, U = 1, nu = 0.01), pimpleFoam, backward/linearUpwind",
    reference=dict(St=[0.164, 0.166], Cd=[1.33, 1.35],
                   source="unconfined-domain values reported across the Re = 100 literature (e.g. Park et al. 1998; Posdziech & Grundmann 2007)"),
    meshes=meshes,
)
out.write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=1))
