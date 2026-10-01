"""Collect the v2 runs into results/ahmed_v2.json and results/validation_v2.json.

Usage: python collect_v2.py <runs_root>
  Ahmed runs:    <runs_root>/a<slant>_<level>/   (ahmed/run.sh, case.info written there)
  Cylinder runs: <runs_root>/cyl_*/              (cylinder2d/run.sh), plus the v1 numbers kept for context
"""
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
HERE = Path(__file__).parent
R = HERE / "results"

EXP = {"25": (0.285, "Ahmed, Ramm & Faltin 1984 (SAE 840300), 60 m/s"),
       "35": (0.257, "Ahmed, Ramm & Faltin 1984 (SAE 840300), 60 m/s")}


def forces(case, n=200):
    files = sorted(glob.glob(f"{case}/postProcessing/forceCoeffs/*/coefficient.dat"))
    if not files:
        return None
    d = np.vstack([np.loadtxt(f, comments="#", usecols=(0, 1, 4)) for f in files])
    d = d[np.argsort(d[:, 0])]
    _, k = np.unique(d[:, 0], return_index=True)
    d = d[k]
    w = d[-n:]
    return dict(iters=int(d[-1, 0]), Cd=round(float(w[:, 1].mean()), 4), Cd_std=round(float(w[:, 1].std()), 4),
                Cl=round(float(w[:, 2].mean()), 4))


def wall(case):
    """ExecutionTime of the last simpleFoam iteration, summed over legs (resumed runs restart the clock)."""
    tot = 0.0
    for f in glob.glob(f"{case}/log.simpleFoam*"):
        last = 0.0
        for line in open(f, errors="ignore"):
            m = re.match(r"ExecutionTime = ([\d.]+) s", line)
            if m:
                t = float(m.group(1))
                if t < last:     # a new leg started in the same log
                    tot += last
                last = t
        tot += last
    return round(tot / 60, 1)


slants = {}
for case in sorted(glob.glob(str(root / "a[0-9][0-9]_*"))):
    info = Path(case, "case.info")
    f = forces(case)
    if not info.exists() or not f or f["iters"] < 600:
        continue
    kv = dict(x.split("=") for x in info.read_text().split())
    s, lv = kv["slant"], kv["level"]
    S = slants.setdefault(s, {"exp": EXP[s][0], "exp_src": EXP[s][1], "levels": {}})
    S["levels"][lv] = dict(cells=int(kv["cells"]), Cd=f["Cd"], Cd_std=f["Cd_std"], Cl=f["Cl"], iters=f["iters"],
                           ground=kv["ground"], wall_min=wall(case), run=os.path.basename(case))
for s, S in slants.items():
    order = [l for l in ("xfine", "fine", "medium", "coarse") if l in S["levels"]]
    S["best"] = order[0]
    L = [S["levels"][l]["Cd"] for l in ("coarse", "medium", "fine", "xfine") if l in S["levels"]]
    S["ladder_change_pct"] = round(100 * (L[-1] - L[-2]) / L[-2], 2) if len(L) > 1 else None

bar = ("Top-10% for steady-RANS external aero on the Ahmed body: drag within 5% of the wind-tunnel value on "
       "a mesh whose last refinement moves Cd by 2% or less, at both slants. 35° (fully separated slant) is "
       "the case steady RANS is known to handle; 25° is the classic failure case, where the experiment has a "
       "slant separation bubble that reattaches and RANS mispredicts it in either direction depending on "
       "model, wall treatment and mesh. Getting 25° right takes hybrid RANS/LES (DDES/IDDES) on tens of "
       "millions of cells.")
note = ("Cd uses the half-model frontal area. Surfaces, vortices and particles come from the finest mesh of each "
        "slant; the slice and the Cd label follow the selected mesh level.")
(R / "ahmed_v2.json").write_text(json.dumps({"bar": bar, "note": note, "slants": slants}, indent=1))

# ---------- cylinder
v1 = json.loads((R / "validation.json").read_text())
runs = []
for m in v1.get("meshes", []):
    runs.append(dict(label=f"v1 inlet 10D, {m['level'].replace('walls ', '')}", St=m["St"], Cd=m["Cd_mean"],
                     cells=m.get("cells"), source="v1 (inlet 10D)"))
an = HERE / "cylinder2d" / "analyze.py"
for case in sorted(glob.glob(str(root / "cyl_*"))):
    meta = json.loads(Path(case, "case.json").read_text()) if Path(case, "case.json").exists() else {}
    t0 = str(meta.get("window", [100])[0])
    try:
        out = subprocess.run([sys.executable, str(an), case, t0], capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        continue
    r = json.loads(out.strip().splitlines()[-1])
    runs.append(dict(label=meta.get("label", os.path.basename(case)), St=r["St"], Cd=r["Cd_mean"], cycles=r["cycles"],
                     cells=meta.get("cells"), window=r["t_window"], source="v2", case=os.path.basename(case)))
ref = {"St": [0.164, 0.166], "Cd": [1.33, 1.35],
       "src": "Williamson 1996; Park, Kwon & Choi 1998 (St 0.165, Cd 1.33); Qu et al. 2013 (St 0.1648, Cd 1.326)"}
v2 = [r for r in runs if r["source"] == "v2"]


def score(r):
    return abs(r["St"] - 0.165) / 0.165 + abs(r["Cd"] - 1.34) / 1.34


anim = [r for r in v2 if r.get("case") == "cyl_x20_frames"]   # the viewer animates this run, so it is the headline
best = anim[0] if anim else (min(v2, key=score) if v2 else runs[-1])
(R / "validation_v2.json").write_text(json.dumps({"runs": runs, "reference": ref, "best": best}, indent=1))
print(json.dumps({"slants": {k: {l: v["Cd"] for l, v in S["levels"].items()} for k, S in slants.items()},
                  "cyl": [(r["label"], r["St"], r["Cd"]) for r in runs]}, indent=1))
