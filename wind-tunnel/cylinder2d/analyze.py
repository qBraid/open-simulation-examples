"""Strouhal number and drag from forceCoeffs output (D = 1, U = 1).

Usage: python analyze.py <case_dir> [t_start]
St from mean period between upward zero-crossings of Cl over t >= t_start.
"""
import glob
import json
import sys
import numpy as np

case = sys.argv[1]
t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
rows = []
for f in sorted(glob.glob(f"{case}/postProcessing/forceCoeffs/*/coefficient.dat")):
    rows.append(np.loadtxt(f, comments="#", usecols=(0, 1, 4)))
d = np.vstack(rows)
d = d[np.argsort(d[:, 0])]
_, keep = np.unique(d[:, 0], return_index=True)
t, cd, cl = d[keep].T
m = t >= t0
t, cd, cl = t[m], cd[m], cl[m]
cl0 = cl - cl.mean()
up = np.where((cl0[:-1] < 0) & (cl0[1:] >= 0))[0]
tc = t[up] - cl0[up] * (t[up + 1] - t[up]) / (cl0[up + 1] - cl0[up])
periods = np.diff(tc)
# average Cd over whole shedding cycles only
w = (t >= tc[0]) & (t <= tc[-1])
res = dict(
    t_window=[round(float(t[0]), 3), round(float(t[-1]), 3)],
    cycles=int(len(periods)),
    St=round(float(1.0 / periods.mean()), 4),
    St_cycle_std=round(float(np.std(1.0 / periods)), 5),
    Cd_mean=round(float(np.trapezoid(cd[w], t[w]) / (t[w][-1] - t[w][0])), 4),
    Cl_rms=round(float(np.sqrt(np.mean(cl0[w] ** 2))), 4),
    Cl_amp=round(float((cl[w].max() - cl[w].min()) / 2), 4),
)
print(json.dumps(res))
