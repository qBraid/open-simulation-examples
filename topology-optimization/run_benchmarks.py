"""Run the Python ports on the published benchmark set and save designs and histories.

Usage: python run_benchmarks.py [mbb|cantilever|top3d|all]
Writes results/py_<case>.npz and appends to results/py_runs.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from topopt import top3d, top88

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

MBB = [(60, 20, 0.5, 3.0, 2.4), (150, 50, 0.5, 3.0, 6.0), (300, 100, 0.5, 3.0, 16.0)]
CANT = [(32, 20, 0.4, 3.0, 1.2)]


def save(name, res, meta):
    snaps = res["snaps"]
    np.savez_compressed(OUT / f"py_{name}.npz",
                        x=res["x"], hist=np.array([h[:4] for h in res["hist"]]),
                        snap_iters=np.array([s[0] for s in snaps]),
                        snaps=np.stack([np.asarray(s[1], np.float32) for s in snaps]))
    runs = json.loads((OUT / "py_runs.json").read_text()) if (OUT / "py_runs.json").exists() else {}
    runs[name] = {**meta, "compliance": res["c"], "iterations": res["iters"], "seconds": round(res["seconds"], 2)}
    (OUT / "py_runs.json").write_text(json.dumps(runs, indent=1))
    print(f"{name}: c={res['c']:.4f} iters={res['iters']} t={res['seconds']:.1f}s", flush=True)


def main(which):
    if which in ("mbb", "all"):
        for ft in (1, 2):
            for nelx, nely, vf, p, r in MBB:
                res = top88(nelx, nely, vf, p, r, ft, bc="mbb", snapshot_every=5 if nelx == 150 else 0)
                save(f"top88_mbb_{nelx}x{nely}_ft{ft}", res, dict(nelx=nelx, nely=nely, volfrac=vf, penal=p, rmin=r, ft=ft, bc="mbb"))
    if which in ("cantilever", "all"):
        for ft in (1, 2):
            for nelx, nely, vf, p, r in CANT:
                res = top88(nelx, nely, vf, p, r, ft, bc="cantilever", snapshot_every=2)
                save(f"top88_cant_{nelx}x{nely}_ft{ft}", res, dict(nelx=nelx, nely=nely, volfrac=vf, penal=p, rmin=r, ft=ft, bc="cantilever"))
    if which in ("top3d", "all"):
        res = top3d(60, 20, 4, 0.3, 3.0, 1.5, snapshot_every=1)
        save("top3d_60x20x4", res, dict(nelx=60, nely=20, nelz=4, volfrac=0.3, penal=3.0, rmin=1.5))
    if which in ("top3d200", "all"):  # same run forced to the reference's 200 iterations (tolx = 0)
        res = top3d(60, 20, 4, 0.3, 3.0, 1.5, snapshot_every=0, tolx=0.0, maxloop=200)
        save("top3d_60x20x4_200it", res, dict(nelx=60, nely=20, nelz=4, volfrac=0.3, penal=3.0, rmin=1.5, note="forced 200 iterations"))


if __name__ == "__main__":
    t = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"total {time.time() - t:.1f}s")
