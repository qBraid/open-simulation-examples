"""NVIDIA cuOpt (GPU) on the same CVRPLIB X subset and budget as bench.py.

cuOpt solves on the GPU; its routes are re-costed here with the TSPLIB rounded
integer distances (the convention BKS use) and validated before scoring.
Needs a CUDA 12 GPU and `pip install --extra-index-url https://pypi.nvidia.com cuopt-cu12`.
Runs in its own venv (cuOpt pins numpy 2.4).

    python cuopt_bench.py --list xsub.txt --dir X --factor 0.1 --out results/xbench_cuopt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from datetime import date

import numpy as np


def euc2d(coords):
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    return np.floor(d + 0.5).astype(np.int64)


def solve(path, time_limit, extra_vehicles=10, mode="default", min_vehicles=None):
    import cudf
    import vrplib
    from cuopt import routing

    raw = vrplib.read_instance(path)
    coords = np.asarray(raw["node_coord"], dtype=float)
    demand = np.asarray(raw["demand"], dtype=np.int64)
    Q = int(raw["capacity"])
    D = euc2d(coords)
    n = len(demand)
    kmin = math.ceil(demand.sum() / Q)
    nv = kmin + extra_vehicles

    dm = routing.DataModel(n, nv, n - 1)
    dm.add_cost_matrix(cudf.DataFrame(D.astype(np.float32)))
    dm.set_order_locations(cudf.Series(np.arange(1, n, dtype=np.int32)))
    dm.add_capacity_dimension("demand", cudf.Series(demand[1:].astype(np.int32)),
                              cudf.Series(np.full(nv, Q, dtype=np.int32)))
    dm.set_vehicle_locations(cudf.Series(np.zeros(nv, dtype=np.int32)),
                             cudf.Series(np.zeros(nv, dtype=np.int32)))
    if mode == "fixedcost":  # replace vehicle-count-first with an explicit (zero) fleet cost
        dm.set_vehicle_fixed_costs(cudf.Series(np.zeros(nv, dtype=np.float32)))
        dm.set_objective_function(cudf.Series([routing.Objective.COST, routing.Objective.VEHICLE_FIXED_COST]),
                                  cudf.Series(np.array([1.0, 1.0], dtype=np.float32)))
    if min_vehicles:
        dm.set_min_vehicles(int(min_vehicles))
    ss = routing.SolverSettings()
    ss.set_time_limit(float(time_limit))
    t0 = time.perf_counter()
    sol = routing.Solve(dm, ss)
    runtime = time.perf_counter() - t0
    if sol.get_status() != 0:
        return {"cost": None, "error": str(sol.get_message()), "runtime": runtime}
    df = sol.get_route().to_pandas()
    routes = []
    for _, g in df.groupby("truck_id", sort=False):
        r = [int(x) for x in g["location"].tolist() if int(x) != 0]
        if r:
            routes.append(r)
    seen = sorted(c for r in routes for c in r)
    assert seen == list(range(1, n)), "cuOpt solution misses or repeats customers"
    for r in routes:
        assert demand[r].sum() <= Q, "capacity violated"
    cost = int(sum(D[a, b] for r in routes for a, b in zip([0, *r], [*r, 0])))
    return {"cost": cost, "routes": len(routes), "objective": float(sol.get_total_objective()),
            "runtime": round(runtime, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--factor", type=float, default=0.1)
    ap.add_argument("--out", default="results/xbench_cuopt")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=10**9)
    ap.add_argument("--mode", default="default", choices=["default", "fixedcost"])
    ap.add_argument("--kplus", type=int, default=0, help="if > 0, request min_vehicles = k_min + kplus")
    a = ap.parse_args()
    import vrplib

    os.makedirs(a.out, exist_ok=True)
    names = [l.strip() for l in open(a.list) if l.strip()][a.start:a.stop]
    for name in names:
        out_path = os.path.join(a.out, f"{name}.json")
        if os.path.exists(out_path):
            continue
        p = os.path.join(a.dir, f"{name}.vrp")
        bks = float(vrplib.read_solution(os.path.join(a.dir, f"{name}.sol"))["cost"])
        n = len(vrplib.read_instance(p)["demand"]) - 1
        tlim = round(a.factor * 2.4 * n, 1)
        try:
            kw = {"mode": a.mode}
            if a.kplus > 0:
                import math as _m
                raw = vrplib.read_instance(p)
                kw["min_vehicles"] = _m.ceil(sum(raw["demand"]) / raw["capacity"]) + a.kplus
            r = solve(p, tlim, **kw)
            r["mode"] = a.mode + (f"+k{a.kplus}" if a.kplus else "")
        except Exception as e:
            r = {"cost": None, "error": repr(e)}
        rec = {"instance": name, "n": n, "bks": bks, "time_limit_s": tlim, **r,
               "gap_pct": round(100 * (r["cost"] - bks) / bks, 4) if r.get("cost") else None,
               "stamp": {"date": date.today().isoformat(),
                         "host": "qBraid gpu-l4 instance (NVIDIA L4 24 GB)",
                         "solver": "cuopt-cu12 26.08", "python": platform.python_version(),
                         "protocol": f"Tmax = {a.factor} x 2.4 n s, 1 seed"}}
        with open(out_path, "w") as f:
            json.dump(rec, f)
        print(json.dumps({k: rec.get(k) for k in ("instance", "n", "cost", "gap_pct", "runtime", "error")}),
              flush=True)


if __name__ == "__main__":
    main()
