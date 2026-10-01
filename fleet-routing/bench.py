"""CVRPLIB X benchmark (Uchoa et al. 2017): gap to best-known solution (BKS).

The published protocol (PyVRP, HGS-CVRP) gives each instance Tmax = 2.4 n seconds
(n = customers) on a reference CPU. This harness scales that budget by --factor,
races the CPU solvers in separate single-thread processes, and checkpoints one
JSON per instance so a capped queue job can resume where it stopped.

    python bench.py --list xsub.txt --dir X --factor 0.1 --out results/xbench
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import re
import time
from datetime import date


def bks_of(path):
    import vrplib

    return float(vrplib.read_solution(path)["cost"])


def compress(trace, keep=60):
    """Keep at most `keep` incumbent points (first, last and an even spread)."""
    pts = [[round(t, 2), c] for t, c, _ in trace if c is not None]
    if len(pts) <= keep:
        return pts
    step = (len(pts) - 1) / (keep - 1)
    return [pts[round(i * step)] for i in range(keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--factor", type=float, default=0.1)
    ap.add_argument("--solvers", default="pyvrp,ortools")
    ap.add_argument("--out", default="results/xbench")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=10**9)
    ap.add_argument("--host", default="qBraid pool gpu-l4 box (cgroup ~5 CPUs), 1 thread per solver")
    ap.add_argument("--sequential", action="store_true",
                    help="run the solvers one after another (same per-solver budget) on a single core")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    from race import race
    from vrp.instance import load

    names = [l.strip() for l in open(a.list) if l.strip()][a.start:a.stop]
    os.makedirs(a.out, exist_ok=True)
    for name in names:
        out_path = os.path.join(a.out, f"{name}.json")
        if os.path.exists(out_path):
            continue
        inst = load(os.path.join(a.dir, f"{name}.vrp"))
        bks = bks_of(os.path.join(a.dir, f"{name}.sol"))
        n = inst.n - 1
        tlim = round(a.factor * 2.4 * n, 1)
        t0 = time.time()
        if a.sequential:
            parts = [race(inst, tlim, (s,)) for s in a.solvers.split(",")]
            results = {k: v for p in parts for k, v in p["results"].items()}
            feas = [r for r in results.values() if r.get("cost") is not None]
            win = min(feas, key=lambda r: r["cost"]) if feas else None
            res = {"results": results, "winner": win["solver"] if win else None,
                   "best_cost": win["cost"] if win else None, "best_routes": win["routes"] if win else None}
        else:
            res = race(inst, tlim, tuple(a.solvers.split(",")))
        rec = {
            "instance": name, "n": n, "bks": bks, "factor": a.factor, "time_limit_s": tlim,
            "winner": res["winner"], "best": res["best_cost"],
            "gap_pct": round(100 * (res["best_cost"] - bks) / bks, 4) if res["best_cost"] else None,
            "solvers": {
                s: {"cost": r.get("cost"),
                    "gap_pct": round(100 * (r["cost"] - bks) / bks, 4) if r.get("cost") else None,
                    "routes": len(r.get("routes") or []),
                    "trace": compress(r.get("trace", [])),
                    "error": r.get("error")}
                for s, r in res["results"].items()
            },
            "best_routes": res["best_routes"],
            "stamp": {"date": date.today().isoformat(), "host": a.host,
                      "python": platform.python_version(), "wall_s": round(time.time() - t0, 1),
                      "protocol": f"Tmax = {a.factor} x 2.4 n s, 1 seed, 1 thread per solver"},
        }
        with open(out_path, "w") as f:
            json.dump(rec, f)
        print(json.dumps({k: rec[k] for k in ("instance", "n", "bks", "best", "gap_pct", "winner")}),
              {s: v["gap_pct"] for s, v in rec["solvers"].items()}, flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
