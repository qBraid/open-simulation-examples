"""Solver racing: run several CVRP solvers at once and keep the best answer.

Each solver runs in its own process under the same wall-clock budget. The
processes are separate for two reasons. It is the local shape of racing across
qBraid instances (one solver per machine). And HiGHS (highspy 1.15) and
OR-Tools 9.15 bundle incompatible HiGHS symbols, so they cannot be imported
into the same Python process.

    python race.py data/X-n101-k25.vrp --time 60 --bks 27591 --out results/x-n101-k25.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import time
from datetime import date


def _worker(name, inst, time_limit, q, kwargs=None):
    from vrp.solvers import SOLVERS

    try:
        q.put((name, SOLVERS[name](inst, time_limit, **(kwargs or {}))))
    except Exception as e:  # report, never hang the race
        q.put((name, {"solver": name, "cost": None, "error": repr(e), "trace": []}))


def race(inst, time_limit, solvers=("pyvrp", "ortools", "highs")):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(s, inst, time_limit, q)) for s in solvers]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    results = {}
    for _ in procs:
        name, res = q.get(timeout=time_limit + 120)
        results[name] = res
    for p in procs:
        p.join(timeout=10)
    wall = time.perf_counter() - t0
    feasible = [r for r in results.values() if r.get("cost") is not None]
    def first_reach(r):  # tie-break: who reached the best cost first
        return min((t for t, c, _ in r.get("trace", []) if c is not None and c <= r["cost"]),
                   default=float("inf"))

    winner = min(feasible, key=lambda r: (r["cost"], first_reach(r))) if feasible else None
    bounds = [r["lower_bound"] for r in results.values() if r.get("lower_bound") is not None]
    return {
        "results": results,
        "proven_optimal": any(r.get("optimal") for r in results.values()),
        "winner": winner["solver"] if winner else None,
        "best_cost": winner["cost"] if winner else None,
        "best_routes": winner["routes"] if winner else None,
        "lower_bound": max(bounds) if bounds else None,
        "wall_time": round(wall, 1),
    }


def incumbent_at(trace, t):
    """Best feasible cost found by time t (None if none yet)."""
    best = None
    for tt, c, _ in trace:
        if tt <= t and c is not None:
            best = c if best is None else min(best, c)
    return best


def stamp(time_limit):
    return {
        "date": date.today().isoformat(),
        "host": "qBraid subscription pod (shared, 8 vCPU tier)",
        "python": platform.python_version(),
        "processes": 3,
        "threads_per_solver": 1,
        "time_limit_s": time_limit,
        "cost_credits": 0,
        "env": "fleet-routing/requirements.txt",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--bks", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--solvers", default="pyvrp,ortools,highs")
    ap.add_argument("--certify", type=float, default=0,
                    help="after the race, give HiGHS this many seconds with the winner as incumbent")
    a = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from vrp.instance import load

    inst = load(a.instance)
    out = race(inst, a.time, tuple(a.solvers.split(",")))
    if a.certify and out["best_routes"] and not out["proven_optimal"]:
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_worker, args=("highs", inst, a.certify, q,
                                               {"incumbent": out["best_routes"]}))
        p.start()
        _, cert = q.get(timeout=a.certify + 120)
        p.join(timeout=10)
        cert["solver"] = "highs-certify"
        out["certify"] = {k: cert.get(k) for k in ("optimal", "lower_bound", "cuts", "runtime", "trace")}
        out["lower_bound"] = max(out["lower_bound"] or 0, cert.get("lower_bound") or 0)
        out["proven_optimal"] = bool(cert.get("optimal"))
    out["instance"] = inst.name
    out["n_customers"] = inst.n - 1
    out["bks"] = a.bks
    out["stamp"] = stamp(a.time)
    checkpoints = [t for t in (1, 5, 10, 30, 60, 120, 300) if t <= a.time]
    table = {}
    for name, r in out["results"].items():
        row = {f"{t}s": incumbent_at(r.get("trace", []), t) for t in checkpoints}
        row["final"] = r.get("cost")
        if a.bks and r.get("cost"):
            row["gap_final_pct"] = round(100 * (r["cost"] - a.bks) / a.bks, 3)
        if r.get("lower_bound") is not None:
            row["lower_bound"] = r["lower_bound"]
        table[name] = row
    out["table"] = table
    if out["lower_bound"] and out["best_cost"]:
        out["certified_gap_pct"] = round(100 * (out["best_cost"] - out["lower_bound"]) / out["best_cost"], 3)
    print(json.dumps({"winner": out["winner"], "best": out["best_cost"], "bks": a.bks,
                      "lower_bound": out["lower_bound"], "proven_optimal": out["proven_optimal"],
                      "certified_gap_pct": out.get("certified_gap_pct"), "table": table}))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f)
