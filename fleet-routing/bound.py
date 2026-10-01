"""Certified lower bound for the asymmetric (one-way street) Chicago instance.

The race's HiGHS bound symmetrised road distances (min(d_ij, d_ji)), which throws
away one-way information. Here the model is the directed two-index ACVRP:

    min  sum d_ij x_ij              x_ij binary, i != j (true asymmetric metres)
    s.t. out(i) = in(i) = 1        every customer
         out(0) = in(0) >= k_min   depot
         x(delta+(S)) >= ceil(d(S)/Q)  rounded capacity cuts (RCI), added lazily

Fractional separation uses the symmetric view y_ij = x_ij + x_ji (a directed cut
has exactly half the symmetric cut value), reusing the race's component and
greedy separators. Phase 2 runs the MIP with PyVRP's routes as a start and exact
separation on integer solutions; the reported bound is HiGHS's MIP dual bound,
which is valid for the full model because every added row is a valid inequality.

    python bound.py --city results/city.json --time 1500 --threads 2 --out results/city_bound.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="results/city.json")
    ap.add_argument("--time", type=float, default=600)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--lp-rounds", type=int, default=400)
    ap.add_argument("--out", default="results/city_bound.json")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    import highspy

    from vrp.solvers import _components, _greedy_sets

    c = json.load(open(a.city))
    D = np.array(c["dist"], dtype=float)
    d = np.array(c["demand"], dtype=np.int64)
    Q = int(c["meta"]["capacity"])
    n = len(d)
    kmin = math.ceil(d.sum() / Q)
    best = int(c["best_cost"])
    best_routes = c["best_routes"]

    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    aidx = {e: k for k, e in enumerate(arcs)}
    m = len(arcs)
    out_of = [[] for _ in range(n)]
    into = [[] for _ in range(n)]
    for k, (i, j) in enumerate(arcs):
        out_of[i].append(k)
        into[j].append(k)
    sedges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", a.threads)
    inf = highspy.kHighsInf
    cost = np.array([D[i, j] for i, j in arcs])
    h.addCols(m, cost, np.zeros(m), np.ones(m), 0, np.array([], dtype=np.int32),
              np.array([], dtype=np.int32), np.array([]))
    for v in range(1, n):
        h.addRow(1.0, 1.0, len(out_of[v]), np.array(out_of[v], dtype=np.int32), np.ones(len(out_of[v])))
        h.addRow(1.0, 1.0, len(into[v]), np.array(into[v], dtype=np.int32), np.ones(len(into[v])))
    h.addRow(float(kmin), inf, len(out_of[0]), np.array(out_of[0], dtype=np.int32), np.ones(len(out_of[0])))
    idx = out_of[0] + into[0]
    h.addRow(0.0, 0.0, len(idx), np.array(idx, dtype=np.int32),
             np.array([1.0] * len(out_of[0]) + [-1.0] * len(into[0])))

    added = set()

    def add_cut(S):
        key = frozenset(S)
        if key in added:
            return False
        Sset = set(S)
        cols = [aidx[(i, j)] for i in S for j in range(n) if j not in Sset]
        rhs = float(math.ceil(d[list(S)].sum() / Q))
        h.addRow(rhs, inf, len(cols), np.array(cols, dtype=np.int32), np.ones(len(cols)))
        added.add(key)
        return True

    def violated(x, S):
        Sset = set(S)
        val = sum(x[aidx[(i, j)]] for i in S for j in range(n) if j not in Sset)
        return val < math.ceil(d[list(S)].sum() / Q) - 1e-6

    def mincut_sets(y):
        """Exact fractional-capacity separation (one max-flow per customer seed).

        Source s feeds every customer i with 2 d_i / Q; the depot is the sink. With the
        seed forced on the source side, the min cut is min_S x(delta(S)) - 2 d(S)/Q + const,
        so its source side is the customer set most likely to violate a capacity cut.
        """
        import networkx as nx

        G = nx.DiGraph()
        for k, (i, j) in enumerate(sedges):
            if y[k] > 1e-9:
                G.add_edge(i, j, capacity=float(y[k]))
                G.add_edge(j, i, capacity=float(y[k]))
        for i in range(1, n):
            G.add_edge("s", i, capacity=2.0 * d[i] / Q)
        found = []
        for seed in range(1, n):
            G.add_edge("s", seed, capacity=1e9)
            try:
                _, (side, _) = nx.minimum_cut(G, "s", 0)
            finally:
                G["s"][seed]["capacity"] = 2.0 * d[seed] / Q
            S = sorted(v for v in side if v != "s")
            if S:
                found.append(S)
        return found

    def sym(x):
        y = np.zeros(len(sedges))
        for k, (i, j) in enumerate(sedges):
            y[k] = x[aidx[(i, j)]] + x[aidx[(j, i)]]
        return y

    t0 = time.perf_counter()
    trace = []
    lb = 0.0
    # Phase 1: LP + cutting planes.
    for r in range(a.lp_rounds):
        if time.perf_counter() - t0 > 0.35 * a.time:
            break
        h.run()
        x = np.array(h.getSolution().col_value)
        lb = max(lb, h.getInfo().objective_function_value)
        trace.append([round(time.perf_counter() - t0, 1), round(lb, 1), len(added), "lp"])
        y = sym(x)
        cands = [S for thr in (0.99, 0.75, 0.5, 0.3, 0.1) for S in _components(n, sedges, y, thr)]
        cands += _greedy_sets(n, sedges, y, d, Q, max_sets=120)
        new = sum(add_cut(S) for S in cands if violated(x, S))
        if new == 0:  # heuristics dry: fall back to the exact fractional separator
            new = sum(add_cut(S) for S in mincut_sets(y) if violated(x, S))
        if new == 0:
            break
    lp_bound = lb
    lp_time = time.perf_counter() - t0

    # Phase 2: MIP from PyVRP's solution; exact separation on integer points.
    h.changeColsIntegrality(m, np.arange(m, dtype=np.int32),
                            np.array([highspy.HighsVarType.kInteger] * m))
    xs = np.zeros(m)
    for rt in best_routes:
        p = [0, *rt, 0]
        for i, j in zip(p, p[1:]):
            xs[aidx[(i, j)]] = 1
    optimal = False
    while True:
        left = a.time - (time.perf_counter() - t0)
        if left < 5:
            break
        h.setOptionValue("time_limit", float(left))
        sol = h.getSolution()
        sol.col_value = xs.tolist()
        sol.value_valid = True
        h.setSolution(sol)
        h.run()
        info, status = h.getInfo(), h.getModelStatus()
        lb = max(lb, info.mip_dual_bound if status != highspy.HighsModelStatus.kOptimal
                 else info.objective_function_value)
        trace.append([round(time.perf_counter() - t0, 1), round(lb, 1), len(added), str(status)])
        if status != highspy.HighsModelStatus.kOptimal:
            break
        x = np.round(np.array(h.getSolution().col_value))
        comps = _components(n, sedges, sym(x), 0.5)
        # exact for integer x: a customer component is a route; check depot-free cycles and loads
        new = sum(add_cut(S) for S in comps if violated(x, S))
        if new == 0:
            cost_int = float(cost @ x)
            if cost_int <= best:
                optimal = True
                lb = cost_int
            break
        if math.ceil(lb - 1e-6) >= best:
            optimal = True
            break

    out = {
        "instance": c["instance"], "n_customers": n - 1, "capacity": Q, "k_min": kmin,
        "best_cost_m": best, "best_by": c["winner"],
        "symmetric_bound_m": c["lower_bound"],
        "directed_lp_bound_m": round(lp_bound, 1), "lp_rounds_s": round(lp_time, 1),
        "directed_mip_bound_m": round(lb, 1), "proven_optimal": optimal,
        "certified_gap_pct": round(100 * (best - lb) / best, 3),
        "symmetric_gap_pct": round(100 * (best - c["lower_bound"]) / best, 3),
        "cuts": len(added), "trace": trace,
        "stamp": {"date": time.strftime("%Y-%m-%d"), "time_limit_s": a.time, "threads": a.threads,
                  "solver": f"HiGHS {highspy.__version__ if hasattr(highspy, '__version__') else '1.15'}"},
    }
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(json.dumps({k: v for k, v in out.items() if k != "trace"}))


if __name__ == "__main__":
    main()
