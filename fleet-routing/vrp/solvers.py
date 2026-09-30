"""Three CVRP solvers with a common interface.

Each solver takes an Instance and a time limit, and returns a dict:
    {"solver", "cost", "routes", "trace": [[t, incumbent, lower_bound|None], ...],
     "optimal": bool, "runtime"}

- pyvrp:   hybrid genetic search (HGS), state of the art heuristic for CVRP
- ortools: OR-Tools routing, guided local search
- highs:   exact two-index MIP on HiGHS with rounded-capacity cuts added in a
           cutting-plane loop. It is the only one of the three that produces a
           lower bound, which is what turns "a good answer" into "a certified gap".
"""

from __future__ import annotations

import math
import time

import numpy as np

from .instance import Instance, check


# --------------------------------------------------------------------- PyVRP
def solve_pyvrp(inst: Instance, time_limit: float, seed: int = 1) -> dict:
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime

    m = Model()
    m.add_vehicle_type(num_available=inst.n - 1, capacity=int(inst.capacity))
    locs = [m.add_location(x=float(x), y=float(y)) for x, y in inst.coords]
    m.add_depot(locs[0])
    for i in range(1, inst.n):
        m.add_client(locs[i], delivery=int(inst.demand[i]))
    for i, a in enumerate(locs):
        for j, b in enumerate(locs):
            if i != j:
                m.add_edge(a, b, distance=int(inst.dist[i, j]))

    t0 = time.perf_counter()
    res = m.solve(stop=MaxRuntime(time_limit), seed=seed, display=False)
    runtime = time.perf_counter() - t0

    trace, best, t = [], math.inf, 0.0
    for dt, d in zip(res.stats.runtimes, res.stats.data):
        t += dt
        if d.best_feas and d.best_cost < best:
            best = d.best_cost
            trace.append([round(t, 3), int(best), None])

    routes = [[int(a.idx) + 1 for a in r if a.is_client()] for r in res.best.routes()]
    cost = check(inst, routes)
    return {"solver": "pyvrp", "cost": cost, "routes": routes, "trace": trace,
            "optimal": False, "runtime": runtime}


# ------------------------------------------------------------------ OR-Tools
def solve_ortools(inst: Instance, time_limit: float, extra_vehicles: int = 5) -> dict:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    nv = inst.min_vehicles + extra_vehicles
    mgr = pywrapcp.RoutingIndexManager(inst.n, nv, 0)
    routing = pywrapcp.RoutingModel(mgr)
    dist = inst.dist

    def d_cb(i, j):
        return int(dist[mgr.IndexToNode(i), mgr.IndexToNode(j)])

    t_idx = routing.RegisterTransitCallback(d_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(t_idx)
    dem = inst.demand
    q_idx = routing.RegisterUnaryTransitCallback(lambda i: int(dem[mgr.IndexToNode(i)]))
    routing.AddDimensionWithVehicleCapacity(q_idx, 0, [int(inst.capacity)] * nv, True, "load")

    trace, t0 = [], time.perf_counter()
    best = [math.inf]

    def on_solution():
        c = routing.CostVar().Value()
        if c < best[0]:
            best[0] = c
            trace.append([round(time.perf_counter() - t0, 3), int(c), None])

    routing.AddAtSolutionCallback(on_solution)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(int(time_limit * 1000))
    sol = routing.SolveWithParameters(p)
    runtime = time.perf_counter() - t0
    if sol is None:
        return {"solver": "ortools", "cost": None, "routes": [], "trace": trace,
                "optimal": False, "runtime": runtime}

    routes = []
    for v in range(nv):
        i, r = routing.Start(v), []
        i = sol.Value(routing.NextVar(i))
        while not routing.IsEnd(i):
            r.append(mgr.IndexToNode(i))
            i = sol.Value(routing.NextVar(i))
        if r:
            routes.append(r)
    cost = check(inst, routes)
    return {"solver": "ortools", "cost": cost, "routes": routes, "trace": trace,
            "optimal": False, "runtime": runtime}


# --------------------------------------------------------------------- HiGHS
def _components(n, edges, x, thr):
    """Connected components of customers 1..n-1 over edges with x >= thr."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k, (i, j) in enumerate(edges):
        if i and j and x[k] >= thr:
            parent[find(i)] = find(j)
    comp = {}
    for v in range(1, n):
        comp.setdefault(find(v), []).append(v)
    return list(comp.values())


def _greedy_sets(n, edges, x, demand, Q, max_sets=60):
    """Greedy growth separation for rounded capacity cuts on a fractional x.

    From every customer seed, grow S by the vertex most strongly connected to it
    and record S whenever x(delta(S)) = 2|S| - 2 x(E(S)) < 2 ceil(d(S)/Q).
    Heuristic (CVRPSEP does this better), but cheap and effective at n ~ 100.
    """
    X = np.zeros((n, n))
    for k, (i, j) in enumerate(edges):
        X[i, j] = X[j, i] = x[k]
    found = {}
    for seed in range(1, n):
        inS = np.zeros(n, dtype=bool)
        inS[seed] = True
        conn = X[:, seed].copy()
        conn[0] = -1
        xE, dS, size = 0.0, int(demand[seed]), 1
        while size < n - 1:
            c = np.where(inS, -1, conn)
            v = int(np.argmax(c))
            if c[v] <= 1e-9:
                break
            inS[v] = True
            xE += conn[v]
            conn += X[:, v]
            conn[0] = -1
            dS += int(demand[v])
            size += 1
            viol = 2 * math.ceil(dS / Q) - (2 * size - 2 * xE)
            if viol > 1e-4:
                key = frozenset(np.flatnonzero(inS).tolist())
                found[key] = max(found.get(key, 0), viol)
    return [sorted(k) for k, _ in sorted(found.items(), key=lambda kv: -kv[1])[:max_sets]]


def solve_highs(inst: Instance, time_limit: float, lp_rounds: int = 200,
                incumbent: list[list[int]] | None = None) -> dict:
    """Two-index CVRP MIP with rounded capacity inequalities, cutting-plane loop.

    x_ij for i<j; depot edges may take value 2 (a single-customer route).
      degree:  x(delta(i)) = 2 for every customer i
      depot:   x(delta(0)) >= 2*ceil(sum d / Q)
      RCI:     x(delta(S)) >= 2*ceil(d(S)/Q)    added lazily for violated S
    For an integer solution, checking the connected components is exact, so
    when the MIP returns a solution with no violated component it is optimal.
    """
    import highspy

    n, Q, d = inst.n, inst.capacity, inst.demand
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    eidx = {e: k for k, e in enumerate(edges)}
    m = len(edges)
    inc = [[] for _ in range(n)]
    for k, (i, j) in enumerate(edges):
        inc[i].append(k)
        inc[j].append(k)

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    inf = highspy.kHighsInf
    # Undirected model. For asymmetric (road) distances, min(d_ij, d_ji) makes the
    # MIP a relaxation: its bound stays valid, and routes are re-costed directed.
    sym = np.minimum(inst.dist, inst.dist.T)
    asym = bool((inst.dist != inst.dist.T).any())
    cost = np.array([float(sym[i, j]) for i, j in edges])
    ub = np.array([2.0 if i == 0 else 1.0 for i, _ in edges])
    h.addCols(m, cost, np.zeros(m), ub, 0, np.array([], dtype=np.int32),
              np.array([], dtype=np.int32), np.array([]))
    for v in range(1, n):
        h.addRow(2.0, 2.0, len(inc[v]), np.array(inc[v], dtype=np.int32), np.ones(len(inc[v])))
    h.addRow(2.0 * inst.min_vehicles, inf, len(inc[0]), np.array(inc[0], dtype=np.int32),
             np.ones(len(inc[0])))

    added = set()

    def cut_for(S):
        key = frozenset(S)
        if key in added:
            return None
        Sset = set(S)
        idx = [eidx[(min(a, b), max(a, b))] for a in S for b in range(n) if b not in Sset]
        rhs = 2.0 * math.ceil(d[list(S)].sum() / Q)
        return key, idx, rhs

    def separate(x, thresholds, greedy=False):
        new = 0
        cands = [S for thr in thresholds for S in _components(n, edges, x, thr)]
        if greedy:
            cands += _greedy_sets(n, edges, x, d, Q)
        for S in cands:
            c = cut_for(S)
            if c is None:
                continue
            key, idx, rhs = c
            if x[idx].sum() < rhs - 1e-6:
                h.addRow(rhs, inf, len(idx), np.array(idx, dtype=np.int32), np.ones(len(idx)))
                added.add(key)
                new += 1
        return new

    t0 = time.perf_counter()
    trace, lb = [], 0.0

    # Phase 1: strengthen the LP relaxation with heuristic fractional separation.
    for _ in range(lp_rounds):
        if time.perf_counter() - t0 > 0.3 * time_limit:
            break
        h.run()
        x = np.array(h.getSolution().col_value)
        lb = max(lb, h.getInfo().objective_function_value)
        trace.append([round(time.perf_counter() - t0, 3), None, round(lb, 2)])
        if separate(x, (0.99, 0.75, 0.5, 0.3, 0.1), greedy=True) == 0:
            break

    # Phase 2: MIP + exact separation on integer solutions.
    integrality = np.array([highspy.HighsVarType.kInteger] * m)
    h.changeColsIntegrality(m, np.arange(m, dtype=np.int32), integrality)
    optimal, routes, best, xs = False, [], None, None
    if incumbent:
        # A verified feasible solution from another racer: an upper bound and a MIP start.
        routes, best = incumbent, check(inst, incumbent)
        xs = np.zeros(m)
        for r in incumbent:
            path = [0, *r, 0]
            for a, b in zip(path, path[1:]):
                xs[eidx[(min(a, b), max(a, b))]] += 1
        trace.append([round(time.perf_counter() - t0, 3), best, round(lb, 2)])
    while True:
        if best is not None and math.ceil(lb - 1e-6) >= best:
            optimal = True  # integer costs: bound meets incumbent
            trace.append([round(time.perf_counter() - t0, 3), best, float(best)])
            break
        left = time_limit - (time.perf_counter() - t0)
        if left <= 0.5:
            break
        h.setOptionValue("time_limit", float(left))
        if xs is not None:  # re-offer the MIP start; rows added since may drop it
            sol = h.getSolution()
            sol.col_value = xs.tolist()
            sol.value_valid = True
            h.setSolution(sol)
        h.run()
        info = h.getInfo()
        status = h.getModelStatus()
        x = np.array(h.getSolution().col_value)
        if status == highspy.HighsModelStatus.kOptimal:
            lb = max(lb, info.objective_function_value)
        else:
            lb = max(lb, info.mip_dual_bound)
        if status != highspy.HighsModelStatus.kOptimal:
            trace.append([round(time.perf_counter() - t0, 3), None, round(lb, 2)])
            break
        if separate(np.round(x), (0.5,)) == 0:
            found = _orient(inst, _extract_routes(n, edges, np.round(x).astype(int)))
            c = check(inst, found)
            if best is None or c < best:
                routes, best = found, c
            if not asym:
                lb = float(best)
            optimal = math.ceil(lb - 1e-6) >= best
            trace.append([round(time.perf_counter() - t0, 3), best, round(lb, 2)])
            break
        trace.append([round(time.perf_counter() - t0, 3), None, round(lb, 2)])

    return {"solver": "highs", "cost": best, "routes": routes, "trace": trace,
            "optimal": optimal, "lower_bound": round(lb, 2), "cuts": len(added),
            "incumbent_from_race": bool(incumbent),
            "runtime": time.perf_counter() - t0}


def _orient(inst, routes):
    """Drive each route in its cheaper direction (matters for one-way streets)."""
    from .instance import route_cost

    return [r if route_cost(inst, [r]) <= route_cost(inst, [r[::-1]]) else r[::-1] for r in routes]


def _extract_routes(n, edges, x):
    adj = [[] for _ in range(n)]
    for k, (i, j) in enumerate(edges):
        for _ in range(int(x[k])):
            adj[i].append(j)
            adj[j].append(i)
    routes, used = [], set()
    for start in list(adj[0]):
        if start in used:
            continue
        r, prev, cur = [], 0, start
        while cur != 0:
            r.append(cur)
            used.add(cur)
            nxt = [v for v in adj[cur] if v != prev]
            if len(nxt) == 0:  # single-customer route (x_0i = 2)
                break
            prev, cur = cur, nxt[0]
        routes.append(r)
    return routes


SOLVERS = {"pyvrp": solve_pyvrp, "ortools": solve_ortools, "highs": solve_highs}
