"""Real-map scenario: one depot and N delivery stops on a real road network.

Road distances come from OpenStreetMap through osmnx (shortest driving
distance in metres, respecting one-way streets). The route geometries are the
actual street polylines, so the viewer draws vehicles driving real roads.

    python city.py --place "Chicago Loop" --stops 80 --time 60 --out results/city.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

CITIES = {
    # name: (lat, lon, radius_m, depot offset description)
    "chicago": (41.8810, -87.6320, 2600, "Chicago Loop and near neighbourhoods"),
}


def build(city="chicago", n_stops=80, seed=7, capacity=100):
    import networkx as nx
    import osmnx as ox
    from scipy.sparse.csgraph import dijkstra

    lat, lon, radius, label = CITIES[city]
    t0 = time.perf_counter()
    G = ox.graph_from_point((lat, lon), dist=radius, network_type="drive", simplify=True)
    G = ox.truncate.largest_component(G, strongly=True)
    t_fetch = time.perf_counter() - t0

    nodes = list(G.nodes)
    rng = np.random.default_rng(seed)
    ys = np.array([G.nodes[v]["y"] for v in nodes])
    xs = np.array([G.nodes[v]["x"] for v in nodes])
    # Depot: the node closest to the south-west corner, like a distribution centre at the edge.
    depot = nodes[int(np.argmin((ys - ys.min()) ** 2 + (xs - xs.min()) ** 2))]
    stops = list(rng.choice([v for v in nodes if v != depot], size=n_stops, replace=False))
    sites = [depot, *stops]

    idx = {v: k for k, v in enumerate(nodes)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="length", format="csr")
    D, P = dijkstra(A, directed=True, indices=[idx[s] for s in sites], return_predecessors=True)
    dist = np.rint(D[:, [idx[s] for s in sites]]).astype(np.int64)  # metres, asymmetric

    def path(i, j):
        """Street polyline (lat, lon) from site i to site j."""
        target, seq = idx[sites[j]], []
        while target != idx[sites[i]] and target >= 0:
            seq.append(target)
            target = P[i, target]
        seq.append(idx[sites[i]])
        seq.reverse()
        out = []
        for a, b in zip(seq, seq[1:]):
            u, v = nodes[a], nodes[b]
            data = min(G.get_edge_data(u, v).values(), key=lambda e: e.get("length", 1e18))
            if "geometry" in data:
                pts = [(y, x) for x, y in data["geometry"].coords]
            else:
                pts = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
            out.extend(pts if not out else pts[1:])
        return out

    demand = [0, *rng.integers(5, 26, size=n_stops).tolist()]
    coords = [(G.nodes[s]["y"], G.nodes[s]["x"]) for s in sites]

    # Light road backdrop for the viewer: every street segment, decimated to 5 decimals.
    roads = []
    for u, v, data in G.edges(data=True):
        if "geometry" in data:
            pts = [(round(y, 5), round(x, 5)) for x, y in data["geometry"].coords]
        else:
            pts = [(round(G.nodes[u]["y"], 5), round(G.nodes[u]["x"], 5)),
                   (round(G.nodes[v]["y"], 5), round(G.nodes[v]["x"], 5))]
        roads.append(pts)

    meta = {"city": label, "graph_nodes": len(nodes), "graph_edges": G.number_of_edges(),
            "fetch_s": round(t_fetch, 1), "capacity": capacity, "seed": seed,
            "distance": "OSM shortest driving distance, metres (asymmetric, one-way aware)"}
    return coords, demand, capacity, dist, path, roads, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="chicago")
    ap.add_argument("--stops", type=int, default=80)
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--out", default="results/city.json")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from race import incumbent_at, race, stamp
    from vrp.instance import from_matrix

    coords, demand, cap, dist, path, roads, meta = build(a.city, a.stops)
    inst = from_matrix(f"{a.city}-{a.stops}", coords, demand, cap, dist)
    out = race(inst, a.time)

    # Street geometry for every route each solver found.
    geo = {}
    for name, r in out["results"].items():
        geo[name] = [[p for k, (i, j) in enumerate(zip([0, *rt], [*rt, 0]))
                      for p in (path(i, j) if k == 0 else path(i, j)[1:])]
                     for rt in (r.get("routes") or [])]
    out.update({"instance": inst.name, "n_customers": a.stops, "coords": coords,
                "demand": demand, "meta": meta, "dist": dist.tolist(), "geometry": geo, "roads": roads,
                "stamp": stamp(a.time)})
    out["table"] = {n: {**{f"{t}s": incumbent_at(r.get("trace", []), t) for t in (1, 5, 10, 30, 60)},
                        "final": r.get("cost"), "routes": len(r.get("routes") or []),
                        "lower_bound": r.get("lower_bound")}
                    for n, r in out["results"].items()}
    if out["lower_bound"] and out["best_cost"]:
        out["certified_gap_pct"] = round(100 * (out["best_cost"] - out["lower_bound"]) / out["best_cost"], 3)
    os.makedirs("results", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(json.dumps({"meta": meta, "winner": out["winner"], "best_m": out["best_cost"],
                      "lower_bound": out["lower_bound"], "gap": out.get("certified_gap_pct"),
                      "table": out["table"]}))
