"""Build viewer.html: inline every result into one self-contained three.js page.

Reads results/city.json, city_bound.json, scenery.json, xbench/*.json,
xbench_cuopt/*.json, xbench_full/*.json, the anchors and qaoa.json.

    python build_viewer.py            # writes viewer.html
"""

from __future__ import annotations

import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")

# Published reference: Tmax = 2.4 n s, X set, 10 seeds (Wouda, Lan & Kool 2024, INFORMS JoC)
REFS = {"PyVRP (published, full budget)": 0.22, "HGS-CVRP (published, full budget)": 0.11}


def load(name, default=None):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def main():
    city = load("city.json")
    cbound = load("city_bound_900.json") or load("city_bound.json", {})
    scen = load("scenery.json")
    qaoa = load("qaoa.json")
    anchors = [load(f) for f in ("e-n22-k4.json", "a-n32-k5.json", "x-n101-k25.json")]

    lat0, lon0 = scen["origin"]
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110574.0

    def q(lat, lon):  # same 0.5 m integer grid as scenery.json
        return [int(round((lon - lon0) * kx * 2)), int(round((lat - lat0) * ky * 2))]

    solvers = {}
    for name, r in city["results"].items():
        routes = []
        for rt, geom in zip(r.get("routes") or [], city["geometry"].get(name, [])):
            flat = []
            for p in geom:
                flat += q(*p)
            routes.append({"stops": rt, "load": int(sum(city["demand"][i] for i in rt)),
                           "m": int(sum(city["dist"][a][b] for a, b in zip([0, *rt], [*rt, 0]))),
                           "path": flat})
        solvers[name] = {"cost": r.get("cost"), "lower_bound": r.get("lower_bound"),
                         "trace": [[t, c] for t, c, _ in r.get("trace", [])], "routes": routes}
    minor = []
    for seg in city["roads"]:
        flat = []
        for p in seg:
            flat += q(*p)
        minor.append(flat)
    scen["roads"]["minor"] = minor

    lb_dir = cbound.get("directed_mip_bound_m")
    best = city["best_cost"]
    lb = max(city["lower_bound"] or 0, lb_dir or 0)
    city_out = {
        "name": city["meta"]["city"], "stamp": city["stamp"], "meta": city["meta"],
        "capacity": city["meta"]["capacity"], "stops": [q(*c) for c in city["coords"]],
        "demand": city["demand"], "solvers": solvers, "winner": city["winner"], "best": best,
        "bound": {"symmetric": city["lower_bound"], "directed": lb_dir, "best": lb,
                  "gap": round(100 * (best - lb) / best, 2),
                  "symmetric_gap": round(100 * (best - city["lower_bound"]) / best, 2),
                  "cuts": cbound.get("cuts"), "stamp": cbound.get("stamp")},
    }

    # X benchmark --------------------------------------------------------------
    cu = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(R, "xbench_cuopt", "*.json"))}
    f30 = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(R, "xbench_full30", "*.json"))}
    xs = []
    for p in sorted(glob.glob(os.path.join(R, "xbench", "*.json"))):
        b = json.load(open(p))
        coords = []
        vrp = os.path.join(HERE, "data", "X", b["instance"] + ".vrp")
        if os.path.exists(vrp):
            import vrplib

            coords = [int(v) for xy in vrplib.read_instance(vrp)["node_coord"] for v in xy]
        sol = {}
        for s, v in b["solvers"].items():
            tr = [[round(t / b["time_limit_s"], 4), round(100 * (c - b["bks"]) / b["bks"], 3)] for t, c in v["trace"]]
            sol[s] = {"cost": v["cost"], "gap": v["gap_pct"], "trace": tr}
        c = cu.get(b["instance"])
        if c and c.get("cost"):
            sol["cuopt"] = {"cost": c["cost"], "gap": c["gap_pct"], "trace": [], "mode": c.get("mode", "default")}
        fb = f30.get(b["instance"])
        if fb:
            v = fb["solvers"]["pyvrp"]
            sol["pyvrp_full"] = {"cost": v["cost"], "gap": v["gap_pct"],
                                 "trace": [[round(t / fb["time_limit_s"], 4), round(100 * (c - b["bks"]) / b["bks"], 3)] for t, c in v["trace"]]}
            winner, best, gap = "pyvrp_full", v["cost"], v["gap_pct"]
            if fb.get("best_routes"):
                b = {**b, "best_routes": fb["best_routes"]}
        else:
            winner, best, gap = b["winner"], b["best"], b["gap_pct"]
        if not fb and b["gap_pct"] is not None and b["gap_pct"] < gap:
            winner, best, gap = b["winner"], b["best"], b["gap_pct"]
        if "cuopt" in sol and sol["cuopt"]["gap"] is not None and sol["cuopt"]["gap"] < gap:
            winner, best, gap = "cuopt", sol["cuopt"]["cost"], sol["cuopt"]["gap"]
        xs.append({"name": b["instance"], "n": b["n"], "bks": b["bks"], "tl": b["time_limit_s"],
                   "winner": winner, "best": best, "gap": gap, "solvers": sol,
                   "coords": coords, "routes": b.get("best_routes")})
    xs.sort(key=lambda r: r["n"])
    full = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(R, "xbench_full", "*.json")))]
    summary = load("xbench_summary.json", {})

    data = {
        "city": city_out,
        "scenery": {k: scen[k] for k in ("buildings", "roads", "water", "shore", "piers", "stats")},
        "bench": {"instances": xs, "refs": REFS, "summary": summary,
                  "full": [{"name": f["instance"], "n": f["n"], "gap": f["gap_pct"], "tl": f["time_limit_s"]}
                           for f in full]},
        "anchors": [{"instance": a["instance"], "n": a["n_customers"], "bks": a["bks"], "best": a["best_cost"],
                     "winner": a["winner"], "lower_bound": a["lower_bound"],
                     "proven": a.get("proven_optimal", False), "gap": a.get("certified_gap_pct"),
                     "time": a["stamp"]["time_limit_s"]} for a in anchors],
        "qaoa": qaoa,
    }
    with open(os.path.join(HERE, "viewer_template.html")) as f:
        html = f.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = os.path.join(HERE, "viewer.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
